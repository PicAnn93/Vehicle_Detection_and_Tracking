# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load and save model variable
load_model = False
save_model = True
use_pretrained = False

# model checkpoint file name
if use_pretrained:
    checkpoint_file = "best_checkpoint.pth.tar"
else :
    checkpoint_file = "checkpoint.pth.tar"

# Anchor boxes for each feature map scaled between 0 and 1
# 3 feature maps at 3 different scales based on YOLOv3 paper
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]

# Batch size for training
batch_size = 16

# Learning rate for training
leanring_rate = 1e-5

# Number of epochs for training
epochs = 200

# Image size
image_size = 416

# Grid cell sizes
s = [image_size // 32, image_size // 16, image_size // 8]

# Class labels
class_labels = ["car", "threewheel", "bus", "truck", "motorbike", "van"]


def iou(box1, box2, is_pred=True):
    if is_pred:
        # Calculates the coordinates of the top-left (x1, y1)
        #  and bottom-right (x2, y2) corners for both boxes.

        # Box coordinates of prediction
        b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2

        # Box coordinates of ground truth
        b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2


        # Get the coordinates of the intersection rectangle by taking the
        # maximum of the top-left coordinates and the minimum of the bottom-right coordinates.
        x1 = torch.max(b1_x1, b2_x1)
        y1 = torch.max(b1_y1, b2_y1)
        x2 = torch.min(b1_x2, b2_x2)
        y2 = torch.min(b1_y2, b2_y2)
        # Make sure the intersection is at least 0
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        # clamp(0) is used to ensure that the intersection area
        # is not negative in cases where the boxes do not overlap.

        # Calculate the union area by adding the areas of the two boxes and
        # subtracting the intersection area to avoid double-counting the overlapping region.
        box1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
        box2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))
        union = box1_area + box2_area - intersection

        # Calculate the IoU score by dividing the intersection area by the
        # union area.
        # A small epsilon is added to the union to prevent division by zero.
        epsilon = 1e-6
        iou_score = intersection / (union + epsilon)

        # Return IoU score
        return iou_score

    else:
        # The case where is_pred is false is used to handle the case
        # where box1 and box2 are in the format [width, height].

        # Calculate intersection area
        intersection_area = torch.min(box1[..., 0], box2[..., 0]) * \
                            torch.min(box1[..., 1], box2[..., 1])

        # Calculate union area
        box1_area = box1[..., 0] * box1[..., 1]
        box2_area = box2[..., 0] * box2[..., 1]
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU score
        iou_score = intersection_area / union_area

        # Return IoU score
        return iou_score

def nms(bboxes, iou_threshold, threshold):
    # Filter out bounding boxes with confidence below the threshold
    bboxes = [box for box in bboxes if box[1] > threshold]

    # Sort by confidence in descending order based on their confidence scores
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    bboxes_nms = []

    while bboxes:
        # Take the box with highest confidence
        chosen_box = bboxes.pop(0)
        bboxes_nms.append(chosen_box)

        # Remove boxes with high IoU with the chosen box
        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0] or  # Different class
            iou(torch.tensor(chosen_box[2:]), torch.tensor(box[2:])) < iou_threshold
        ]

    return bboxes_nms


def convert_cells_to_bboxes(predictions, anchors, s, is_predictions=True):
    # Extract basic information from the input
    batch_size = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5] # raw boxes coordinate predictions

    if is_predictions: # If we consider the model's raw input
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2) # reshape anchors to match the dimensions of the predictions
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2]) # Apply a sigmoid function
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors # scale the predicted width and height based on the anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        # Find the class with the highest probability among the class predictions
        # and gets its index
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else: # we are considering the ground truth labels
        # Take score and class label
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    # Create cell indices
    cell_indices = (
        torch.arange(s)
        .repeat(predictions.shape[0], 3, s, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )

    # Calculate coordinates
    # converts the cell-relative predictions into image-relative coordinates (normalized between 0 and 1)

    x = 1 / s * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / s * (box_predictions[..., 1:2] +
                 cell_indices.permute(0, 1, 3, 2, 4))
    width_height = 1 / s * box_predictions[..., 2:4]

    # Concatenate the best_class, scores, x, y, and width_height tensors along the last dimension to form the
    # complete bounding box information for each anchor box in each cell and reshapes this into a tensor where
    # each row represents a potential bounding box and has the format [class_label, confidence, x, y, width, height].
    converted_bboxes = torch.cat(
        (best_class, scores, x, y, width_height), dim=-1
    ).reshape(batch_size, num_anchors * s * s, 6)

    return converted_bboxes.tolist()



def plot_image(image, boxes):
    # Get the color map from matplotlib
    colour_map = plt.get_cmap("tab20b")
    # Get different colors from the color map for different classes
    colors = [colour_map(i) for i in np.linspace(0, 1, len(class_labels))]

    # Read the image with OpenCV
    img = np.array(image)
    # Gett the height and width of the image
    h, w, _ = img.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Add image to plot
    ax.imshow(img)

    # Plot the bounding boxes and labels over the image
    for box in boxes:
        # Get the class from the box
        class_pred = box[0]
        # Get the center x and y coordinates
        box = box[2:]

        # Get the upper left corner coordinates: converts from center-coordinates to corner-coordinates
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2

        # Create a Rectangle patch with the bounding box
        rect = patches.Rectangle(
            (upper_left_x * w, upper_left_y * h),
            box[2] * w,
            box[3] * h,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )

        # Add the patch to the Axes
        ax.add_patch(rect)

        # Add class name to the patch
        plt.text(
            upper_left_x * w,
            upper_left_y * h,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    # Display the plot
    plt.show()






# Function to load checkpoint
def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("==> Loading checkpoint")
    # Define the path to the folder
    
    drive_path = "checkpoint_pretrained"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Combine the drive path and the filename
    full_path = os.path.join(drive_path, checkpoint_file)
    checkpoint = torch.load(full_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    # Check if optimizer state exists in checkpoint before loading
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        print("Optimizer state not found in checkpoint. Skipping optimizer loading.")


    for param_group in optimizer.param_groups:
        param_group["lr"] = lr




from typing import List

# Create a dataset class to load the images and labels from the folder
class Dataset(torch.utils.data.Dataset):

    # This is the constructor of the class. It initializes the dataset with various parameters
    def __init__(
        self, image_dir: str, label_dir: str, anchors: List[List[List[float]]],
        image_size=416, grid_sizes=[13, 26, 52],
        num_classes=6, transform=None
    ):

        # Image and label directories
        self.image_dir = image_dir
        self.label_dir = label_dir

        # Get all image files from the image directory
        self.image_files = sorted([
            file for file in os.listdir(image_dir)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ])


        # Image size
        self.image_size = image_size

        # Transformations
        self.transform = transform
        # Grid sizes for each scale
        self.grid_sizes = grid_sizes
        # Anchor boxes
        self.anchors = torch.tensor(
            anchors[0] + anchors[1] + anchors[2])
        # Number of anchor boxes
        self.num_anchors = self.anchors.shape[0]
        # Number of anchor boxes per scale
        self.num_anchors_per_scale = self.num_anchors // 3
        # Number of classes
        self.num_classes = num_classes
        # Ignore IoU threshold
        self.ignore_iou_thresh = 0.5

    # Returns the total number of samples in the dataset, which is the number of image files.
    def __len__(self):
        return len(self.image_files)

    # Loads and proces a single sample (image and its labels) given an index idx
    def __getitem__(self, idx):
        # Get image file
        img_file = self.image_files[idx]
        # Create corresponding label file name
        label_file = os.path.splitext(img_file)[0] + '.txt'

        # Load image
        img_path = os.path.join(self.image_dir, img_file)
        image = np.array(Image.open(img_path).convert("RGB"))

        # Load bounding boxes from label file
        label_path = os.path.join(self.label_dir, label_file)

        if os.path.exists(label_path):
            # Load bounding boxes in format [class_label, x, y, width, height]
            bboxes_raw = np.loadtxt(label_path, delimiter=' ', ndmin=2)

            # Separate bounding boxes and class labels for Albumentations
            bboxes = []
            class_labels = []

            for bbox in bboxes_raw:
                # Extract class label
                class_labels.append(int(bbox[0]))
                # Extract bbox coordinates in YOLO format [x, y, width, height]
                bboxes.append([bbox[1], bbox[2], bbox[3], bbox[4]])
        else:
            bboxes = []
            class_labels = []

        # Apply augmentations if provided
        if self.transform:
            augs = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image = augs["image"]
            bboxes = augs["bboxes"]
            class_labels = augs["class_labels"]

        # Recombine bboxes with class labels after transformation
        # Convert back to [x, y, width, height, class_label] format for internal processing
        combined_bboxes = []
        for bbox, class_label in zip(bboxes, class_labels):
            combined_bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3], class_label])

        # prepares the targets tensor for each of the YOLO output scales.
        # These tensors are initially filled with zeros and will be populated with the ground truth information in a format suitable for calculating the YOLO loss.
        targets = [torch.zeros((self.num_anchors_per_scale, s, s, 6))
                for s in self.grid_sizes]


        for box in combined_bboxes:
            # Calculate IoU between the box and all anchors
            # to determine which anchor is most appropriate for this box.
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors, is_pred=False)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3

            # Transform for training
            # A.Compose allows to chain multiple augmentation techniques together.
            train_transform = A.Compose(
                [
                    # Rescale an image so that maximum side is equal to image_size
                    A.LongestMaxSize(max_size=image_size),
                    # Pad remaining areas with zeros
                    A.PadIfNeeded(
                        min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT
                    ),
                    # Random color jittering: randomly changes the brightness, contrast, saturation, and hue of the image.
                    A.ColorJitter(
                        brightness=0.5, contrast=0.5,
                        saturation=0.5, hue=0.5, p=0.5
                    ),
                    # Flip the image horizontally
                    A.HorizontalFlip(p=0.5),
                    # Normalize the image
                    A.Normalize(
                        mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
                    ),
                    # Convert the image to PyTorch tensor
                    ToTensorV2()
                ],
                # Augmentation for bounding boxes:
                # specifies how the bounding boxes should be handled during the transformations
                bbox_params=A.BboxParams(
                                format="yolo",
                                min_visibility=0.4,
                                label_fields=["class_labels"]  # Added label field
                            )
            )

            # Transform for testing
            test_transform = A.Compose(
                [
                    # Rescale an image so that maximum side is equal to image_size
                    A.LongestMaxSize(max_size=image_size),
                    # Pad remaining areas with zeros
                    A.PadIfNeeded(
                        min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT
                    ),
                    # Normalize the image
                    A.Normalize(
                        mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
                    ),
                    # Convert the image to PyTorch tensor
                    ToTensorV2()
                ],
                # Augmentation for bounding boxes
                bbox_params=A.BboxParams(
                                format="yolo",
                                min_visibility=0.4,
                                label_fields=["class_labels"]  # Added label field
                            )
            )

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                s = self.grid_sizes[scale_idx]
                i, j = int(s * y), int(s * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = s * x - j, s * y - i
                    width_cell, height_cell = width * s, height * s
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)







# Defining CNN Block
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_batch_norm, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
        self.use_batch_norm = use_batch_norm

    def forward(self, x):
        # Applying convolution
        x = self.conv(x)
        # Applying BatchNorm and activation if needed
        if self.use_batch_norm:
            x = self.bn(x)
            return self.activation(x)
        else:
            return x

"""### Residual block

Now we will define residual block. We will be looping the layers in the residual block based on number defined in the architecture.
"""

# Defining residual block
class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()

        # Defining all the layers in a list and adding them based on number of
        # repeats mentioned in the design
        res_layers = []
        for _ in range(num_repeats):
            res_layers += [
                nn.Sequential(
                    nn.Conv2d(channels, channels // 2, kernel_size=1),
                    nn.BatchNorm2d(channels // 2),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.LeakyReLU(0.1)
                )
            ]
        self.layers = nn.ModuleList(res_layers)
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    # Defining forward pass
    def forward(self, x):
        for layer in self.layers:
            residual = x
            x = layer(x)
            if self.use_residual:
                x = x + residual
        return x

"""### Scale Prediction"""

# Defining scale prediction class
class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Defining the layers in the network
        self.pred = nn.Sequential(
            nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*in_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(2*in_channels, (num_classes + 5) * 3, kernel_size=1),
        )
        self.num_classes = num_classes

    # Defining the forward pass and reshaping the output to the desired output
    # format: (batch_size, 3, grid_size, grid_size, num_classes + 5)
    def forward(self, x):
        output = self.pred(x)
        output = output.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3))
        output = output.permute(0, 1, 3, 4, 2)
        return output

"""### YOLOv3 model

We will now use these components to code the YOLOv3 network.
"""

# Class for defining YOLOv3 model
class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Layers list for YOLOv3
        self.layers = nn.ModuleList([
            CNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1),
            CNNBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ResidualBlock(64, num_repeats=1),
            CNNBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ResidualBlock(128, num_repeats=2),
            CNNBlock(128, 256, kernel_size=3, stride=2, padding=1),
            ResidualBlock(256, num_repeats=8),
            CNNBlock(256, 512, kernel_size=3, stride=2, padding=1),
            ResidualBlock(512, num_repeats=8),
            CNNBlock(512, 1024, kernel_size=3, stride=2, padding=1),
            ResidualBlock(1024, num_repeats=4),
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0),
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            ResidualBlock(1024, use_residual=False, num_repeats=1),
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0),
            ScalePrediction(512, num_classes=num_classes),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2),
            CNNBlock(768, 256, kernel_size=1, stride=1, padding=0),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            ResidualBlock(512, use_residual=False, num_repeats=1),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
            ScalePrediction(256, num_classes=num_classes),
            CNNBlock(256, 128, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2),
            CNNBlock(384, 128, kernel_size=1, stride=1, padding=0),
            CNNBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ResidualBlock(256, use_residual=False, num_repeats=1),
            CNNBlock(256, 128, kernel_size=1, stride=1, padding=0),
            ScalePrediction(128, num_classes=num_classes)
        ])

    # Forward pass for YOLOv3 with route connections and scale predictions
    def forward(self, x):
        outputs = []
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        return outputs

"""## Training

For training the model, we need to define a loss function on which our model can optimize. The paper discusses that the YOLO (v3) architecture was optimized on a combination of four losses: no object loss, object loss, box coordinate loss, and class loss. The loss function is defined as:

$L(x, y, w, h, c, p, pc, tc, tx, ty, tw, th)= \lambda_{coord} \times L_{coord} + \lambda_{obj}\times L_{obj}+ \lambda_{noobj} \times L_{noonbj}+ \lambda_{class}\times L_{class}$

where,

- $λ_{coord}$, $λ_{obj}$, $λ_{noobj}$, and $λ_{class}$ are constants that weight the different components of the loss function (they are set to 1 in the paper).
- $L_{coord}$ penalizes the errors in the bounding box coordinates.
- $L_{obj}$ penalizes the confidence predictions for object detection.
- $L_{noobj}$ penalizes the confidence predictions for background regions.
- $L_{class}$ penalizes the errors in the class predictions.
"""

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, target, anchors):
        # Identifying which cells in target have objects
        # and which have no objects
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        # Calculating No object loss
        no_object_loss = self.bce(
            (pred[..., 0:1][no_obj]), (target[..., 0:1][no_obj]),
        )


        # Reshaping anchors to match predictions
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        # Box prediction confidence
        box_preds = torch.cat([self.sigmoid(pred[..., 1:3]),
                               torch.exp(pred[..., 3:5]) * anchors
                            ],dim=-1)
        # Calculating intersection over union for prediction and target
        ious = iou(box_preds[obj], target[..., 1:5][obj]).detach()
        # Calculating Object loss
        object_loss = self.mse(self.sigmoid(pred[..., 0:1][obj]),
                               ious * target[..., 0:1][obj])


        # Predicted box coordinates
        pred[..., 1:3] = self.sigmoid(pred[..., 1:3])
        # Target box coordinates
        target[..., 3:5] = torch.log(1e-6 + target[..., 3:5] / anchors)
        # Calculating box coordinate loss
        box_loss = self.mse(pred[..., 1:5][obj],
                            target[..., 1:5][obj])


        # Claculating class loss
        class_loss = self.cross_entropy((pred[..., 5:][obj]),
                                   target[..., 5][obj].long())

        # Total loss
        return (
            box_loss
            + object_loss
            + no_object_loss
            + class_loss
        )


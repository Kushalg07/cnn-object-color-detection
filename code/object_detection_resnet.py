# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import VOCDetection
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Define the data transforms
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# Define the label mapping
LABEL_MAP = {
    'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5,
    'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10,
    'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15,
    'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20
}

class VOCDetectionDataset(VOCDetection):
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        annotation = target['annotation']

        boxes = []
        labels = []
        for obj in annotation['object']:
            bbox = obj['bndbox']
            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(LABEL_MAP[obj['name']])  # Use label map to convert to integers

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        # Ensure the image is a PIL Image before applying transforms
        img = T.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

# Load the VOC2012 dataset
dataset = VOCDetectionDataset(root='/kaggle/working/', year='2012', image_set='train', download=True, transform=get_transform(train=True))
dataset_test = VOCDetectionDataset(root='/kaggle/working/', year='2012', image_set='val', download=True, transform=get_transform(train=False))

# Split the dataset into train and test sets
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# Define the number of classes (20 classes + background)
num_classes = 21

# Load a ResNet-50 model with FPN as the backbone
class BackboneWithFPN(nn.Module):
    def __init__(self, backbone, fpn):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.out_channels = 256  # FPN output channels

    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, torch.Tensor):
            features = {'0': features}
        else:
            features = {str(i): feature for i, feature in enumerate(features)}
        return self.fpn(features)

# Define the backbone
backbone = torchvision.models.resnet50(pretrained=True)
backbone = nn.Sequential(*list(backbone.children())[:-2])

# Define the FPN
fpn = torchvision.ops.FeaturePyramidNetwork([256, 512, 1024, 2048], out_channels=256)

# Combine backbone and FPN
backbone_with_fpn = BackboneWithFPN(backbone, fpn)

# Define the anchor generator
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),) * len((32, 64, 128, 256, 512))
)

# Define the RPN head
rpn_head = torchvision.models.detection.rpn.RPNHead(
    in_channels=256,  # This should match the FPN output channels
    num_anchors=anchor_generator.num_anchors_per_location()[0]
)

# Create the Faster R-CNN model
model = FasterRCNN(
    backbone_with_fpn,
    num_classes=num_classes,
    rpn_anchor_generator=anchor_generator,
    rpn_head=rpn_head,
    box_roi_pool=torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )
)

# Replace the classifier head to match the number of classes
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Define the device

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define the optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

# Define the learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Define training loop
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    num_batches = len(data_loader)

    for batch, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

        if batch % 100 == 0:
            print(f"Epoch [{epoch+1}], Batch [{batch+1}/{num_batches}], Loss: {losses.item():.4f}")

    avg_loss = total_loss / num_batches
    print(f"Epoch [{epoch+1}] completed. Average Loss: {avg_loss:.4f}")

    return avg_loss

# Define evaluation function
def calculate_iou(box1, box2):
    # Calculate intersection over union of two boxes
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection / float(area1 + area2 - intersection)
    return iou

def evaluate(model, data_loader_test, device, iou_threshold=0.5):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in data_loader_test:
            images = list(image.to(device) for image in images)
            outputs = model(images)

            for i, target in enumerate(targets):
                pred_boxes = outputs[i]['boxes'].cpu().numpy()
                pred_labels = outputs[i]['labels'].cpu().numpy()
                pred_scores = outputs[i]['scores'].cpu().numpy()

                target_boxes = target['boxes'].cpu().numpy()
                target_labels = target['labels'].cpu().numpy()

                for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
                    if pred_score > 0.5:  # Consider only predictions with score > 0.5
                        all_predictions.append((pred_box, pred_label))

                for target_box, target_label in zip(target_boxes, target_labels):
                    all_targets.append((target_box, target_label))

    true_positives = Counter()
    false_positives = Counter()
    false_negatives = Counter()
    ious = []

    for pred_box, pred_label in all_predictions:
        matched = False
        for target_box, target_label in all_targets:
            iou = calculate_iou(pred_box, target_box)
            if pred_label == target_label and iou >= iou_threshold:
                true_positives[pred_label] += 1
                ious.append(iou)
                matched = True
                break
        if not matched:
            false_positives[pred_label] += 1

    for target_box, target_label in all_targets:
        matched = False
        for pred_box, pred_label in all_predictions:
            if pred_label == target_label and calculate_iou(pred_box, target_box) >= iou_threshold:
                matched = True
                break
        if not matched:
            false_negatives[target_label] += 1

    precisions = {}
    recalls = {}
    f1_scores = {}

    for label in set(list(true_positives.keys()) + list(false_positives.keys()) + list(false_negatives.keys())):
        tp = true_positives[label]
        fp = false_positives[label]
        fn = false_negatives[label]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precisions[label] = precision
        recalls[label] = recall
        f1_scores[label] = f1

    mean_precision = np.mean(list(precisions.values()))
    mean_recall = np.mean(list(recalls.values()))
    mean_f1 = np.mean(list(f1_scores.values()))
    mean_iou = np.mean(ious) if ious else 0

    # Calculate mAP
    ap_sum = 0
    for label in precisions:
        ap_sum += precisions[label]
    mean_ap = ap_sum / len(precisions) if precisions else 0

    return {
        'Mean Precision': mean_precision,
        'Mean Recall': mean_recall,
        'Mean F1 Score': mean_f1,
        'Mean IoU': mean_iou,
        'mAP': mean_ap
    }

# Training loop
num_epochs = 20
epoch_losses = []
final_metrics = None

for epoch in range(num_epochs):
    avg_loss = train_one_epoch(model, optimizer, data_loader, device, epoch)
    epoch_losses.append(avg_loss)
    final_metrics = evaluate(model, data_loader_test, device)

import matplotlib.pyplot as plt

# Plot the loss
def plot_smooth_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.show()

# After training
# Print final performance metrics
print("\nFinal Performance Metrics:")
for metric, value in final_metrics.items():
    print(f"{metric}: {value:.4f}")

# Plot the smoothed loss
plot_smooth_loss(epoch_losses)

import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb
from PIL import Image

# Color analysis functions
def get_color_name(requested_color):
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))

    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(requested_color)
    return names[index]

def extract_circular_region(image, center, radius):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, center, radius, 255, -1)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def get_dominant_color(image, k=5):
    if len(image.shape) == 3 and image.shape[2] == 3:
        pixels = image.reshape((-1, 3))
    else:
        return None, None, None

    pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]

    if pixels.size == 0:
        return None, None, None

    kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300)
    kmeans.fit(pixels)

    mean_color = np.mean(pixels, axis=0)
    distances = np.linalg.norm(kmeans.cluster_centers_ - mean_color, axis=1)
    dominant_color = kmeans.cluster_centers_[np.argmin(distances)]

    return dominant_color.astype(int), kmeans.cluster_centers_, np.bincount(kmeans.labels_)

# Image loading and preprocessing
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def preprocess_image(image):
    image_tensor = F.to_tensor(image)
    return image_tensor.unsqueeze(0)

# Visualization function
def visualize_predictions_with_color(image, prediction, label_map, score_threshold=0.5, k=5, radius=50):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    reversed_label_map = {v: k for k, v in label_map.items()}

    boxes = prediction[0]['boxes'].cpu().detach().numpy()
    labels = prediction[0]['labels'].cpu().detach().numpy()
    scores = prediction[0]['scores'].cpu().detach().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score >= score_threshold:
            xmin, ymin, xmax, ymax = box.astype(int)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            text = f"{reversed_label_map[label]}: {score:.2f}"
            ax.text(xmin, ymin, text, bbox=dict(facecolor='yellow', alpha=0.5), fontsize=12, color='black')

            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)

            circular_region = extract_circular_region(np.array(image), (center_x, center_y), radius)

            if circular_region.size > 0:
                dominant_color, cluster_centers, counts = get_dominant_color(circular_region, k=k)

                if dominant_color is not None:
                    color_name = get_color_name(dominant_color)

                    color_rect = patches.Rectangle((xmin, ymax), 20, 20, linewidth=1, edgecolor='none', facecolor=tuple(dominant_color/255))
                    ax.add_patch(color_rect)
                    ax.text(xmin + 25, ymax + 5, f"Color: {color_name} ({dominant_color})", fontsize=12, color='black', verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))

                    plt.figure(figsize=(8, 4))
                    plt.bar(range(k), counts, color=[cluster_centers[i]/255 for i in range(k)])
                    plt.title(f'Color Distribution for {reversed_label_map[label]}')
                    plt.xlabel('Clusters')
                    plt.ylabel('Frequency')
                    plt.show()
                else:
                    print(f"No valid colors found in the region for {reversed_label_map[label]}")
            else:
                print(f"Skipping empty crop: {(xmin, ymin, xmax, ymax)}")

    plt.axis('off')
    plt.show()

# Main testing function
def test_object_detection_and_color_analysis(model, image_path, label_map, device='cuda'):
    try:
        model.eval()
        model.to(device)

        # Load and preprocess the image
        image = load_image(image_path)
        image_tensor = preprocess_image(image)

        # Perform inference
        with torch.no_grad():
            prediction = model(image_tensor.to(device))

        # Visualize results with color analysis
        visualize_predictions_with_color(image, prediction, label_map)
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    # model should be defined and trained in your training script
    # model = your trained model

    image_path = "/kaggle/input/test-images/bus-58_jpg.rf.fad69f9e8db5a40e72e48be36639764f.jpg"

    LABEL_MAP = {
        'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5,
        'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10,
        'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15,
        'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_object_detection_and_color_analysis(model, image_path, LABEL_MAP, device)


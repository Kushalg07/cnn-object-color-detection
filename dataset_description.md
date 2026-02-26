# 📊 Dataset Description
## PASCAL VOC 2012 – Object Detection Dataset

---

## 📌 Overview

This project uses the **PASCAL VOC 2012 (Visual Object Classes)** dataset for training and evaluating the object detection model.

PASCAL VOC is one of the most widely used benchmark datasets in computer vision for object detection and segmentation tasks.

---

## 📂 Dataset Source

Official Dataset:
http://host.robots.ox.ac.uk/pascal/VOC/

Kaggle Mirror:
https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset

---

## 📊 Dataset Statistics

- Total Images: 11,530
- Total Annotated Objects: 27,450
- Total Segmentation Masks: 6,929
- Number of Classes: 20

---

## 🏷 Object Classes

The dataset contains 20 object categories:

1. Person  
2. Bird  
3. Cat  
4. Cow  
5. Dog  
6. Horse  
7. Sheep  
8. Aeroplane  
9. Bicycle  
10. Boat  
11. Bus  
12. Car  
13. Motorbike  
14. Train  
15. Bottle  
16. Chair  
17. Dining Table  
18. Potted Plant  
19. Sofa  
20. TV/Monitor  

---

## 🗂 Dataset Structure

The dataset is divided into:

VOC2012/
│
├── JPEGImages/        (All images)
├── Annotations/       (XML bounding box annotations)
├── ImageSets/
│   ├── Main/
│   └── SegmentationClass/
└── SegmentationObject/

Each image has a corresponding XML file containing:

- Object class label
- Bounding box coordinates
- Object difficulty flag
- Truncation information

---

## 📦 Annotation Format

Annotations are stored in XML format.

Each object contains:

- <name> (class label)
- <bndbox>
    - xmin
    - ymin
    - xmax
    - ymax

These annotations were parsed and converted into PyTorch-compatible tensor format for Faster R-CNN training.

---

## 🔄 Data Split

The dataset includes:

- Training Set
- Validation Set
- Test Set (evaluation server)

For this project:

- Training + Validation were used for model training and evaluation.
- Random horizontal flipping was applied during training.

---

## 🖼 Image Properties

- Format: JPG
- Resolution: Varies (real-world images)
- Scene complexity: High
- Multiple objects per image are possible
- Objects at different scales and orientations

---

## ⚠ Dataset Challenges

PASCAL VOC presents several challenges:

- Multi-object scenes
- Small object detection
- Occlusion
- Complex backgrounds
- Lighting variations
- Scale variations

These challenges make it suitable for evaluating real-world detection systems.

---

## 🎨 Color Analysis Consideration

The original PASCAL VOC dataset does not contain color annotations.

For this project:

- Dominant color is extracted from detected object regions.
- K-Means clustering is used for color quantization.
- Extracted RGB values are mapped to CSS3 color names.

Thus, color analysis is implemented as a post-detection enhancement.

---

## 📏 Evaluation Metrics

The dataset supports evaluation using:

- Mean Average Precision (mAP)
- Intersection over Union (IoU)
- Precision
- Recall
- F1 Score

In this project, mAP and IoU were the primary evaluation metrics.

---

## 📌 Why PASCAL VOC?

Chosen because:

- Standard benchmark dataset
- Widely cited in research
- Balanced number of classes
- Real-world image diversity
- Suitable for Faster R-CNN training

---

## 👨‍💻 Maintainer

Kushal G  
M.Tech – Data Science  

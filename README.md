# 🎨 A.I. Vision in Color
## Object Detection with Color Recognition and Explainable AI

---

## 📌 Project Overview

This project presents an advanced object detection system built using:

- Faster R-CNN (ResNet50 + FPN backbone)
- Region Proposal Network (RPN)
- Color Extraction using K-Means Clustering
- Grad-CAM Explainability
- HOG Visualization

The system detects objects in images and simultaneously identifies their dominant color, providing enhanced image understanding.

---

## 🎯 Problem Statement

Traditional object detection models identify and localize objects but do not analyze object color.

This project extends object detection by integrating:

- Bounding box prediction
- Dominant color extraction
- Color naming (CSS3 matching)
- Model interpretability using Grad-CAM

This enables richer scene understanding for robotics, surveillance, autonomous systems, and AR applications.

---

## 📊 Dataset

Dataset used:
PASCAL VOC 2012

- 11,530 images
- 27,450 annotated objects
- 20 object classes
- Bounding box annotations

Dataset Link:
https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset

---

## 🏗 Model Architecture

<img width="1140" height="666" alt="Simple_OD_model drawio" src="https://github.com/user-attachments/assets/baf802ad-7ec6-4bf6-8051-ff260596ee99" />

### 🔹 Backbone
- ResNet-50 (Pretrained on ImageNet)

### 🔹 Feature Extraction
- Feature Pyramid Network (FPN)

### 🔹 Region Proposal
- Anchor-based RPN
- Multi-scale anchor generation

### 🔹 Detection Head
- ROI Align
- Classification branch
- Bounding box regression branch

### 🔹 Optimization
- SGD Optimizer
- Learning Rate Scheduling
- Weight Decay Regularization

---

## 🎨 Color Analysis Module

For each detected object:

1. Extract circular region from bounding box
2. Apply K-Means clustering
3. Compute dominant color
4. Match to nearest CSS3 color name
5. Display color with bounding box

---

## 🔍 Explainable AI

### Grad-CAM
- Highlights regions influencing predictions
- Provides transparency in model decision-making

### HOG Visualization
- Visualizes gradient orientations
- Shows structural feature extraction

---

## 📈 Experimental Results

### Best Configuration:
- Learning Rate: 0.005
- Epochs: 30–50

### Performance (Best Model):

- mAP: 0.8119
- Mean F1 Score: 0.6242
- Mean IoU: 0.7096
- Mean Precision: 0.8119
- Mean Recall: 0.5951

---

## 📊 Hyperparameter Analysis

Experiments conducted with varying:

- Learning Rates (0.01 → 0.0001)
- Epochs (10 → 50)

Key Insight:
Moderate learning rates (0.001–0.005) performed best.
Very high or very low learning rates degraded performance.

---

## 🛠 Tech Stack

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

## 🚀 Applications

- Autonomous vehicles
- Smart surveillance
- Robotics
- Augmented reality
- Industrial inspection

---

## 🔮 Future Improvements

- DETR / Vision Transformer integration
- EfficientNet backbone
- Small object detection enhancement
- Model compression for edge devices
- Real-time deployment optimization

---

## 👨‍💻 Author

Kushal G  
M.Tech – Data Science 

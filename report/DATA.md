# COCO 2017 Dataset Exploratory Data Analysis Report

## 1. Overview
The COCO 2017 dataset contains images and annotations for object detection, segmentation, and captioning.

### 1.1. Image Statistics
- **Training Images:** 118287
- **Validation Images:** 5000

### 1.2. Annotation Statistics
- **Training Annotations:** 860001
- **Validation Annotations:** 36781
- **Total Categories:** 80

### 1.3. Category Distribution
- The dataset is imbalanced, with some categories (e.g., "person") having significantly more instances than others (e.g., "toaster").
- Image size distribution shows a wide range from 320x320 to 640x640.
- Class distribution remains consistent across training and validation sets.

### 1.4. Instance Distribution
- The number of instances per image varies widely, with many images containing only a few objects and some containing dozens.
- Each image have at least one instance, average instances per image is around 7.3. There are some images with nearly 100 instances.

## 2. Completeness
- Bounding boxes and segmentation masks were verified to be complete across the dataset.
- Instances per image distributions show a typical long-tail where most images have few objects, but some have many.

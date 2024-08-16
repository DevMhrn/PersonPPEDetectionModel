# Person and PPE Detection

## Setup Instructions

Before getting started, make sure you have the necessary packages installed. You can do this by running the following command:

```bash
pip install ultralytics opencv-python argparse lxml
```

# Approach Documentation

## Overview of the Process

### 1. Dataset Preparation

**Person Detection:**

- The dataset was initially filtered to focus on detecting persons.
- Images and labels were divided into `train`, `valid`, and `test` subfolders.

**PPE Detection:**

- After training the person detection model, the images were cropped to isolate detected persons.
- These cropped images were then organized similarly into `train`, `valid`, and `test` subfolders for PPE detection training.

### 2. Training and Inference

**Person Detection Model:**

- The filtered dataset was used to train the person detection model.

**PPE Detection Model:**

- The cropped images, focused on persons, were used to train the PPE detection model.

**Inference:**

- The `inference.py` script was then used to detect persons first and subsequently detect PPE on the cropped images.

### 3. Labeling Issues

- During the training of the PPE detection model, some class labels were incorrectly labeled, causing warnings and leading to inaccuracies in the model’s detection of PPE.
- This issue highlighted the importance of correct labeling for effective model performance.

### [Approach Doc](https://docs.google.com/document/d/1tld96ulpa4bboxbn4vVffRZV3me5BcoesNQqwI08WBI/edit?usp=sharing)

## Conclusion

This process demonstrates the importance of correctly preparing datasets and labeling to ensure effective model training and accurate detection results.
#   P e r s o n P P E D e t e c t i o n M o d e l  
 
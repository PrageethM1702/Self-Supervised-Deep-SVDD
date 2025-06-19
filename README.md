# Blood Cell Anomaly Detection using Self-Supervised Learning

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)

## 📌 Project Overview

A self-supervised deep learning approach to detect anomalies in blood cell images using contrastive learning (SimCLR) and Deep SVDD. Achieves **99% AUC-ROC** by learning robust representations of normal cells and flagging deviations.

[Download Model from Google Drive](https://drive.google.com/file/d/1aHpoNRodYJ1KLcWwXhshgiFMlNRYkYcK/view?usp=sharing)

---

## 🩸 Key Features

- **Self-Supervised Learning**: Uses SimCLR to pretrain on unlabeled data
- **Anomaly Synthesis**: Generates synthetic anomalies via noise/rotations
- **High Accuracy**: 98.7% F1-score on blood cell classification
- **Visual Diagnostics**: Includes prediction visualizations and confusion matrices

## 📊 Dataset

[Blood Cell Image Dataset](https://www.kaggle.com/datasets/unclesamulus/blood-cells-image-dataset) from Kaggle with 8 cell types:
- Normal: `neutrophil`, `lymphocyte`, `monocyte`, `eosinophil`, `basophil`, `erythroblast`, `platelet`, `Ig` cells

![Sample Cells](https://i.imgur.com/JQ6Yc7P.png)

---

## 🚀 Results

| Metric       | Score  |
|--------------|--------|
| Accuracy     | 98.7%  |
| Precision    | 98.1%  |
| Recall       | 99.3%  |
| AUC-ROC      | 99.9%  |

**Confusion Matrix**  
|                | Normal | Anomaly |
|----------------|--------|---------|
| **Normal**     | 3395   | 64      |
| **Anomaly**    | 24     | 3354    |

---

## 💻 Usage

```bash
# Single image
python run.py --input_path ./sample.jpg

# Folder of images
python run.py --input_path ./test_images/

# Show image preview
python run.py --input_path ./test_images/ --show


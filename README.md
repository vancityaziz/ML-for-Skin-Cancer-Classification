# Mixture-of-Experts Inspired Ensemble Model for Malignant Skin Lesion Classification

This project implements a Mixture-of-Experts (MoE) inspired ensemble of fine-tuned convolutional neural networks (EfficientNet-B0, B2, B3, and MobileNet-V2) to classify dermoscopic skin images into seven diagnostic categories.

## Performance Summary
Our ensemble architecture significantly outperformed the baseline Support Vector Machine (SVM) and individual CNN models.

| Model | Overall Accuracy | Balanced Accuracy (BACC) | AUC |
| :--- | :--- | :--- | :--- |
| **SVM Baseline** | 65.9% | 40.9% | N/A |
| **Ensemble Model** | **82.5%** | **74.8%** | **0.966** |

## Model Architecture
The system uses a top-k gating network to selectively weigh the outputs of four "expert" pretrained CNNs:

* **Expert 1:** EfficientNet-B0 (5.3M parameters)
* **Expert 2:** EfficientNet-B2 (9.2M parameters)
* **Expert 3:** EfficientNet-B3 (12.M parameters)
* **Expert 4:** MobileNet-V2 (3.4M parameters)

### Gating Mechanism
The gating network is a fully connected network with a 64-dimensional hidden layer and ReLU activation. It outputs weights via Softmax to determine the influence of each expert on the final prediction.

## Data Processing & Augmentation
The model was trained on the **HAM10000 dataset**. To handle the significant class imbalance (67% of samples are 'nv'), we implemented several techniques:

* **Weighted Cross Entropy Loss:** Misclassifications in minority classes are penalized more heavily based on inverse class frequency.
* **Advanced Augmentation:** Training images underwent RandAugment, RandomResizedCrop with zoom, and CutOut to improve generalization.
* **Preprocessing:** All images were resized to 224x224 pixels and normalized using ImageNet statistics.

## Training Workflow
1. **Fine-tuning:** Individual CNN classifiers were trained for 70 epochs. We froze the base weights for the first 15 epochs before unfreezing for full-network training.
2. **Ensemble Training:** The individual CNNs were frozen, and only the gating network was trained to optimize the combination of expert outputs.

## Contributors
* Mohammad Abdul Aziz
* Kyle Sabado
* Keli Chen
* Uddhava Swaminathan

---
*University of Toronto - APS360 Final Project*

# 🐾 Animal Classification using CNN in PyTorch

This project implements a Convolutional Neural Network (CNN) using PyTorch for classifying images of animals into distinct categories. It includes data preprocessing, model architecture, training, evaluation, and ensemble modeling to enhance accuracy and generalization.

---

## 📁 Dataset

The dataset consists of labeled images of animals, divided into training and validation sets (typically 80:20). Images are resized, normalized, and augmented during preprocessing to improve model performance and robustness.

---

## ⚙️ Model Architecture

The CNN model contains:

* **Convolutional Layers**: Extract spatial features using filters
* **ReLU Activation**: Introduce non-linearity
* **MaxPooling Layers**: Reduce spatial dimensions
* **Dropout Layers**: Prevent overfitting
* **Fully Connected Layers**: Classify features into categories
* **Softmax Output**: Generate probabilities per class

---

## 🚀 Training Details

| Parameter     | Value                           |
| ------------- | ------------------------------- |
| Optimizer     | Adam                            |
| Loss Function | CrossEntropyLoss                |
| Learning Rate | 0.001                           |
| Batch Size    | 32                              |
| Epochs        | 10 for base model, 5 for others |

---

## 📈 Model Performance Summary

A total of **four models** were trained. One was trained for 10 epochs, and the other three for 5 epochs each. An ensemble model was created to combine predictions from all.

### 🔍 Individual Model Performance

| Model         | Epochs | Train Accuracy | Val Accuracy | Val Loss  | Precision | Recall | F1-Score |
| ------------- | ------ | -------------- | ------------ | --------- | --------- | ------ | -------- |
| Model (10 ep) | 10     | 98.50%         | 68.83%       | ↑ gradual | 0.72      | 0.70   | 0.71     |
| Model 1       | 5      | 75.50%         | 70.67%       | 0.7092    | 0.73      | 0.69   | 0.71     |
| Model 2       | 5      | 79.88%         | 68.83%       | 0.7163    | 0.70      | 0.68   | 0.69     |
| Model 3       | 5      | 78.20%         | 69.10%       | 0.7150    | 0.71      | 0.69   | 0.70     |

> 🔍 **Note**: The 10-epoch model showed high training accuracy but suffered from increasing validation loss, indicating **overfitting**.

---

### 🤝 Ensemble Model Performance

Combining the predictions of all four models significantly improved generalization.

| Metric       | Value  |
| ------------ | ------ |
| Val Accuracy | 72.00% |
| Precision    | 0.74   |
| Recall       | 0.72   |
| F1-Score     | 0.73   |

---

## 📌 Key Observations

* **Overfitting Identified**: The 10-epoch model achieved 98.50% training accuracy but underperformed on validation data.
* **Best Generalization**: Model 1 showed the best balance between training and validation accuracy among individual models.
* **Ensemble Strength**: Ensemble model outperformed all individual models with improved and stable metrics.
* **Class-Specific Weakness**: The **"Dogs"** class consistently had lower precision and recall — suggesting data imbalance or harder feature representation.

---

## 📦 Requirements

* Python 3.8+
* PyTorch
* torchvision
* matplotlib
* scikit-learn
* numpy

Install using:

```bash
pip install -r requirements.txt
```

---

## 📬 Results & Conclusion

This project demonstrates the effectiveness of CNN-based models in image classification tasks. Despite high training accuracy, overfitting was managed through model selection and ensembling. The ensemble model delivered the best balance between precision, recall, and F1-score.

### 📌 Future Work:

* Incorporate **data augmentation** techniques.
* Explore **transfer learning** using pre-trained models (ResNet, VGG).
* Address class imbalance, particularly in underperforming categories like "Dogs".

---

## 👨‍💻 Author

**Rahul Kumar Sharma**
Department of Mining Engineering
IIT (ISM) Dhanbad
📧 [20je0749@iitism.ac.in](mailto:20je0749@iitism.ac.in)
📱 +91-9508476508


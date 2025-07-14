# 🐾 Animal Classification using CNN in PyTorch
This project focuses on building a robust CNN-based model using PyTorch to classify animal images into distinct categories. It includes data preprocessing, multiple model training strategies, performance evaluation, and an ensemble approach to enhance classification accuracy and generalization.

---

## 🏆 Achievements

* 🥈 **Silver Medalist on Kaggle** (Top 10%)
* 📈 Achieved **72% Validation Accuracy** using ensemble modeling
* 🧠 Demonstrated effective generalization through multiple model strategies
* 🔗 Project ranked with **12+ upvotes** on Kaggle platform

---

## 📎 Live Notebook on Kaggle

👉 [Animal Classification CNN in PyTorch – Kaggle Notebook](https://www.kaggle.com/code/oops26/animal-classification-cnn-pytorch)

---

## 📁 Dataset Overview

* Contains images of animals grouped into labeled classes
* Split into training and validation sets
* Preprocessed with resizing, normalization, and data augmentation techniques

---

## 🧠 Model Architecture

Implemented with multiple convolutional layers and includes:

* **Conv2D + ReLU**: Feature extraction
* **MaxPooling2D**: Dimensionality reduction
* **Dropout**: Regularization to avoid overfitting
* **Fully Connected Layers**: Class prediction
* **Softmax**: Final output probability distribution

---

## ⚙️ Training Configuration

| Parameter     | Value                      |
| ------------- | -------------------------- |
| Framework     | PyTorch                    |
| Optimizer     | Adam                       |
| Loss Function | CrossEntropyLoss           |
| Learning Rate | 0.001                      |
| Batch Size    | 32                         |
| Epochs        | 10 (1 model), 5 (3 models) |

---

## 📊 Model Performance Summary

A total of **4 models** were trained individually and later combined via ensemble. One model was trained for 10 epochs, and the rest for 5 each.

### 🔍 Individual Model Results

| Model         | Epochs | Train Acc | Val Acc | Val Loss    | Precision | Recall | F1-score |
| ------------- | ------ | --------- | ------- | ----------- | --------- | ------ | -------- |
| Model (10 ep) | 10     | 98.50%    | 68.83%  | ↑ (Overfit) | 0.72      | 0.70   | 0.71     |
| Model 1       | 5      | 75.50%    | 70.67%  | 0.7092      | 0.73      | 0.69   | 0.71     |
| Model 2       | 5      | 79.88%    | 68.83%  | 0.7163      | 0.70      | 0.68   | 0.69     |
| Model 3       | 5      | 78.20%    | 69.10%  | 0.7150      | 0.71      | 0.69   | 0.70     |

### 🤝 Ensemble Model Performance

| Metric       | Value  |
| ------------ | ------ |
| Val Accuracy | 72.00% |
| Precision    | 0.74   |
| Recall       | 0.72   |
| F1-Score     | 0.73   |

> ✅ **Ensembling all four models yielded the highest overall accuracy and more balanced metrics.**

---

## 📈 Training and Validation Trends

* **Overfitting** was observed in the 10-epoch model due to increasing validation loss.
* **Model 1** showed best validation performance on its own.
* **Ensemble model** achieved the best **F1-score (0.73)** and overall performance.
* The **"Dog" class** consistently underperformed compared to others, indicating a possible imbalance or harder feature separability.

---

## 📌 Key Observations

* Ensemble modeling significantly improved validation performance.
* Validation accuracies across models ranged from **68.83% to 72.00%**.
* The ensemble model achieved a **good balance** of precision, recall, and F1-score.
* Dataset augmentation and dropout layers helped mitigate overfitting.

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python train.py
```

(Adjust depending on how the code is organized.)

---

## 📦 Requirements

* Python 3.8+
* PyTorch
* torchvision
* matplotlib
* numpy
* scikit-learn

---

## 📬 Conclusion

This CNN-based animal classification project demonstrates:

* Solid baseline performance with CNN
* Benefits of ensembling in improving generalization
* Critical need for further data tuning (especially class balance)

---

## 🔭 Future Improvements

* Apply **transfer learning** with models like ResNet, EfficientNet
* Augment data with **advanced image transformations**
* Address class imbalance to improve precision and recall

---

## 👨‍💻 Author

* **Rahul Kumar Sharma**
* Department of Mining Engineering at IIT (ISM) Dhanbad.
* 📧 [20je0749@iitism.ac.in](mailto:20je0749@iitism.ac.in)
* 📱 +91-9508476508


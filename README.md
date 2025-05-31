# 🐾 Animal Classification using CNN in PyTorch

This project implements a **Convolutional Neural Network (CNN)** using PyTorch to classify animal images into multiple categories. It covers end-to-end development, including data preprocessing, model building, training, evaluation, and ensemble modeling to boost performance and robustness.

---

## 🏆 Achievements

- 🥈 **Silver Medalist on Kaggle** (Top 10%)
- 📈 Achieved **72% Validation Accuracy** using ensemble modeling
- 🧠 Demonstrated effective generalization through multiple model strategies
- 🔗 Project ranked with **21+ upvotes** on Kaggle platform

---

## 📎 Live Notebook on Kaggle

👉 [Bhagavad Gita Chatbot using NLP – Kaggle Notebook](https://www.kaggle.com/code/oops26/bhagavad-gita-chatbot-using-nlp)  

---

## 📁 Dataset

The dataset contains labeled images of animals split into **training and validation sets (80:20)**.  
Preprocessing steps include:

- Resizing and normalization  
- Augmentation (rotation, flipping)  
- Conversion to tensors for PyTorch compatibility

---

## ⚙️ Model Architecture

The custom CNN model consists of:

- **Convolutional Layers** – to extract features  
- **ReLU Activation** – to introduce non-linearity  
- **MaxPooling** – to reduce feature size  
- **Dropout Layers** – to prevent overfitting  
- **Fully Connected Layers** – for classification  
- **Softmax Output** – to predict probabilities per class

---

## 🚀 Training Details

| Parameter       | Value            |
|----------------|------------------|
| Optimizer       | Adam             |
| Loss Function   | CrossEntropyLoss |
| Learning Rate   | 0.001            |
| Batch Size      | 32               |
| Epochs          | 10 (base model), 5 (others) |

---

## 📈 Model Performance Summary

A total of **4 models** were trained. The best performing metrics came from the **ensemble model**, combining all.

### 🔍 Individual Models

| Model         | Epochs | Train Acc | Val Acc | Val Loss | Precision | Recall | F1-Score |
|---------------|--------|-----------|---------|----------|-----------|--------|----------|
| Model (10 ep) | 10     | 98.50%    | 68.83%  | ↑ gradual | 0.72      | 0.70   | 0.71     |
| Model 1       | 5      | 75.50%    | 70.67%  | 0.7092   | 0.73      | 0.69   | 0.71     |
| Model 2       | 5      | 79.88%    | 68.83%  | 0.7163   | 0.70      | 0.68   | 0.69     |
| Model 3       | 5      | 78.20%    | 69.10%  | 0.7150   | 0.71      | 0.69   | 0.70     |

### 🤝 Ensemble Model

| Metric     | Value  |
|------------|--------|
| Val Acc    | 72.00% |
| Precision  | 0.74   |
| Recall     | 0.72   |
| F1-Score   | 0.73   |

---

## 📌 Key Observations

- ⚠️ **Overfitting Identified** in 10-epoch base model  
- ✅ **Best Validation Accuracy** from Model 1  
- 🤝 **Ensemble Model** provided best generalization  
- 🐶 **"Dogs" class** showed lower precision and recall, suggesting possible data imbalance

---

## 💡 Future Work

- Enhance data augmentation strategies
- Apply **Transfer Learning** using pretrained CNNs like ResNet or VGG
- Balance underperforming classes using SMOTE or weighted loss
- Deploy using **Streamlit**, **Flask**, or FastAPI for real-time demo

---

## 📦 Installation

```bash
git clone https://github.com/your-username/animal-classification-cnn.git
cd animal-classification-cnn
pip install -r requirements.txt


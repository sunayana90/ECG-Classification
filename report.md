# ECG Signal Classification Report

---

## 📝 Overview

This report documents the methodology, results, and observations for the ECG heartbeat classification project using a 1D Convolutional Neural Network (1D CNN) trained on the MIT-BIH Arrhythmia dataset.

The objective was to classify ECG signals into five heartbeat categories:

- **0:** Normal beat
- **1:** Supraventricular premature beat
- **2:** Premature ventricular contraction
- **3:** Fusion of ventricular and normal beat
- **4:** Unknown beat

---

## 📊 Dataset

**Source:** [MIT-BIH Arrhythmia Dataset on Kaggle](https://www.kaggle.com/shayanfazeli/heartbeat)

**Files Used:**
- `mitbih_train.csv`
- `mitbih_test.csv`

**Data Characteristics:**
- Each sample contains 187 time steps of ECG signal.
- Labels are encoded as integers 0–4.

**Train Set:**
- 87,554 samples

**Test Set:**
- 21,892 samples

---

## ⚙️ Model Architecture

The model is a simple yet effective 1D CNN implemented in PyTorch with the following structure:

1. **Conv1D Layer:**
   - Channels: 1 → 32
   - Kernel Size: 7
   - Padding: 3
   - Activation: ReLU
   - Batch Normalization
   - Max Pooling

2. **Conv1D Layer:**
   - Channels: 32 → 64
   - Kernel Size: 5
   - Padding: 2
   - Activation: ReLU
   - Batch Normalization
   - Max Pooling

3. **Fully Connected Layer:**
   - Units: 128
   - Activation: ReLU
   - Dropout (0.3)

4. **Output Layer:**
   - Units: 5 (number of classes)

**Optimizer:** Adam  
**Loss Function:** CrossEntropyLoss  
**Batch Size:** 64  
**Epochs:** 10  

---

## 📈 Evaluation Metrics

After training for 10 epochs, the model achieved:

- **Overall Accuracy:** ~98%
- **Macro-averaged F1 Score:** ~0.91

**Detailed Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Normal) | 0.99 | 1.00 | 0.99 | 18,118 |
| 1 | 0.94 | 0.74 | 0.83 | 556 |
| 2 | 0.96 | 0.95 | 0.95 | 1,448 |
| 3 | 0.86 | 0.73 | 0.79 | 162 |
| 4 | 1.00 | 0.98 | 0.99 | 1,608 |

- **Weighted Average F1:** ~0.98
- **Macro Average F1:** ~0.91

---

## 🧩 Confusion Matrix

![Confusion Matrix](https://github.com/sunayana90/ECG-Classification/blob/main/confusion-metrix.png)

```
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
```

---

## 🔍 Observations

✅ **Strengths:**
- Excellent performance on the majority class (Normal beats) and class 4.
- High precision and recall for most classes.
- Simple architecture achieving strong results.

⚠️ **Weaknesses:**
- Lower recall for classes 1 and 3 (supraventricular premature and fusion beats).
- Class imbalance in the dataset likely contributed to misclassification.

---

## ✨ Future Improvements

To further enhance the model’s performance, especially on minority classes:

- Use **class-weighted loss** to penalize misclassification of rare classes.
- Apply **data augmentation** (e.g., noise injection, scaling, shifting).
- Train for more epochs or apply early stopping.
- Experiment with **more complex architectures**.
- Implement model explainability techniques like **Grad-CAM** for interpretability.

---

## 📝 Conclusion

This project demonstrates that a relatively simple 1D CNN architecture can effectively classify ECG signals with high accuracy. Further tuning and advanced preprocessing can improve recall on minority arrhythmia classes.

---

**Author:** Sunayana Yadav

**Date:** 28 June 2025


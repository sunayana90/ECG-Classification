# ECG Signal Classification using 1D CNN

This project implements a 1D Convolutional Neural Network (1D CNN) to classify ECG (Electrocardiogram) signals into five heartbeat categories using the MIT-BIH Arrhythmia dataset.

---

## 📂 Project Structure

```
ecg-classification/
├── ecg_classification.ipynb     # Jupyter Notebook with code and results
├── ecg_cnn.pth                  # Trained model weights
├── report.md                    # Detailed evaluation report and insights
└── README.md                    # Project documentation
```

---

## 🚀 Project Overview

**Objective:**  
Develop a robust deep learning pipeline to classify ECG heartbeats into five classes:

- 0: Normal beat
- 1: Supraventricular premature beat
- 2: Premature ventricular contraction
- 3: Fusion of ventricular and normal beat
- 4: Unknown beat

---

## 🛠️ Setup Instructions

### 1️⃣ Clone this repository

```bash
git clone <https://github.com/sunayana90/ECG-Classification>
cd ecg-classification
```

### 2️⃣ Install dependencies

Install required Python packages:

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn
```

Or use Google Colab, which has most dependencies pre-installed.

---

## 📈 Training the Model

Open `ecg_classification.ipynb` in Google Colab or Jupyter Notebook, and run all cells sequentially.

The notebook includes:
- Dataset download via Kaggle API
- Data preprocessing & normalization
- Custom PyTorch Dataset & DataLoader
- 1D CNN model architecture
- Model training loop
- Evaluation metrics & confusion matrix visualization

---

## 🧪 Evaluation Results

**Test Accuracy:** ~98%  
**Macro-averaged F1 Score:** ~0.91

**Classification Report:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0 (Normal) | 0.99 | 1.00 | 0.99 |
| 1 | 0.94 | 0.74 | 0.83 |
| 2 | 0.96 | 0.95 | 0.95 |
| 3 | 0.86 | 0.73 | 0.79 |
| 4 | 1.00 | 0.98 | 0.99 |

For detailed analysis, see [report.md](https://github.com/sunayana90/ECG-Classification/blob/main/report.md).

---

## 💾 Inference with Trained Model

To load the trained model weights:

```python
import torch

model = ECG1DCNN(num_classes=5)
model.load_state_dict(torch.load("ecg_cnn.pth"))
model.eval()
```

Then you can run predictions on new ECG samples.

---

## ✨ Possible Improvements

- Use class-weighted loss to improve minority class recall
- Apply data augmentation techniques
- Train for more epochs or experiment with different architectures
- Implement model explainability (e.g., Grad-CAM)

---

## 📄 License

This project is shared for educational purposes only.

---

## 🙏 Acknowledgements

- [MIT-BIH Arrhythmia Database](https://www.kaggle.com/shayanfazeli/heartbeat)
- PyTorch Team
- Google Colab

# 🔍 Image Blur Classification

<p align="center">
  <strong>Sharp vs. Blurred Image Classification using Handcrafted Image-Quality Features and a Multi-Layer Perceptron (MLP)</strong>
</p>

<p align="center">
  <a href="https://colab.research.google.com/drive/193vrYTJBat8rydM1tnBbaYfpOZVs4Snn">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
  </a>
  <a href="https://github.com/Randsalem19">
    <img src="https://img.shields.io/badge/GitHub-Randsalem19-181717?style=flat&logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="https://www.linkedin.com/in/rand-majed-salem/">
    <img src="https://img.shields.io/badge/LinkedIn-Rand%20Majed%20Salem-0A66C2?style=flat&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
  <a href="https://www.kaggle.com/randsalem">
    <img src="https://img.shields.io/badge/Kaggle-randsalem-20BEFF?style=flat&logo=kaggle&logoColor=white" alt="Kaggle">
  </a>
</p>

---

## 📌 Project Overview

This project builds a lightweight machine-learning pipeline to classify images as either **Sharp** or **Blurred**.

Instead of training directly on raw pixels, the system extracts four handcrafted image-quality features and uses them as input to a **Scikit-learn Multi-Layer Perceptron (MLP)** classifier.

Two optimization strategies are evaluated using the same network architecture:

- **Adam**
- **Stochastic Gradient Descent (SGD)**

The best recorded result was achieved using **Adam**, with **94.81% test accuracy**.

---

## ✨ Project Highlights

- Binary classification: **Sharp vs. Blurred**
- Handcrafted spatial and frequency-domain features
- Image preprocessing with **OpenCV**
- Feature standardization with **StandardScaler**
- Neural-network classification using **MLPClassifier**
- Comparison between **Adam** and **SGD**
- Evaluation using accuracy, precision, recall, F1-score, and confusion matrices
- Prediction workflow for unseen images
- Fully runnable notebook on **Google Colab**

---

## 🚀 Run the Project

Open the complete notebook directly in Google Colab:

### [▶ Open Image Blur Classification in Google Colab](https://colab.research.google.com/drive/193vrYTJBat8rydM1tnBbaYfpOZVs4Snn)

---

## 📊 Dataset

The project uses the public **Blur Dataset** from Kaggle:

**Dataset:** https://www.kaggle.com/datasets/kwentar/blur-dataset

The selected subset contains:

| Category | Images |
|---|---:|
| Sharp | 300 |
| Defocus Blur | 300 |
| Motion Blur | 300 |
| **Total** | **900** |

For binary classification:

```text
0 → Sharp
1 → Blur
```

Motion-blurred and defocus-blurred images are combined into one **Blur** class.

---

## ⚙️ Machine Learning Pipeline

```text
Image Dataset
      ↓
Image Loading with OpenCV
      ↓
Grayscale Conversion
      ↓
Resize to 256 × 256
      ↓
Handcrafted Feature Extraction
      ↓
Feature Standardization
      ↓
Stratified 70/30 Train-Test Split
      ↓
MLP Neural Network
      ↓
Adam vs. SGD Comparison
      ↓
Model Evaluation
      ↓
Prediction on New Images
```

---

## 🧠 Feature Engineering

Each image is represented using four numerical features:

| Feature | Description |
|---|---|
| **Variance of Laplacian** | Measures edge sharpness and local detail |
| **Tenengrad** | Measures edge strength using Sobel gradients |
| **Image Contrast** | Measures variation in image intensity |
| **High-Frequency Energy (FFT)** | Measures high-frequency detail associated with sharpness |

Final feature vector:

```text
[Laplacian Variance, Tenengrad, Contrast, FFT Energy]
```

---

## 🔧 Data Preparation

The dataset is split using a **stratified 70/30 train-test split**:

| Split | Samples |
|---|---:|
| Training | 630 |
| Testing | 270 |

Features are standardized using `StandardScaler`.

---

## 🤖 Model Architecture

The classifier is a Multi-Layer Perceptron with two hidden layers:

```text
4 Input Features
      ↓
Hidden Layer — 32 Neurons — ReLU
      ↓
Hidden Layer — 16 Neurons — ReLU
      ↓
Binary Classification
      ↓
Sharp / Blur
```

Core configuration:

```python
MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    max_iter=300,
    random_state=42
)
```

---

## 📈 Results

| Optimizer | Test Accuracy |
|---|---:|
| **Adam** | **94.81%** |
| SGD | 92.59% |

### Adam

```text
Confusion Matrix:
[[ 87   3]
 [ 11 169]]
```

### SGD

```text
Confusion Matrix:
[[ 82   8]
 [ 12 168]]
```

Adam produced the strongest recorded performance and is used in the final prediction workflow.

---

## 🔮 Predicting New Images

For each new image, the system:

1. Loads the image
2. Converts it to grayscale
3. Resizes it to `256 × 256`
4. Extracts the four handcrafted features
5. Applies the fitted scaler
6. Runs the trained MLP model
7. Returns the predicted class and confidence score

Example:

```text
Prediction: Sharp
Confidence: 0.9933
```

---

## 🛠️ Technologies Used

`Python` · `OpenCV` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Jupyter Notebook` · `Google Colab` · `Kaggle`

---

## 📁 Repository Structure

```text
image-blur-classification/
│
├── Image_Blur_Classification_Rand_Majed_Salem.ipynb
├── Image_Blur_Classification_Report_Rand_Majed_Salem.pdf
├── README.md
└── requirements.txt
```

---

## 📄 Project Report

A detailed report covering the methodology, implementation, experiments, and results is available here:

[📄 View Project Report](Image_Blur_Classification_Report_Rand_Majed_Salem.pdf)

---

## ⚠️ Limitations

- The selected subset contains more blurred images than sharp images.
- The classifier relies on handcrafted features rather than end-to-end representation learning.
- Performance may vary on images from substantially different capture conditions.
- Both MLP configurations use a fixed `max_iter=300`, so further tuning may improve convergence.

---

## 🔭 Future Improvements

- Compare the current method with **CNN-based models**
- Perform systematic hyperparameter optimization
- Evaluate additional blur-detection features
- Test on additional external datasets
- Estimate **blur severity**, not only binary blur classification
- Deploy the model as a lightweight web application or REST API

---

## 👩‍💻 Author

### **Rand Majed Salem**

Data Science & Artificial Intelligence

[![GitHub](https://img.shields.io/badge/GitHub-Randsalem19-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Randsalem19)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Rand%20Majed%20Salem-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rand-majed-salem/)
[![Kaggle](https://img.shields.io/badge/Kaggle-randsalem-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/randsalem)

---

<p align="center">
  <strong>⭐ If you find this project useful or interesting, consider starring the repository.</strong>
</p>

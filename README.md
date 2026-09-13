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

Image blur is a common image-quality problem caused by motion, defocus, camera shake, or acquisition conditions. It reduces edge clarity and fine visual detail, which can negatively affect downstream computer-vision systems.

This project builds a lightweight and interpretable machine-learning pipeline to classify images into two categories:

- **Sharp**
- **Blurred**

Instead of training directly on raw pixels, the project extracts a compact set of handcrafted image-quality features from each image and uses them as inputs to a **Scikit-learn Multi-Layer Perceptron (`MLPClassifier`)**.

Two optimization strategies — **Adam** and **Stochastic Gradient Descent (SGD)** — are evaluated using the same network architecture to compare their performance on the blur-classification task.

---

## ✨ Project Highlights

- Binary classification of **Sharp vs. Blurred** images.
- Four handcrafted image-quality features combining spatial and frequency-domain information.
- Image preprocessing with **OpenCV**.
- Feature standardization using **StandardScaler**.
- Neural-network classification using **MLPClassifier**.
- Direct comparison between **Adam** and **SGD**.
- Evaluation using accuracy, precision, recall, F1-score, and confusion matrices.
- Reusable prediction pipeline for unseen images.
- Fully runnable notebook available on **Google Colab**.

---

## 🚀 Run the Project

The complete notebook can be opened directly in Google Colab:

### [▶ Open Image Blur Classification in Google Colab](https://colab.research.google.com/drive/193vrYTJBat8rydM1tnBbaYfpOZVs4Snn)

---

## 📊 Dataset

The project uses the public **Blur Dataset** available on Kaggle:

**Dataset:** https://www.kaggle.com/datasets/kwentar/blur-dataset

The original dataset contains three image categories:

- Sharp images
- Motion-blurred images
- Defocus-blurred images

For this binary classification task, motion blur and defocus blur are combined into a single **Blur** class.

### Dataset Subset Used

| Category | Number of Images |
|---|---:|
| Sharp | 300 |
| Defocus Blur | 300 |
| Motion Blur | 300 |
| **Total** | **900** |

### Final Labels

```text
0 → Sharp
1 → Blur
```

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

## 🖼️ Image Preprocessing

Each image is processed through the same preparation pipeline before feature extraction:

1. Load the image using OpenCV.
2. Convert it to grayscale.
3. Resize it to **256 × 256 pixels**.
4. Extract the four image-quality features used by the classifier.

Using grayscale images reduces computational complexity while retaining the structural information needed to analyze image sharpness.

---

## 🧠 Feature Engineering

Each image is represented using four numerical image-quality features.

| Feature | Description |
|---|---|
| **Variance of Laplacian** | Measures edge sharpness and local detail. |
| **Tenengrad** | Uses Sobel gradients to measure edge strength. |
| **Image Contrast** | Measures variation in image intensity values. |
| **High-Frequency Energy (FFT)** | Measures frequency-domain detail associated with sharp visual content. |

The final input vector is:

```text
[Laplacian Variance, Tenengrad, Contrast, FFT Energy]
```

This approach provides a compact and interpretable representation instead of feeding the full image directly into the model.

---

## 🔧 Data Preparation

The 900-image subset is divided using a **stratified 70/30 train-test split**.

| Split | Samples |
|---|---:|
| Training | 630 |
| Testing | 270 |

The numerical features are standardized using `StandardScaler`.

The scaler is fitted only on the training data and then applied to the testing data to keep the evaluation pipeline consistent.

---

## 🤖 Model Architecture

The project uses a **Multi-Layer Perceptron (MLP)** neural network with two hidden layers.

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

### Core Configuration

```python
MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    max_iter=300,
    random_state=42
)
```

Two solvers are evaluated independently:

```text
Adam
SGD
```

---

## 📈 Experimental Results

### 🥇 Adam Optimizer

Adam achieved the strongest overall performance in the recorded experiment.

| Metric | Result |
|---|---:|
| **Accuracy** | **94.81%** |
| Sharp Precision | 89% |
| Sharp Recall | 97% |
| Sharp F1-score | 93% |
| Blur Precision | 98% |
| Blur Recall | 94% |
| Blur F1-score | 96% |

### Confusion Matrix — Adam

```text
[[ 87   3]
 [ 11 169]]
```

Out of the 270 test images, the Adam model correctly classified **256 images**.

---

### SGD Optimizer

| Metric | Result |
|---|---:|
| **Accuracy** | **92.59%** |
| Sharp Precision | 87% |
| Sharp Recall | 91% |
| Sharp F1-score | 89% |
| Blur Precision | 95% |
| Blur Recall | 93% |
| Blur F1-score | 94% |

### Confusion Matrix — SGD

```text
[[ 82   8]
 [ 12 168]]
```

---

## 🏆 Adam vs. SGD

| Optimizer | Test Accuracy |
|---|---:|
| **Adam** | **94.81%** |
| SGD | 92.59% |

Adam produced the best classification accuracy in this experiment and is therefore used by the notebook's final prediction workflow.

---

## 🔮 Predicting New Images

The notebook includes a reusable prediction workflow for unseen images.

For each new image, the system:

1. Loads the image.
2. Converts it to grayscale.
3. Resizes it to `256 × 256`.
4. Extracts the same four handcrafted features.
5. Applies the fitted scaler.
6. Passes the feature vector to the trained Adam MLP model.
7. Returns the predicted class and confidence score.

Example output:

```text
Prediction: Sharp
Confidence: 0.9933
```

or:

```text
Prediction: Blur
Confidence: 0.9778
```

---

## 🛠️ Technologies Used

- **Python**
- **OpenCV**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Jupyter Notebook**
- **Google Colab**
- **Kaggle**

---

## 📁 Repository Structure

```text
Image-Blur-Classification/
│
├── notebooks/
│   └── Image_Blur_Classification.ipynb
│
├── docs/
│   └── Image_Blur_Classification_Report.pdf
│
├── .gitignore
├── CREDITS.md
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 📄 Project Report

A detailed project report covering the methodology, implementation, experiments, and results is included in the repository:

[`docs/Image_Blur_Classification_Report.pdf`](docs/Image_Blur_Classification_Report.pdf)

---

## ⚠️ Limitations

- The selected subset contains more blurred images than sharp images.
- The classifier relies on handcrafted features rather than end-to-end image representation learning.
- Results may vary on images captured under conditions that differ significantly from the source dataset.
- Both MLP configurations use a fixed `max_iter=300`, so further convergence and hyperparameter analysis could improve performance.

---

## 🔭 Future Improvements

Possible extensions include:

- Compare the current approach with **Convolutional Neural Networks (CNNs)**.
- Evaluate additional blur-detection features.
- Perform systematic hyperparameter optimization.
- Test the model on additional external datasets.
- Expand the dataset with more real-world blur conditions.
- Estimate **blur severity** instead of only performing binary classification.
- Deploy the classifier as a lightweight web application or REST API.

---

## 👩‍💻 Author

### **Rand Majed Salem**

Data Science & Artificial Intelligence

[![GitHub](https://img.shields.io/badge/GitHub-Randsalem19-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Randsalem19)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Rand%20Majed%20Salem-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rand-majed-salem/)
[![Kaggle](https://img.shields.io/badge/Kaggle-randsalem-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/randsalem)

### Project Links

- **Google Colab:** https://colab.research.google.com/drive/193vrYTJBat8rydM1tnBbaYfpOZVs4Snn
- **GitHub Profile:** https://github.com/Randsalem19
- **LinkedIn:** https://www.linkedin.com/in/rand-majed-salem/
- **Kaggle:** https://www.kaggle.com/randsalem

---


<p align="center">
  <strong>⭐ If you find this project useful or interesting, consider starring the repository.</strong>
</p>

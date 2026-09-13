# 🔍 Image Blur Classification

> A machine-learning project for classifying images as **Sharp** or **Blurred** using handcrafted image-quality features and a Multi-Layer Perceptron (MLP).

<p align="center">
  <a href="https://colab.research.google.com/drive/193vrYTJBat8rydM1tnBbaYfpOZVs4Snn"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
  <a href="https://github.com/Randsalem19"><img src="https://img.shields.io/badge/GitHub-Randsalem19-181717?style=flat&logo=github" alt="GitHub"></a>
  <a href="https://www.linkedin.com/in/rand-majed-salem/"><img src="https://img.shields.io/badge/LinkedIn-Rand%20Majed%20Salem-0A66C2?style=flat&logo=linkedin" alt="LinkedIn"></a>
  <a href="https://www.kaggle.com/randsalem"><img src="https://img.shields.io/badge/Kaggle-randsalem-20BEFF?style=flat&logo=kaggle&logoColor=white" alt="Kaggle"></a>
</p>

## Overview

Image blur reduces edge clarity and fine visual detail, which can degrade the performance of downstream computer-vision systems. This project uses a lightweight, interpretable pipeline to classify images into two categories:

- **Sharp**
- **Blurred**

Instead of training directly on raw pixels, the notebook extracts four handcrafted image-quality features and uses them as inputs to a **Scikit-learn `MLPClassifier`**. Two optimizers, **Adam** and **SGD**, are compared using the same network architecture.

## Open in Google Colab

Run the project interactively in Google Colab:

**[Open the notebook in Colab](https://colab.research.google.com/drive/193vrYTJBat8rydM1tnBbaYfpOZVs4Snn)**

## Project Objectives

- Distinguish sharp images from blurred images.
- Compare spatial- and frequency-domain blur indicators.
- Train an MLP on compact handcrafted features.
- Compare Adam and SGD optimization.
- Evaluate the models with accuracy, precision, recall, F1-score, and confusion matrices.
- Provide a reusable prediction function for unseen images.

## Dataset

The project uses the public **Blur Dataset** from Kaggle:

**Dataset:** https://www.kaggle.com/datasets/kwentar/blur-dataset

The original data contains three categories:

- Sharp images
- Motion-blurred images
- Defocus-blurred images

For binary classification, motion blur and defocus blur are merged into one **Blur** class.

### Subset used in the notebook

| Category | Images |
|---|---:|
| Sharp | 300 |
| Defocus blur | 300 |
| Motion blur | 300 |
| **Total** | **900** |

Final labels:

- `0` → Sharp
- `1` → Blur

## Pipeline

```text
Kaggle Dataset
      ↓
Grayscale + Resize (256×256)
      ↓
Handcrafted Feature Extraction
      ↓
StandardScaler
      ↓
Stratified 70/30 Train-Test Split
      ↓
MLP Classifier
      ↓
Adam vs. SGD
      ↓
Evaluation + Prediction
```

## Feature Extraction

Each image is represented by four numerical features:

| Feature | Purpose |
|---|---|
| Variance of Laplacian | Measures edge sharpness |
| Tenengrad | Measures Sobel-gradient strength |
| Image Contrast | Measures intensity variation |
| High-Frequency Energy (FFT) | Measures high-frequency visual content |

The final feature vector is:

```text
[Laplacian Variance, Tenengrad, Contrast, FFT Energy]
```

## Data Preparation

The 900-image subset is split using a stratified **70/30 train-test split**.

| Split | Samples |
|---|---:|
| Training | 630 |
| Testing | 270 |

Features are standardized with `StandardScaler`, fitted on the training set only and then applied to the test set.

## Model Architecture

The project uses a Multi-Layer Perceptron with the following configuration:

```text
4 Input Features
      ↓
Dense Layer — 32 neurons — ReLU
      ↓
Dense Layer — 16 neurons — ReLU
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
    random_state=42,
)
```

The same architecture is evaluated with two solvers:

- `adam`
- `sgd`

## Results

### Adam

| Metric | Result |
|---|---:|
| Accuracy | **94.81%** |
| Sharp Precision | 89% |
| Sharp Recall | 97% |
| Sharp F1-score | 93% |
| Blur Precision | 98% |
| Blur Recall | 94% |
| Blur F1-score | 96% |

Confusion matrix:

```text
[[ 87   3]
 [ 11 169]]
```

### SGD

| Metric | Result |
|---|---:|
| Accuracy | **92.59%** |
| Sharp Precision | 87% |
| Sharp Recall | 91% |
| Sharp F1-score | 89% |
| Blur Precision | 95% |
| Blur Recall | 93% |
| Blur F1-score | 94% |

Confusion matrix:

```text
[[ 82   8]
 [ 12 168]]
```

### Optimizer Comparison

| Optimizer | Accuracy |
|---|---:|
| **Adam** | **94.81%** |
| SGD | 92.59% |

Adam achieved the strongest performance in the recorded experiment and is used by the notebook's final prediction function.

## Predicting New Images

The notebook includes a reusable `predict_image()` function that:

1. Loads an image.
2. Converts it to grayscale.
3. Resizes it to `256 × 256`.
4. Extracts the same four features used during training.
5. Applies the fitted scaler.
6. Predicts with the Adam MLP.
7. Reports the predicted class and confidence score.

Example recorded outputs include:

```text
Prediction: Sharp
Confidence: 0.9933
```

and

```text
Prediction: Blur
Confidence: 0.9778
```

## Technologies

- Python
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook / Google Colab
- Kaggle CLI

## Repository Structure

```text
Image-Blur-Classification/
├── notebooks/
│   └── Image_Blur_Classification.ipynb
├── docs/
│   └── Image_Blur_Classification_Report.pdf
├── .gitignore
├── CREDITS.md
├── LICENSE
├── README.md
└── requirements.txt
```

## Project Report

The original project report is preserved in:

[`docs/Image_Blur_Classification_Report.pdf`](docs/Image_Blur_Classification_Report.pdf)

## Limitations

- The selected subset is class-imbalanced: 300 sharp images versus 600 blurred images.
- The system relies on handcrafted features instead of end-to-end visual representation learning.
- Performance may change on images from cameras, blur patterns, or domains that differ from the source dataset.
- Both recorded MLP runs reached the configured `max_iter=300`, so convergence and hyperparameter tuning remain areas for improvement.

## Future Work

- Compare the handcrafted-feature MLP with CNN-based approaches.
- Add more blur-quality features and feature-selection experiments.
- Perform hyperparameter tuning and convergence analysis.
- Evaluate on additional external datasets.
- Estimate blur severity instead of only binary classes.
- Deploy the classifier through a lightweight web application or API.

## 👩‍💻 Author / Repository Owner

**Rand Majed Salem**  
Data Science & Artificial Intelligence

- GitHub: https://github.com/Randsalem19
- LinkedIn: https://www.linkedin.com/in/rand-majed-salem/
- Kaggle: https://www.kaggle.com/randsalem
- Colab: https://colab.research.google.com/drive/193vrYTJBat8rydM1tnBbaYfpOZVs4Snn

## Credits & Provenance

This portfolio repository was prepared and maintained by **Rand Majed Salem** from project materials supplied for this repository. The included original report attributes its preparation to **Nagham Ehsan Zidiah** and is intentionally preserved unmodified. See [`CREDITS.md`](CREDITS.md) for provenance details.

## License

This repository is distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

If you find the project useful, consider giving the repository a ⭐.

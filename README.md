# multitask-art-classifier

A deep learning project implementing a **Multi-Task Learning (MTL)** convolutional network to simultaneously predict art styles (genres) and identify artists from fine art images.

---

## 📌 Project Overview

Identifying painting styles and attributing authorship are complex computer vision tasks due to overlapping artistic influences, brushwork variations, and limited training data. This project solves both problems jointly by training a shared feature extractor with two task-specific prediction heads.

* **Primary Task:** Art Style / Genre Classification.
* **Secondary Task:** Artist Identification.
* **Dataset:** [Best Artworks of All Time (Kaggle)](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time) filtered to the top 10 most represented artists.

---

## 🏗️ Architecture

* **Backbone:** Pre-trained **EfficientNet-B0** on ImageNet (Transfer Learning).
* **Shared Representation:** `GlobalMaxPooling2D` followed by `BatchNormalization`, `Dropout (0.3)`, and a 512-unit `Dense (ReLU)` bottleneck layer.
* **Multi-Task Heads:**
  * `genre_out`: Softmax layer predicting artistic movements.
  * `artist_out`: Softmax layer predicting specific artists.
* **Loss Strategy:** Weighted Categorical Crossentropy:
  $$\mathcal{L}_{total} = 1.0 \times \mathcal{L}_{genre} + 0.7 \times \mathcal{L}_{artist}$$
* **Fine-tuning:** End-to-end unfreezing of the base model using a reduced learning rate ($10^{-5}$) with `EarlyStopping` and `ModelCheckpoint`.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Framework:** TensorFlow 2.x / Keras
* **Hardware:** NVIDIA Tesla T4 GPU (Google Colab)
* **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

---

## 🚀 Pipeline & Usage

1. **Environment Setup:** Mount Google Drive and supply Kaggle API credentials (`kaggle.json`).
2. **Data Ingestion & Filtering:** Automatically download the dataset and filter paintings for the target artists.
3. **Data Generation:** Custom multi-output generator with real-time geometric augmentations (rotation, shift, zoom, horizontal flip).
4. **Training & Fine-tuning:** Two-stage training regime (frozen feature extraction followed by full-network fine-tuning).
5. **Evaluation:** Detailed classification reports and confusion matrices for both tasks.

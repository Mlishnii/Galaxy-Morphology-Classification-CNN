# Galaxy Morphology Classification with CNN

A convolutional neural network that classifies galaxy images into three morphological types — **Spiral**, **Elliptical**, and **Irregular** — using the Galaxy10 DECals dataset.

## Overview

Modern sky surveys (e.g. SDSS) image hundreds of millions of galaxies, making manual morphological classification infeasible. This project builds an automated CNN pipeline to classify galaxy images, reducing reliance on manual expert labeling.

## Dataset

- **Source:** [Galaxy10 DECals](https://astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5) (Leung & Bovy, 2019) — 17,736 images from the DECam Legacy Survey
- **Classes used:** Spiral, Elliptical, Irregular (relabeled from the original 10 Galaxy10 classes)
- **Samples:** 1,000 images per class (3,000 total, perfectly balanced)
- **Split:** 70% train / 15% validation / 15% test (stratified)

## Methodology

- **Preprocessing:** resize to 64×64, pixel normalization, one-hot encoding
- **Augmentation:** rotation, horizontal/vertical flip, zoom, shift (Keras `ImageDataGenerator`)
- **Architecture:** custom CNN — 3 Conv2D blocks (32/64/128 filters) with BatchNorm + MaxPooling, followed by a Dense(256) layer with Dropout(0.5)
- **Training:** Adam optimizer, categorical cross-entropy loss, batch size 32, early stopping (patience=8)

## CNN Architecture

| Layer | Output Shape | Parameters |
|---|---|---|
| Conv2D (32 filters, 3×3) + ReLU | 64×64×32 | 896 |
| BatchNormalization + MaxPool | 32×32×32 | 128 |
| Conv2D (64 filters, 3×3) + ReLU | 32×32×64 | 18,496 |
| BatchNormalization + MaxPool | 16×16×64 | 256 |
| Conv2D (128 filters, 3×3) + ReLU | 16×16×128 | 73,856 |
| BatchNormalization + MaxPool | 8×8×128 | 512 |
| Flatten | 8,192 | 0 |
| Dense (256) + ReLU | 256 | 2,097,408 |
| Dropout (0.5) | 256 | 0 |
| Dense (3, Softmax) | 3 | 771 |
| **Total Trainable Parameters** | | **2,191,875** |

## Results

Training stopped at epoch 26 via early stopping (best weights restored from epoch 18). Best validation accuracy: **71.28%**.

| Class      | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Spiral     | 0.65      | 0.67   | 0.66     |
| Elliptical | 0.79      | 0.67   | 0.73     |
| Irregular  | 0.60      | 0.67   | 0.64     |
| **Overall accuracy** |   |        | **67.11%** |

![Confusion Matrix](images/confusion_matrix.png)
![Sample Predictions](images/sample_predictions.png)
![Accuracy & Loss Curves](images/accuracy_loss_curves.png)

## Key Findings

- Elliptical galaxies achieved the highest precision (0.79), while Irregular galaxies had the lowest (0.60) — the model still confuses Irregular galaxies with other classes more often due to their lack of a defining shape.
- All three classes had identical recall (0.67), meaning the model detects roughly two-thirds of each class correctly regardless of type.
- Data augmentation and dropout effectively controlled overfitting on a relatively small dataset (2,167 training images).

## Future Work

- Transfer learning with pretrained architectures (ResNet, VGG)
- Higher input resolution to preserve fine structural details
- Larger training set

## Project Structure

```
galaxy-morphology-cnn-classification/
├── README.md
├── Galaxy_Classification_CNN.ipynb
├── report/
│   └── Galaxy_Classification_Report.pdf
├── images/
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   └── accuracy_loss_curves.png
└── requirements.txt
```

## Documentation

- 📄 [Full project report](report/Galaxy_Classification_Report.pdf)

## How to Run

1. Download the dataset:
   ```bash
   wget https://astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `Galaxy_Classification_CNN.ipynb` in Jupyter or Google Colab and run all cells (GPU recommended).

## References

- Leung, H., & Bovy, J. *astroNN: A Python package for deep learning in astrophysics.* Galaxy10 DECals Dataset Documentation.
- Chollet, F., et al. (2015). *Keras: The Python Deep Learning library.*

## Author

**Melika Asgharnezhad**
Course project — University of Mazandaran, supervised by Dr. Aghajani

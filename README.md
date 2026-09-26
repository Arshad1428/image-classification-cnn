# Image Classification using CNN and Transfer Learning

## Overview

Image classification system built using PyTorch and the Intel Image Classification dataset.

The project implements:

- Custom CNN
- ResNet18 transfer learning
- Image preprocessing
- Data augmentation
- Train/validation/test split
- Model checkpointing
- Evaluation metrics
- MLflow experiment tracking
- Reproducibility
- Image inference
- Streamlit application
- Automated tests

## Dataset

Intel Image Classification dataset.

Classes:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

## Models

### Custom CNN

Three convolutional blocks followed by adaptive pooling and fully connected classification layers.

### ResNet18

Pretrained ResNet18 with the final classification layer replaced for six classes.

## Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

## Experiment Tracking

MLflow is used to track:

- Model type
- Learning rate
- Batch size
- Epochs
- Random seed
- Training loss
- Validation loss
- Training accuracy
- Validation accuracy
- Test metrics
- Model artifacts

## Reproducibility

Random seeds are controlled through the project configuration.

## Project Structure

```
image-classification-cnn/
├── app.py                     # Streamlit inference app
├── main.py                    # Training entry point
├── pyproject.toml             # Package metadata (installable via pip)
├── requirements.txt
├── data/
│   ├── raw/                   # Dataset goes here (see Dataset Setup)
│   └── processed/
├── models/                    # Saved checkpoints (.pth)
├── outputs/
│   ├── figures/                # Confusion matrices, plots
│   └── reports/                # Classification reports
├── experiments/                # MLflow run artifacts
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_custom_cnn.ipynb
│   └── 03_transfer_learning.ipynb
├── src/image_classifier/
│   ├── config.py               # Env-driven configuration
│   ├── data_loader.py          # ImageFolder datasets + splits
│   ├── preprocessing.py        # Train/eval transforms
│   ├── seed.py                 # Reproducibility
│   ├── utils.py                # Device selection, param counting
│   ├── models/
│   │   ├── cnn.py               # Custom CNN
│   │   └── transfer_learning.py # ResNet18 transfer learning
│   ├── training/
│   │   ├── train.py             # Train/validate loops
│   │   └── checkpoint.py        # Save/load checkpoints
│   ├── evaluation/
│   │   └── evaluate.py          # Metrics, confusion matrix, reports
│   └── inference/
│       └── predict.py           # Single-image prediction
└── tests/                      # pytest unit tests
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .                # installs image_classifier as an importable package

cp .env.example .env            # adjust values as needed
```

## Dataset Setup

Download the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) (e.g. via the Kaggle CLI: `kaggle datasets download -d puneet6060/intel-image-classification`), then unzip it so the folder layout matches:

```
data/raw/intel/
├── seg_train/seg_train/{buildings,forest,glacier,mountain,sea,street}/*.jpg
└── seg_test/seg_test/{buildings,forest,glacier,mountain,sea,street}/*.jpg
```

`create_dataloaders()` reads directly from this path, splitting `seg_train` 80/20 into train/validation and using `seg_test` as the held-out test set.

## Running

Train the custom CNN (default):

```bash
python main.py
```

Train with ResNet18 transfer learning instead:

```bash
MODEL_TYPE=resnet18 FREEZE_BACKBONE=true python main.py
```

Launch the inference app once a model checkpoint exists in `models/`:

```bash
streamlit run app.py
```

Run the test suite:

```bash
pytest
```

View MLflow experiment runs:

```bash
mlflow ui --backend-store-uri mlruns
```
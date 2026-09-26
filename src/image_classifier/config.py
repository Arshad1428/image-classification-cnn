import os
from pathlib import Path

from dotenv import load_dotenv


# --------------------------------------------------
# Project root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load .env from project root
load_dotenv(PROJECT_ROOT / ".env")


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

PROJECT_NAME = os.getenv(
    "PROJECT_NAME",
    "image-classification-cnn"
)

DATA_DIR = PROJECT_ROOT / os.getenv(
    "DATA_DIR",
    "data"
)

RAW_DATA_DIR = PROJECT_ROOT / os.getenv(
    "RAW_DATA_DIR",
    "data/raw"
)

PROCESSED_DATA_DIR = PROJECT_ROOT / os.getenv(
    "PROCESSED_DATA_DIR",
    "data/processed"
)

MODEL_DIR = PROJECT_ROOT / os.getenv(
    "MODEL_DIR",
    "models"
)

OUTPUT_DIR = PROJECT_ROOT / os.getenv(
    "OUTPUT_DIR",
    "outputs"
)


# --------------------------------------------------
# Image configuration
# --------------------------------------------------

IMAGE_SIZE = int(
    os.getenv("IMAGE_SIZE", "224")
)

IMAGE_CHANNELS = int(
    os.getenv("IMAGE_CHANNELS", "3")
)


# --------------------------------------------------
# Training configuration
# --------------------------------------------------

BATCH_SIZE = int(
    os.getenv("BATCH_SIZE", "32")
)

EPOCHS = int(
    os.getenv("EPOCHS", "10")
)

LEARNING_RATE = float(
    os.getenv("LEARNING_RATE", "0.001")
)


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

RANDOM_SEED = int(
    os.getenv("RANDOM_SEED", "42")
)


# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = os.getenv(
    "DEVICE",
    "auto"
)


# --------------------------------------------------
# MLflow
# --------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "mlruns"
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "image-classification"
)
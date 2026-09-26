import os
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.optim import Adam

from image_classifier.config import (
    EPOCHS,
    LEARNING_RATE,
    MODEL_DIR,
    MODEL_VERSION,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_SEED,
)
from image_classifier.data_loader import create_dataloaders
from image_classifier.evaluation.evaluate import evaluate_model
from image_classifier.models.cnn import CustomCNN
from image_classifier.models.transfer_learning import create_resnet18
from image_classifier.seed import set_seed
from image_classifier.training.checkpoint import save_checkpoint
from image_classifier.training.train import (
    train_one_epoch,
    validate_one_epoch,
)
from image_classifier.utils import (
    count_parameters,
    get_device,
)


MODEL_TYPE = os.getenv("MODEL_TYPE", "cnn")
FREEZE_BACKBONE = (
    os.getenv("FREEZE_BACKBONE", "false").lower()
    == "true"
)


def create_model():
    if MODEL_TYPE == "cnn":
        return CustomCNN(num_classes=6)

    if MODEL_TYPE == "resnet18":
        return create_resnet18(
            num_classes=6,
            freeze_backbone=FREEZE_BACKBONE,
        )

    raise ValueError(
        f"Unknown MODEL_TYPE: {MODEL_TYPE}"
    )


def main():
    set_seed(RANDOM_SEED)

    device = get_device()

    print(f"Device: {device}")
    print(f"Model: {MODEL_TYPE}")

    train_loader, val_loader, test_loader = (
        create_dataloaders()
    )

    model = create_model().to(device)

    print(
        f"Trainable parameters: "
        f"{count_parameters(model):,}"
    )

    loss_fn = nn.CrossEntropyLoss()

    optimizer = Adam(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
    )

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )

    best_val_loss = float("inf")

    model_path = (
        MODEL_DIR
        / f"{MODEL_TYPE}_{MODEL_VERSION}.pth"
    )

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model_type": MODEL_TYPE,
                "model_version": MODEL_VERSION,
                "epochs": EPOCHS,
                "batch_size": train_loader.batch_size,
                "learning_rate": LEARNING_RATE,
                "random_seed": RANDOM_SEED,
                "device": str(device),
                "freeze_backbone": FREEZE_BACKBONE,
            }
        )

        for epoch in range(EPOCHS):
            print(
                f"\nEpoch {epoch + 1}/{EPOCHS}"
            )

            train_loss, train_accuracy = (
                train_one_epoch(
                    model,
                    train_loader,
                    loss_fn,
                    optimizer,
                    device,
                )
            )

            val_loss, val_accuracy = (
                validate_one_epoch(
                    model,
                    val_loader,
                    loss_fn,
                    device,
                )
            )

            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Train Accuracy: "
                f"{train_accuracy:.4f}"
            )

            print(
                f"Val Loss: {val_loss:.4f} | "
                f"Val Accuracy: "
                f"{val_accuracy:.4f}"
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                step=epoch,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    loss=val_loss,
                    path=model_path,
                )

                print(
                    f"Best model saved: {model_path}"
                )

        results = evaluate_model(
            model,
            test_loader,
            device,
        )

        print("\nTest Results")
        print(
            f"Accuracy: "
            f"{results['accuracy']:.4f}"
        )
        print(
            f"Precision: "
            f"{results['precision']:.4f}"
        )
        print(
            f"Recall: "
            f"{results['recall']:.4f}"
        )
        print(
            f"F1 Score: "
            f"{results['f1']:.4f}"
        )

        print("\nClassification Report")
        print(
            results["classification_report"]
        )

        mlflow.log_metrics(
            {
                "test_accuracy": results[
                    "accuracy"
                ],
                "test_precision": results[
                    "precision"
                ],
                "test_recall": results[
                    "recall"
                ],
                "test_f1": results[
                    "f1"
                ],
            }
        )

        mlflow.log_artifact(
            str(model_path)
        )

        print(
            f"\nModel saved to: {model_path}"
        )


if __name__ == "__main__":
    main()
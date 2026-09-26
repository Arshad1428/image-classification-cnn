from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
    )

    report = classification_report(
        all_labels,
        all_predictions,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix,
        "classification_report": report,
        "all_predictions": all_predictions,
        "all_labels": all_labels,
    }


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: list[str],
    output_path: str | Path | None = None,
) -> None:
    """Plot a confusion matrix and optionally save the figure."""
    figure, axis = plt.subplots()
    image = axis.imshow(confusion_matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)

    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )
    axis.tick_params(axis="x", rotation=45)

    threshold = confusion_matrix.max() / 2 if confusion_matrix.size else 0
    for row_index in range(confusion_matrix.shape[0]):
        for column_index in range(confusion_matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                confusion_matrix[row_index, column_index],
                ha="center",
                va="center",
                color="white"
                if confusion_matrix[row_index, column_index] > threshold
                else "black",
            )

    figure.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight")

    plt.close(figure)


def save_classification_report(
    report: str,
    output_path: str | Path,
) -> None:
    """Save a classification report as a UTF-8 text file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
import torch

from image_classifier.models.cnn import CustomCNN
from image_classifier.models.transfer_learning import (
    create_resnet18,
)


def test_custom_cnn_output_shape():
    model = CustomCNN(num_classes=6)

    x = torch.randn(
        2,
        3,
        224,
        224,
    )

    output = model(x)

    assert output.shape == (2, 6)


def test_resnet18_output_shape():
    model = create_resnet18(
        num_classes=6,
        freeze_backbone=True,
    )

    x = torch.randn(
        2,
        3,
        224,
        224,
    )

    output = model(x)

    assert output.shape == (2, 6)
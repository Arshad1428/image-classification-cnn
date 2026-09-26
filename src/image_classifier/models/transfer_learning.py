import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def create_resnet18(num_classes=6, freeze_backbone=False):
    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        num_classes,
    )

    return model
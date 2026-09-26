import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from image_classifier.config import BATCH_SIZE, RAW_DATA_DIR, RANDOM_SEED
from image_classifier.preprocessing import (
    get_eval_transforms,
    get_train_transforms,
)


def create_dataloaders():
    train_root = RAW_DATA_DIR / "intel" / "seg_train" / "seg_train"
    test_root = RAW_DATA_DIR / "intel" / "seg_test" / "seg_test"

    # Dataset used to determine the samples and class labels.
    base_dataset = ImageFolder(
        root=train_root,
        transform=None,
    )

    # Reproducible 80/20 split.
    train_size = int(0.8 * len(base_dataset))
    val_size = len(base_dataset) - train_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)

    indices = torch.randperm(
        len(base_dataset),
        generator=generator,
    ).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create separate datasets so train and validation
    # can have different transformations.
    train_dataset = ImageFolder(
        root=train_root,
        transform=get_train_transforms(),
    )

    val_dataset = ImageFolder(
        root=train_root,
        transform=get_eval_transforms(),
    )

    test_dataset = ImageFolder(
        root=test_root,
        transform=get_eval_transforms(),
    )

    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader
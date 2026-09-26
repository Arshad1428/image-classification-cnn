from image_classifier.data_loader import (
    create_dataloaders,
)


def test_dataloaders():
    train_loader, val_loader, test_loader = (
        create_dataloaders()
    )

    assert len(train_loader.dataset) > 0
    assert len(val_loader.dataset) > 0
    assert len(test_loader.dataset) > 0

    images, labels = next(
        iter(train_loader)
    )

    assert images.shape[1:] == (
        3,
        224,
        224,
    )

    assert labels.ndim == 1
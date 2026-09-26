import torch
from PIL import Image

from image_classifier.preprocessing import (
    get_eval_transforms,
    get_train_transforms,
)


def test_train_transform_shape():
    image = Image.new(
        "RGB",
        (150, 150),
    )

    transform = get_train_transforms()

    output = transform(image)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 224, 224)


def test_eval_transform_shape():
    image = Image.new(
        "RGB",
        (150, 150),
    )

    transform = get_eval_transforms()

    output = transform(image)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 224, 224)
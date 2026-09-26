import torch

from image_classifier.inference.predict import (
    predict_image,
)
from image_classifier.models.cnn import CustomCNN


def test_prediction(tmp_path):
    from PIL import Image

    image_path = tmp_path / "test.jpg"

    image = Image.new(
        "RGB",
        (150, 150),
    )

    image.save(image_path)

    model = CustomCNN(num_classes=6)

    predicted_class, confidence = (
        predict_image(
            model=model,
            image_path=image_path,
            class_names=[
                "buildings",
                "forest",
                "glacier",
                "mountain",
                "sea",
                "street",
            ],
            device=torch.device("cpu"),
        )
    )

    assert predicted_class in [
        "buildings",
        "forest",
        "glacier",
        "mountain",
        "sea",
        "street",
    ]

    assert 0 <= confidence <= 1
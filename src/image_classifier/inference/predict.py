import torch
from PIL import Image

from image_classifier.preprocessing import get_eval_transforms


def predict_image(model, image_path, class_names, device):
    model.eval()

    image = Image.open(image_path).convert("RGB")

    transform = get_eval_transforms()
    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        predicted_index = probabilities.argmax(dim=1).item()

        confidence = probabilities[0, predicted_index].item()

    predicted_class = class_names[predicted_index]

    return predicted_class, confidence
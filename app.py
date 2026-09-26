import os

import streamlit as st
import torch

from image_classifier.models.cnn import CustomCNN
from image_classifier.models.transfer_learning import (
    create_resnet18,
)
from image_classifier.inference.predict import (
    predict_image,
)
from image_classifier.preprocessing import (
    get_eval_transforms,
)
from image_classifier.utils import get_device


CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]

MODEL_TYPE = os.getenv(
    "MODEL_TYPE",
    "cnn",
)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    f"models/{MODEL_TYPE}_v1.0.0.pth",
)


@st.cache_resource
def load_model():
    device = get_device()

    if MODEL_TYPE == "cnn":
        model = CustomCNN(
            num_classes=len(CLASS_NAMES)
        )
    else:
        model = create_resnet18(
            num_classes=len(CLASS_NAMES)
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, device


st.title("Intel Image Classification")
st.write(
    "Classify natural and urban scene images "
    "using a PyTorch CNN."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded image",
        use_container_width=True,
    )

    model, device = load_model()

    if st.button("Predict"):
        with open(
            "temp_uploaded_image.jpg",
            "wb",
        ) as file:
            file.write(
                uploaded_file.getbuffer()
            )

        predicted_class, confidence = (
            predict_image(
                model,
                "temp_uploaded_image.jpg",
                CLASS_NAMES,
                device,
            )
        )

        st.success(
            f"Prediction: {predicted_class}"
        )

        st.info(
            f"Confidence: {confidence:.2%}"
        )
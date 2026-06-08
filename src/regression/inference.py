import threading
from typing import Dict, Any

import torch
from PIL import Image
from torchvision import transforms

from .model import ViTRegressor


_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()


def get_inference_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_regression_model(model_path: str):
    global _MODEL_CACHE

    with _MODEL_CACHE_LOCK:
        if model_path in _MODEL_CACHE:
            return _MODEL_CACHE[model_path]

        device = get_inference_device()
        checkpoint = torch.load(model_path, map_location="cpu")

        model = ViTRegressor(
            model_name=checkpoint["model_name"],
            output_dim=len(checkpoint["target_columns"]),
            dropout=checkpoint.get("dropout", 0.1),
            pretrained=False
        )

        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device)
        model.eval()

        cached = (
            model,
            checkpoint["target_columns"],
            checkpoint["img_size"],
            checkpoint["target_min"],
            checkpoint["target_max"],
            device,
            checkpoint
        )

        _MODEL_CACHE[model_path] = cached

        return cached


def predict_regression_image(model_path: str, image_path: str) -> Dict[str, Any]:
    model, target_columns, img_size, target_min, target_max, device, checkpoint = load_regression_model(model_path)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        normalized_outputs = model(x)[0].detach().cpu()

    scores = normalized_outputs * (target_max - target_min) + target_min

    predictions = {
        trait: round(float(score), 4)
        for trait, score in zip(target_columns, scores)
    }

    return {
        "target_columns": target_columns,
        "predictions": predictions,
        "normalized_outputs": [
            round(float(value), 6)
            for value in normalized_outputs
        ],
        "experiment_id": checkpoint.get("experiment_id"),
        "experiment_name": checkpoint.get("experiment_name"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_phase": checkpoint.get("phase")
    }
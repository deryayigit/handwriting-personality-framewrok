import math
import threading
from typing import Dict, Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .model import ViTClassifier


_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()


def get_inference_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_model(model_path: str):
    global _MODEL_CACHE

    with _MODEL_CACHE_LOCK:
        if model_path in _MODEL_CACHE:
            return _MODEL_CACHE[model_path]

        device = get_inference_device()
        checkpoint = torch.load(model_path, map_location="cpu")

        model = ViTClassifier(
            model_name=checkpoint["model_name"],
            num_classes=len(checkpoint["class_names"]),
            dropout=checkpoint.get("dropout", 0.1),
            pretrained=False
        )

        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device)
        model.eval()

        cached = (
            model,
            checkpoint["class_names"],
            checkpoint["img_size"],
            device,
            checkpoint
        )

        _MODEL_CACHE[model_path] = cached

        return cached


def compute_entropy(probs):
    return -sum(p * math.log(p + 1e-8) for p in probs)


def predict_image(model_path: str, image_path: str) -> Dict[str, Any]:
    model, class_names, img_size, device, checkpoint = load_model(model_path)

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
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().tolist()

    trait_scores = [p * 100 for p in probs]
    predicted_index = max(
        range(len(trait_scores)),
        key=lambda i: trait_scores[i]
    )

    model_confidence = max(trait_scores)

    entropy = compute_entropy(probs)
    max_entropy = math.log(len(probs))
    entropy_confidence = (1 - entropy / max_entropy) * 100

    return {
        "class_names": class_names,
        "traits": trait_scores,
        "predicted_index": predicted_index,
        "predicted_class": class_names[predicted_index],
        "model_confidence": model_confidence,
        "entropy_confidence": entropy_confidence,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_phase": checkpoint.get("phase"),
        "experiment_id": checkpoint.get("experiment_id"),
        "experiment_name": checkpoint.get("experiment_name")
    }
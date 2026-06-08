from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str]
) -> Dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0

    y_true = []
    y_pred = []

    for x, y, _ in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)
        predictions = torch.argmax(logits, dim=1)

        total_loss += loss.item() * x.size(0)
        total += x.size(0)

        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())

    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )

    return {
        "loss": round(total_loss / max(1, total), 6),
        "accuracy": round(float(acc), 6),
        "macro_precision": round(float(precision), 6),
        "macro_recall": round(float(recall), 6),
        "macro_f1": round(float(f1), 6),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }


def compute_class_weights(items: List[Dict[str, Any]], num_classes: int) -> torch.Tensor:
    labels = [int(item["label"]) for item in items]
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()

    weights = []
    for count in counts:
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * count))

    return torch.tensor(weights, dtype=torch.float32)
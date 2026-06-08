from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.utils import get_device, save_json
from src.regression.config import RegressionConfig
from src.regression.dataset import HandwritingRegressionDataset
from src.regression.model import ViTRegressor


def _to_builtin_int_list(values: List[int]) -> List[int]:
    return [int(v) for v in values]


@torch.no_grad()
def evaluate_regression_dominant_trait(
    cfg: RegressionConfig,
    split: str = "test",
    model_path: str = None
) -> Dict[str, Any]:
    if split not in cfg.dataset_splits:
        raise ValueError(f"Split not found in dataset_splits: {split}")

    device = get_device()

    if model_path is None:
        model_path = str(cfg.run_dir / "model_best.pt")

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")

    target_columns = checkpoint.get("target_columns", cfg.target_columns)
    target_min = checkpoint.get("target_min", cfg.target_min)
    target_max = checkpoint.get("target_max", cfg.target_max)

    dataset = HandwritingRegressionDataset(
        csv_path=cfg.dataset_splits[split],
        image_dir=cfg.image_dir,
        target_columns=target_columns,
        img_size=cfg.img_size,
        target_min=target_min,
        target_max=target_max,
        train=False
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda")
    )

    model = ViTRegressor(
        model_name=checkpoint["model_name"],
        output_dim=len(target_columns),
        dropout=checkpoint.get("dropout", cfg.dropout),
        pretrained=False
    )

    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    records = []

    for x, targets, image_paths in tqdm(loader, desc=f"dominant trait eval ({split})", leave=False):
        x = x.to(device, non_blocking=True)

        predicted_scores_normalized = model(x).detach().cpu().numpy()
        true_scores_normalized = targets.detach().cpu().numpy()

        predicted_scores = predicted_scores_normalized * (target_max - target_min) + target_min
        true_scores = true_scores_normalized * (target_max - target_min) + target_min

        true_indices = np.argmax(true_scores, axis=1)
        pred_indices = np.argmax(predicted_scores, axis=1)

        y_true.extend(true_indices.tolist())
        y_pred.extend(pred_indices.tolist())

        for i in range(len(image_paths)):
            true_index = int(true_indices[i])
            pred_index = int(pred_indices[i])

            records.append({
                "image_path": image_paths[i],
                "true_dominant_trait": target_columns[true_index],
                "predicted_dominant_trait": target_columns[pred_index],
                "is_correct": bool(true_index == pred_index),
                "true_scores": {
                    trait: round(float(score), 4)
                    for trait, score in zip(target_columns, true_scores[i])
                },
                "predicted_scores": {
                    trait: round(float(score), 4)
                    for trait, score in zip(target_columns, predicted_scores[i])
                }
            })

    labels = list(range(len(target_columns)))

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_columns,
        zero_division=0,
        output_dict=True
    )

    metrics = {
        "experiment_id": cfg.experiment_id,
        "experiment_name": cfg.experiment_name,
        "task": "regression_dominant_trait_evaluation",
        "split": split,
        "model_path": str(model_path),
        "evaluation_logic": (
            "True dominant trait is computed with argmax over real Big Five scores. "
            "Predicted dominant trait is computed with argmax over predicted Big Five scores."
        ),
        "target_columns": target_columns,
        "total_samples": len(y_true),
        "dominant_trait_accuracy": round(float(accuracy), 6),
        "dominant_trait_macro_precision": round(float(precision), 6),
        "dominant_trait_macro_recall": round(float(recall), 6),
        "dominant_trait_macro_f1": round(float(macro_f1), 6),
        "dominant_trait_confusion_matrix": cm.tolist(),
        "dominant_trait_classification_report": report,
        "y_true": _to_builtin_int_list(y_true),
        "y_pred": _to_builtin_int_list(y_pred),
        "records": records
    }

    output_path = cfg.run_dir / f"{split}_dominant_trait_metrics.json"
    save_json(metrics, output_path)

    print(f"[OK] Dominant trait evaluation saved: {output_path}")
    print(f"[OK] Dominant Trait Accuracy: {metrics['dominant_trait_accuracy']:.4f}")
    print(f"[OK] Dominant Trait Macro F1: {metrics['dominant_trait_macro_f1']:.4f}")

    return metrics
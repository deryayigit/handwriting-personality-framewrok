from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def denormalize(values, target_min: float, target_max: float):
    return values * (target_max - target_min) + target_min


def pearson_corr(y_true, y_pred):
    if len(y_true) < 2:
        return 0.0

    true_std = np.std(y_true)
    pred_std = np.std(y_pred)

    if true_std == 0 or pred_std == 0:
        return 0.0

    return float(np.corrcoef(y_true, y_pred)[0, 1])


@torch.no_grad()
def evaluate_regression_model(
    model,
    loader: DataLoader,
    device: torch.device,
    target_columns: List[str],
    target_min: float,
    target_max: float
) -> Dict[str, Any]:
    model.eval()

    all_true = []
    all_pred = []

    for x, y, _ in tqdm(loader, desc="eval", leave=False):
        x = x.to(device)
        y = y.to(device)

        preds = model(x)

        all_true.append(y.detach().cpu().numpy())
        all_pred.append(preds.detach().cpu().numpy())

    y_true = np.vstack(all_true)
    y_pred = np.vstack(all_pred)

    y_true_denorm = denormalize(y_true, target_min, target_max)
    y_pred_denorm = denormalize(y_pred, target_min, target_max)

    mse = float(np.mean((y_true_denorm - y_pred_denorm) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true_denorm - y_pred_denorm)))

    per_trait = {}

    pcc_values = []

    for i, trait in enumerate(target_columns):
        trait_mse = float(np.mean((y_true_denorm[:, i] - y_pred_denorm[:, i]) ** 2))
        trait_rmse = float(np.sqrt(trait_mse))
        trait_mae = float(np.mean(np.abs(y_true_denorm[:, i] - y_pred_denorm[:, i])))
        trait_pcc = pearson_corr(y_true_denorm[:, i], y_pred_denorm[:, i])

        pcc_values.append(trait_pcc)

        per_trait[trait] = {
            "mse": round(trait_mse, 6),
            "rmse": round(trait_rmse, 6),
            "mae": round(trait_mae, 6),
            "pcc": round(trait_pcc, 6)
        }

    return {
        "mse": round(mse, 6),
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "mean_pcc": round(float(np.mean(pcc_values)), 6),
        "per_trait": per_trait
    }
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.common.utils import seed_everything, ensure_dir, save_json, save_yaml, get_device
from src.regression.config import RegressionConfig
from src.regression.dataset import HandwritingRegressionDataset
from src.regression.metrics import evaluate_regression_model
from src.regression.model import ViTRegressor, freeze_backbone


def save_checkpoint(
    model: nn.Module,
    cfg: RegressionConfig,
    output_path: Path,
    epoch_global: int,
    phase_name: str,
    metrics: Dict[str, Any]
) -> None:
    ensure_dir(output_path.parent)

    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": cfg.model_name,
            "target_columns": cfg.target_columns,
            "img_size": cfg.img_size,
            "dropout": cfg.dropout,
            "experiment_id": cfg.experiment_id,
            "experiment_name": cfg.experiment_name,
            "dataset_id": cfg.dataset_id,
            "target_min": cfg.target_min,
            "target_max": cfg.target_max,
            "epoch": epoch_global,
            "phase": phase_name,
            "metrics": metrics
        },
        output_path
    )


def build_regression_loss(cfg: RegressionConfig) -> nn.Module:
    if cfg.loss == "mse":
        return nn.MSELoss()

    if cfg.loss == "smooth_l1":
        return nn.SmoothL1Loss(beta=1.0)

    raise ValueError(f"Unsupported regression loss: {cfg.loss}")


def compute_dominant_trait_labels_from_csv(
    csv_path: str,
    target_columns: List[str]
) -> np.ndarray:
    df = pd.read_csv(csv_path)
    scores = df[target_columns].values.astype(np.float32)
    return np.argmax(scores, axis=1)


def compute_dominant_trait_class_weights(
    labels: np.ndarray,
    num_traits: int
) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_traits)
    total = counts.sum()

    weights = np.zeros(num_traits, dtype=np.float32)

    for index, count in enumerate(counts):
        if count > 0:
            weights[index] = total / (num_traits * count)
        else:
            weights[index] = 0.0

    return weights


def create_regression_weighted_sampler(
    csv_path: str,
    target_columns: List[str]
) -> WeightedRandomSampler:
    labels = compute_dominant_trait_labels_from_csv(
        csv_path=csv_path,
        target_columns=target_columns
    )

    class_weights = compute_dominant_trait_class_weights(
        labels=labels,
        num_traits=len(target_columns)
    )

    sample_weights = [
        float(class_weights[label])
        for label in labels
    ]

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )


def create_sample_weight_lookup(
    csv_path: str,
    image_dir: str,
    target_columns: List[str]
) -> Dict[str, float]:
    df = pd.read_csv(csv_path)

    labels = np.argmax(
        df[target_columns].values.astype(np.float32),
        axis=1
    )

    class_weights = compute_dominant_trait_class_weights(
        labels=labels,
        num_traits=len(target_columns)
    )

    image_root = Path(image_dir)
    weight_lookup = {}

    for index, row in df.iterrows():
        image_path = str(image_root / row["image"])
        label = int(labels[index])
        weight_lookup[image_path] = float(class_weights[label])

    return weight_lookup


@torch.no_grad()
def evaluate_dominant_trait_from_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_columns: List[str],
    target_min: float,
    target_max: float
) -> Dict[str, Any]:
    model.eval()

    y_true = []
    y_pred = []

    for x, y, _ in tqdm(loader, desc="dominant eval", leave=False):
        x = x.to(device, non_blocking=True)

        pred_norm = model(x).detach().cpu().numpy()
        true_norm = y.detach().cpu().numpy()

        pred_scores = pred_norm * (target_max - target_min) + target_min
        true_scores = true_norm * (target_max - target_min) + target_min

        true_idx = np.argmax(true_scores, axis=1)
        pred_idx = np.argmax(pred_scores, axis=1)

        y_true.extend(true_idx.tolist())
        y_pred.extend(pred_idx.tolist())

    labels = list(range(len(target_columns)))

    acc = accuracy_score(y_true, y_pred)

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

    return {
        "dominant_trait_accuracy": round(float(acc), 6),
        "dominant_trait_macro_precision": round(float(precision), 6),
        "dominant_trait_macro_recall": round(float(recall), 6),
        "dominant_trait_macro_f1": round(float(macro_f1), 6),
        "dominant_trait_confusion_matrix": cm.tolist(),
        "dominant_trait_classification_report": report,
        "y_true": [int(v) for v in y_true],
        "y_pred": [int(v) for v in y_pred]
    }


def train_regression_single_run(
    cfg: RegressionConfig,
    train_csv: str,
    val_csv: str,
    test_csv: Optional[str],
    run_dir: Path,
    split_strategy: str
) -> Path:
    seed_everything(cfg.seed)
    device = get_device()

    ensure_dir(run_dir)
    ensure_dir(run_dir / "checkpoints")

    save_json(cfg.to_dict(), run_dir / "config.json")
    save_yaml(cfg.to_dict(), run_dir / "config.yaml")

    split_info = {
        "split_strategy": split_strategy,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "test_csv": test_csv,
        "loss": cfg.loss,
        "use_weighted_sampler": cfg.use_weighted_sampler,
        "use_weighted_loss": cfg.use_weighted_loss,
        "augmentation_strength": cfg.augmentation_strength,
        "best_selection_metric": "val_mean_pcc"
    }

    save_json(split_info, run_dir / "split_info.json")

    train_dataset = HandwritingRegressionDataset(
        csv_path=train_csv,
        image_dir=cfg.image_dir,
        target_columns=cfg.target_columns,
        img_size=cfg.img_size,
        target_min=cfg.target_min,
        target_max=cfg.target_max,
        train=True,
        augmentation_strength=cfg.augmentation_strength
    )

    val_dataset = HandwritingRegressionDataset(
        csv_path=val_csv,
        image_dir=cfg.image_dir,
        target_columns=cfg.target_columns,
        img_size=cfg.img_size,
        target_min=cfg.target_min,
        target_max=cfg.target_max,
        train=False
    )

    train_sampler = None

    if cfg.use_weighted_sampler:
        train_sampler = create_regression_weighted_sampler(
            csv_path=train_csv,
            target_columns=cfg.target_columns
        )
        print("[INFO] Regression WeightedRandomSampler enabled.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda")
    )

    test_loader = None

    if test_csv is not None:
        test_dataset = HandwritingRegressionDataset(
            csv_path=test_csv,
            image_dir=cfg.image_dir,
            target_columns=cfg.target_columns,
            img_size=cfg.img_size,
            target_min=cfg.target_min,
            target_max=cfg.target_max,
            train=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda")
        )

    model = ViTRegressor(
        model_name=cfg.model_name,
        output_dim=len(cfg.target_columns),
        dropout=cfg.dropout,
        pretrained=True
    ).to(device)

    criterion = build_regression_loss(cfg)

    sample_weight_lookup = None

    if cfg.use_weighted_loss:
        sample_weight_lookup = create_sample_weight_lookup(
            csv_path=train_csv,
            image_dir=cfg.image_dir,
            target_columns=cfg.target_columns
        )
        print("[INFO] Regression weighted loss enabled.")

    amp_enabled = cfg.use_amp and device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled
    )

    metrics_log = {
        "experiment_id": cfg.experiment_id,
        "experiment_name": cfg.experiment_name,
        "task": "regression",
        "loss": cfg.loss,
        "use_weighted_sampler": cfg.use_weighted_sampler,
        "use_weighted_loss": cfg.use_weighted_loss,
        "augmentation_strength": cfg.augmentation_strength,
        "best_selection_metric": "val_mean_pcc",
        "history": [],
        "best_val_mean_pcc": -1.0,
        "best_epoch": None,
        "checkpoint_epochs": cfg.checkpoint_epochs,
        "saved_checkpoints": []
    }

    best_val_mean_pcc = -1.0
    best_model_path = run_dir / "model_best.pt"

    def run_epochs(
        epochs: int,
        lr: float,
        phase_name: str,
        global_start_epoch: int
    ) -> None:
        nonlocal best_val_mean_pcc

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=cfg.weight_decay
        )

        for local_epoch in range(1, epochs + 1):
            global_epoch = global_start_epoch + local_epoch

            model.train()

            total_loss = 0.0
            total = 0

            progress = tqdm(
                train_loader,
                desc=f"{phase_name} epoch {local_epoch}/{epochs}",
                leave=False
            )

            for x, y, image_paths in progress:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    "cuda",
                    enabled=amp_enabled
                ):
                    preds = model(x)

                    if cfg.use_weighted_loss:
                        per_element_loss = (preds - y) ** 2
                        per_sample_loss = per_element_loss.mean(dim=1)

                        weights = torch.tensor(
                            [
                                sample_weight_lookup.get(path, 1.0)
                                for path in image_paths
                            ],
                            dtype=torch.float32,
                            device=device
                        )

                        loss = (per_sample_loss * weights).mean()
                    else:
                        loss = criterion(preds, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item() * x.size(0)
                total += x.size(0)

                progress.set_postfix(loss=loss.item())

            train_loss = total_loss / max(1, total)

            val_metrics = evaluate_regression_model(
                model=model,
                loader=val_loader,
                device=device,
                target_columns=cfg.target_columns,
                target_min=cfg.target_min,
                target_max=cfg.target_max
            )

            dominant_metrics = evaluate_dominant_trait_from_loader(
                model=model,
                loader=val_loader,
                device=device,
                target_columns=cfg.target_columns,
                target_min=cfg.target_min,
                target_max=cfg.target_max
            )

            entry = {
                "global_epoch": global_epoch,
                "phase": phase_name,
                "local_epoch": local_epoch,
                "train_loss_normalized": round(train_loss, 6),
                "val_mse": val_metrics["mse"],
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_mean_pcc": val_metrics["mean_pcc"],
                "val_dominant_trait_accuracy": dominant_metrics["dominant_trait_accuracy"],
                "val_dominant_trait_macro_f1": dominant_metrics["dominant_trait_macro_f1"],
                "val_per_trait": val_metrics["per_trait"]
            }

            metrics_log["history"].append(entry)

            if val_metrics["mean_pcc"] > best_val_mean_pcc:
                best_val_mean_pcc = val_metrics["mean_pcc"]

                metrics_log["best_val_mean_pcc"] = val_metrics["mean_pcc"]
                metrics_log["best_epoch"] = global_epoch
                metrics_log["best_val_metrics"] = val_metrics
                metrics_log["best_val_dominant_trait_metrics"] = dominant_metrics

                save_checkpoint(
                    model=model,
                    cfg=cfg,
                    output_path=best_model_path,
                    epoch_global=global_epoch,
                    phase_name=phase_name,
                    metrics=val_metrics
                )

                save_json(val_metrics, run_dir / "best_val_metrics.json")
                save_json(dominant_metrics, run_dir / "best_val_dominant_trait_metrics.json")

            if global_epoch in cfg.checkpoint_epochs:
                checkpoint_path = run_dir / "checkpoints" / f"epoch_{global_epoch:03d}.pt"

                save_checkpoint(
                    model=model,
                    cfg=cfg,
                    output_path=checkpoint_path,
                    epoch_global=global_epoch,
                    phase_name=phase_name,
                    metrics=val_metrics
                )

                metrics_log["saved_checkpoints"].append(str(checkpoint_path))

            save_json(metrics_log, run_dir / "metrics.json")

            print(
                f"[{phase_name}] global epoch {global_epoch:02d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_rmse={val_metrics['rmse']:.4f} | "
                f"val_mae={val_metrics['mae']:.4f} | "
                f"val_mean_pcc={val_metrics['mean_pcc']:.4f} | "
                f"val_dom_acc={dominant_metrics['dominant_trait_accuracy']:.4f} | "
                f"val_dom_f1={dominant_metrics['dominant_trait_macro_f1']:.4f} | "
                f"best_pcc={best_val_mean_pcc:.4f}"
            )

    if cfg.training_strategy == "two_stage":
        freeze_backbone(model, freeze=True)
        run_epochs(cfg.head_epochs, cfg.lr_head, "head", global_start_epoch=0)

        freeze_backbone(model, freeze=False)
        run_epochs(cfg.finetune_epochs, cfg.lr_finetune, "finetune", global_start_epoch=cfg.head_epochs)

    elif cfg.training_strategy == "full_train":
        freeze_backbone(model, freeze=False)
        run_epochs(cfg.total_epochs, cfg.lr_finetune, "full_train", global_start_epoch=0)

    else:
        raise ValueError(f"Unsupported training strategy: {cfg.training_strategy}")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    if test_loader is not None:
        test_metrics = evaluate_regression_model(
            model=model,
            loader=test_loader,
            device=device,
            target_columns=cfg.target_columns,
            target_min=cfg.target_min,
            target_max=cfg.target_max
        )

        test_dominant_metrics = evaluate_dominant_trait_from_loader(
            model=model,
            loader=test_loader,
            device=device,
            target_columns=cfg.target_columns,
            target_min=cfg.target_min,
            target_max=cfg.target_max
        )

        save_json(test_metrics, run_dir / "test_metrics_best.json")
        save_json(test_dominant_metrics, run_dir / "test_dominant_trait_metrics.json")

        print(f"[OK] Test RMSE: {test_metrics['rmse']:.4f}")
        print(f"[OK] Test MAE: {test_metrics['mae']:.4f}")
        print(f"[OK] Test Mean PCC: {test_metrics['mean_pcc']:.4f}")
        print(f"[OK] Test Dominant Accuracy: {test_dominant_metrics['dominant_trait_accuracy']:.4f}")
        print(f"[OK] Test Dominant Macro F1: {test_dominant_metrics['dominant_trait_macro_f1']:.4f}")

    print(f"[OK] Best model saved: {best_model_path}")

    return best_model_path


def run_train_val_test_regression_experiment(cfg: RegressionConfig) -> Path:
    return train_regression_single_run(
        cfg=cfg,
        train_csv=cfg.dataset_splits["train"],
        val_csv=cfg.dataset_splits["val"],
        test_csv=cfg.dataset_splits["test"],
        run_dir=cfg.run_dir,
        split_strategy="predefined_train_val_test"
    )


def dominant_labels_from_dataframe(
    df: pd.DataFrame,
    target_columns: List[str]
) -> np.ndarray:
    scores = df[target_columns].values.astype(np.float32)
    return np.argmax(scores, axis=1)


def run_kfold_regression_experiment(cfg: RegressionConfig) -> Path:
    print(f"[INFO] Running K-Fold regression experiment: {cfg.experiment_id}")

    seed_everything(cfg.seed)

    ensure_dir(cfg.run_dir)
    save_json(cfg.to_dict(), cfg.run_dir / "config.json")
    save_yaml(cfg.to_dict(), cfg.run_dir / "config.yaml")

    all_frames = []

    for split_name in ["train", "val", "test"]:
        if split_name in cfg.dataset_splits:
            split_df = pd.read_csv(cfg.dataset_splits[split_name])
            split_df["source_split"] = split_name
            all_frames.append(split_df)

    if not all_frames:
        raise RuntimeError("No CSV splits found for K-Fold regression.")

    all_df = pd.concat(all_frames, ignore_index=True)

    labels = dominant_labels_from_dataframe(
        all_df,
        cfg.target_columns
    )

    class_counts = np.bincount(labels, minlength=len(cfg.target_columns))

    if np.min(class_counts) < cfg.k_folds:
        raise ValueError(
            f"K-Fold cannot be applied safely. "
            f"Minimum dominant trait count is {int(np.min(class_counts))}, "
            f"but k_folds is {cfg.k_folds}."
        )

    skf = StratifiedKFold(
        n_splits=cfg.k_folds,
        shuffle=True,
        random_state=cfg.seed
    )

    save_json(
        {
            "split_strategy": "stratified_kfold_on_dominant_trait",
            "total_samples": len(all_df),
            "k_folds": cfg.k_folds,
            "target_columns": cfg.target_columns,
            "dominant_trait_distribution": {
                cfg.target_columns[i]: int(class_counts[i])
                for i in range(len(cfg.target_columns))
            },
            "loss": cfg.loss,
            "use_weighted_sampler": cfg.use_weighted_sampler,
            "use_weighted_loss": cfg.use_weighted_loss,
            "augmentation_strength": cfg.augmentation_strength,
            "note": (
                "For regression K-Fold, dominant trait labels are created by "
                "argmax over real Big Five scores and used only for stratified splitting."
            )
        },
        cfg.run_dir / "kfold_dataset_info.json"
    )

    fold_results = []

    fold_val_rmse = []
    fold_val_mae = []
    fold_val_pcc = []
    fold_val_dom_acc = []
    fold_val_dom_f1 = []

    for fold_index, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels),
        start=1
    ):
        print(f"\n========== {cfg.experiment_id} | Fold {fold_index}/{cfg.k_folds} ==========")

        fold_dir = cfg.run_dir / f"fold_{fold_index}"
        ensure_dir(fold_dir)

        train_df = all_df.iloc[train_idx].copy()
        val_df = all_df.iloc[val_idx].copy()

        train_csv = fold_dir / "train.csv"
        val_csv = fold_dir / "val.csv"

        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)

        best_model_path = train_regression_single_run(
            cfg=cfg,
            train_csv=str(train_csv),
            val_csv=str(val_csv),
            test_csv=None,
            run_dir=fold_dir,
            split_strategy="stratified_kfold_on_dominant_trait"
        )

        metrics_path = fold_dir / "metrics.json"

        if not metrics_path.exists():
            raise FileNotFoundError(f"Fold metrics not found: {metrics_path}")

        best_val_metrics_path = fold_dir / "best_val_metrics.json"
        best_dom_metrics_path = fold_dir / "best_val_dominant_trait_metrics.json"

        if not best_val_metrics_path.exists():
            raise FileNotFoundError(f"Best val metrics not found: {best_val_metrics_path}")

        if not best_dom_metrics_path.exists():
            raise FileNotFoundError(f"Best dominant trait metrics not found: {best_dom_metrics_path}")

        with open(best_val_metrics_path, "r", encoding="utf-8") as f:
            best_val_metrics = json.load(f)

        with open(best_dom_metrics_path, "r", encoding="utf-8") as f:
            best_dom_metrics = json.load(f)

        fold_val_rmse.append(best_val_metrics["rmse"])
        fold_val_mae.append(best_val_metrics["mae"])
        fold_val_pcc.append(best_val_metrics["mean_pcc"])
        fold_val_dom_acc.append(best_dom_metrics["dominant_trait_accuracy"])
        fold_val_dom_f1.append(best_dom_metrics["dominant_trait_macro_f1"])

        fold_results.append(
            {
                "fold": fold_index,
                "train_size": len(train_df),
                "val_size": len(val_df),
                "best_model_path": str(best_model_path),
                "best_val_rmse": best_val_metrics["rmse"],
                "best_val_mae": best_val_metrics["mae"],
                "best_val_mean_pcc": best_val_metrics["mean_pcc"],
                "best_val_dominant_accuracy": best_dom_metrics["dominant_trait_accuracy"],
                "best_val_dominant_macro_f1": best_dom_metrics["dominant_trait_macro_f1"]
            }
        )

    summary = {
        "experiment_id": cfg.experiment_id,
        "experiment_name": cfg.experiment_name,
        "task": "regression_kfold",
        "split_strategy": "stratified_kfold_on_dominant_trait",
        "k_folds": cfg.k_folds,
        "loss": cfg.loss,
        "use_weighted_sampler": cfg.use_weighted_sampler,
        "use_weighted_loss": cfg.use_weighted_loss,
        "augmentation_strength": cfg.augmentation_strength,
        "average_val_rmse": round(float(np.mean(fold_val_rmse)), 6),
        "std_val_rmse": round(float(np.std(fold_val_rmse)), 6),
        "average_val_mae": round(float(np.mean(fold_val_mae)), 6),
        "std_val_mae": round(float(np.std(fold_val_mae)), 6),
        "average_val_mean_pcc": round(float(np.mean(fold_val_pcc)), 6),
        "std_val_mean_pcc": round(float(np.std(fold_val_pcc)), 6),
        "average_val_dominant_accuracy": round(float(np.mean(fold_val_dom_acc)), 6),
        "std_val_dominant_accuracy": round(float(np.std(fold_val_dom_acc)), 6),
        "average_val_dominant_macro_f1": round(float(np.mean(fold_val_dom_f1)), 6),
        "std_val_dominant_macro_f1": round(float(np.std(fold_val_dom_f1)), 6),
        "results": fold_results
    }

    save_json(summary, cfg.run_dir / "summary.json")

    print(f"\n[OK] K-Fold regression completed: {cfg.experiment_id}")
    print(f"Average Val RMSE: {summary['average_val_rmse']:.4f}")
    print(f"Average Val MAE: {summary['average_val_mae']:.4f}")
    print(f"Average Val Mean PCC: {summary['average_val_mean_pcc']:.4f}")
    print(f"Average Dominant Accuracy: {summary['average_val_dominant_accuracy']:.4f}")
    print(f"Average Dominant Macro F1: {summary['average_val_dominant_macro_f1']:.4f}")

    return cfg.run_dir


def run_regression_experiment(cfg: RegressionConfig) -> Path:
    print(f"[INFO] Running regression experiment: {cfg.experiment_id} - {cfg.experiment_name}")

    if cfg.split_type == "train_val_test":
        return run_train_val_test_regression_experiment(cfg)

    if cfg.split_type == "kfold":
        return run_kfold_regression_experiment(cfg)

    raise ValueError(f"Unsupported regression split_type: {cfg.split_type}")
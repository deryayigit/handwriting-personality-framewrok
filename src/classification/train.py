import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.common.utils import seed_everything, ensure_dir, save_json, save_yaml, get_device
from .config import ClassificationConfig
from .dataset import HandwritingDataset, list_dataset_items
from .metrics import evaluate_model, compute_class_weights
from .model import ViTClassifier, freeze_backbone


def save_checkpoint(
    model: nn.Module,
    cfg: ClassificationConfig,
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
            "class_names": cfg.class_names,
            "img_size": cfg.img_size,
            "dropout": cfg.dropout,
            "experiment_id": cfg.experiment_id,
            "experiment_name": cfg.experiment_name,
            "dataset_id": cfg.dataset_id,
            "training_strategy": cfg.training_strategy,
            "class_weight": cfg.class_weight,
            "use_weighted_sampler": cfg.use_weighted_sampler,
            "augmentation_strength": cfg.augmentation_strength,
            "epoch": epoch_global,
            "phase": phase_name,
            "metrics": metrics
        },
        output_path
    )


def stratified_train_val_split(
    items: List[Dict[str, Any]],
    val_ratio: float,
    seed: int
):
    labels = [item["label"] for item in items]

    train_items, val_items = train_test_split(
        items,
        test_size=val_ratio,
        stratify=labels,
        random_state=seed
    )

    return train_items, val_items


def count_distribution(
    items: Optional[List[Dict[str, Any]]],
    class_names: List[str]
) -> Dict[str, int]:
    if not items:
        return {}

    counts = {name: 0 for name in class_names}

    for item in items:
        counts[class_names[int(item["label"])]] += 1

    return counts


def create_weighted_sampler(
    items: List[Dict[str, Any]],
    num_classes: int
) -> WeightedRandomSampler:
    labels = [int(item["label"]) for item in items]
    counts = np.bincount(labels, minlength=num_classes)

    class_sample_weights = []

    for count in counts:
        if count == 0:
            class_sample_weights.append(0.0)
        else:
            class_sample_weights.append(1.0 / count)

    sample_weights = [
        class_sample_weights[label]
        for label in labels
    ]

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )


def train_single_run(
    cfg: ClassificationConfig,
    train_items: List[Dict[str, Any]],
    val_items: List[Dict[str, Any]],
    test_items: Optional[List[Dict[str, Any]]],
    run_dir: Path,
    split_strategy: str
) -> Path:
    seed_everything(cfg.seed)
    device = get_device()

    ensure_dir(run_dir)
    ensure_dir(run_dir / "checkpoints")

    save_json(cfg.to_dict(), run_dir / "config.json")
    save_yaml(cfg.to_dict(), run_dir / "config.yaml")

    label_map = {
        name: index
        for index, name in enumerate(cfg.class_names)
    }

    save_json(
        {
            "class_names": cfg.class_names,
            "label_map": label_map
        },
        run_dir / "label.json"
    )

    split_info = {
        "split_strategy": split_strategy,
        "training_strategy": cfg.training_strategy,
        "class_weight": cfg.class_weight,
        "use_weighted_sampler": cfg.use_weighted_sampler,
        "augmentation_strength": cfg.augmentation_strength,
        "best_selection_metric": "val_macro_f1",
        "train_size": len(train_items),
        "val_size": len(val_items),
        "test_size": len(test_items) if test_items is not None else 0,
        "train_distribution": count_distribution(train_items, cfg.class_names),
        "val_distribution": count_distribution(val_items, cfg.class_names),
        "test_distribution": count_distribution(test_items, cfg.class_names)
        if test_items is not None else {}
    }

    save_json(split_info, run_dir / "split_info.json")

    train_dataset = HandwritingDataset(
        train_items,
        img_size=cfg.img_size,
        train=True,
        augmentation_strength=cfg.augmentation_strength
    )

    val_dataset = HandwritingDataset(
        val_items,
        img_size=cfg.img_size,
        train=False
    )

    train_sampler = None

    if cfg.use_weighted_sampler:
        train_sampler = create_weighted_sampler(
            train_items,
            cfg.num_classes
        )
        print("[INFO] WeightedRandomSampler enabled.")

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

    if test_items is not None:
        test_dataset = HandwritingDataset(
            test_items,
            img_size=cfg.img_size,
            train=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda")
        )

    model = ViTClassifier(
        model_name=cfg.model_name,
        num_classes=cfg.num_classes,
        dropout=cfg.dropout,
        pretrained=True
    ).to(device)

    class_weights = None

    if cfg.class_weight:
        class_weights = compute_class_weights(
            train_items,
            cfg.num_classes
        ).to(device)

        print(f"[INFO] Class weights: {class_weights.detach().cpu().tolist()}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg.label_smoothing
    )

    amp_enabled = (
        cfg.use_amp and
        device.type == "cuda"
    )

    if cfg.use_amp and device.type != "cuda":
        print("[INFO] AMP requested but CUDA is not available. Training will run without AMP.")

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled
    )

    metrics_log: Dict[str, Any] = {
        "experiment_id": cfg.experiment_id,
        "experiment_name": cfg.experiment_name,
        "training_strategy": cfg.training_strategy,
        "class_weight": cfg.class_weight,
        "use_weighted_sampler": cfg.use_weighted_sampler,
        "augmentation_strength": cfg.augmentation_strength,
        "best_selection_metric": "val_macro_f1",
        "history": [],
        "best_val_accuracy": 0.0,
        "best_val_macro_f1": 0.0,
        "best_epoch": None,
        "checkpoint_epochs": cfg.checkpoint_epochs,
        "saved_checkpoints": []
    }

    best_val_macro_f1 = -1.0
    best_model_path = run_dir / "model_best.pt"

    def run_epochs(
        epochs: int,
        lr: float,
        phase_name: str,
        global_start_epoch: int
    ):
        nonlocal best_val_macro_f1

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=cfg.weight_decay
        )

        for local_epoch in range(1, epochs + 1):
            global_epoch = global_start_epoch + local_epoch

            model.train()

            total_loss = 0.0
            correct = 0
            total = 0

            progress = tqdm(
                train_loader,
                desc=f"{phase_name} epoch {local_epoch}/{epochs}",
                leave=False
            )

            for x, y, _ in progress:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    "cuda",
                    enabled=amp_enabled
                ):
                    logits = model(x)
                    loss = criterion(logits, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                predictions = torch.argmax(logits, dim=1)

                total_loss += loss.item() * x.size(0)
                correct += (predictions == y).sum().item()
                total += x.size(0)

                progress.set_postfix(
                    loss=loss.item(),
                    acc=correct / max(1, total)
                )

            train_loss = total_loss / max(1, total)
            train_acc = correct / max(1, total)

            val_metrics = evaluate_model(
                model,
                val_loader,
                device,
                cfg.class_names
            )

            entry = {
                "global_epoch": global_epoch,
                "phase": phase_name,
                "local_epoch": local_epoch,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_acc, 6),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_macro_precision": val_metrics["macro_precision"],
                "val_macro_recall": val_metrics["macro_recall"]
            }

            metrics_log["history"].append(entry)

            if val_metrics["macro_f1"] > best_val_macro_f1:
                best_val_macro_f1 = val_metrics["macro_f1"]

                metrics_log["best_val_accuracy"] = val_metrics["accuracy"]
                metrics_log["best_val_macro_f1"] = val_metrics["macro_f1"]
                metrics_log["best_epoch"] = global_epoch
                metrics_log["best_val_confusion_matrix"] = val_metrics["confusion_matrix"]
                metrics_log["best_val_classification_report"] = val_metrics["classification_report"]

                save_checkpoint(
                    model=model,
                    cfg=cfg,
                    output_path=best_model_path,
                    epoch_global=global_epoch,
                    phase_name=phase_name,
                    metrics=val_metrics
                )

                save_json(
                    val_metrics,
                    run_dir / "best_val_metrics.json"
                )

            if global_epoch in cfg.checkpoint_epochs:
                checkpoint_path = (
                    run_dir /
                    "checkpoints" /
                    f"epoch_{global_epoch:03d}.pt"
                )

                save_checkpoint(
                    model=model,
                    cfg=cfg,
                    output_path=checkpoint_path,
                    epoch_global=global_epoch,
                    phase_name=phase_name,
                    metrics=val_metrics
                )

                metrics_log["saved_checkpoints"].append(str(checkpoint_path))

                save_json(
                    val_metrics,
                    run_dir / "checkpoints" / f"epoch_{global_epoch:03d}_metrics.json"
                )

            save_json(metrics_log, run_dir / "metrics.json")

            print(
                f"[{phase_name}] global epoch {global_epoch:02d} | "
                f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"val loss={val_metrics['loss']:.4f} "
                f"acc={val_metrics['accuracy']:.3f} | "
                f"macro_f1={val_metrics['macro_f1']:.3f} | "
                f"best_macro_f1={best_val_macro_f1:.3f}"
            )

    if cfg.training_strategy == "two_stage":
        freeze_backbone(model, freeze=True)

        run_epochs(
            cfg.head_epochs,
            cfg.lr_head,
            "head",
            global_start_epoch=0
        )

        freeze_backbone(model, freeze=False)

        run_epochs(
            cfg.finetune_epochs,
            cfg.lr_finetune,
            "finetune",
            global_start_epoch=cfg.head_epochs
        )

    elif cfg.training_strategy == "full_train":
        freeze_backbone(model, freeze=False)

        run_epochs(
            cfg.total_epochs,
            cfg.lr_finetune,
            "full_train",
            global_start_epoch=0
        )

    else:
        raise ValueError(f"Unsupported training_strategy: {cfg.training_strategy}")

    if test_loader is not None:
        print("[INFO] Evaluating best model on independent test split...")

        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=True)

        test_metrics = evaluate_model(
            model,
            test_loader,
            device,
            cfg.class_names
        )

        save_json(test_metrics, run_dir / "test_metrics_best.json")

    print(f"[OK] Best model saved by val_macro_f1: {best_model_path}")

    return best_model_path


def run_train_test_experiment(cfg: ClassificationConfig) -> Path:
    all_train_items = list_dataset_items(
        cfg.dataset_splits["train"],
        cfg.class_names
    )

    test_items = list_dataset_items(
        cfg.dataset_splits["test"],
        cfg.class_names
    )

    train_items, val_items = stratified_train_val_split(
        items=all_train_items,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed
    )

    return train_single_run(
        cfg=cfg,
        train_items=train_items,
        val_items=val_items,
        test_items=test_items,
        run_dir=cfg.run_dir,
        split_strategy="train_folder_split_80_20_test_folder_final"
    )


def run_train_val_test_experiment(cfg: ClassificationConfig) -> Path:
    train_items = list_dataset_items(
        cfg.dataset_splits["train"],
        cfg.class_names
    )

    val_items = list_dataset_items(
        cfg.dataset_splits["val"],
        cfg.class_names
    )

    test_items = list_dataset_items(
        cfg.dataset_splits["test"],
        cfg.class_names
    )

    return train_single_run(
        cfg=cfg,
        train_items=train_items,
        val_items=val_items,
        test_items=test_items,
        run_dir=cfg.run_dir,
        split_strategy="predefined_train_val_test"
    )


def run_kfold_experiment(cfg: ClassificationConfig) -> Path:
    all_items = []

    if "train" in cfg.dataset_splits:
        all_items.extend(
            list_dataset_items(
                cfg.dataset_splits["train"],
                cfg.class_names
            )
        )

    if "test" in cfg.dataset_splits:
        all_items.extend(
            list_dataset_items(
                cfg.dataset_splits["test"],
                cfg.class_names
            )
        )

    if not all_items:
        raise ValueError("No dataset items found for K-Fold experiment.")

    labels = [
        item["label"]
        for item in all_items
    ]

    class_counts = np.bincount(labels, minlength=cfg.num_classes)

    if np.min(class_counts) < cfg.k_folds:
        raise ValueError(
            f"K-Fold cannot be applied safely. "
            f"Minimum class count is {int(np.min(class_counts))}, "
            f"but k_folds is {cfg.k_folds}."
        )

    skf = StratifiedKFold(
        n_splits=cfg.k_folds,
        shuffle=True,
        random_state=cfg.seed
    )

    ensure_dir(cfg.run_dir)
    save_json(cfg.to_dict(), cfg.run_dir / "config.json")
    save_yaml(cfg.to_dict(), cfg.run_dir / "config.yaml")

    fold_results = []
    fold_accuracies = []
    fold_macro_f1 = []

    merged_distribution = count_distribution(
        all_items,
        cfg.class_names
    )

    save_json(
        {
            "split_strategy": "kfold_on_train_test_merged",
            "merge_note": (
                "For K-Fold experiments, the original train and test folders "
                "are merged into a single data pool. No independent test set "
                "is used."
            ),
            "total_samples": len(all_items),
            "merged_distribution": merged_distribution,
            "k_folds": cfg.k_folds,
            "class_names": cfg.class_names
        },
        cfg.run_dir / "kfold_dataset_info.json"
    )

    for fold_index, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels),
        start=1
    ):
        print(
            f"\n========== {cfg.experiment_id} | "
            f"Fold {fold_index}/{cfg.k_folds} =========="
        )

        train_items = [
            all_items[i]
            for i in train_idx
        ]

        val_items = [
            all_items[i]
            for i in val_idx
        ]

        fold_dir = cfg.run_dir / f"fold_{fold_index}"

        best_model_path = train_single_run(
            cfg=cfg,
            train_items=train_items,
            val_items=val_items,
            test_items=None,
            run_dir=fold_dir,
            split_strategy="kfold_on_train_test_merged"
        )

        metrics_path = fold_dir / "metrics.json"

        if not metrics_path.exists():
            raise FileNotFoundError(
                f"K-Fold metrics file was not created: {metrics_path}"
            )

        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Invalid JSON file generated during K-Fold training. "
                f"Fold directory: {fold_dir}"
            ) from e

        best_acc = metrics["best_val_accuracy"]
        best_macro_f1 = metrics["best_val_macro_f1"]
        best_epoch = metrics["best_epoch"]

        fold_accuracies.append(best_acc)
        fold_macro_f1.append(best_macro_f1)

        fold_results.append(
            {
                "fold": fold_index,
                "train_size": len(train_items),
                "val_size": len(val_items),
                "train_distribution": count_distribution(train_items, cfg.class_names),
                "val_distribution": count_distribution(val_items, cfg.class_names),
                "best_selection_metric": "val_macro_f1",
                "best_val_accuracy": best_acc,
                "best_val_macro_f1": best_macro_f1,
                "best_epoch": best_epoch,
                "best_model_path": str(best_model_path),
                "metrics_path": str(metrics_path)
            }
        )

    summary = {
        "experiment_id": cfg.experiment_id,
        "experiment_name": cfg.experiment_name,
        "dataset_id": cfg.dataset_id,
        "training_strategy": cfg.training_strategy,
        "class_weight": cfg.class_weight,
        "use_weighted_sampler": cfg.use_weighted_sampler,
        "augmentation_strength": cfg.augmentation_strength,
        "k_folds": cfg.k_folds,
        "best_selection_metric": "val_macro_f1",
        "split_strategy": "kfold_on_train_test_merged",
        "test_usage_note": (
            "For K-Fold experiments, the original train and test folders are "
            "merged into a single data pool. No independent test set is used."
        ),
        "val_ratio_note": (
            "val_ratio is not used in K-Fold experiments because validation "
            "splits are generated by StratifiedKFold."
        ),
        "total_samples": len(all_items),
        "merged_distribution": merged_distribution,
        "fold_val_accuracies": fold_accuracies,
        "fold_val_macro_f1": fold_macro_f1,
        "average_val_accuracy": round(float(np.mean(fold_accuracies)), 6),
        "std_val_accuracy": round(float(np.std(fold_accuracies)), 6),
        "average_val_macro_f1": round(float(np.mean(fold_macro_f1)), 6),
        "std_val_macro_f1": round(float(np.std(fold_macro_f1)), 6),
        "results": fold_results
    }

    save_json(summary, cfg.run_dir / "summary.json")

    print(f"\n[OK] K-Fold experiment completed: {cfg.experiment_id}")
    print(f"Total Samples: {summary['total_samples']}")
    print(f"Average Val Accuracy: {summary['average_val_accuracy']:.4f}")
    print(f"Average Val Macro F1: {summary['average_val_macro_f1']:.4f}")

    return cfg.run_dir


def run_classification_experiment(cfg: ClassificationConfig) -> Path:
    print(
        f"[INFO] Running experiment: "
        f"{cfg.experiment_id} - {cfg.experiment_name}"
    )

    if cfg.split_type == "train_test":
        return run_train_test_experiment(cfg)

    if cfg.split_type == "train_val_test":
        return run_train_val_test_experiment(cfg)

    if cfg.split_type == "kfold":
        return run_kfold_experiment(cfg)

    raise ValueError(f"Unsupported split_type: {cfg.split_type}")
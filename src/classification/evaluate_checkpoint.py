import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.common.registry import load_experiment_registry, get_dataset
from src.classification.model import ViTClassifier


CLASS_NAMES = [
    "Agreeableness",
    "Conscientiousness",
    "Extraversion",
    "Neuroticism",
    "Openness"
]


IMAGE_COLUMNS = [
    "image",
    "image_path",
    "filepath",
    "file_path",
    "path",
    "filename"
]


LABEL_COLUMNS = [
    "label",
    "class",
    "class_name",
    "target"
]


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def find_label_file(run_dir: Path, fold_dir: Optional[Path]) -> Optional[Path]:
    candidates = []

    if fold_dir:
        candidates.extend([
            fold_dir / "label.json",
            fold_dir / "labels.json"
        ])

    candidates.extend([
        run_dir / "label.json",
        run_dir / "labels.json"
    ])

    for path in candidates:
        if path.exists():
            return path

    return None


def load_class_names(run_dir: Path, fold_dir: Optional[Path]) -> List[str]:
    label_file = find_label_file(run_dir, fold_dir)

    if not label_file:
        return CLASS_NAMES

    data = read_json(label_file)

    if not data:
        return CLASS_NAMES

    if isinstance(data, list):
        return data

    if "class_names" in data:
        return data["class_names"]

    if "classes" in data:
        return data["classes"]

    if "idx_to_class" in data:
        idx_to_class = data["idx_to_class"]
        return [
            idx_to_class[str(i)]
            for i in range(len(idx_to_class))
        ]

    if "class_to_idx" in data:
        class_to_idx = data["class_to_idx"]
        return [
            class_name
            for class_name, _ in sorted(class_to_idx.items(), key=lambda x: x[1])
        ]

    return CLASS_NAMES


def find_config_file(run_dir: Path, fold_dir: Optional[Path]) -> Optional[Path]:
    candidates = []

    if fold_dir:
        candidates.append(fold_dir / "config.yaml")

    candidates.append(run_dir / "config.yaml")

    for path in candidates:
        if path.exists():
            return path

    return None


def load_model_config(run_dir: Path, fold_dir: Optional[Path]) -> Dict[str, Any]:
    config_file = find_config_file(run_dir, fold_dir)

    if not config_file:
        return {}

    return read_yaml(config_file)


def get_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])


class FolderClassificationDataset(Dataset):
    def __init__(self, root: Path, class_names: List[str], transform=None):
        self.root = Path(root)
        self.class_names = class_names
        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(class_names)
        }
        self.transform = transform
        self.samples = []

        extensions = {".png", ".jpg", ".jpeg", ".bmp"}

        for class_name in class_names:
            class_dir = self.root / class_name

            if not class_dir.exists():
                continue

            for path in class_dir.rglob("*"):
                if path.suffix.lower() in extensions:
                    self.samples.append((path, self.class_to_idx[class_name]))

        if not self.samples:
            raise RuntimeError(f"No images found in folder: {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, str(path)


class CSVClassificationDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        class_names: List[str],
        image_root: Optional[Path],
        transform=None
    ):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(csv_path)
        self.class_names = class_names
        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(class_names)
        }
        self.image_root = Path(image_root) if image_root else self.csv_path.parent
        self.transform = transform

        self.image_column = self._find_column(IMAGE_COLUMNS)
        self.label_column = self._find_column(LABEL_COLUMNS)

        if not self.image_column:
            raise RuntimeError(f"No image column found in {csv_path}")

    def _find_column(self, candidates: List[str]) -> Optional[str]:
        for column in candidates:
            if column in self.df.columns:
                return column

        return None

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, raw_path: str) -> Path:
        path = Path(str(raw_path))

        if path.is_absolute():
            return path

        candidate = self.image_root / path

        if candidate.exists():
            return candidate

        return self.csv_path.parent / path

    def _resolve_label(self, row) -> int:
        if self.label_column:
            value = row[self.label_column]

            if isinstance(value, str):
                return self.class_to_idx[value]

            return int(value)

        image_path = self._resolve_image_path(row[self.image_column])
        class_name = image_path.parent.name

        return self.class_to_idx[class_name]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = self._resolve_image_path(row[self.image_column])
        label = self._resolve_label(row)

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, str(image_path)


def get_dataset_image_root(dataset_info: Dict[str, Any]) -> Optional[Path]:
    for key in ["image_dir", "images_dir", "root", "root_dir", "dataset_dir"]:
        value = dataset_info.get(key)

        if value:
            return Path(value)

    return None


def get_eval_source_for_experiment(
    experiment_id: str,
    experiment_info: Dict[str, Any],
    run_dir: Path,
    fold_dir: Optional[Path]
) -> Tuple[Path, Optional[Path]]:
    if fold_dir:
        for name in ["val.csv", "valid.csv", "test.csv"]:
            candidate = fold_dir / name

            if candidate.exists():
                return candidate, None

    dataset_id = experiment_info.get("dataset_id")
    dataset_info = get_dataset(dataset_id)
    splits = dataset_info.get("splits", {})

    for split_name in ["test", "val", "valid"]:
        split_path = splits.get(split_name)

        if split_path:
            return Path(split_path), get_dataset_image_root(dataset_info)

    raise RuntimeError(
        f"No evaluation split found for {experiment_id}. "
        f"Expected fold val.csv/test.csv or dataset test/val split."
    )


def load_model(
    checkpoint_path: Path,
    run_dir: Path,
    fold_dir: Optional[Path],
    class_names: List[str],
    device: torch.device
):
    config = load_model_config(run_dir, fold_dir)

    model_name = config.get("model_name", "vit_base_patch16_224")
    dropout = float(config.get("dropout", 0.1))

    model = ViTClassifier(
    model_name=model_name,
    num_classes=len(class_names),
    pretrained=False,
    dropout=dropout
)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        cleaned_state_dict[new_key] = value

    model.load_state_dict(cleaned_state_dict, strict=False)
    model.to(device)
    model.eval()

    return model


def evaluate_model(
    model,
    dataloader,
    class_names: List[str],
    device: torch.device
) -> Dict[str, Any]:
    y_true = []
    y_pred = []
    records = []

    with torch.no_grad():
        for images, labels, paths in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

            for path, true_idx, pred_idx in zip(paths, labels.cpu().numpy(), preds.cpu().numpy()):
                records.append({
                    "image_path": path,
                    "true_label": class_names[int(true_idx)],
                    "predicted_label": class_names[int(pred_idx)],
                    "correct": bool(int(true_idx) == int(pred_idx))
                })

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
        output_dict=True,
        zero_division=0
    )

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "accuracy": round(float(accuracy), 6),
        "macro_f1": round(float(macro_f1), 6),
        "macro_precision": round(float(macro_precision), 6),
        "macro_recall": round(float(macro_recall), 6),
        "confusion_matrix": cm.astype(int).tolist(),
        "classification_report": report,
        "class_names": class_names,
        "total_samples": int(len(y_true)),
        "correct_samples": int(sum(int(t == p) for t, p in zip(y_true, y_pred))),
        "predictions": records
    }


def evaluate_checkpoint(
    experiment_id: str,
    checkpoint_path: Path,
    output_path: Path,
    eval_source: Path,
    image_root: Optional[Path],
    run_dir: Path,
    fold_dir: Optional[Path],
    batch_size: int = 16,
    device_name: Optional[str] = None
) -> Dict[str, Any]:
    device = torch.device(
        device_name
        if device_name
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    class_names = load_class_names(run_dir, fold_dir)

    config = load_model_config(run_dir, fold_dir)
    img_size = int(config.get("img_size", 224))

    transform = get_transform(img_size)

    if eval_source.is_file() and eval_source.suffix.lower() == ".csv":
        dataset = CSVClassificationDataset(
            csv_path=eval_source,
            class_names=class_names,
            image_root=image_root,
            transform=transform
        )
    else:
        dataset = FolderClassificationDataset(
            root=eval_source,
            class_names=class_names,
            transform=transform
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    model = load_model(
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        fold_dir=fold_dir,
        class_names=class_names,
        device=device
    )

    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        class_names=class_names,
        device=device
    )

    metrics["experiment_id"] = experiment_id
    metrics["checkpoint_path"] = str(checkpoint_path)
    metrics["eval_source"] = str(eval_source)
    metrics["output_path"] = str(output_path)

    write_json(output_path, metrics)

    print(f"[DONE] Saved evaluation metrics: {output_path}")
    print(f"[INFO] Accuracy: {metrics['accuracy']}")
    print(f"[INFO] Macro F1: {metrics['macro_f1']}")
    print(f"[INFO] Total samples: {metrics['total_samples']}")
    print(f"[INFO] Correct samples: {metrics['correct_samples']}")

    return metrics


def evaluate_experiment(experiment_id: str):
    registry = load_experiment_registry()

    if experiment_id not in registry:
        raise RuntimeError(f"Experiment not found in registry: {experiment_id}")

    experiment_info = registry[experiment_id]

    if experiment_info.get("task") != "classification":
        raise RuntimeError(f"{experiment_id} is not a classification experiment.")

    run_dir = Path("runs") / "classification" / experiment_id

    if not run_dir.exists():
        raise RuntimeError(f"Run folder not found: {run_dir}")

    checkpoint_jobs = []

    root_checkpoint = run_dir / "model_best.pt"

    if root_checkpoint.exists():
        checkpoint_jobs.append((root_checkpoint, run_dir, run_dir / "test_metrics_best.json"))

    for fold_dir in sorted(run_dir.glob("fold_*")):
        checkpoint = fold_dir / "model_best.pt"

        if checkpoint.exists():
            checkpoint_jobs.append((checkpoint, fold_dir, fold_dir / "test_metrics_best.json"))

    if not checkpoint_jobs:
        raise RuntimeError(f"No model_best.pt checkpoint found under: {run_dir}")

    for checkpoint_path, current_dir, output_path in checkpoint_jobs:
        fold_dir = current_dir if current_dir.name.startswith("fold_") else None

        eval_source, image_root = get_eval_source_for_experiment(
            experiment_id=experiment_id,
            experiment_info=experiment_info,
            run_dir=run_dir,
            fold_dir=fold_dir
        )

        print("=" * 80)
        print(f"[INFO] Experiment: {experiment_id}")
        print(f"[INFO] Checkpoint: {checkpoint_path}")
        print(f"[INFO] Eval source: {eval_source}")
        print(f"[INFO] Output: {output_path}")

        evaluate_checkpoint(
            experiment_id=experiment_id,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            eval_source=eval_source,
            image_root=image_root,
            run_dir=run_dir,
            fold_dir=fold_dir
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        required=True,
        help="Classification experiment ID, e.g. EXP-004_3"
    )

    args = parser.parse_args()

    evaluate_experiment(args.experiment)


if __name__ == "__main__":
    main()
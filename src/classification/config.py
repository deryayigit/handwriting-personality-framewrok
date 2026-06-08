from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class ClassificationConfig:
    experiment_id: str
    experiment_name: str
    dataset_id: str
    split_type: str
    training_strategy: str
    class_weight: bool
    preprocessing: str

    dataset_splits: Dict[str, str]
    class_names: List[str]

    total_epochs: int = 20
    head_epochs: int = 5
    finetune_epochs: int = 15
    checkpoint_epochs: List[int] = None

    val_ratio: float = 0.2
    k_folds: int = 5
    seed: int = 42

    model_name: str = "vit_base_patch16_224"
    num_classes: int = 5
    img_size: int = 224
    dropout: float = 0.1

    batch_size: int = 4
    num_workers: int = 0

    lr_head: float = 3e-4
    lr_finetune: float = 5e-5
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    use_amp: bool = True

    use_weighted_sampler: bool = False
    augmentation_strength: str = "standard"

    runs_root: Path = Path("runs") / "classification"

    def __post_init__(self):
        if self.checkpoint_epochs is None:
            self.checkpoint_epochs = [1, 3, 5, 10, 15, 20]

        self.num_classes = len(self.class_names)

    @property
    def run_dir(self) -> Path:
        return self.runs_root / self.experiment_id

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "dataset_id": self.dataset_id,
            "split_type": self.split_type,
            "training_strategy": self.training_strategy,
            "class_weight": self.class_weight,
            "preprocessing": self.preprocessing,
            "dataset_splits": self.dataset_splits,
            "class_names": self.class_names,
            "total_epochs": self.total_epochs,
            "head_epochs": self.head_epochs,
            "finetune_epochs": self.finetune_epochs,
            "checkpoint_epochs": self.checkpoint_epochs,
            "val_ratio": self.val_ratio,
            "k_folds": self.k_folds,
            "seed": self.seed,
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "img_size": self.img_size,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "lr_head": self.lr_head,
            "lr_finetune": self.lr_finetune,
            "weight_decay": self.weight_decay,
            "label_smoothing": self.label_smoothing,
            "use_amp": self.use_amp,
            "use_weighted_sampler": self.use_weighted_sampler,
            "augmentation_strength": self.augmentation_strength,
            "run_dir": str(self.run_dir),
            "checkpoints_dir": str(self.checkpoints_dir)
        }


def build_classification_config(
    experiment_id: str,
    experiment: Dict[str, Any],
    dataset: Dict[str, Any]
) -> ClassificationConfig:
    return ClassificationConfig(
        experiment_id=experiment_id,
        experiment_name=experiment["name"],
        dataset_id=experiment["dataset_id"],
        split_type=experiment["split_type"],
        training_strategy=experiment["training_strategy"],
        class_weight=experiment["class_weight"],
        preprocessing=experiment.get("preprocessing", "original"),
        dataset_splits=dataset["splits"],
        class_names=dataset["class_names"],
        total_epochs=experiment.get("total_epochs", 20),
        head_epochs=experiment.get("head_epochs", 5),
        finetune_epochs=experiment.get("finetune_epochs", 15),
        checkpoint_epochs=experiment.get(
            "checkpoint_epochs",
            [1, 3, 5, 10, 15, 20]
        ),
        val_ratio=experiment.get("val_ratio", 0.2),
        k_folds=experiment.get("k_folds", 5),
        use_weighted_sampler=experiment.get("use_weighted_sampler", False),
        augmentation_strength=experiment.get("augmentation_strength", "standard")
    )
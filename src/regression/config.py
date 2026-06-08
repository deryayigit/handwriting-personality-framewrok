from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List


@dataclass
class RegressionConfig:
    experiment_id: str
    experiment_name: str
    dataset_id: str
    split_type: str
    training_strategy: str
    preprocessing: str
    loss: str

    dataset_splits: Dict[str, str]
    image_dir: str
    target_columns: List[str]

    total_epochs: int = 20
    head_epochs: int = 5
    finetune_epochs: int = 15
    checkpoint_epochs: List[int] = None

    k_folds: int = 5
    seed: int = 42

    model_name: str = "vit_base_patch16_224"
    img_size: int = 224
    dropout: float = 0.1

    batch_size: int = 4
    num_workers: int = 0

    lr_head: float = 3e-4
    lr_finetune: float = 5e-5
    weight_decay: float = 1e-4
    use_amp: bool = True

    target_min: float = 0.0
    target_max: float = 40.0

    use_weighted_sampler: bool = False
    use_weighted_loss: bool = False
    augmentation_strength: str = "standard"

    runs_root: Path = Path("runs") / "regression"

    def __post_init__(self):
        if self.checkpoint_epochs is None:
            self.checkpoint_epochs = [1, 3, 5, 10, 15, 20]

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
            "preprocessing": self.preprocessing,
            "loss": self.loss,
            "dataset_splits": self.dataset_splits,
            "image_dir": self.image_dir,
            "target_columns": self.target_columns,
            "total_epochs": self.total_epochs,
            "head_epochs": self.head_epochs,
            "finetune_epochs": self.finetune_epochs,
            "checkpoint_epochs": self.checkpoint_epochs,
            "k_folds": self.k_folds,
            "seed": self.seed,
            "model_name": self.model_name,
            "img_size": self.img_size,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "lr_head": self.lr_head,
            "lr_finetune": self.lr_finetune,
            "weight_decay": self.weight_decay,
            "use_amp": self.use_amp,
            "target_min": self.target_min,
            "target_max": self.target_max,
            "use_weighted_sampler": self.use_weighted_sampler,
            "use_weighted_loss": self.use_weighted_loss,
            "augmentation_strength": self.augmentation_strength,
            "runs_root": str(self.runs_root),
            "run_dir": str(self.run_dir),
            "checkpoints_dir": str(self.checkpoints_dir)
        }


def build_regression_config(
    experiment_id: str,
    experiment: Dict[str, Any],
    dataset: Dict[str, Any]
) -> RegressionConfig:
    return RegressionConfig(
        experiment_id=experiment_id,
        experiment_name=experiment["name"],
        dataset_id=experiment["dataset_id"],
        split_type=experiment["split_type"],
        training_strategy=experiment["training_strategy"],
        preprocessing=experiment.get("preprocessing", "original"),
        loss=experiment.get("loss", "mse"),
        dataset_splits=dataset["splits"],
        image_dir=dataset["images"],
        target_columns=dataset["target_columns"],
        total_epochs=experiment.get("total_epochs", 20),
        head_epochs=experiment.get("head_epochs", 5),
        finetune_epochs=experiment.get("finetune_epochs", 15),
        checkpoint_epochs=experiment.get("checkpoint_epochs", [1, 3, 5, 10, 15, 20]),
        k_folds=experiment.get("k_folds", 5),
        target_min=experiment.get("target_min", 0.0),
        target_max=experiment.get("target_max", 40.0),
        use_weighted_sampler=experiment.get("use_weighted_sampler", False),
        use_weighted_loss=experiment.get("use_weighted_loss", False),
        augmentation_strength=experiment.get("augmentation_strength", "standard")
    )
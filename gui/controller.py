import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

from PIL import Image, ImageOps

from src.common.registry import load_experiment_registry, get_dataset
from src.common.preprocessing import enhance_with_cpp
from src.classification.inference import predict_image


class AppController:
    def __init__(self):
        self.classification_runs_root = Path("runs") / "classification"
        self.regression_runs_root = Path("runs") / "regression"

        self.default_analysis_experiment = "EXP-004_3"
        self.default_analysis_checkpoint = "fold_1_model_best"

    def _read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _fold_sort_key(self, path: Path):
        name = path.name

        if name.startswith("fold_"):
            try:
                return int(name.split("_")[1])
            except Exception:
                return 9999

        return 9999

    def _get_runs_root(self, task_type: str) -> Path:
        if task_type == "regression":
            return self.regression_runs_root

        return self.classification_runs_root

    def list_experiments(self, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        registry = load_experiment_registry()
        experiments = []

        for experiment_id, info in registry.items():
            current_task = info.get("task", "classification")

            if task_type is not None and current_task != task_type:
                continue

            run_dir = self._get_runs_root(current_task) / experiment_id

            experiments.append({
                "experiment_id": experiment_id,
                "name": info.get("name", experiment_id),
                "task": current_task,
                "dataset_id": info.get("dataset_id"),
                "split_type": info.get("split_type"),
                "training_strategy": info.get("training_strategy"),
                "class_weight": info.get("class_weight", False),
                "preprocessing": info.get("preprocessing", "original"),
                "use_weighted_sampler": info.get("use_weighted_sampler", False),
                "augmentation_strength": info.get("augmentation_strength", "standard"),
                "k_folds": info.get("k_folds"),
                "run_exists": run_dir.exists(),
                "run_dir": str(run_dir)
            })

        experiments.sort(key=lambda item: item["experiment_id"])
        return experiments

    def list_classification_experiments(self) -> List[Dict[str, Any]]:
        return self.list_experiments(task_type="classification")

    def list_regression_experiments(self) -> List[Dict[str, Any]]:
        return self.list_experiments(task_type="regression")

    def get_experiment_by_id(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        for experiment in self.list_experiments():
            if experiment["experiment_id"] == experiment_id:
                return experiment

        return None

    def list_checkpoints(self, experiment_id: str) -> List[Dict[str, Any]]:
        experiment = self.get_experiment_by_id(experiment_id)
        task_type = experiment.get("task", "classification") if experiment else "classification"

        run_dir = self._get_runs_root(task_type) / experiment_id
        checkpoints = []

        root_best = run_dir / "model_best.pt"

        if root_best.exists():
            checkpoints.append({
                "name": "model_best",
                "path": str(root_best),
                "metrics_path": str(run_dir / "metrics.json"),
                "best_metrics_path": str(run_dir / "test_metrics_best.json"),
                "fold": None,
                "task": task_type
            })

        for fold_dir in sorted(run_dir.glob("fold_*"), key=self._fold_sort_key):
            fold_best = fold_dir / "model_best.pt"

            if fold_best.exists():
                checkpoints.append({
                    "name": f"{fold_dir.name}_model_best",
                    "path": str(fold_best),
                    "metrics_path": str(fold_dir / "metrics.json"),
                    "best_metrics_path": str(fold_dir / "best_val_metrics.json"),
                    "fold": fold_dir.name,
                    "task": task_type
                })

        checkpoint_dir = run_dir / "checkpoints"

        if checkpoint_dir.exists():
            for path in sorted(checkpoint_dir.glob("epoch_*.pt")):
                checkpoints.append({
                    "name": path.stem,
                    "path": str(path),
                    "metrics_path": str(run_dir / "metrics.json"),
                    "best_metrics_path": str(checkpoint_dir / f"{path.stem}_metrics.json"),
                    "fold": None,
                    "task": task_type
                })

        return checkpoints

    def get_default_analysis_checkpoint(self) -> Optional[Dict[str, Any]]:
        checkpoints = self.list_checkpoints(self.default_analysis_experiment)

        if not checkpoints:
            return None

        for checkpoint in checkpoints:
            if checkpoint["name"] == self.default_analysis_checkpoint:
                return checkpoint

        return checkpoints[0]

    def get_preferred_checkpoint(
        self,
        experiment_id: str,
        preferred_name: str = "fold_1_model_best"
    ) -> Optional[Dict[str, Any]]:
        checkpoints = self.list_checkpoints(experiment_id)

        if not checkpoints:
            return None

        for checkpoint in checkpoints:
            if checkpoint["name"] == preferred_name:
                return checkpoint

        for checkpoint in checkpoints:
            if checkpoint["name"] == "model_best":
                return checkpoint

        return checkpoints[0]

    def get_metrics_for_checkpoint(
        self,
        checkpoint: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not checkpoint:
            return None

        metrics_path = Path(checkpoint.get("metrics_path", ""))

        if metrics_path.exists():
            return self._read_json(metrics_path)

        return None

    def get_best_metrics_for_checkpoint(
        self,
        checkpoint: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not checkpoint:
            return None

        possible_paths = []

        best_path = checkpoint.get("best_metrics_path")

        if best_path:
            possible_paths.append(Path(best_path))

        model_path = Path(checkpoint.get("path", ""))

        if model_path.exists():
            parent = model_path.parent

            possible_paths.extend([
                parent / "best_val_metrics.json",
                parent / "test_metrics_best.json"
            ])

            if parent.name.startswith("fold_"):
                experiment_root = parent.parent
                possible_paths.append(experiment_root / "test_metrics_best.json")

        for path in possible_paths:
            if path.exists():
                data = self._read_json(path)

                if data:
                    return data

        metrics_path = Path(checkpoint.get("metrics_path", ""))

        if metrics_path.exists():
            data = self._read_json(metrics_path)

            if data:
                for key in [
                    "best_val_metrics",
                    "best_metrics",
                    "test_metrics",
                    "best_test_metrics"
                ]:
                    value = data.get(key)

                    if isinstance(value, dict):
                        return value

                return data

        return None

    def get_dataset_sample_root(self, dataset_id: str) -> Optional[Path]:
        try:
            dataset = get_dataset(dataset_id)
            splits = dataset.get("splits", {})

            split_path = (
                splits.get("test")
                or splits.get("val")
                or splits.get("valid")
                or splits.get("train")
            )

            if split_path:
                return Path(split_path)

        except Exception:
            return None

        return None

    def build_comparison_row(
        self,
        experiment: Dict[str, Any],
        checkpoint: Dict[str, Any],
        metrics: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        metrics = metrics or {}
        task_type = experiment.get("task", "classification")

        if task_type == "regression":
            return {
                "experiment_id": experiment.get("experiment_id", "—"),
                "dataset": experiment.get("dataset_id", "—"),
                "split_type": experiment.get("split_type", "—"),
                "class_weight": "—",
                "preprocessing": experiment.get("preprocessing", "original"),
                "sampler": "ON" if experiment.get("use_weighted_sampler") else "OFF",
                "augmentation": experiment.get("augmentation_strength", "standard"),
                "checkpoint": checkpoint.get("name", "—"),
                "accuracy": metrics.get("dominant_accuracy", "—"),
                "macro_f1": "—",
                "precision": f"RMSE={metrics.get('rmse', '—')}",
                "recall": f"MAE={metrics.get('mae', '—')}"
            }

        report = metrics.get("classification_report", {})
        macro_avg = report.get("macro avg", {})

        return {
            "experiment_id": experiment.get("experiment_id", "—"),
            "dataset": experiment.get("dataset_id", "—"),
            "split_type": experiment.get("split_type", "—"),
            "class_weight": "ON" if experiment.get("class_weight") else "OFF",
            "preprocessing": experiment.get("preprocessing", "original"),
            "sampler": "ON" if experiment.get("use_weighted_sampler") else "OFF",
            "augmentation": experiment.get("augmentation_strength", "standard"),
            "checkpoint": checkpoint.get("name", "—"),
            "accuracy": metrics.get("accuracy", "—"),
            "macro_f1": metrics.get("macro_f1", "—"),
            "precision": metrics.get("macro_precision", macro_avg.get("precision", "—")),
            "recall": metrics.get("macro_recall", macro_avg.get("recall", "—"))
        }

    def prepare_gui_image(self, image_path: str) -> str:
        output_dir = Path("runs") / "gui_processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"selected_original_{uuid.uuid4().hex}.png"

        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image.save(output_path)

        return str(output_path)

    def preprocess_image(self, image_path: str) -> str:
        output_dir = Path("runs") / "gui_processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"processed_clahe_{uuid.uuid4().hex}.png"

        processed_path = enhance_with_cpp(
            image_path=image_path,
            output_path=str(output_path)
        )

        return str(processed_path)

    def analyze_image(
        self,
        model_path: str,
        image_path: str,
        use_preprocessing: bool,
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        if not model_path:
            raise FileNotFoundError("Model path is empty.")

        model_path_obj = Path(model_path)

        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not image_path:
            raise FileNotFoundError("Image path is empty.")

        image_path_obj = Path(image_path)

        if not image_path_obj.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        processed_path = image_path

        if use_preprocessing:
            processed_path = self.preprocess_image(image_path)

        if task_type == "regression":
            try:
                from src.regression.inference import predict_image_regression
            except Exception as error:
                raise ImportError(
                    "Regression inference function could not be imported. "
                    "Expected: src.regression.inference.predict_image_regression"
                ) from error

            result = predict_image_regression(
                model_path=str(model_path_obj),
                image_path=str(processed_path)
            )
        else:
            result = predict_image(
                model_path=str(model_path_obj),
                image_path=str(processed_path)
            )

        result["processed_path"] = str(processed_path)
        result["use_preprocessing"] = use_preprocessing
        result["model_path"] = str(model_path_obj)
        result["task_type"] = task_type

        return result
from pathlib import Path
from typing import Dict, Any, List

from .utils import load_json


DATASET_REGISTRY_PATH = Path("dataset_registry.json")
EXPERIMENT_REGISTRY_PATH = Path("experiment_registry.json")


def load_dataset_registry() -> Dict[str, Any]:
    if not DATASET_REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Missing dataset registry: {DATASET_REGISTRY_PATH}")
    return load_json(DATASET_REGISTRY_PATH)


def load_experiment_registry() -> Dict[str, Any]:
    if not EXPERIMENT_REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Missing experiment registry: {EXPERIMENT_REGISTRY_PATH}")
    return load_json(EXPERIMENT_REGISTRY_PATH)


def get_dataset(dataset_id: str) -> Dict[str, Any]:
    registry = load_dataset_registry()
    if dataset_id not in registry:
        raise KeyError(f"Dataset not found in registry: {dataset_id}")
    return registry[dataset_id]


def get_experiment(experiment_id: str) -> Dict[str, Any]:
    registry = load_experiment_registry()
    if experiment_id not in registry:
        raise KeyError(f"Experiment not found in registry: {experiment_id}")
    return registry[experiment_id]


def list_experiments() -> List[str]:
    return list(load_experiment_registry().keys())
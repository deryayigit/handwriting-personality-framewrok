from pathlib import Path
from typing import List, Dict, Any

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


class HandwritingDataset(Dataset):
    def __init__(
        self,
        items: List[Dict[str, Any]],
        img_size: int,
        train: bool = True,
        augmentation_strength: str = "standard"
    ):
        self.items = items
        self.img_size = img_size
        self.train = train
        self.augmentation_strength = augmentation_strength

        if train:
            if augmentation_strength == "strong":
                self.transform = T.Compose([
                    T.Resize((img_size, img_size)),
                    T.RandomRotation(8),
                    T.RandomAffine(
                        degrees=0,
                        translate=(0.04, 0.04),
                        scale=(0.92, 1.08),
                        shear=3
                    ),
                    T.ColorJitter(
                        brightness=0.18,
                        contrast=0.18
                    ),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]
                    ),
                ])
            else:
                self.transform = T.Compose([
                    T.Resize((img_size, img_size)),
                    T.RandomRotation(3),
                    T.ColorJitter(
                        brightness=0.1,
                        contrast=0.1
                    ),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]
                    ),
                ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                ),
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        image_path = Path(item["path"])
        label = int(item["label"])

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, label, str(image_path)


def list_dataset_items(split_root: str, class_names: List[str]) -> List[Dict[str, Any]]:
    root = Path(split_root)

    if not root.exists():
        raise FileNotFoundError(f"Dataset split folder not found: {root}")

    items = []

    for label, class_name in enumerate(class_names):
        class_dir = root / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")

        for image_path in class_dir.rglob("*"):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append({
                    "path": str(image_path),
                    "label": label,
                    "class_name": class_name
                })

    return items


def merge_items_from_splits(
    split_paths: List[str],
    class_names: List[str]
) -> List[Dict[str, Any]]:
    all_items = []

    for split_path in split_paths:
        all_items.extend(
            list_dataset_items(split_path, class_names)
        )

    return all_items
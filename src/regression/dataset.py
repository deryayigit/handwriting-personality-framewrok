from pathlib import Path
from typing import List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class HandwritingRegressionDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        target_columns: List[str],
        img_size: int,
        target_min: float,
        target_max: float,
        train: bool = True,
        augmentation_strength: str = "standard"
    ):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.target_columns = target_columns
        self.target_min = target_min
        self.target_max = target_max
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
                    )
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
                    )
                ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = self.image_dir / row["image"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        targets = row[self.target_columns].astype("float32").values
        targets = torch.tensor(targets, dtype=torch.float32)

        targets = (targets - self.target_min) / (self.target_max - self.target_min)

        return image, targets, str(image_path)
from pathlib import Path
import shutil

import pandas as pd


DS1_ROOT = Path("datasets/ds1")
DS4_ROOT = Path("datasets/ds4_regression")
DS4_IMAGE_DIR = DS4_ROOT / "images"
DS4_LABELS_PATH = DS4_ROOT / "labels.csv"

TARGET_ROOT = Path("datasets/ds5")
TARGET_TRAIN_ROOT = TARGET_ROOT / "train"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

CLASS_NAMES = [
    "Agreeableness",
    "Conscientiousness",
    "Extraversion",
    "Neuroticism",
    "Openness"
]

DS4_TARGET_COLUMNS = [
    "Extroversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness to Experience"
]

DS4_TO_CLASS_NAME = {
    "Extroversion": "Extraversion",
    "Agreeableness": "Agreeableness",
    "Conscientiousness": "Conscientiousness",
    "Neuroticism": "Neuroticism",
    "Openness to Experience": "Openness"
}


def ensure_class_folders() -> None:
    for class_name in CLASS_NAMES:
        (TARGET_TRAIN_ROOT / class_name).mkdir(parents=True, exist_ok=True)


def copy_ds1_images() -> int:
    copied_count = 0

    for split_name in ["train", "test"]:
        split_root = DS1_ROOT / split_name

        if not split_root.exists():
            raise FileNotFoundError(f"DS1 split folder not found: {split_root}")

        for class_name in CLASS_NAMES:
            class_dir = split_root / class_name

            if not class_dir.exists():
                raise FileNotFoundError(f"DS1 class folder not found: {class_dir}")

            for image_path in class_dir.rglob("*"):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                target_name = f"ds1_{split_name}_{image_path.stem}{image_path.suffix.lower()}"
                target_path = TARGET_TRAIN_ROOT / class_name / target_name

                shutil.copy2(image_path, target_path)
                copied_count += 1

    return copied_count


def get_dominant_trait(row: pd.Series) -> str:
    scores = row[DS4_TARGET_COLUMNS].astype(float)
    dominant_column = scores.idxmax()
    return DS4_TO_CLASS_NAME[dominant_column]


def copy_ds4_images() -> int:
    if not DS4_LABELS_PATH.exists():
        raise FileNotFoundError(f"DS4 labels file not found: {DS4_LABELS_PATH}")

    if not DS4_IMAGE_DIR.exists():
        raise FileNotFoundError(f"DS4 image folder not found: {DS4_IMAGE_DIR}")

    df = pd.read_csv(DS4_LABELS_PATH)

    required_columns = ["image"] + DS4_TARGET_COLUMNS

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required DS4 column: {column}")

    copied_count = 0
    missing_count = 0

    for _, row in df.iterrows():
        image_name = row["image"]
        source_path = DS4_IMAGE_DIR / image_name

        if not source_path.exists():
            missing_count += 1
            continue

        class_name = get_dominant_trait(row)

        target_name = f"ds4_{Path(image_name).stem}{Path(image_name).suffix.lower()}"
        target_path = TARGET_TRAIN_ROOT / class_name / target_name

        shutil.copy2(source_path, target_path)
        copied_count += 1

    if missing_count > 0:
        print(f"[WARNING] Missing DS4 images skipped: {missing_count}")

    return copied_count


def count_distribution() -> dict:
    counts = {}

    for class_name in CLASS_NAMES:
        class_dir = TARGET_TRAIN_ROOT / class_name
        counts[class_name] = len([
            path for path in class_dir.rglob("*")
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ])

    return counts


def main() -> None:
    if TARGET_ROOT.exists():
        print(f"[INFO] Removing existing DS5 folder: {TARGET_ROOT}")
        shutil.rmtree(TARGET_ROOT)

    ensure_class_folders()

    ds1_count = copy_ds1_images()
    ds4_count = copy_ds4_images()

    print("[DONE] DS5 classification dataset created.")
    print(f"[INFO] DS1 images copied: {ds1_count}")
    print(f"[INFO] DS4 dominant-labeled images copied: {ds4_count}")
    print(f"[INFO] Total images: {ds1_count + ds4_count}")

    print("\n[INFO] DS5 class distribution:")
    for class_name, count in count_distribution().items():
        print(f"{class_name}: {count}")


if __name__ == "__main__":
    main()
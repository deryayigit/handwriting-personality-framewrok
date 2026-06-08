from pathlib import Path
import re
import shutil

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split


SOURCE_ROOT = Path("datasets/hienwrite")
SOURCE_IMAGE_DIR = SOURCE_ROOT / "images"
SOURCE_CSV_PATH = SOURCE_ROOT / "labels.csv"

TARGET_ROOT = Path("datasets/ds4_regression")
TARGET_IMAGE_DIR = TARGET_ROOT / "images"

SEED = 42

TRAIT_COLUMNS = [
    "Extroversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness to Experience"
]


def is_english_image(image_name: str) -> bool:
    return Path(image_name).stem.endswith("e")


def get_writer_id(image_name: str) -> str:
    stem = Path(image_name).stem
    match = re.match(r"(\d+)e$", stem)

    if not match:
        raise ValueError(f"Cannot extract writer id from image name: {image_name}")

    return match.group(1)


def is_valid_image(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def main() -> None:
    if not SOURCE_IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {SOURCE_IMAGE_DIR}")

    if not SOURCE_CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {SOURCE_CSV_PATH}")

    if TARGET_ROOT.exists():
        print(f"[INFO] Removing existing target folder: {TARGET_ROOT}")
        shutil.rmtree(TARGET_ROOT)

    TARGET_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SOURCE_CSV_PATH)

    required_columns = ["image"] + TRAIT_COLUMNS

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    english_df = df[df["image"].apply(is_english_image)].copy()
    english_df["writer_id"] = english_df["image"].apply(get_writer_id)

    copied_rows = []
    missing_images = []
    invalid_images = []

    for _, row in english_df.iterrows():
        image_name = row["image"]
        source_path = SOURCE_IMAGE_DIR / image_name
        target_path = TARGET_IMAGE_DIR / image_name

        if not source_path.exists():
            missing_images.append(image_name)
            continue

        if not is_valid_image(source_path):
            invalid_images.append(image_name)
            continue

        shutil.copy2(source_path, target_path)
        copied_rows.append(row)

    final_df = pd.DataFrame(copied_rows)

    if final_df.empty:
        raise RuntimeError("No valid English images were copied.")

    writer_ids = final_df["writer_id"].unique()

    train_ids, temp_ids = train_test_split(
        writer_ids,
        test_size=0.30,
        random_state=SEED
    )

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=SEED
    )

    train_df = final_df[final_df["writer_id"].isin(train_ids)].copy()
    val_df = final_df[final_df["writer_id"].isin(val_ids)].copy()
    test_df = final_df[final_df["writer_id"].isin(test_ids)].copy()

    final_df.to_csv(TARGET_ROOT / "labels.csv", index=False)
    train_df.to_csv(TARGET_ROOT / "train.csv", index=False)
    val_df.to_csv(TARGET_ROOT / "val.csv", index=False)
    test_df.to_csv(TARGET_ROOT / "test.csv", index=False)

    print("[DONE] DS4 regression dataset prepared.")
    print(f"[INFO] Total valid English samples: {len(final_df)}")
    print(f"[INFO] Train samples: {len(train_df)}")
    print(f"[INFO] Val samples: {len(val_df)}")
    print(f"[INFO] Test samples: {len(test_df)}")
    print(f"[INFO] Missing images: {len(missing_images)}")
    print(f"[INFO] Invalid images: {len(invalid_images)}")

    if missing_images:
        print("\n[WARNING] Missing image files:")
        for image_name in missing_images:
            print(f"- {image_name}")

    if invalid_images:
        print("\n[WARNING] Invalid image files:")
        for image_name in invalid_images:
            print(f"- {image_name}")

    print("\n[INFO] Split writer counts:")
    print(f"Train writers: {len(train_ids)}")
    print(f"Val writers: {len(val_ids)}")
    print(f"Test writers: {len(test_ids)}")

    print("\n[INFO] Target score ranges:")
    print(final_df[TRAIT_COLUMNS].describe())


if __name__ == "__main__":
    main()
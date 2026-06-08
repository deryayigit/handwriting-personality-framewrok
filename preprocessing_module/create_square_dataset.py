from pathlib import Path
import shutil
from PIL import Image, ImageOps


SOURCE_ROOT = Path("datasets/ds1")
TARGET_ROOT = Path("datasets/ds1_square")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

EXPECTED_SPLITS = ["train", "test"]
EXPECTED_CLASSES = [
    "Agreeableness",
    "Conscientiousness",
    "Extraversion",
    "Neuroticism",
    "Openness"
]


def center_crop_to_square(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    width, height = image.size
    square_size = min(width, height)

    left = (width - square_size) // 2
    top = (height - square_size) // 2
    right = left + square_size
    bottom = top + square_size

    return image.crop((left, top, right, bottom))


def count_images(split_root: Path) -> dict:
    counts = {}

    for class_name in EXPECTED_CLASSES:
        class_dir = split_root / class_name
        counts[class_name] = 0

        if class_dir.exists():
            for image_path in class_dir.rglob("*"):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    counts[class_name] += 1

    return counts


def process_split(split_name: str) -> None:
    source_split_dir = SOURCE_ROOT / split_name
    target_split_dir = TARGET_ROOT / split_name

    if not source_split_dir.exists():
        raise FileNotFoundError(f"Source split folder not found: {source_split_dir}")

    for class_name in EXPECTED_CLASSES:
        source_class_dir = source_split_dir / class_name

        if not source_class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {source_class_dir}")

    for image_path in source_split_dir.rglob("*"):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_path = image_path.relative_to(source_split_dir)
        output_path = target_split_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image = Image.open(image_path)
        square_image = center_crop_to_square(image)
        square_image.save(output_path)

        print(f"[OK] {image_path} -> {output_path}")


def main() -> None:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE_ROOT}")

    if TARGET_ROOT.exists():
        print(f"[INFO] Removing existing target folder: {TARGET_ROOT}")
        shutil.rmtree(TARGET_ROOT)

    print(f"[INFO] Source dataset: {SOURCE_ROOT}")
    print(f"[INFO] Target dataset: {TARGET_ROOT}")

    for split_name in EXPECTED_SPLITS:
        process_split(split_name)

    print("\n[INFO] Source distribution:")
    for split_name in EXPECTED_SPLITS:
        counts = count_images(SOURCE_ROOT / split_name)
        print(f"{split_name}: {counts}")

    print("\n[INFO] Target distribution:")
    for split_name in EXPECTED_SPLITS:
        counts = count_images(TARGET_ROOT / split_name)
        print(f"{split_name}: {counts}")

    print("\n[DONE] DS1 square dataset created successfully.")


if __name__ == "__main__":
    main()
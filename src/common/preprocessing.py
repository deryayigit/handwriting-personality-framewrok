from pathlib import Path
import subprocess

from PIL import Image, ImageOps

from .utils import ensure_dir


MAX_PREPROCESS_SIZE = 1024


def convert_image_to_pgm(input_image_path: str, output_pgm_path: Path) -> None:
    image = Image.open(input_image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("L")
    image.thumbnail((MAX_PREPROCESS_SIZE, MAX_PREPROCESS_SIZE), Image.Resampling.LANCZOS)
    image.save(output_pgm_path)


def convert_pgm_to_png(input_pgm_path: Path, output_png_path: Path) -> None:
    image = Image.open(input_pgm_path)
    image.save(output_png_path)


def enhance_with_cpp(image_path: str, output_path: str) -> str:
    temp_dir = Path("runs") / "temp"
    ensure_dir(temp_dir)

    enhancer_exe = Path("preprocessing_module") / "image_enhancer.exe"

    input_pgm_path = temp_dir / "input.pgm"
    output_pgm_path = temp_dir / "output.pgm"
    output_png_path = Path(output_path)

    ensure_dir(output_png_path.parent)

    convert_image_to_pgm(image_path, input_pgm_path)

    if not enhancer_exe.exists():
        raise FileNotFoundError(f"C++ enhancer executable not found: {enhancer_exe}")

    result = subprocess.run(
        [str(enhancer_exe), str(input_pgm_path), str(output_pgm_path)],
        capture_output=True,
        text=True,
        timeout=30
    )

    if result.returncode != 0:
        raise RuntimeError(
            "C++ image enhancer failed.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    if not output_pgm_path.exists():
        raise FileNotFoundError(f"Expected output file was not created: {output_pgm_path}")

    convert_pgm_to_png(output_pgm_path, output_png_path)
    return str(output_png_path)
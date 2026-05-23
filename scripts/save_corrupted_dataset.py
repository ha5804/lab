import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.corruptions import apply_corruption

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CORRUPTIONS = [
    "gaussian_noise",
    "motion_blur",
    "brightness",
    "contrast",
    "jpeg_compression",
    "downsample_upsample",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Save corrupted MVTec/ViSA images to a local directory.")
    parser.add_argument("--dataset", choices=["mvtec", "visa", "all"], default="all")
    parser.add_argument(
        "--corruption",
        required=True,
        help="Corruption name, comma-separated names, or all.",
    )
    parser.add_argument("--severity", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        args.corruptions = parse_corruptions(args.corruption)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def parse_corruptions(value):
    if value == "all":
        return CORRUPTIONS

    corruptions = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(corruptions) - set(CORRUPTIONS))
    if unknown:
        raise ValueError(f"Unknown corruption(s): {unknown}. Choose from: {CORRUPTIONS}")
    return corruptions


def dataset_roots(dataset):
    if dataset == "mvtec":
        return [REPO_ROOT / "data" / "MVTec"]
    if dataset == "visa":
        return [REPO_ROOT / "data" / "Visa"]
    return [REPO_ROOT / "data" / "MVTec", REPO_ROOT / "data" / "Visa"]


def should_corrupt(path):
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False
    return "ground_truth" not in path.parts and "Masks" not in path.parts


def save_image(src_path, dst_path, corruption, severity):
    with Image.open(src_path) as image:
        image = image.convert("RGB")
        image = apply_corruption(image, corruption=corruption, severity=severity)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(dst_path)


def copy_file(src_path, dst_path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def save_dataset(root, output_dir, corruption, severity, overwrite):
    if not root.exists():
        print(f"skip missing dataset root: {root}")
        return 0, 0

    target_root = output_dir / f"{root.name}_corruption" / corruption
    image_count = 0
    copied_count = 0

    for src_path in root.rglob("*"):
        if not src_path.is_file():
            continue

        dst_path = target_root / src_path.relative_to(root)
        if dst_path.exists() and not overwrite:
            continue

        if should_corrupt(src_path):
            save_image(src_path, dst_path, corruption, severity)
            image_count += 1
        else:
            copy_file(src_path, dst_path)
            copied_count += 1

    print(f"saved {root.name}: corrupted_images={image_count}, copied_files={copied_count}, output={target_root}")
    return image_count, copied_count


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    total_images = 0
    total_copied = 0

    for root in dataset_roots(args.dataset):
        for corruption in args.corruptions:
            image_count, copied_count = save_dataset(
                root,
                output_dir,
                corruption,
                args.severity,
                args.overwrite,
            )
            total_images += image_count
            total_copied += copied_count

    print(f"done: corrupted_images={total_images}, copied_files={total_copied}")


if __name__ == "__main__":
    main()

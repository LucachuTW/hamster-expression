from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from . import config
from .predictor import EmotionPredictor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run emotion inference on all images in a folder")
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing images to process",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=config.WEIGHTS_PATH,
        help=f"Path to model checkpoint (default: {config.WEIGHTS_PATH})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional torch.device string (cpu, cuda, etc.)",
    )
    return parser.parse_args()


def collect_images(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Images directory {folder} does not exist")
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a directory")
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> int:
    args = parse_args()
    image_paths = collect_images(args.images)
    if not image_paths:
        print(f"No supported image files found in {args.images}", file=sys.stderr)
        return 1

    predictor = EmotionPredictor(weights_path=args.weights, device=args.device)

    for image_path in image_paths:
        try:
            boxes = predictor.predict(Image.open(image_path))
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[ERROR] {image_path.name}: {exc}", file=sys.stderr)
            continue

        if not boxes:
            print(f"{image_path.name}: no detections")
            continue

        top = max(boxes, key=lambda box: box.score)
        print(f"{image_path.name}: {top.label} ({top.score:.2f}) - {len(boxes)} box(es)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

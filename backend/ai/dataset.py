from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from . import config
from .transforms import build_transforms


class EmotionDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        split: str = "train",
        image_size: int = config.IMAGE_SIZE,
        grid_size: int = config.GRID_SIZE,
        num_boxes: int = config.NUM_BOXES,
        num_classes: int = config.NUM_CLASSES,
        augment: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.images_dir = self.root_dir / split / "images"
        self.labels_dir = self.root_dir / split / "labels"
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        paths = []
        for ext in exts:
            paths.extend(self.images_dir.glob(ext))
        self.image_paths = sorted(paths)
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.images_dir}")
        self.transform = build_transforms(image_size, augment=augment)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        boxes = self._read_label(label_path)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        target = self._build_target(boxes)
        return image, target

    def _read_label(self, label_path: Path) -> List[Tuple[int, float, float, float, float]]:
        if not label_path.exists():
            return []
        boxes: List[Tuple[int, float, float, float, float]] = []
        with label_path.open("r", encoding="utf-8") as file:
            for line in file.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, w, h = [float(x) for x in parts]
                boxes.append((int(cls), x, y, w, h))
        return boxes

    def _build_target(self, boxes: List[Tuple[int, float, float, float, float]]) -> torch.Tensor:
        cell_dim = config.NUM_BOXES * 5 + self.num_classes
        target = torch.zeros((self.grid_size, self.grid_size, cell_dim), dtype=torch.float32)
        for cls, x_center, y_center, width, height in boxes:
            grid_x = int(x_center * self.grid_size)
            grid_y = int(y_center * self.grid_size)
            grid_x = min(self.grid_size - 1, max(0, grid_x))
            grid_y = min(self.grid_size - 1, max(0, grid_y))
            x_cell = x_center * self.grid_size - grid_x
            y_cell = y_center * self.grid_size - grid_y
            box = torch.tensor([x_cell, y_cell, width, height, 1.0])
            slot = target[grid_y, grid_x, : 5 * self.num_boxes]
            slot_empty = True
            if slot.numel() >= 5:
                existing_conf = slot[4::5]
                slot_empty = bool((existing_conf == 0).any())
            if slot_empty:
                start = 0
                target[grid_y, grid_x, start : start + 5] = box
                target[grid_y, grid_x, 5 * self.num_boxes + cls] = 1.0
        return target

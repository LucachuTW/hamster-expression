from __future__ import annotations

import io
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from . import config
from .model import FEATURE_CHANNELS, EmotionDetector
from .transforms import build_transforms
from .utils import PredictionBox, cells_to_bboxes, non_max_suppression


class EmotionPredictor:
    def __init__(
        self,
        weights_path: Path = config.WEIGHTS_PATH,
        device: Optional[str] = None,
        image_size: Optional[int] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {weights_path}. Train the model before running inference."
            )
        state_dict = torch.load(weights_path, map_location=self.device)
        inferred_size = image_size or self._infer_image_size(state_dict)
        self.image_size = inferred_size
        self.model = EmotionDetector(image_size=self.image_size).to(self.device)
        self.transforms = build_transforms(self.image_size, augment=False)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @staticmethod
    def _infer_image_size(state_dict: dict) -> int:
        head_weight = state_dict.get("head.1.weight")
        if head_weight is None:
            return config.IMAGE_SIZE
        flat_dim = head_weight.shape[1]
        channels_last = FEATURE_CHANNELS[-1]
        if flat_dim % channels_last != 0:
            return config.IMAGE_SIZE
        feature_map_area = flat_dim // channels_last
        feature_map_size = math.isqrt(feature_map_area)
        if feature_map_size * feature_map_size != feature_map_area:
            return config.IMAGE_SIZE
        spatial_downscale = 2 ** (len(FEATURE_CHANNELS) - 1)
        return feature_map_size * spatial_downscale

    def _prepare_image(self, image: Image.Image | np.ndarray | bytes) -> torch.Tensor:
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise ValueError("Unsupported image type")
        tensor = self.transforms(image).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, image: Image.Image | np.ndarray | bytes) -> List[PredictionBox]:
        tensor = self._prepare_image(image)
        with torch.no_grad():
            raw_preds = self.model(tensor)
        boxes = cells_to_bboxes(raw_preds)[0]
        return non_max_suppression(boxes)

    def top_prediction(self, image: Image.Image | np.ndarray | bytes) -> Optional[PredictionBox]:
        boxes = self.predict(image)
        return max(boxes, key=lambda box: box.score) if boxes else None

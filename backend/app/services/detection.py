from __future__ import annotations

import base64
import logging
from typing import List

from backend.ai.utils import PredictionBox

from ..dependencies import get_predictor

logger = logging.getLogger("detection_service")


class DetectionService:
    def __init__(self) -> None:
        self.predictor = get_predictor()

    def predict_from_bytes(self, image_bytes: bytes) -> List[PredictionBox]:
        logger.debug("Running prediction on %d bytes", len(image_bytes))
        return self.predictor.predict(image_bytes)

    def predict_from_base64(self, encoded: str) -> List[PredictionBox]:
        try:
            image_bytes = base64.b64decode(encoded.split(",")[-1])
        except Exception as exc:  # pylint:disable=broad-except
            logger.error("Failed to decode base64 payload: %s", exc)
            raise
        return self.predict_from_bytes(image_bytes)

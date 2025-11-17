from functools import lru_cache

from backend.ai.predictor import EmotionPredictor

from .config import settings


@lru_cache(maxsize=1)
def get_predictor() -> EmotionPredictor:
    return EmotionPredictor(weights_path=settings.weights_path)

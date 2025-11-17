from typing import List

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    name: str
    version: str


class StreamRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image string")


class BoundingBox(BaseModel):
    label: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


class PredictionResponse(BaseModel):
    emotion: str
    confidence: float
    boxes: List[BoundingBox]

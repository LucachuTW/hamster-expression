from typing import List

from fastapi import APIRouter, File, UploadFile

from backend.ai.utils import PredictionBox

from ..schemas import BoundingBox, PredictionResponse, StreamRequest
from ..services.detection import DetectionService

router = APIRouter(prefix="/api", tags=["inference"])
service = DetectionService()


def to_response(boxes: List[PredictionBox]) -> PredictionResponse:
    if boxes:
        top = max(boxes, key=lambda box: box.score)
        emotion = top.label
        confidence = top.score
    else:
        emotion = "unknown"
        confidence = 0.0
    return PredictionResponse(
        emotion=emotion,
        confidence=confidence,
        boxes=[
            BoundingBox(
                label=box.label,
                score=box.score,
                x1=box.x1,
                y1=box.y1,
                x2=box.x2,
                y2=box.y2,
            )
            for box in boxes
        ],
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_from_file(file: UploadFile = File(...)) -> PredictionResponse:
    contents = await file.read()
    boxes = service.predict_from_bytes(contents)
    return to_response(boxes)


@router.post("/predict/stream", response_model=PredictionResponse)
async def predict_from_stream(payload: StreamRequest) -> PredictionResponse:
    boxes = service.predict_from_base64(payload.image)
    return to_response(boxes)

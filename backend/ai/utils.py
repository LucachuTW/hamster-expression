from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from . import config


@dataclass
class PredictionBox:
    label: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


def cells_to_bboxes(
    predictions: torch.Tensor,
    conf_threshold: float = 0.4,
) -> List[List[PredictionBox]]:
    preds = predictions.detach().cpu()
    batch_size = preds.size(0)
    S = config.GRID_SIZE
    C = config.NUM_CLASSES
    bboxes_batch: List[List[PredictionBox]] = []
    preds = preds.view(batch_size, S, S, config.NUM_BOXES * 5 + C)
    for batch_idx in range(batch_size):
        sample_boxes: List[PredictionBox] = []
        for y in range(S):
            for x in range(S):
                cell = preds[batch_idx, y, x]
                conf = torch.sigmoid(cell[4]).item()
                if conf < conf_threshold:
                    continue
                x_offset, y_offset, width, height = torch.sigmoid(cell[0]), torch.sigmoid(cell[1]), torch.sigmoid(cell[2]), torch.sigmoid(cell[3])
                x_center = (x + x_offset.item()) / S
                y_center = (y + y_offset.item()) / S
                w = width.item()
                h = height.item()
                x1 = max(0.0, x_center - w / 2)
                y1 = max(0.0, y_center - h / 2)
                x2 = min(1.0, x_center + w / 2)
                y2 = min(1.0, y_center + h / 2)
                class_probs = torch.softmax(cell[5:], dim=0)
                class_idx = int(torch.argmax(class_probs))
                label = config.LABELS[class_idx]
                score = conf * class_probs[class_idx].item()
                sample_boxes.append(PredictionBox(label=label, score=score, x1=x1, y1=y1, x2=x2, y2=y2))
        bboxes_batch.append(sample_boxes)
    return bboxes_batch


def intersection_over_union(box_a: PredictionBox, box_b: PredictionBox) -> float:
    x1 = max(box_a.x1, box_b.x1)
    y1 = max(box_a.y1, box_b.y1)
    x2 = min(box_a.x2, box_b.x2)
    y2 = min(box_a.y2, box_b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a.x2 - box_a.x1) * (box_a.y2 - box_a.y1)
    area_b = (box_b.x2 - box_b.x1) * (box_b.y2 - box_b.y1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def non_max_suppression(boxes: List[PredictionBox], iou_threshold: float = 0.5) -> List[PredictionBox]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
    selected: List[PredictionBox] = []
    while boxes:
        current = boxes.pop(0)
        selected.append(current)
        boxes = [
            box
            for box in boxes
            if box.label != current.label or intersection_over_union(current, box) < iou_threshold
        ]
    return selected

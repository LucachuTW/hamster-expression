import torch
from torch import nn
from torch.nn import functional as F

from . import config


class YoloLoss(nn.Module):
    def __init__(
        self,
        grid_size: int = config.GRID_SIZE,
        num_boxes: int = config.NUM_BOXES,
        num_classes: int = config.NUM_CLASSES,
        lambda_coord: float = config.LAMBDA_COORD,
        lambda_noobj: float = config.LAMBDA_NOOBJ,
    ) -> None:
        super().__init__()
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = predictions.view(-1, self.S, self.S, self.B * 5 + self.C)
        obj_mask = targets[..., 4] > 0
        noobj_mask = ~obj_mask

        pred_boxes = torch.sigmoid(preds[..., :4])
        target_boxes = targets[..., :4]
        pred_conf = torch.sigmoid(preds[..., 4])
        target_conf = targets[..., 4]
        pred_classes = preds[..., 5:]
        target_classes = targets[..., 5:]

        coord_loss = torch.tensor(0.0, device=preds.device)
        class_loss = torch.tensor(0.0, device=preds.device)
        if obj_mask.any():
            coord_loss = F.mse_loss(pred_boxes[obj_mask], target_boxes[obj_mask], reduction="sum")
            class_loss = F.mse_loss(pred_classes[obj_mask], target_classes[obj_mask], reduction="sum")

        noobj_loss = F.mse_loss(pred_conf[noobj_mask], target_conf[noobj_mask], reduction="sum")
        obj_loss = F.mse_loss(pred_conf[obj_mask], target_conf[obj_mask], reduction="sum") if obj_mask.any() else 0.0

        total_loss = (
            self.lambda_coord * coord_loss
            + obj_loss
            + self.lambda_noobj * noobj_loss
            + class_loss
        )
        batch_size = predictions.size(0)
        return total_loss / batch_size

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .dataset import EmotionDataset
from .loss import YoloLoss
from .model import EmotionDetector

logger = logging.getLogger("emotion_training")


def create_dataloaders(data_dir: Path, image_size: int) -> Tuple[DataLoader, DataLoader]:
    train_dataset = EmotionDataset(
        root_dir=data_dir,
        split="train",
        image_size=image_size,
        augment=True,
    )
    val_dataset = EmotionDataset(
        root_dir=data_dir,
        split="valid",
        image_size=image_size,
        augment=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: EmotionDetector,
    dataloader: DataLoader,
    criterion: YoloLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0
    progress = tqdm(dataloader, desc=f"Epoch {epoch} [train]", leave=False)
    for images, targets in progress:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        running_loss += loss.item()
        progress.set_postfix(loss=loss.item())
    return running_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate(
    model: EmotionDetector,
    dataloader: DataLoader,
    criterion: YoloLoss,
    device: torch.device,
    epoch: int,
) -> float:
    model.eval()
    running_loss = 0.0
    progress = tqdm(dataloader, desc=f"Epoch {epoch} [val]", leave=False)
    for images, targets in progress:
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item()
        progress.set_postfix(loss=loss.item())
    return running_loss / max(1, len(dataloader))


def save_checkpoint(model: EmotionDetector, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info("Saved model checkpoint to %s", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the custom emotion detection model.")
    parser.add_argument("--data-dir", type=Path, default=config.DATASET_DIR, help="Path to dataset root.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs.")
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE, help="Image resize dimension.")
    parser.add_argument("--lr", type=float, default=config.LR, help="Learning rate.")
    parser.add_argument("--weights", type=Path, default=config.WEIGHTS_PATH, help="Where to store trained weights.")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda:0")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, val_loader = create_dataloaders(args.data_dir, args.image_size)
    model = EmotionDetector(image_size=args.image_size).to(device)
    criterion = YoloLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = evaluate(model, val_loader, criterion, device, epoch)
        logger.info("Epoch %d | train_loss=%.4f | val_loss=%.4f", epoch, train_loss, val_loss)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, args.weights)
    logger.info("Training finished. Best val loss %.4f", best_val)


if __name__ == "__main__":
    main()

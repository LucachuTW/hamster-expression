import torch
from torch import nn

from . import config

FEATURE_CHANNELS = [3, 32, 64, 128, 256, 512]


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EmotionDetector(nn.Module):
    def __init__(
        self,
        image_size: int = config.IMAGE_SIZE,
        grid_size: int = config.GRID_SIZE,
        num_boxes: int = config.NUM_BOXES,
        num_classes: int = config.NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        channels = FEATURE_CHANNELS
        layers = []
        in_c = channels[0]
        for out_c in channels[1:]:
            layers.append(ConvBlock(in_c, out_c, kernel_size=3, stride=1, padding=1))
            layers.append(nn.MaxPool2d(2, 2))
            in_c = out_c
        self.features = nn.Sequential(*layers)
        spatial_downscale = 2 ** (len(channels) - 1)
        feature_map_size = image_size // spatial_downscale
        flat_dim = feature_map_size * feature_map_size * channels[-1]
        head_dim = grid_size * grid_size * (num_boxes * 5 + num_classes)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            nn.Linear(1024, head_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x.view(-1, self.grid_size, self.grid_size, self.num_boxes * 5 + self.num_classes)

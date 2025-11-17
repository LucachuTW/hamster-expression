from typing import Optional

import torchvision.transforms as T


def build_transforms(image_size: int, augment: bool = False) -> Optional[T.Compose]:
    transforms = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ]
    if augment:
        transforms.insert(
            0,
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        )
    return T.Compose(transforms)

"""
dataset.py -- wraps our image folders in a PyTorch Dataset with transforms.

CONCEPTS:

* A neural network doesn't see "images" -- it sees tensors (multi-dimensional
  arrays of numbers). An RGB image becomes a tensor of shape
  (3, height, width): one 2D grid of numbers per color channel, each pixel
  value normalized to roughly [-1, 1] or [0, 1].

* transforms.Resize / ToTensor / Normalize turn a JPEG on disk into that
  tensor. We resize everything to the same fixed size (e.g. 128x128) because
  the network's input layer expects a fixed shape.

* Data augmentation (RandomHorizontalFlip, RandomRotation, ColorJitter) is
  applied ONLY to the training set. It creates realistic variations of each
  image on the fly (flipped, slightly rotated, slightly recolored) so the
  model sees more diversity than raw file count suggests, which reduces
  overfitting -- the model can't just memorize exact pixels if it never sees
  the exact same tensor twice.

* The validation set gets NO random augmentation (just resize + normalize),
  because we want a stable, repeatable measurement of how well the model
  generalizes.
"""
from pathlib import Path

import torch
from torchvision import datasets, transforms

IMAGE_SIZE = 224  # matches what ResNet18's pretrained weights were trained on

# ImageNet mean/std are a common, reasonable default for normalizing natural
# images even when not using an ImageNet-pretrained model -- they just center
# pixel values in a range gradient descent handles well.
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool):
    ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    ops += [
        transforms.ToTensor(),  # PIL image (0-255 ints) -> tensor (0-1 floats), shape (3, H, W)
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ]
    return transforms.Compose(ops)


def load_datasets(data_root: str = "data/kanto151"):
    """
    Returns (train_dataset, val_dataset, class_names).

    Uses torchvision's ImageFolder, which expects:
        data_root/train/<class_name>/*.jpg
        data_root/val/<class_name>/*.jpg
    and automatically builds the label <-> integer index mapping from the
    sorted folder names -- that's `class_names` below, and it's the same
    ordering the model's output layer will use.
    """
    root = Path(data_root)
    train_dir, val_dir = root / "train", root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit(
            f"Expected '{train_dir}' and '{val_dir}' to exist. "
            f"Run prepare_data.py first."
        )

    train_dataset = datasets.ImageFolder(train_dir, transform=build_transforms(train=True))
    val_dataset = datasets.ImageFolder(val_dir, transform=build_transforms(train=False))

    assert train_dataset.classes == val_dataset.classes, (
        "Train and val folders have different class lists -- did prepare_data.py "
        "run correctly for both splits?"
    )

    return train_dataset, val_dataset, train_dataset.classes


def get_device():
    """Pick the fastest available device: CUDA GPU > Apple Silicon MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

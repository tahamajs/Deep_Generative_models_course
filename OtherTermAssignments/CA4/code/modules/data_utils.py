"""Data utilities for notebook
Keep lightweight, easy-to-read functions that the notebook can import.
"""
from typing import Tuple
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def load_image_dataset(path: str, image_size: int = 256, batch_size: int = 32, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """Load images using torchvision ImageFolder with common transforms.

    Returns a DataLoader for the folder at `path`.
    Replace or extend transforms as needed in the notebook.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    ds = datasets.ImageFolder(path, transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    return dl


def debug_batch(dl: DataLoader):
    """Return a single batch and shapes for quick checks."""
    x, y = next(iter(dl))
    return x, y

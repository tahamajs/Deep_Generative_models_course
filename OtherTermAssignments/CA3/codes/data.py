"""
MNIST data loading utilities.
"""

from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import DataConfig
from utils import set_seed


def mnist_dataloaders(
    cfg: DataConfig, normalize_to_minus1_1: bool = False
) -> Tuple[DataLoader, DataLoader]:
    set_seed(cfg.seed)
    transform_list = [transforms.ToTensor()]
    if normalize_to_minus1_1:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    transform = transforms.Compose(transform_list)
    data_root = Path(__file__).resolve().parent.parent / "data" / "mnist"
    train_ds = datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    return train_loader, test_loader

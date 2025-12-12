"""
Shared utilities for CA3 experiments: seeding, I/O, and visualization.
"""

from pathlib import Path
import random
import numpy as np
import torch
from torchvision.utils import make_grid, save_image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_grid(
    images: torch.Tensor, path: Path, nrow: int = 8, normalize: bool = True
) -> None:
    ensure_dir(path.parent)
    grid = make_grid(images, nrow=nrow, normalize=normalize, value_range=(0, 1))
    save_image(grid, path)

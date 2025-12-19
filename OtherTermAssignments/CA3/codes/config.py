"""
Central configuration dataclasses for CA3 assignments.
All defaults align with the HW3 specification.
"""

from dataclasses import dataclass, field
from pathlib import Path
import torch


def default_device() -> torch.device:
    """Prefer CUDA when available; fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class RunPaths:
    root: Path = Path(__file__).resolve().parent.parent
    images: Path = field(default_factory=lambda: RunPaths.root / "images")  # type: ignore
    checkpoints: Path = field(default_factory=lambda: RunPaths.root / "checkpoints")  # type: ignore

    def ensure(self) -> None:
        for path in [self.images, self.checkpoints]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42
    image_size: int = 28
    channels: int = 1


@dataclass
class EBMConfig:
    epochs: int = 10
    lr: float = 2e-4
    lambda_reg: float = 1e-3
    langevin_step_size: float = 0.1
    langevin_steps: int = 60
    langevin_noise_scale: float = 1.0
    clamp_min: float = 0.0
    clamp_max: float = 1.0
    log_interval: int = 200
    sample_grid: int = 16
    device: torch.device = field(default_factory=default_device)


@dataclass
class NCSNConfig:
    epochs: int = 30
    lr: float = 2e-4
    batch_size: int = 64
    num_workers: int = 4
    sigma_begin: float = 30.0
    sigma_end: float = 0.01
    num_levels: int = 10
    langevin_steps: int = 150
    step_lr: float = 2e-5
    image_size: int = 28
    channels: int = 1
    embed_dim: int = 256
    conditional: bool = False
    num_classes: int = 10
    device: torch.device = field(default_factory=default_device)

    @property
    def sigmas(self):
        return torch.exp(
            torch.linspace(
                torch.log(torch.tensor(self.sigma_begin)),
                torch.log(torch.tensor(self.sigma_end)),
                self.num_levels,
            )
        ).to(self.device)

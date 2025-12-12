"""
Annealed Langevin Dynamics sampling and denoising for NCSN.
"""
from typing import Optional
import torch
from torch import nn
from .config import NCSNConfig


@torch.no_grad()
def annealed_langevin_dynamics(
    model: nn.Module, cfg: NCSNConfig, sigmas: torch.Tensor, x: torch.Tensor, y: Optional[torch.Tensor] = None
) -> torch.Tensor:
    for sigma in sigmas:
        step_size = cfg.step_lr * (sigma / sigmas[-1]) ** 2
        for _ in range(cfg.langevin_steps):
            noise = torch.randn_like(x)
            score = model(x, sigma.expand(x.size(0)), y)
            x = x + step_size * score + (2 * step_size) ** 0.5 * noise
        x = x.clamp(-1.0, 1.0)
    return x


@torch.no_grad()
def sample(model: nn.Module, cfg: NCSNConfig, num_samples: int, y: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = torch.randn(num_samples, cfg.channels, cfg.image_size, cfg.image_size, device=cfg.device)
    sigmas = cfg.sigmas
    return annealed_langevin_dynamics(model, cfg, sigmas, x, y)


@torch.no_grad()
def denoise(
    model: nn.Module,
    cfg: NCSNConfig,
    images: torch.Tensor,
    noise_level: float,
    y: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # use a single sigma value for denoising
    sigma = torch.tensor([noise_level], device=images.device)
    return annealed_langevin_dynamics(model, cfg, sigma, images, y)

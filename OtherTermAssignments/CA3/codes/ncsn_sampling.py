"""
Annealed Langevin Dynamics sampling and denoising for NCSN.
"""

from typing import Optional
import torch
from torch import nn

try:
    from .config import NCSNConfig
except ImportError:
    from config import NCSNConfig


@torch.no_grad()
def annealed_langevin_dynamics(
    model: nn.Module,
    cfg: NCSNConfig,
    sigmas: torch.Tensor,
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run annealed Langevin dynamics to sample/denoise (returns final tensor)."""
    for sigma in sigmas:
        step_size = cfg.step_lr * (sigma / sigmas[-1]) ** 2
        for _ in range(cfg.langevin_steps):
            noise = torch.randn_like(x)
            score = model(x, sigma.expand(x.size(0)), y)
            x = x + step_size * score + (2 * step_size) ** 0.5 * noise
        x = x.clamp(-1.0, 1.0)
    return x


@torch.no_grad()
def annealed_langevin_dynamics_with_trajectory(
    model: nn.Module,
    cfg: NCSNConfig,
    sigmas: torch.Tensor,
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    record_every: int = 10,
):
    """Run ALD and record intermediate snapshots for visualization.

    Returns a list of CPU tensors (snapshots collected at intervals).
    """
    traj = [x.detach().cpu().clone()]
    for sigma in sigmas:
        step_size = cfg.step_lr * (sigma / sigmas[-1]) ** 2
        for t in range(cfg.langevin_steps):
            noise = torch.randn_like(x)
            score = model(x, sigma.expand(x.size(0)), y)
            x = x + step_size * score + (2 * step_size) ** 0.5 * noise
            if (t + 1) % record_every == 0:
                traj.append(x.detach().cpu().clone())
        x = x.clamp(-1.0, 1.0)
        traj.append(x.detach().cpu().clone())
    return traj


@torch.no_grad()
def sample(
    model: nn.Module,
    cfg: NCSNConfig,
    num_samples: int,
    y: Optional[torch.Tensor] = None,
    return_trajectory: bool = False,
    record_every: int = 10,
) -> torch.Tensor:
    x = torch.randn(
        num_samples, cfg.channels, cfg.image_size, cfg.image_size, device=cfg.device
    )
    sigmas = cfg.sigmas
    if return_trajectory:
        return annealed_langevin_dynamics_with_trajectory(model, cfg, sigmas, x, y, record_every)
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

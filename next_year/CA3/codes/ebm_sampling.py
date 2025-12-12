"""
Langevin dynamics utilities for Energy-Based Models.
"""
import torch
from torch import nn
from .config import EBMConfig


class LangevinSampler:
    def __init__(self, model: nn.Module, cfg: EBMConfig):
        self.model = model
        self.cfg = cfg

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        step_size = self.cfg.langevin_step_size
        noise_scale = self.cfg.langevin_noise_scale * (2 * step_size) ** 0.5
        xk = x
        for _ in range(self.cfg.langevin_steps):
            xk = xk.detach().requires_grad_(True)
            energy = self.model(xk).sum()
            grad = torch.autograd.grad(energy, xk)[0]
            xk = xk - step_size * grad + noise_scale * torch.randn_like(xk)
            xk = xk.clamp(self.cfg.clamp_min, self.cfg.clamp_max)
        return xk.detach()


def sample_from_noise(model: nn.Module, cfg: EBMConfig, shape) -> torch.Tensor:
    init = torch.rand(shape, device=cfg.device)
    sampler = LangevinSampler(model, cfg)
    return sampler(init)

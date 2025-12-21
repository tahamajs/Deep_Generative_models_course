"""
Weighted denoising score matching loss for NCSN.
"""

import torch
from torch import nn
from config import NCSNConfig
from typing import Optional


def dsm_loss(
    model: nn.Module,
    x: torch.Tensor,
    cfg: NCSNConfig,
    sigmas: torch.Tensor,
    y: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size = x.size(0)
    # pick a noise level per-sample
    idx = torch.randint(0, cfg.num_levels, (batch_size,), device=x.device)
    sigma = sigmas[idx]
    noise = torch.randn_like(x)
    x_noisy = x + sigma.view(-1, 1, 1, 1) * noise

    score = model(x_noisy, sigma, y)
    target = -noise / sigma.view(-1, 1, 1, 1)
    loss = (
        0.5 * (score - target).pow(2).reshape(batch_size, -1).mean(dim=1) * (sigma**2)
    )
    return loss.mean()

"""Sampling utilities (DDIM/DDPM wrappers) — minimal placeholder implementations."""
import torch


def sample_ddim(model, shape, steps=50, device='cpu'):
    """Minimal deterministic sampling placeholder that returns Gaussian samples passed through model."""
    model.to(device)
    model.eval()
    x = torch.randn(shape, device=device)
    with torch.no_grad():
        for _ in range(steps):
            x = model(x)
    return x

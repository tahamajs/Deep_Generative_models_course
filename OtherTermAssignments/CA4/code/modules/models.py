"""Model definitions and lightweight builders.
Add full model code here or import from other modules.
"""
import torch
import torch.nn as nn


def build_simple_unet(in_channels=3, base_channels=64):
    """Return a very small U-Net-like module for quick tests.
    Replace with full U-Net implementation when needed.
    """
    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(base_channels, base_channels, 3, padding=1),
                nn.ReLU()
            )
            self.dec = nn.Sequential(
                nn.Conv2d(base_channels, base_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(base_channels, in_channels, 1)
            )
        def forward(self, x, t=None, cond=None):
            h = self.enc(x)
            return self.dec(h)
    return TinyUNet()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

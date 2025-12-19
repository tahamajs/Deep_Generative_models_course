"""
Noise Conditional Score Network for MNIST (unconditional + class-conditional).
Implements Gaussian Fourier embeddings and FiLM-conditioned residual blocks.
"""

from typing import Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F
from .config import NCSNConfig


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim: int, scale: float = 1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        # sigma: (B,)
        proj = sigma[:, None] * self.W[None, :]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class AdaptiveResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, embed_dim: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(embed_dim, out_ch * 2)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        scale_shift = self.emb_proj(emb)
        scale, shift = torch.chunk(scale_shift, 2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class ScoreNet(nn.Module):
    def __init__(self, cfg: NCSNConfig):
        super().__init__()
        self.cfg = cfg
        self.sigma_embed = GaussianFourierProjection(cfg.embed_dim, scale=1.0)
        self.emb_mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )
        self.input_conv = nn.Conv2d(cfg.channels, 64, kernel_size=3, padding=1)

        self.enc1 = AdaptiveResBlock(64, 64, cfg.embed_dim)
        self.down1 = nn.AvgPool2d(2)
        self.enc2 = AdaptiveResBlock(64, 128, cfg.embed_dim)
        self.down2 = nn.AvgPool2d(2)
        self.enc3 = AdaptiveResBlock(128, 256, cfg.embed_dim)

        self.dec1 = AdaptiveResBlock(256 + 128, 128, cfg.embed_dim)
        self.dec2 = AdaptiveResBlock(128 + 64, 64, cfg.embed_dim)
        self.out_conv = nn.Conv2d(64, cfg.channels, kernel_size=3, padding=1)

        self.conditional = cfg.conditional
        if self.conditional:
            self.label_emb = nn.Embedding(cfg.num_classes, cfg.embed_dim)

    def forward(
        self, x: torch.Tensor, sigma: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        emb = self.sigma_embed(sigma)
        if self.conditional:
            if y is None:
                raise ValueError("Class labels required for conditional ScoreNet.")
            emb = emb + self.label_emb(y)
        emb = self.emb_mlp(emb)

        h0 = self.input_conv(x)
        h1 = self.enc1(h0, emb)
        h2 = self.enc2(self.down1(h1), emb)
        h3 = self.enc3(self.down2(h2), emb)

        u1 = F.interpolate(h3, scale_factor=2, mode="nearest")
        d1 = self.dec1(torch.cat([u1, h2], dim=1), emb)
        u2 = F.interpolate(d1, scale_factor=2, mode="nearest")
        d2 = self.dec2(torch.cat([u2, h1], dim=1), emb)
        out = self.out_conv(F.silu(d2))
        return out

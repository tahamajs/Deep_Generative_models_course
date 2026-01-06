"""Model definitions and builders extracted from the notebook.

Includes:
- `SinusoidalPositionEmbeddings`
- `SelfAttention`
- `ResidualBlock`
- `UNet` (noise-prediction U-Net)

These are direct notebook translations intended for import into the notebook.
"""
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SelfAttention(nn.Module):
    """Self-attention mechanism for feature maps."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).transpose(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff(attention_value) + attention_value
        return attention_value.transpose(1, 2).view(-1, self.channels, size, size)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, kernel_size=4):
        super().__init__()

        self.act = nn.SiLU()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.transform = nn.Conv2d(out_channels, out_channels, kernel_size, 2, 1)

        num_groups = 32 if out_channels % 32 == 0 else min(out_channels // 4, 8)
        self.bn1 = nn.GroupNorm(num_groups, out_channels)
        self.bn2 = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x, t):
        h = self.act(self.bn1(self.conv1(x)))

        time_emb = self.act(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb

        h = self.act(self.bn2(self.conv2(h)))
        return h


class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, base_channels=64):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.inc = nn.Conv2d(c_in, base_channels, kernel_size=3, padding=1)

        # --- Encoder (Downsampling) ---
        self.down1 = ResidualBlock(base_channels, base_channels * 2, time_dim)
        self.pool1 = nn.Conv2d(base_channels * 2, base_channels * 2, 4, 2, 1)
        self.sa1 = SelfAttention(base_channels * 2)

        self.down2 = ResidualBlock(base_channels * 2, base_channels * 4, time_dim)
        self.pool2 = nn.Conv2d(base_channels * 4, base_channels * 4, 4, 2, 1)
        self.sa2 = SelfAttention(base_channels * 4)

        self.down3 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)
        self.pool3 = nn.Conv2d(base_channels * 4, base_channels * 4, 4, 2, 1)
        self.sa3 = SelfAttention(base_channels * 4)

        # --- Bottleneck ---
        self.bot1 = ResidualBlock(base_channels * 4, base_channels * 8, time_dim)
        self.bot2 = ResidualBlock(base_channels * 8, base_channels * 8, time_dim)
        self.bot3 = ResidualBlock(base_channels * 8, base_channels * 4, time_dim)
        self.bot_sa = SelfAttention(base_channels * 4)

        # --- Decoder (Upsampling) ---
        self.up_trans1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 4, 2, 1)
        self.up1 = ResidualBlock(base_channels * 8, base_channels * 2, time_dim)
        self.sa4 = SelfAttention(base_channels * 2)

        self.up_trans2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, 2, 1)
        self.up2 = ResidualBlock(base_channels * 2 + base_channels * 4, base_channels, time_dim)
        self.sa5 = SelfAttention(base_channels)

        self.up_trans3 = nn.ConvTranspose2d(base_channels, base_channels, 4, 2, 1)
        self.up3 = ResidualBlock(base_channels + base_channels * 2, base_channels, time_dim)
        self.sa6 = SelfAttention(base_channels)

        self.outc = nn.Conv2d(base_channels, c_out, kernel_size=1)

    def _safe_concat(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffX != 0 or diffY != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return torch.cat([x1, x2], dim=1)

    def forward(self, x, t):
        t = self.time_mlp(t)

        x1 = self.inc(x)

        h1 = self.down1(x1, t)
        h1 = self.sa1(h1)
        x2 = self.pool1(h1)

        h2 = self.down2(x2, t)
        h2 = self.sa2(h2)
        x3 = self.pool2(h2)

        h3 = self.down3(x3, t)
        h3 = self.sa3(h3)
        x4 = self.pool3(h3)

        mid = self.bot1(x4, t)
        mid = self.bot2(mid, t)
        mid = self.bot3(mid, t)
        mid = self.bot_sa(mid)

        x = self.up_trans1(mid)
        x = self._safe_concat(x, h3)
        x = self.up1(x, t)
        x = self.sa4(x)

        x = self.up_trans2(x)
        x = self._safe_concat(x, h2)
        x = self.up2(x, t)
        x = self.sa5(x)

        x = self.up_trans3(x)
        x = self._safe_concat(x, h1)
        x = self.up3(x, t)
        x = self.sa6(x)

        return self.outc(x)


def build_simple_unet(in_channels=3, base_channels=64):
    """Compatibility helper: return a tiny U-Net for quick tests."""
    return UNet(c_in=in_channels, c_out=in_channels, time_dim=128, base_channels=base_channels)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

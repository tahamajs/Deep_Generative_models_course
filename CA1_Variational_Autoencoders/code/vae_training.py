"""
VAE training script for CA1.

Implements the beta-VAE architecture and training loop described in HW1:
- 128x128 RGB faces, normalized to [-1, 1]
- Encoder/decoder per provided layer table (4 strided conv downs, 4 deconvs up)
- ELBO loss with KL + reconstruction; supports beta scaling
- Train/val split 0.8/0.2; logs per-epoch recon/KL

Usage (from repo root):
  source .venv/bin/activate
  python CA1_Variational_Autoencoders/code/vae_training.py --data-root CA1_Variational_Autoencoders/train
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    """Configuration dataclass for VAE training hyperparameters and paths."""
    data_root: Path
    out_dir: Path
    image_size: int = 128
    batch_size: int = 128
    lr: float = 5e-4
    epochs: int = 1000
    beta: float = 1.0  # beta-VAE coefficient
    val_split: float = 0.2
    num_workers: int = 4
    seed: int = 42
    sample_grid: int = 32  # number of images for recon/gen grids


class BetaVAE(nn.Module):
    """Convolutional beta-VAE for 128x128 RGB images with configurable latent dimension and dropout."""
    def __init__(self, latent_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        # Encoder: 128 -> 8 (4 downsamples), channel depth grows 32 -> 64 -> 128 -> 256
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # After 4 downsamples: 128 -> 8, channels=256 => 256*8*8 = 16384
        hidden_dim = 256 * 8 * 8
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder mirrors encoder
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input images to latent mean and log-variance."""
        h = self.enc(x)
        h = torch.flatten(h, start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample latent vector z using the reparameterization trick."""
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector z to reconstructed image."""
        h = self.fc_dec(z)
        h = h.view(h.size(0), 256, 8, 8)
        return self.dec(h)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: returns reconstruction, mean, and log-variance."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def elbo_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ELBO loss: reconstruction + beta-scaled KL divergence."""
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    loss = recon_loss + beta * kl
    return loss, recon_loss, kl


def get_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders with deterministic split and transforms."""
    tfm = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.ImageFolder(root=str(cfg.data_root), transform=tfm)
    val_len = int(len(dataset) * cfg.val_split)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def save_samples(
    model: BetaVAE,
    device: torch.device,
    data: torch.Tensor,
    out_dir: Path,
    step_label: str,
    num_samples: int,
) -> None:
    """Save reconstruction and generation image grids for a batch of data and random samples."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        data = data.to(device)[:num_samples]
        recon, _, _ = model(data)
        random_z = torch.randn(num_samples, model.fc_mu.out_features, device=device)
        gen = model.decode(random_z)
        save_image(
            torch.cat([data, recon], dim=0) * 0.5 + 0.5,
            out_dir / f"recon_{step_label}.png",
            nrow=int(math.sqrt(num_samples * 2)),
        )
        save_image(
            gen * 0.5 + 0.5,
            out_dir / f"gen_{step_label}.png",
            nrow=int(math.sqrt(num_samples)),
        )
    model.train()


def train(cfg: TrainConfig) -> None:
    """Run the full training loop for beta-VAE and save periodic samples/logs."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(cfg)
    model = BetaVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.out_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,split,loss,recon,kl\n")

    for epoch in range(1, cfg.epochs + 1):
        for split, loader in [("train", train_loader), ("val", val_loader)]:
            model.train() if split == "train" else model.eval()
            running = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "count": 0}
            with torch.set_grad_enabled(split == "train"):
                for batch in loader:
                    imgs, _ = batch
                    imgs = imgs.to(device)
                    recon, mu, logvar = model(imgs)
                    loss, recon_loss, kl = elbo_loss(
                        recon, imgs, mu, logvar, beta=cfg.beta
                    )
                    if split == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    bs = imgs.size(0)
                    running["loss"] += loss.item() * bs
                    running["recon"] += recon_loss.item() * bs
                    running["kl"] += kl.item() * bs
                    running["count"] += bs
            for key in ["loss", "recon", "kl"]:
                running[key] /= max(1, running["count"])
            with log_path.open("a") as f:
                f.write(
                    f"{epoch},{split},{running['loss']:.4f},{running['recon']:.4f},{running['kl']:.4f}\n"
                )

        # Save samples periodically (every 50 epochs or first epoch)
        if epoch == 1 or epoch % 50 == 0:
            sample_batch = next(iter(train_loader))[0]
            save_samples(
                model=model,
                device=device,
                data=sample_batch,
                out_dir=cfg.out_dir,
                step_label=f"epoch{epoch}",
                num_samples=min(cfg.sample_grid, sample_batch.size(0)),
            )
        # Optional: early stop for smoke tests
        if cfg.epochs <= 5 and epoch == cfg.epochs:
            break


def parse_args() -> TrainConfig:
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train beta-VAE on face dataset (CA1)."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to image root (ImageFolder).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("CA1_Variational_Autoencoders/output")
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return TrainConfig(
        data_root=args.data_root,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        image_size=args.image_size,
        val_split=args.val_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)

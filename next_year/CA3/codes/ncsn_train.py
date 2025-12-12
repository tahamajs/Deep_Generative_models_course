"""
Training pipeline for Noise Conditional Score Network (unconditional or conditional).
"""

from pathlib import Path
from typing import Dict, Any, Optional
import torch
from torch import optim
from tqdm import tqdm

from .config import NCSNConfig, DataConfig, RunPaths
from .data import mnist_dataloaders
from .ncsn_model import ScoreNet
from .ncsn_loss import dsm_loss
from .ncsn_sampling import sample
from .utils import save_grid, set_seed, ensure_dir


def train(
    cfg: NCSNConfig, output_dir: Path, conditional: bool = False
) -> Dict[str, Any]:
    cfg.conditional = conditional
    set_seed(42)

    data_cfg = DataConfig(
        batch_size=cfg.batch_size, num_workers=cfg.num_workers, channels=cfg.channels
    )
    train_loader, _ = mnist_dataloaders(data_cfg, normalize_to_minus1_1=True)

    device = cfg.device
    model = ScoreNet(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    sigmas = cfg.sigmas

    ensure_dir(output_dir)
    history = {"loss": []}
    checkpoint_path = output_dir / ("ncsn_cond.pt" if conditional else "ncsn.pt")

    for epoch in range(1, cfg.epochs + 1):
        progress = tqdm(
            train_loader, desc=f"NCSN Epoch {epoch}/{cfg.epochs}", leave=False
        )
        for x, labels in progress:
            x = x.to(device)
            x = x * 2 - 1  # map to [-1, 1]
            y = labels.to(device) if conditional else None

            loss = dsm_loss(model, x, cfg, sigmas, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            history["loss"].append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.3f}")

        with torch.no_grad():
            y_samples: Optional[torch.Tensor] = None
            if conditional:
                y_samples = torch.arange(0, 16, device=device) % cfg.num_classes
            samples = sample(model, cfg, num_samples=16, y=y_samples)
            samples = (samples + 1) / 2.0
            save_grid(samples, output_dir / f"samples_epoch{epoch}.png", nrow=4)

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            checkpoint_path,
        )

    return history


def main():
    paths = RunPaths()
    paths.ensure()
    cfg = NCSNConfig()
    train(cfg, paths.images / "ncsn", conditional=False)
    train(cfg, paths.images / "ncsn_cond", conditional=True)


if __name__ == "__main__":
    main()

"""
Training pipeline for Noise Conditional Score Network (unconditional or conditional).
"""

import matplotlib.pyplot as plt
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch import optim
from tqdm import tqdm

from config import NCSNConfig, DataConfig, RunPaths
from data import mnist_dataloaders
from ncsn_model import ScoreNet
from ncsn_loss import dsm_loss
from ncsn_sampling import sample
from utils import save_grid, set_seed, ensure_dir, write_run_info


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
    write_run_info(
        output_dir,
        configs={
            "data": asdict(data_cfg),
            "model": asdict(cfg),
            "conditional": {"enabled": conditional},
        },
        notes={"script": "ncsn_train.py"},
        device=str(device),
    )

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

        # Interactive/demo sampling (no_grad is safe here because sample() uses ALD and handles grad internally)
        y_samples: Optional[torch.Tensor] = None
        if conditional:
            y_samples = torch.arange(0, 16, device=device) % cfg.num_classes
        samples = sample(model, cfg, num_samples=16, y=y_samples)
        samples = (samples + 1) / 2.0
        save_grid(samples.detach().cpu(), output_dir / f"samples_epoch{epoch}.png", nrow=4)

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            checkpoint_path,
        )

    return history


def train_interactive(
    cfg: NCSNConfig, output_dir: Path, conditional: bool = False, epochs: int = 1, show: bool = True
) -> Dict[str, Any]:
    """Interactive/demo training for quick runs inside notebooks.

    Runs a small number of epochs, saves demo samples/loss plots to `output_dir`,
    and optionally displays sample grids inline when `show` is True.
    """
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
    write_run_info(
        output_dir,
        configs={
            "data": asdict(data_cfg),
            "model": asdict(cfg),
            "conditional": {"enabled": conditional},
        },
        notes={"script": "ncsn_train.py (interactive)"},
        device=str(device),
    )

    for epoch in range(1, epochs + 1):
        progress = tqdm(
            train_loader, desc=f"NCSN demo epoch {epoch}/{epochs}", leave=False
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

        y_samples: Optional[torch.Tensor] = None
        if conditional:
            y_samples = torch.arange(0, 16, device=device) % cfg.num_classes
        samples = sample(model, cfg, num_samples=16, y=y_samples)
        samples = (samples + 1) / 2.0
        save_grid(samples.detach().cpu(), output_dir / f"samples_epoch{epoch}.png", nrow=4)

        if show:
            try:
                import matplotlib.pyplot as plt
                grid = make_grid(samples, nrow=4, normalize=True, value_range=(0, 1))
                img_arr = grid.permute(1, 2, 0).detach().cpu().numpy()
                plt.figure(figsize=(4, 4))
                plt.axis("off")
                plt.imshow(img_arr)
                plt.show()
            except Exception:
                pass

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            checkpoint_path,
        )

    return history

    # Save loss visualization
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(history["loss"], label="DSM loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("NCSN DSM Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "ncsn_loss.png")
        plt.close()
    except Exception:
        # Best-effort plotting
        pass

    return history


def main():
    paths = RunPaths()
    paths.ensure()
    cfg = NCSNConfig()
    train(cfg, paths.images / "ncsn", conditional=False)
    train(cfg, paths.images / "ncsn_cond", conditional=True)


if __name__ == "__main__":
    main()

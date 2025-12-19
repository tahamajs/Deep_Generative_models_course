"""
Sampling and denoising utilities for trained NCSN models.
"""

from pathlib import Path
from typing import Optional, Sequence
import torch

from .config import NCSNConfig, DataConfig, RunPaths
from .data import mnist_dataloaders
from .ncsn_model import ScoreNet
from .ncsn_sampling import sample, annealed_langevin_dynamics
from .utils import save_grid, ensure_dir


def load_model(
    checkpoint: Path, cfg: NCSNConfig, conditional: bool = False
) -> ScoreNet:
    cfg.conditional = conditional
    model = ScoreNet(cfg).to(cfg.device)
    state = torch.load(checkpoint, map_location=cfg.device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def generate_and_denoise(
    checkpoint: Path,
    output_dir: Path,
    cfg: NCSNConfig,
    conditional: bool = False,
    noise_levels: Sequence[float] = (0.2, 0.4, 0.6),
) -> None:
    ensure_dir(output_dir)
    data_cfg = DataConfig(batch_size=16)
    train_loader, _ = mnist_dataloaders(data_cfg, normalize_to_minus1_1=True)
    model = load_model(checkpoint, cfg, conditional)

    y_samples: Optional[torch.Tensor] = None
    if conditional:
        y_samples = torch.arange(0, 16, device=cfg.device) % cfg.num_classes
    samples = sample(model, cfg, num_samples=16, y=y_samples)
    save_grid((samples + 1) / 2.0, output_dir / "ncsn_samples.png", nrow=4)

    x_real, labels = next(iter(train_loader))
    x_real = x_real.to(cfg.device)[:16] * 2 - 1
    y = labels.to(cfg.device)[:16] if conditional else None

    for nl in noise_levels:
        noisy = x_real + nl * torch.randn_like(x_real)
        sigmas = torch.tensor([nl], device=cfg.device)
        denoised = annealed_langevin_dynamics(model, cfg, sigmas, noisy.clone(), y)
        save_grid((noisy + 1) / 2.0, output_dir / f"noisy_{nl:.2f}.png", nrow=4)
        save_grid((denoised + 1) / 2.0, output_dir / f"denoised_{nl:.2f}.png", nrow=4)


def main():
    paths = RunPaths()
    paths.ensure()
    cfg = NCSNConfig()
    generate_and_denoise(
        paths.images / "ncsn" / "ncsn.pt",
        paths.images / "ncsn_infer",
        cfg,
        conditional=False,
    )
    generate_and_denoise(
        paths.images / "ncsn_cond" / "ncsn_cond.pt",
        paths.images / "ncsn_cond_infer",
        cfg,
        conditional=True,
    )


if __name__ == "__main__":
    main()

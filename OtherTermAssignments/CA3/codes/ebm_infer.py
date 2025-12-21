"""
Inference helpers for the trained EBM: generation from noise and denoising.
"""

from pathlib import Path
import torch

from config import DataConfig, EBMConfig, RunPaths
from data import mnist_dataloaders
from ebm_model import ConvEnergyModel
from ebm_sampling import sample_from_noise, LangevinSampler
from utils import save_grid, set_seed, ensure_dir


def load_model(checkpoint: Path, device: torch.device) -> ConvEnergyModel:
    model = ConvEnergyModel().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def generate_and_denoise(
    checkpoint: Path, output_dir: Path, data_cfg: DataConfig, ebm_cfg: EBMConfig
) -> None:
    set_seed(data_cfg.seed)
    train_loader, _ = mnist_dataloaders(data_cfg)
    ensure_dir(output_dir)
    model = load_model(checkpoint, ebm_cfg.device)
    sampler = LangevinSampler(model, ebm_cfg)

    # Sampling requires gradients for input; enable gradients for Langevin updates
    samples = sample_from_noise(model, ebm_cfg, (16, 1, 28, 28))
    save_grid(samples.detach().cpu(), output_dir / "ebm_samples_final.png", nrow=4)

    # Denoise a few training digits
    x_real, _ = next(iter(train_loader))
    x_real = x_real[:16].to(ebm_cfg.device)
    noise = torch.randn_like(x_real) * 0.3
    noisy = (x_real + noise).clamp(0.0, 1.0)
    denoised = sampler(noisy)
    save_grid(x_real, output_dir / "ebm_real.png", nrow=4)
    save_grid(noisy, output_dir / "ebm_noisy.png", nrow=4)
    save_grid(denoised, output_dir / "ebm_denoised.png", nrow=4)


def main():
    paths = RunPaths()
    ckpt = paths.images / "ebm" / "ebm_ckpt.pt"
    generate_and_denoise(ckpt, paths.images / "ebm_infer", DataConfig(), EBMConfig())


if __name__ == "__main__":
    main()

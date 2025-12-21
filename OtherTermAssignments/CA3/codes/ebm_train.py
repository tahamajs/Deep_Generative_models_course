"""
Training loop for the MNIST Energy-Based Model with Langevin sampling.
Saves sample grids during training and checkpoints the model.
"""

import matplotlib.pyplot as plt
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any

import torch
from torch import optim
from tqdm import tqdm

from config import DataConfig, EBMConfig, RunPaths
from data import mnist_dataloaders
from ebm_model import ConvEnergyModel
from ebm_sampling import LangevinSampler, sample_from_noise
from utils import save_grid, set_seed, ensure_dir, write_run_info


def train(
    cfg_data: DataConfig, cfg_model: EBMConfig, output_dir: Path
) -> Dict[str, Any]:
    set_seed(cfg_data.seed)
    train_loader, test_loader = mnist_dataloaders(cfg_data)
    device = cfg_model.device

    model = ConvEnergyModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg_model.lr)
    sampler = LangevinSampler(model, cfg_model)

    history = {"loss": [], "E_real": [], "E_fake": []}
    ensure_dir(output_dir)
    checkpoint_path = output_dir / "ebm_ckpt.pt"
    write_run_info(
        output_dir,
        configs={"data": asdict(cfg_data), "model": asdict(cfg_model)},
        notes={"script": "ebm_train.py"},
        device=str(device),
    )

    for epoch in range(1, cfg_model.epochs + 1):
        progress = tqdm(
            train_loader, desc=f"EBM Epoch {epoch}/{cfg_model.epochs}", leave=False
        )
        for step, (x_real, _) in enumerate(progress, start=1):
            x_real = x_real.to(device)
            x_fake = sampler(torch.rand_like(x_real))

            E_real = model(x_real)
            E_fake = model(x_fake)

            data_term = E_real.mean() - E_fake.detach().mean()
            reg_term = cfg_model.lambda_reg * (
                E_real.pow(2).mean() + E_fake.detach().pow(2).mean()
            )
            loss = data_term + reg_term

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            history["loss"].append(loss.item())
            history["E_real"].append(E_real.mean().item())
            history["E_fake"].append(E_fake.mean().item())

            if step % cfg_model.log_interval == 0:
                progress.set_postfix(
                    loss=f"{loss.item():.3f}",
                    E_real=f"{E_real.mean().item():.3f}",
                    E_fake=f"{E_fake.mean().item():.3f}",
                )

        # Save training samples each epoch
        with torch.no_grad():
            samples = sample_from_noise(
                model, cfg_model, (cfg_model.sample_grid, 1, 28, 28)
            )
            save_grid(samples, output_dir / f"ebm_samples_epoch{epoch}.png", nrow=4)

            # Denoising a few test digits via Langevin starting from noisy images
            x_test, _ = next(iter(test_loader))
            x_test = x_test[: cfg_model.sample_grid].to(device)
            noise = torch.randn_like(x_test) * 0.3
            noisy = (x_test + noise).clamp(0.0, 1.0)
            denoised = sampler(noisy)
            save_grid(noisy, output_dir / f"ebm_noisy_epoch{epoch}.png", nrow=4)
            save_grid(denoised, output_dir / f"ebm_denoised_epoch{epoch}.png", nrow=4)

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            checkpoint_path,
        )

    # Save simple visualizations of loss and energies
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(history["loss"], label="loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("EBM Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "ebm_loss.png")
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.plot(history["E_real"], label="E_real")
        plt.plot(history["E_fake"], label="E_fake")
        plt.xlabel("Step")
        plt.ylabel("Energy")
        plt.title("EBM Energies")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "ebm_energy.png")
        plt.close()
    except Exception:
        # Best-effort visualization; continue even if matplotlib is unavailable.
        pass

    return history


def main():
    paths = RunPaths()
    paths.ensure()
    data_cfg = DataConfig()
    ebm_cfg = EBMConfig()
    train(data_cfg, ebm_cfg, paths.images / "ebm")


if __name__ == "__main__":
    main()

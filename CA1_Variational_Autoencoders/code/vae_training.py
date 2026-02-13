"""
VAE training and report-asset generation script for CA1.

This script is the canonical pipeline for:
- Training beta-VAE on 128x128 RGB faces
- Saving checkpoints and per-epoch metrics
- Generating all report figures under images/
- Writing run summaries for report metric synchronization
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    """Configuration for VAE training, checkpointing, and report asset generation."""

    data_root: Path
    out_dir: Path
    images_dir: Path
    checkpoint_dir: Path
    image_size: int = 128
    batch_size: int = 64
    lr: float = 5e-4
    epochs: int = 120
    beta: float = 1.0
    val_split: float = 0.2
    num_workers: int = 0
    seed: int = 42
    sample_grid: int = 32
    latent_dim: int = 32
    dropout: float = 0.2
    device: str = "auto"
    resume: Path | None = None
    save_report_plots: bool = True
    analysis_samples: int = 500
    save_descriptive_copies: bool = True


class BetaVAE(nn.Module):
    """Convolutional beta-VAE for 128x128 RGB images."""

    def __init__(self, latent_dim: int = 32, dropout: float = 0.2):
        super().__init__()

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

        hidden_dim = 256 * 8 * 8
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

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
        h = self.enc(x)
        h = torch.flatten(h, start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        h = h.view(h.size(0), 256, 8, 8)
        return self.dec(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


@dataclass
class DataBundle:
    """Container for datasets/loaders and class metadata."""

    dataset: Dataset
    train_set: Subset
    val_set: Subset
    train_loader: DataLoader
    val_loader: DataLoader
    analysis_loader: DataLoader
    class_names: List[str]
    class_to_idx: Dict[str, int]


def resolve_device(device_arg: str) -> torch.device:
    """Resolve execution device with priority cuda > mps > cpu when auto."""
    requested = device_arg.lower().strip()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")

    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("Requested --device mps, but MPS is not available.")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unknown --device value: {device_arg}")


def _loader_kwargs(cfg: TrainConfig, device: torch.device) -> Dict[str, object]:
    """Build DataLoader kwargs with safe defaults for macOS/MPS."""
    kwargs: Dict[str, object] = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def get_data_bundle(cfg: TrainConfig, device: torch.device) -> DataBundle:
    """Create train/val/analysis loaders with deterministic split."""
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
    split_generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=split_generator)

    loader_kwargs = _loader_kwargs(cfg, device)
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    analysis_loader = DataLoader(train_set, shuffle=False, **loader_kwargs)

    return DataBundle(
        dataset=dataset,
        train_set=train_set,
        val_set=val_set,
        train_loader=train_loader,
        val_loader=val_loader,
        analysis_loader=analysis_loader,
        class_names=list(dataset.classes),
        class_to_idx=dict(dataset.class_to_idx),
    )


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


def train_epoch(
    model: BetaVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    beta: float,
    device: torch.device,
) -> Dict[str, float]:
    """Run one training epoch and return mean losses."""
    model.train()
    running = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "count": 0}

    for imgs, _ in loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(imgs)
        loss, recon_loss, kl = elbo_loss(recon, imgs, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        running["loss"] += loss.item() * bs
        running["recon"] += recon_loss.item() * bs
        running["kl"] += kl.item() * bs
        running["count"] += bs

    denom = max(1, running["count"])
    return {
        "loss": running["loss"] / denom,
        "recon": running["recon"] / denom,
        "kl": running["kl"] / denom,
    }


def eval_epoch(
    model: BetaVAE,
    loader: DataLoader,
    beta: float,
    device: torch.device,
) -> Dict[str, float]:
    """Run one validation epoch and return mean losses."""
    model.eval()
    running = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "count": 0}

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            recon, mu, logvar = model(imgs)
            loss, recon_loss, kl = elbo_loss(recon, imgs, mu, logvar, beta=beta)

            bs = imgs.size(0)
            running["loss"] += loss.item() * bs
            running["recon"] += recon_loss.item() * bs
            running["kl"] += kl.item() * bs
            running["count"] += bs

    denom = max(1, running["count"])
    return {
        "loss": running["loss"] / denom,
        "recon": running["recon"] / denom,
        "kl": running["kl"] / denom,
    }


def _denorm(x: torch.Tensor) -> torch.Tensor:
    """Map image tensor from [-1, 1] to [0, 1]."""
    return (x * 0.5 + 0.5).clamp(0.0, 1.0)


def _save_with_optional_copy(src: Path, dst_dir: Path, dst_name: str, enabled: bool) -> None:
    """Copy generated asset to an additional descriptive filename when enabled."""
    if not enabled:
        return
    shutil.copy2(src, dst_dir / dst_name)


def save_training_samples(
    model: BetaVAE,
    device: torch.device,
    batch: torch.Tensor,
    out_dir: Path,
    epoch: int,
    sample_grid: int,
) -> None:
    """Save lightweight reconstruction and generation samples during training."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        data = batch[:sample_grid].to(device)
        recon, _, _ = model(data)
        z = torch.randn(data.size(0), model.fc_mu.out_features, device=device)
        gen = model.decode(z)

        recon_path = out_dir / f"recon_epoch{epoch}.png"
        gen_path = out_dir / f"gen_epoch{epoch}.png"

        save_image(
            _denorm(torch.cat([data, recon], dim=0).cpu()),
            recon_path,
            nrow=max(1, data.size(0)),
        )
        save_image(_denorm(gen.cpu()), gen_path, nrow=max(1, int(math.sqrt(data.size(0)))))


def _checkpoint_payload(
    model: BetaVAE,
    optimizer: torch.optim.Optimizer,
    history: List[Dict[str, float]],
    epoch: int,
    best_val_loss: float,
    cfg: TrainConfig,
) -> Dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
    }


def save_checkpoint(
    path: Path,
    model: BetaVAE,
    optimizer: torch.optim.Optimizer,
    history: List[Dict[str, float]],
    epoch: int,
    best_val_loss: float,
    cfg: TrainConfig,
) -> None:
    """Save training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            history=history,
            epoch=epoch,
            best_val_loss=best_val_loss,
            cfg=cfg,
        ),
        path,
    )


def load_checkpoint(
    ckpt_path: Path,
    model: BetaVAE,
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, List[Dict[str, float]], float]:
    """Load checkpoint and return start_epoch, history, best_val_loss."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])

    last_epoch = int(payload.get("epoch", 0))
    history = list(payload.get("history", []))
    best_val_loss = float(payload.get("best_val_loss", float("inf")))
    return last_epoch + 1, history, best_val_loss


def _extract_latent_stats(
    model: BetaVAE,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract latent means/logvars and labels for analysis."""
    model.eval()

    all_mu: List[np.ndarray] = []
    all_logvar: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    seen = 0

    with torch.no_grad():
        for imgs, labels in loader:
            if seen >= max_samples:
                break
            remaining = max_samples - seen
            take = min(remaining, imgs.size(0))

            imgs = imgs[:take].to(device)
            labels = labels[:take]

            mu, logvar = model.encode(imgs)
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
            all_labels.append(labels.numpy())

            seen += take

    if not all_mu:
        raise RuntimeError("No samples were available for latent analysis.")

    return np.concatenate(all_mu), np.concatenate(all_logvar), np.concatenate(all_labels)


def _find_base_image_for_class(
    loader: DataLoader,
    target_label: int,
    device: torch.device,
) -> torch.Tensor:
    """Find one sample image from loader matching target class label."""
    for imgs, labels in loader:
        mask = labels == target_label
        if mask.any():
            idx = int(torch.nonzero(mask, as_tuple=False)[0].item())
            return imgs[idx : idx + 1].to(device)
    raise RuntimeError(f"Could not find sample with label {target_label} for interpolation.")


def generate_report_assets(
    model: BetaVAE,
    device: torch.device,
    data: DataBundle,
    history: List[Dict[str, float]],
    cfg: TrainConfig,
) -> None:
    """Generate all report figures and overwrite existing report image files."""
    images_dir = cfg.images_dir
    images_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: dataset sample preview
    preview_imgs, _ = next(iter(data.analysis_loader))
    preview_grid = make_grid(_denorm(preview_imgs[:4]), nrow=4)
    fig_path = images_dir / "output_cell_11_img_0.png"
    save_image(preview_grid, fig_path)
    _save_with_optional_copy(fig_path, images_dir, "dataset_preview.png", cfg.save_descriptive_copies)

    # Figure 2: reconstruction grids (3 batches)
    model.eval()
    val_iter = iter(data.val_loader)
    with torch.no_grad():
        for idx in range(3):
            try:
                imgs, _ = next(val_iter)
            except StopIteration:
                val_iter = iter(data.val_loader)
                imgs, _ = next(val_iter)

            n = min(8, imgs.size(0))
            inp = imgs[:n].to(device)
            recon, _, _ = model(inp)
            recon_grid = torch.cat([inp.cpu(), recon.cpu()], dim=0)
            fig_path = images_dir / f"output_cell_19_img_{idx}.png"
            save_image(_denorm(recon_grid), fig_path, nrow=n)
            _save_with_optional_copy(
                fig_path,
                images_dir,
                f"reconstruction_batch_{idx + 1}.png",
                cfg.save_descriptive_copies,
            )

    # Figure 3: training curves
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    train_recon = [float(row["train_recon"]) for row in history]
    val_recon = [float(row["val_recon"]) for row in history]
    train_kl = [float(row["train_kl"]) for row in history]
    val_kl = [float(row["val_kl"]) for row in history]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train ELBO", linewidth=1.8)
    plt.plot(epochs, val_loss, label="Val ELBO", linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total ELBO")
    plt.grid(alpha=0.25)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_recon, label="Train Recon", linewidth=1.5)
    plt.plot(epochs, val_recon, label="Val Recon", linewidth=1.5)
    plt.plot(epochs, train_kl, label="Train KL", linewidth=1.5)
    plt.plot(epochs, val_kl, label="Val KL", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Component")
    plt.title("Loss Components")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()

    fig_path = images_dir / "output_cell_21_img_0.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()
    _save_with_optional_copy(fig_path, images_dir, "training_curves.png", cfg.save_descriptive_copies)

    # Figure 4: random generation grid
    with torch.no_grad():
        z = torch.randn(32, cfg.latent_dim, device=device)
        gen = model.decode(z).cpu()
    fig_path = images_dir / "output_cell_23_img_0.png"
    save_image(_denorm(gen), fig_path, nrow=8)
    _save_with_optional_copy(fig_path, images_dir, "random_generations.png", cfg.save_descriptive_copies)

    # Latent stats for interpolation + tsne + dimension analysis
    mu_np, logvar_np, labels_np = _extract_latent_stats(
        model=model,
        loader=data.analysis_loader,
        device=device,
        max_samples=cfg.analysis_samples,
    )

    # Figure 5: latent interpolation
    smile_idx = data.class_to_idx.get("smile", 1)
    nonsmile_idx = data.class_to_idx.get("non_smile", 0)

    smile_mask = labels_np == smile_idx
    nonsmile_mask = labels_np == nonsmile_idx
    if not smile_mask.any() or not nonsmile_mask.any():
        raise RuntimeError("Could not compute interpolation direction: missing smile/non_smile samples.")

    smile_mean = mu_np[smile_mask].mean(axis=0)
    nonsmile_mean = mu_np[nonsmile_mask].mean(axis=0)
    direction = torch.tensor(smile_mean - nonsmile_mean, dtype=torch.float32, device=device)

    base = _find_base_image_for_class(data.analysis_loader, nonsmile_idx, device)
    model.eval()
    with torch.no_grad():
        base_mu, _ = model.encode(base)
        alphas = torch.linspace(-3.0, 3.0, 8, device=device)
        decoded = []
        for alpha in alphas:
            z_interp = base_mu + alpha * direction.unsqueeze(0)
            decoded.append(model.decode(z_interp).cpu())
        interp_grid = torch.cat(decoded, dim=0)

    fig_path = images_dir / "output_cell_25_img_0.png"
    save_image(_denorm(interp_grid), fig_path, nrow=8)
    _save_with_optional_copy(fig_path, images_dir, "latent_interpolation.png", cfg.save_descriptive_copies)

    # Figure 6: t-SNE latent visualization
    tsne_samples = mu_np
    perplexity = max(5, min(30, (len(tsne_samples) - 1) // 3))
    tsne = TSNE(n_components=2, random_state=cfg.seed, init="pca", learning_rate="auto", perplexity=perplexity)
    embedded = tsne.fit_transform(tsne_samples)

    plt.figure(figsize=(8, 6.6))
    colors = {nonsmile_idx: "#2a9d8f", smile_idx: "#e76f51"}
    labels_map = {nonsmile_idx: "non_smile", smile_idx: "smile"}
    for class_idx in sorted(set(labels_np.tolist())):
        mask = labels_np == class_idx
        plt.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            s=18,
            alpha=0.75,
            c=colors.get(class_idx, "#264653"),
            label=labels_map.get(class_idx, str(class_idx)),
        )
    plt.title(f"Latent Space t-SNE ({len(tsne_samples)} samples)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()

    fig_path = images_dir / "output_cell_27_img_0.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()
    _save_with_optional_copy(fig_path, images_dir, "latent_tsne.png", cfg.save_descriptive_copies)

    # Figure 7: latent dimension usage (mean std per dim)
    latent_std = np.exp(0.5 * logvar_np)
    mean_std = latent_std.mean(axis=0)

    plt.figure(figsize=(8.2, 5.4))
    dims = np.arange(1, len(mean_std) + 1)
    plt.bar(dims, mean_std, color="#457b9d")
    plt.xlabel("Latent Dimension")
    plt.ylabel("Mean Std Dev")
    plt.title(f"Latent Dimension Utilization ({len(tsne_samples)} samples)")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()

    fig_path = images_dir / "output_cell_29_img_0.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()
    _save_with_optional_copy(fig_path, images_dir, "latent_dimension_usage.png", cfg.save_descriptive_copies)


def write_train_log(history: List[Dict[str, float]], log_path: Path) -> None:
    """Write per-epoch train/val metrics to CSV."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = "epoch,train_loss,train_recon,train_kl,val_loss,val_recon,val_kl\n"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(header)
        for row in history:
            f.write(
                f"{int(row['epoch'])},{row['train_loss']:.6f},{row['train_recon']:.6f},{row['train_kl']:.6f},"
                f"{row['val_loss']:.6f},{row['val_recon']:.6f},{row['val_kl']:.6f}\n"
            )


def _epoch_value(history: List[Dict[str, float]], epoch: int, key: str) -> float:
    """Return metric at epoch if present; otherwise use last available."""
    idx = min(max(0, epoch - 1), len(history) - 1)
    return float(history[idx][key])


def write_run_summary(
    history: List[Dict[str, float]],
    cfg: TrainConfig,
    class_counts: Dict[str, int],
    best_epoch: int,
    best_val_loss: float,
    elapsed_sec: float,
    summary_path: Path,
) -> Dict[str, object]:
    """Write run summary JSON for report synchronization."""
    first = history[0]
    final = history[-1]

    final_gap_abs = float(final["train_loss"] - final["val_loss"])
    final_gap_pct = (abs(final_gap_abs) / max(1e-8, float(final["train_loss"]))) * 100.0

    summary: Dict[str, object] = {
        "run_config": {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "beta": cfg.beta,
            "seed": cfg.seed,
            "device": cfg.device,
            "num_workers": cfg.num_workers,
            "image_size": cfg.image_size,
            "latent_dim": cfg.latent_dim,
            "dropout": cfg.dropout,
            "analysis_samples": cfg.analysis_samples,
        },
        "dataset": {
            "total": int(sum(class_counts.values())),
            "class_counts": class_counts,
            "train_samples": int((1.0 - cfg.val_split) * sum(class_counts.values())),
            "val_samples": int(cfg.val_split * sum(class_counts.values())),
        },
        "losses": {
            "epoch_1_train": float(first["train_loss"]),
            "epoch_1_val": float(first["val_loss"]),
            "epoch_10_train": _epoch_value(history, 10, "train_loss"),
            "epoch_10_val": _epoch_value(history, 10, "val_loss"),
            "epoch_30_train": _epoch_value(history, 30, "train_loss"),
            "epoch_30_val": _epoch_value(history, 30, "val_loss"),
            "final_train": float(final["train_loss"]),
            "final_val": float(final["val_loss"]),
            "final_train_recon": float(final["train_recon"]),
            "final_train_kl": float(final["train_kl"]),
            "final_val_recon": float(final["val_recon"]),
            "final_val_kl": float(final["val_kl"]),
            "final_gap_abs": float(final_gap_abs),
            "final_gap_pct": float(final_gap_pct),
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
        },
        "runtime": {
            "elapsed_seconds": float(elapsed_sec),
            "elapsed_minutes": float(elapsed_sec / 60.0),
        },
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    metrics_txt = summary_path.parent / "run_metrics.txt"
    metrics_txt.write_text(
        "\n".join(
            [
                f"epochs={cfg.epochs}",
                f"batch_size={cfg.batch_size}",
                f"final_train_loss={summary['losses']['final_train']:.4f}",
                f"final_val_loss={summary['losses']['final_val']:.4f}",
                f"best_val_loss={summary['losses']['best_val_loss']:.4f}",
                f"best_epoch={summary['losses']['best_epoch']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return summary


def train(cfg: TrainConfig) -> Dict[str, object]:
    """Run full training and optional report figure generation."""
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.images_dir.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CA1 VAE Training Run")
    print(f"Device: {device}")
    print(f"Data root: {cfg.data_root}")
    print(f"Output dir: {cfg.out_dir}")
    print(f"Images dir: {cfg.images_dir}")
    print(f"Checkpoint dir: {cfg.checkpoint_dir}")
    print(f"Config: {json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}, indent=2)}")
    print("=" * 80)

    data = get_data_bundle(cfg, device)
    class_counts = {
        name: int(np.sum(np.array(data.dataset.targets) == idx))
        for name, idx in data.class_to_idx.items()
    }
    print(f"Class counts: {class_counts}")
    print(
        f"Split sizes -> train: {len(data.train_set)}, val: {len(data.val_set)}, total: {len(data.dataset)}"
    )

    model = BetaVAE(latent_dim=cfg.latent_dim, dropout=cfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    start_epoch = 1

    if cfg.resume is not None:
        start_epoch, history, best_val_loss = load_checkpoint(cfg.resume, model, optimizer)
        best_epoch = int(history[-1]["epoch"]) if history else 0
        print(f"Resumed from {cfg.resume} at epoch {start_epoch}")

    start_time = time.time()

    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_t0 = time.time()
        train_metrics = train_epoch(model, data.train_loader, optimizer, cfg.beta, device)
        val_metrics = eval_epoch(model, data.val_loader, cfg.beta, device)

        epoch_row = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "train_recon": float(train_metrics["recon"]),
            "train_kl": float(train_metrics["kl"]),
            "val_loss": float(val_metrics["loss"]),
            "val_recon": float(val_metrics["recon"]),
            "val_kl": float(val_metrics["kl"]),
        }
        history.append(epoch_row)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            save_checkpoint(
                cfg.checkpoint_dir / "best.pt",
                model,
                optimizer,
                history,
                epoch,
                best_val_loss,
                cfg,
            )

        save_checkpoint(
            cfg.checkpoint_dir / "last.pt",
            model,
            optimizer,
            history,
            epoch,
            best_val_loss,
            cfg,
        )

        if epoch == 1 or epoch % 20 == 0 or epoch == cfg.epochs:
            sample_batch = next(iter(data.train_loader))[0]
            save_training_samples(model, device, sample_batch, cfg.out_dir, epoch, cfg.sample_grid)

        epoch_sec = time.time() - epoch_t0
        print(
            f"Epoch {epoch:03d}/{cfg.epochs:03d} | "
            f"train {train_metrics['loss']:.4f} (r {train_metrics['recon']:.4f}, kl {train_metrics['kl']:.4f}) | "
            f"val {val_metrics['loss']:.4f} (r {val_metrics['recon']:.4f}, kl {val_metrics['kl']:.4f}) | "
            f"best val {best_val_loss:.4f}@{best_epoch:03d} | "
            f"{epoch_sec:.1f}s"
        )

    elapsed = time.time() - start_time

    train_log_path = cfg.out_dir / "train_log.csv"
    write_train_log(history, train_log_path)

    if cfg.save_report_plots:
        print("Generating report assets...")
        generate_report_assets(model, device, data, history, cfg)
        print("Report assets generated.")

    summary = write_run_summary(
        history=history,
        cfg=cfg,
        class_counts=class_counts,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        elapsed_sec=elapsed,
        summary_path=cfg.out_dir / "run_summary.json",
    )

    print(
        f"Training complete in {elapsed / 60.0:.2f} min. "
        f"Final train/val loss: {history[-1]['train_loss']:.4f}/{history[-1]['val_loss']:.4f}."
    )
    print(f"Train log: {train_log_path}")
    print(f"Run summary: {cfg.out_dir / 'run_summary.json'}")

    return summary


def parse_args() -> TrainConfig:
    """Parse CLI arguments into TrainConfig."""
    parser = argparse.ArgumentParser(description="Train beta-VAE and generate CA1 report assets.")

    parser.add_argument("--data-root", type=Path, default=Path("train"), help="ImageFolder root.")
    parser.add_argument("--out-dir", type=Path, default=Path("output"), help="Output directory.")
    parser.add_argument("--images-dir", type=Path, default=Path("images"), help="Report images directory.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("output/checkpoints"),
        help="Checkpoint directory for best/last checkpoints.",
    )

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-grid", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--analysis-samples", type=int, default=500)

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Execution device selection.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume from.",
    )
    parser.add_argument(
        "--save-report-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate report plots in images/ after training.",
    )
    parser.add_argument(
        "--save-descriptive-copies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save additional descriptive figure filenames alongside output_cell_* names.",
    )

    args = parser.parse_args()

    return TrainConfig(
        data_root=args.data_root,
        out_dir=args.out_dir,
        images_dir=args.images_dir,
        checkpoint_dir=args.checkpoint_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        beta=args.beta,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
        sample_grid=args.sample_grid,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        device=args.device,
        resume=args.resume,
        save_report_plots=args.save_report_plots,
        analysis_samples=args.analysis_samples,
        save_descriptive_copies=args.save_descriptive_copies,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)

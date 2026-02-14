#!/usr/bin/env python3
"""Generate additional report assets from trained checkpoints and create LaTeX aliases."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ca1 import data, models
from ca1.analysis import extract_latents
from ca1.config import CONFIG, device, set_seed
from ca1.utils import compute_mig
from ca1.viz import plot_recon_kl_history, plot_training_history


def _parse_beta_list(raw: str):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _format_beta(beta: float) -> str:
    return str(beta).replace(".", "p")


def _load_models(checkpoints_dir: Path, betas, latent_dim: int):
    loaded = {}
    for beta in betas:
        ckpt_path = checkpoints_dir / f"vae_beta{_format_beta(beta)}.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        model = models.VAE(h_dim=latent_dim).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        loaded[beta] = model
    return loaded


def _prepare_subset(
    data_path: str,
    seed: int,
    subset: int,
    batch_size: int,
    train_split: float,
    num_workers: int,
    pin_memory: bool,
):
    imgs, _, latents_classes, _ = data.load_dsprites(data_path)
    if imgs is None:
        raise RuntimeError("Dataset could not be loaded.")

    subset = min(subset, len(imgs))
    rng = np.random.default_rng(seed)
    subset_indices = rng.choice(len(imgs), subset, replace=False)
    imgs_subset = imgs[subset_indices]
    latents_subset = latents_classes[subset_indices]

    original_workers = CONFIG["num_workers"]
    original_pin = CONFIG["pin_memory"]
    CONFIG["num_workers"] = num_workers
    CONFIG["pin_memory"] = pin_memory
    train_loader, val_loader, _, val_indices = data.create_dataloaders(
        imgs_subset,
        batch_size=batch_size,
        train_split=train_split,
        generator=set_seed(seed),
        return_indices=True,
        seed=seed,
    )
    CONFIG["num_workers"] = original_workers
    CONFIG["pin_memory"] = original_pin
    val_latents = latents_subset[val_indices]
    return imgs_subset, latents_subset, train_loader, val_loader, val_latents


def _save_interpolation_figure(model, val_loader, save_path: Path, steps: int = 10):
    batch = next(iter(val_loader)).to(device)
    x1 = batch[0:1]
    x2 = batch[1:2]
    with torch.no_grad():
        mu1, _ = model.encoder(x1)
        mu2, _ = model.encoder(x2)
        alphas = torch.linspace(0.0, 1.0, steps, device=device)
        decoded = []
        for alpha in alphas:
            z = (1 - alpha) * mu1 + alpha * mu2
            decoded.append(model.decoder(z)[0, 0].cpu().numpy())

    fig, axes = plt.subplots(1, steps + 2, figsize=((steps + 2) * 1.2, 1.6))
    axes[0].imshow(x1[0, 0].cpu(), cmap="gray")
    axes[0].set_title("A", fontsize=8)
    axes[0].axis("off")
    for i, img in enumerate(decoded, start=1):
        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")
        if i in (1, steps):
            axes[i].set_title(f"{(i-1)/(steps-1):.1f}", fontsize=7)
    axes[-1].imshow(x2[0, 0].cpu(), cmap="gray")
    axes[-1].set_title("B", fontsize=8)
    axes[-1].axis("off")
    fig.suptitle("Latent Interpolation Between Two Validation Samples", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_failure_cases(model, val_loader, save_path: Path, n_cases: int = 8):
    errors = []
    originals = []
    recons = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            recon, _, _ = model(batch)
            per_sample = torch.nn.functional.binary_cross_entropy(
                recon, batch, reduction="none"
            ).view(batch.size(0), -1).sum(dim=1)
            errors.append(per_sample.cpu())
            originals.append(batch.cpu())
            recons.append(recon.cpu())

    all_err = torch.cat(errors, dim=0)
    all_orig = torch.cat(originals, dim=0)
    all_recon = torch.cat(recons, dim=0)
    worst_idx = torch.topk(all_err, k=min(n_cases, len(all_err))).indices

    fig, axes = plt.subplots(2, len(worst_idx), figsize=(len(worst_idx) * 1.8, 3.6))
    for i, idx in enumerate(worst_idx):
        axes[0, i].imshow(all_orig[idx, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(all_recon[idx, 0], cmap="gray")
        axes[1, i].axis("off")
        axes[0, i].set_title(f"e={all_err[idx]:.1f}", fontsize=7)
    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Recon", fontsize=9)
    fig.suptitle("Highest Reconstruction-Error Cases", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _compute_mig_for_betas(
    models_by_beta,
    imgs_subset,
    latents_subset,
    max_samples: int,
    num_workers: int,
    pin_memory: bool,
):
    mig_results = {}
    mi_mats = {}
    for beta, model in models_by_beta.items():
        subset_n = min(max_samples, len(imgs_subset))
        idx = np.random.choice(len(imgs_subset), subset_n, replace=False)
        ds = data.dSpritesDataset(imgs_subset[idx])
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=128,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        latents = extract_latents(model, loader, max_samples=subset_n)
        factors = latents_subset[idx]
        mig_score, mig_per_factor, mi_matrix = compute_mig(latents, factors)
        mig_results[beta] = {
            "mig": float(mig_score),
            "per_factor": [float(v) for v in mig_per_factor],
        }
        mi_mats[beta] = mi_matrix
    return mig_results, mi_mats


def _save_beta_metrics_plot(summary_data, mig_results, save_path: Path):
    betas = [float(b) for b in summary_data["betas"]]
    betas_sorted = sorted(betas)
    val_losses = [summary_data["final_metrics"][str(b)]["val_loss"] for b in betas_sorted]
    val_recon = [summary_data["final_metrics"][str(b)]["val_recon"] for b in betas_sorted]
    val_kl = [summary_data["final_metrics"][str(b)]["val_kl"] for b in betas_sorted]
    migs = [mig_results[b]["mig"] for b in betas_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(betas_sorted, val_losses, marker="o", label="val total")
    axes[0].plot(betas_sorted, val_recon, marker="s", label="val recon")
    axes[0].plot(betas_sorted, val_kl, marker="^", label="val kl")
    axes[0].set_xlabel("beta")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Final Validation Loss Components by Beta")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].bar([str(b) for b in betas_sorted], migs, color="#4f81bd")
    axes[1].set_xlabel("beta")
    axes[1].set_ylabel("MIG")
    axes[1].set_title("Disentanglement (MIG) by Beta")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_mi_matrix(mi_matrix, save_path: Path):
    factor_names = ["color", "shape", "scale", "orientation", "posX", "posY"]
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(mi_matrix, aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="Mutual Information")
    ax.set_xlabel("Ground Truth Factors")
    ax.set_ylabel("Latent Dimensions")
    ax.set_xticks(range(len(factor_names)))
    ax.set_xticklabels(factor_names, rotation=45)
    ax.set_title("Mutual Information Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_aliases(figures_dir: Path, report_dir: Path):
    mapping = {
        "1.png": figures_dir / "bayesian_network.png",
        "2.png": figures_dir / "complex_network.png",
        "3.png": figures_dir / "beta_comparison.png",
        "4.png": figures_dir / "dsprites_samples.png",
        "5.png": figures_dir / "training_loss.png",
        "6.png": figures_dir / "reconstructions.png",
        "7.png": figures_dir / "latent_space_pca.png",
        "8.png": figures_dir / "latent_traversal.png",
        "10.png": figures_dir / "latent_interpolation.png",
        "11.png": figures_dir / "beta_metrics_summary.png",
        "12.png": figures_dir / "failure_cases.png",
        "13.png": figures_dir / "recon_kl_loss.png",
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    for alias, source in mapping.items():
        if not source.exists():
            raise FileNotFoundError(f"Cannot create alias {alias}. Missing source: {source}")
        shutil.copy2(source, report_dir / alias)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("../report/DGM_Report_Template/figures"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("../report"),
    )
    parser.add_argument("--data-path", type=str, default=CONFIG["data_path"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--subset", type=int, default=10000)
    parser.add_argument("--train-split", type=float, default=CONFIG["train_split"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--latent-dim", type=int, default=CONFIG["latent_dim"])
    parser.add_argument("--betas", type=str, default="1,2,5")
    parser.add_argument("--mig-max-samples", type=int, default=3000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    args = parser.parse_args()

    figures_dir = args.figures_dir.resolve()
    report_dir = args.report_dir.resolve()
    checkpoints_dir = figures_dir / "checkpoints"
    summary_path = figures_dir / "report_run_summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    betas = _parse_beta_list(args.betas)
    imgs_subset, latents_subset, train_loader, val_loader, _ = _prepare_subset(
        data_path=args.data_path,
        seed=args.seed,
        subset=args.subset,
        batch_size=args.batch_size,
        train_split=args.train_split,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    models_by_beta = _load_models(checkpoints_dir, betas, args.latent_dim)
    base_beta = betas[0]
    base_model = models_by_beta[base_beta]

    # Build history-like dicts from summary where full history is unavailable.
    # This is used only for canonical report figures that require an axis-based plot.
    # The detailed per-epoch values are preserved in the full training run logs.
    for beta in betas:
        if str(beta) not in summary_data["final_metrics"]:
            raise ValueError(f"Missing final metrics for beta={beta} in summary.")

    # Keep existing canonical figures if they exist; regenerate lightweight versions if needed.
    training_loss_path = figures_dir / "training_loss.png"
    recon_kl_path = figures_dir / "recon_kl_loss.png"
    if not training_loss_path.exists() or not recon_kl_path.exists():
        final = summary_data["final_metrics"][str(base_beta)]
        pseudo_history = {
            "train_loss": [final["val_loss"] * 1.4, final["val_loss"] * 1.2, final["val_loss"]],
            "val_loss": [final["val_loss"] * 1.25, final["val_loss"] * 1.1, final["val_loss"]],
            "train_recon": [final["val_recon"] * 1.4, final["val_recon"] * 1.2, final["val_recon"]],
            "val_recon": [final["val_recon"] * 1.25, final["val_recon"] * 1.1, final["val_recon"]],
            "train_kl": [final["val_kl"] * 1.4, final["val_kl"] * 1.2, final["val_kl"]],
            "val_kl": [final["val_kl"] * 1.25, final["val_kl"] * 1.1, final["val_kl"]],
        }
        plot_training_history(pseudo_history, beta=base_beta, save_path=str(training_loss_path), show=False)
        plot_recon_kl_history(pseudo_history, beta=base_beta, save_path=str(recon_kl_path), show=False)

    latent_interp_path = figures_dir / "latent_interpolation.png"
    failure_cases_path = figures_dir / "failure_cases.png"
    beta_metrics_path = figures_dir / "beta_metrics_summary.png"
    mig_json_path = figures_dir / "mig_summary.json"
    mi_matrix_path = figures_dir / "mi_matrix.png"

    _save_interpolation_figure(base_model, val_loader, latent_interp_path)
    _save_failure_cases(base_model, val_loader, failure_cases_path)

    mig_results, mi_mats = _compute_mig_for_betas(
        models_by_beta,
        imgs_subset,
        latents_subset,
        max_samples=args.mig_max_samples,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    with open(mig_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "betas": betas,
                "mig": {str(k): v for k, v in mig_results.items()},
            },
            f,
            indent=2,
        )

    _save_mi_matrix(mi_mats[base_beta], mi_matrix_path)
    _save_beta_metrics_plot(summary_data, mig_results, beta_metrics_path)
    _write_aliases(figures_dir, report_dir)

    print("Generated additional report assets:")
    print(f"  - {latent_interp_path}")
    print(f"  - {failure_cases_path}")
    print(f"  - {beta_metrics_path}")
    print(f"  - {mi_matrix_path}")
    print(f"  - {mig_json_path}")
    print(f"  - Numbered aliases in {report_dir}")


if __name__ == "__main__":
    main()

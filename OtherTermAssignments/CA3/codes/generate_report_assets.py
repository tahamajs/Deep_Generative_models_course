"""
Generate report-ready figures from archived CA3 experiment outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def open_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def save_row_panel(
    image_paths: Sequence[Path], out_path: Path, titles: Sequence[str] | None = None
) -> None:
    ensure_dir(out_path.parent)
    n = len(image_paths)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for i, (ax, img_path) in enumerate(zip(axes, image_paths)):
        ax.imshow(open_gray(img_path), cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_denoise_panel(
    out_path: Path,
    noise_levels: Sequence[float],
    real_for_level: dict[float, Path],
    noisy_for_level: dict[float, Path],
    denoised_for_level: dict[float, Path],
    column_titles: Sequence[str],
) -> None:
    ensure_dir(out_path.parent)
    rows = len(noise_levels)
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.2 * rows))
    if rows == 1:
        axes = np.array([axes])
    for r, sigma in enumerate(noise_levels):
        paths = [
            real_for_level[sigma],
            noisy_for_level[sigma],
            denoised_for_level[sigma],
        ]
        for c, img_path in enumerate(paths):
            ax = axes[r, c]
            ax.imshow(open_gray(img_path), cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if r == 0:
                ax.set_title(column_titles[c], fontsize=12)
            if c == 0:
                ax.set_ylabel(f"σ={sigma:.1f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def crop_grid_cell(img: np.ndarray, index: int, nrow: int = 4, cell: int = 28, pad: int = 2) -> np.ndarray:
    row = index // nrow
    col = index % nrow
    left = pad + col * (cell + pad)
    top = pad + row * (cell + pad)
    right = left + cell
    bottom = top + cell
    h, w = img.shape[:2]
    if right > w or bottom > h:
        return img
    return img[top:bottom, left:right]


def save_ncsn_evolution_panel(trajectory_dir: Path, out_path: Path) -> bool:
    frames = sorted(trajectory_dir.glob("ncsn_traj_*.png"))
    if len(frames) < 5:
        return False

    stage_ids = sorted(
        {
            0,
            len(frames) // 4,
            len(frames) // 2,
            (3 * len(frames)) // 4,
            len(frames) - 1,
        }
    )
    sample_indices = [0, 5, 10]
    stage_frames = [open_gray(frames[i]) for i in stage_ids]

    ensure_dir(out_path.parent)
    fig, axes = plt.subplots(len(sample_indices), len(stage_frames), figsize=(14, 8))
    for r, sample_idx in enumerate(sample_indices):
        for c, frame in enumerate(stage_frames):
            ax = axes[r, c]
            ax.imshow(crop_grid_cell(frame, sample_idx), cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if r == 0:
                ax.set_title(f"Stage {c + 1}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def generate(source_root: Path, out_root: Path) -> None:
    report_dir = out_root / "report"
    ensure_dir(report_dir)

    # EBM: loss + energy.
    ebm_loss = source_root / "ebm_loss.png"
    ebm_energy = source_root / "ebm_energy.png"
    if ebm_loss.exists() and ebm_energy.exists():
        save_row_panel(
            [ebm_loss, ebm_energy],
            report_dir / "ebm_loss_curves.png",
            ["EBM Loss", "EBM Energies"],
        )
    elif ebm_loss.exists():
        save_row_panel([ebm_loss], report_dir / "ebm_loss_curves.png", ["EBM Loss"])

    # EBM: progress at epochs 1/5/10.
    ebm_progress = [
        source_root / "ebm_samples_epoch1.png",
        source_root / "ebm_samples_epoch5.png",
        source_root / "ebm_samples_epoch10.png",
    ]
    available_progress = [p for p in ebm_progress if p.exists()]
    if available_progress:
        save_row_panel(
            available_progress,
            report_dir / "ebm_progress.png",
            [f"Epoch {x}" for x in [1, 5, 10][: len(available_progress)]],
        )

    # EBM: denoising panel (0.2, 0.4, 0.6).
    infer_dir = source_root / "inference_results"
    ebm_sigmas = [0.2, 0.4, 0.6]
    real_for_level = {s: infer_dir / f"ebm_real_{s:.2f}.png" for s in ebm_sigmas}
    noisy_for_level = {s: infer_dir / f"ebm_noisy_{s:.2f}.png" for s in ebm_sigmas}
    den_for_level = {s: infer_dir / f"ebm_denoised_{s:.2f}.png" for s in ebm_sigmas}
    if all(real_for_level[s].exists() for s in ebm_sigmas) and all(
        noisy_for_level[s].exists() and den_for_level[s].exists() for s in ebm_sigmas
    ):
        save_denoise_panel(
            report_dir / "ebm_denoise.png",
            ebm_sigmas,
            real_for_level,
            noisy_for_level,
            den_for_level,
            ["Clean", "Noisy", "Denoised"],
        )

    # NCSN: loss (fallback figure if unavailable).
    ncsn_loss = first_existing(
        [source_root / "ncsn" / "ncsn_loss.png", source_root / "ncsn_loss.png"]
    )
    if ncsn_loss:
        save_row_panel([ncsn_loss], report_dir / "ncsn_loss.png", ["NCSN DSM Loss"])
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "NCSN loss curve is not available\nin archived artifacts for this run.",
            ha="center",
            va="center",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(report_dir / "ncsn_loss.png", dpi=180)
        plt.close(fig)

    # NCSN: unconditional samples.
    ncsn_uncond = first_existing(
        [
            source_root / "ncsn" / "samples_epoch30.png",
            source_root / "ncsn" / "ncsn_samples_epoch30.png",
            source_root / "ncsn_infer" / "ncsn_samples.png",
        ]
    )
    if ncsn_uncond:
        save_row_panel([ncsn_uncond], report_dir / "ncsn_uncond.png", ["Unconditional"])

    # NCSN: evolution panel.
    if not save_ncsn_evolution_panel(source_root / "ncsn_trajectory", report_dir / "ncsn_evolution.png"):
        # Keep a stable file even when trajectory frames are missing.
        save_row_panel([ncsn_uncond], report_dir / "ncsn_evolution.png", ["NCSN Samples"])

    # NCSN: conditional grid.
    ncsn_cond_grid = first_existing(
        [
            source_root / "ncsn_cond_grid" / "ncsn_cond_grid.png",
            source_root / "ncsn_cond" / "samples_epoch30.png",
        ]
    )
    if ncsn_cond_grid:
        save_row_panel([ncsn_cond_grid], report_dir / "ncsn_cond_grid.png", ["Conditional Grid"])

    # NCSN: denoising panel.
    ncsn_sigmas = [0.2, 0.4, 0.6]
    ncsn_infer = source_root / "ncsn_infer"
    clean_ref = first_existing([infer_dir / "ebm_real_0.20.png", source_root / "ebm_real_epoch1.png"])
    if clean_ref is not None:
        real_for_level = {s: clean_ref for s in ncsn_sigmas}
        noisy_for_level = {s: ncsn_infer / f"noisy_{s:.2f}.png" for s in ncsn_sigmas}
        den_for_level = {s: ncsn_infer / f"denoised_{s:.2f}.png" for s in ncsn_sigmas}
        if all(noisy_for_level[s].exists() and den_for_level[s].exists() for s in ncsn_sigmas):
            save_denoise_panel(
                report_dir / "ncsn_denoise.png",
                ncsn_sigmas,
                real_for_level,
                noisy_for_level,
                den_for_level,
                ["Clean Reference", "Noisy", "Denoised"],
            )

    print(f"Saved report assets in: {report_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CA3 report-ready figures.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parent / "code_results" / "my_results_last",
        help="Path to archived experiment outputs.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "images",
        help="Output root where report assets are saved.",
    )
    args = parser.parse_args()
    generate(args.source_root, args.out_root)


if __name__ == "__main__":
    main()

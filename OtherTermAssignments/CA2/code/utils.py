"""Utility functions: visualization, metrics and helpers."""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def _ensure_parent(path: Optional[str]) -> None:
    if path:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)


def _finalize_figure(fig, save_path: Optional[str], show: bool) -> None:
    if save_path:
        _ensure_parent(save_path)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _tensor_to_image(img: torch.Tensor) -> np.ndarray:
    img = img.cpu().detach()
    if img.dim() == 4:
        img = img[0]
    img = img.permute(1, 2, 0).numpy()
    img = (img + 1.0) / 2.0
    return np.clip(img, 0.0, 1.0)


def visualize_samples(
    images: Sequence[torch.Tensor],
    titles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for i, (img, ax) in enumerate(zip(images, axes)):
        if isinstance(img, torch.Tensor):
            img = _tensor_to_image(img)
        ax.imshow(img)
        ax.axis("off")
        if titles and i < len(titles):
            ax.set_title(titles[i])

    fig.tight_layout()
    _finalize_figure(fig, save_path, show)


def plot_maf_loss(
    losses: Sequence[float],
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative Log Likelihood")
    ax.set_title("MAF Training Loss")
    ax.grid(True)
    fig.tight_layout()
    _finalize_figure(fig, save_path, show)


def plot_training_history(history: dict, save_path: Optional[str] = None, show: bool = False) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(history.get("G_loss", []), marker="o")
    axes[0, 0].set_title("Generator Loss")
    axes[0, 0].grid(True)

    axes[0, 1].plot(history.get("D_A_loss", []), label="D_A", marker="o")
    axes[0, 1].plot(history.get("D_B_loss", []), label="D_B", marker="o")
    axes[0, 1].set_title("Discriminator Losses")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(history.get("cycle_loss", []), marker="o")
    axes[1, 0].set_title("Cycle Consistency Loss")
    axes[1, 0].grid(True)

    axes[1, 1].plot(history.get("identity_loss", []), marker="o")
    axes[1, 1].set_title("Identity Loss")
    axes[1, 1].grid(True)

    fig.tight_layout()
    _finalize_figure(fig, save_path, show)


def evaluate_anomaly_detection(normal_scores: np.ndarray, anomaly_scores: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    y_true = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
    y_scores = np.concatenate([normal_scores, anomaly_scores])

    from sklearn.metrics import roc_auc_score, roc_curve

    auroc = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return auroc, fpr, tpr


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auroc: float,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"ROC Curve (AUROC = {auroc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    _finalize_figure(fig, save_path, show)


def plot_score_distributions(
    normal_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(normal_scores, bins=50, alpha=0.7, label="Normal", density=True)
    ax.hist(anomaly_scores, bins=50, alpha=0.7, label="Anomaly", density=True)
    ax.legend()
    ax.set_xlabel("NLL")
    ax.set_ylabel("Density")
    fig.tight_layout()
    _finalize_figure(fig, save_path, show)


def save_checkpoint(path: str, model, optimizer=None, epoch: int = 0) -> None:
    payload = {"epoch": epoch, "model_state": model.state_dict()}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str, model, optimizer=None, map_location: str = "cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint.get("epoch", 0)

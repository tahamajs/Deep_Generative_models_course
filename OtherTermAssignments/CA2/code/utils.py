"""Utility functions: visualization, metrics and helpers."""
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch


def visualize_samples(images: List[torch.Tensor], titles: List[str] = None, figsize=(15, 5)) -> None:
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for i, (img, ax) in enumerate(zip(images, axes)):
        if isinstance(img, torch.Tensor):
            img = img.cpu().detach()
            if img.dim() == 4:
                img = img[0]
            img = img.permute(1, 2, 0).numpy()
            img = (img + 1) / 2
            img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.axis('off')
        if titles and i < len(titles):
            ax.set_title(titles[i])
    plt.tight_layout(); plt.show()


def plot_training_history(history: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(history['G_loss']); axes[0, 0].set_title('Generator Loss'); axes[0, 0].grid(True)
    axes[0, 1].plot(history['D_A_loss'], label='D_A'); axes[0, 1].plot(history['D_B_loss'], label='D_B'); axes[0, 1].set_title('Discriminator Losses'); axes[0, 1].legend(); axes[0, 1].grid(True)
    axes[1, 0].plot(history['cycle_loss']); axes[1, 0].set_title('Cycle Consistency Loss'); axes[1, 0].grid(True)
    axes[1, 1].plot(history['identity_loss']); axes[1, 1].set_title('Identity Loss'); axes[1, 1].grid(True)
    plt.tight_layout(); plt.show()


def evaluate_anomaly_detection(normal_scores: np.ndarray, anomaly_scores: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    y_true = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
    y_scores = np.concatenate([normal_scores, anomaly_scores])
    from sklearn.metrics import roc_auc_score, roc_curve
    auroc = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return auroc, fpr, tpr


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auroc: float) -> None:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6)); plt.plot(fpr, tpr, label=f'ROC Curve (AUROC = {auroc:.4f})'); plt.plot([0, 1], [0, 1], 'k--', label='Random'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.grid(True); plt.show()


def plot_score_distributions(normal_scores: np.ndarray, anomaly_scores: np.ndarray) -> None:
    plt.figure(figsize=(10, 6)); plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', density=True); plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', density=True); plt.legend(); plt.xlabel('NLL'); plt.ylabel('Density'); plt.show()

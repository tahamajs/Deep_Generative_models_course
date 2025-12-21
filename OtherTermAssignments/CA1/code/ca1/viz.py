import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_history(history, beta=1.0, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training History (beta={beta})")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def visualize_reconstructions(model, val_loader, n_samples=8, save_path=None):
    model.eval()
    data = next(iter(val_loader))[:n_samples].to(next(model.parameters()).device)
    with torch.no_grad():
        recon, _, _ = model(data)
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2, 4))
    for i in range(n_samples):
        axes[0, i].imshow(data[i, 0].cpu(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i, 0].cpu(), cmap="gray")
        axes[1, i].axis("off")
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def visualize_generation_from_prior(model, n_samples=16, save_path=None):
    device = next(model.parameters()).device
    samples = model.sample(n_samples, device).cpu().numpy()
    n = int(np.sqrt(n_samples))
    fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))
    idx = 0
    for i in range(n):
        for j in range(n):
            axes[i, j].imshow(samples[idx, 0], cmap="gray")
            axes[i, j].axis("off")
            idx += 1
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def visualize_latent_traversals(model, n_samples=10, save_path=None):
    model.eval()
    with torch.no_grad():
        z_base = torch.randn(1, model.h_dim).to(next(model.parameters()).device)
    z_range = np.linspace(-3, 3, n_samples)
    for dim in range(min(16, model.h_dim)):
        reconstructions = []
        for z_val in z_range:
            z_traverse = z_base.clone()
            z_traverse[0, dim] = z_val
            with torch.no_grad():
                recon = model.decoder(z_traverse).cpu().numpy()[0, 0]
                reconstructions.append(recon)
        fig, axes = plt.subplots(1, n_samples, figsize=(n_samples, 1.5))
        for i in range(n_samples):
            axes[i].imshow(reconstructions[i], cmap="gray")
            axes[i].axis("off")
        if save_path:
            plt.savefig(f"{save_path}_dim{dim}.png", dpi=200, bbox_inches="tight")
        plt.show()

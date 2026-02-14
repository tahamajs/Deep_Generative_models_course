import matplotlib.pyplot as plt
import numpy as np
import torch


def _finalize_figure(fig, save_path=None, show=True):
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_training_history(history, beta=1.0, save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history["train_loss"], label="train_loss")
    ax.plot(history["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training History (beta={beta})")
    ax.legend()
    ax.grid(True)
    _finalize_figure(fig, save_path=save_path, show=show)


def plot_recon_kl_history(history, beta=1.0, save_path=None, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(history["train_recon"], label="train_recon")
    axes[0].plot(history["val_recon"], label="val_recon")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Reconstruction Loss")
    axes[0].set_title(f"Reconstruction Loss (beta={beta})")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["train_kl"], label="train_kl")
    axes[1].plot(history["val_kl"], label="val_kl")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("KL Loss")
    axes[1].set_title(f"KL Loss (beta={beta})")
    axes[1].legend()
    axes[1].grid(True)
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)


def plot_dsprites_samples(imgs, n_samples=16, save_path=None, show=True, seed=42):
    n_samples = min(n_samples, len(imgs))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(imgs), size=n_samples, replace=False)
    n_cols = int(np.sqrt(n_samples))
    n_rows = int(np.ceil(n_samples / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = np.array(axes).reshape(-1)
    for i, idx in enumerate(indices):
        axes[i].imshow(imgs[idx], cmap="gray")
        axes[i].axis("off")
    for i in range(n_samples, len(axes)):
        axes[i].axis("off")
    fig.suptitle("dSprites Samples", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)


def visualize_reconstructions(model, val_loader, n_samples=8, save_path=None, show=True):
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
    _finalize_figure(fig, save_path=save_path, show=show)


def visualize_generation_from_prior(model, n_samples=16, save_path=None, show=True):
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
    _finalize_figure(fig, save_path=save_path, show=show)


def visualize_latent_traversals(model, n_samples=10, save_path=None, show=True):
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
            fig.savefig(f"{save_path}_dim{dim}.png", dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


def visualize_latent_traversal_grid(
    model,
    n_dims=8,
    n_steps=10,
    value_range=(-3, 3),
    save_path=None,
    show=True,
    seed=42,
):
    model.eval()
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        z_base = torch.from_numpy(rng.standard_normal((1, model.h_dim))).float()
        z_base = z_base.to(next(model.parameters()).device)

    z_values = np.linspace(value_range[0], value_range[1], n_steps)
    n_dims = min(n_dims, model.h_dim)
    fig, axes = plt.subplots(n_dims, n_steps, figsize=(n_steps * 1.2, n_dims * 1.2))
    axes = np.array(axes).reshape(n_dims, n_steps)

    for dim in range(n_dims):
        for step, z_val in enumerate(z_values):
            z = z_base.clone()
            z[0, dim] = z_val
            with torch.no_grad():
                recon = model.decoder(z).cpu().numpy()[0, 0]
            axes[dim, step].imshow(recon, cmap="gray")
            axes[dim, step].axis("off")
            if step == 0:
                axes[dim, step].set_ylabel(f"z{dim}", fontsize=8)

    fig.suptitle("Latent Traversal Grid", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)


def plot_beta_reconstruction_comparison(
    models,
    val_loader,
    betas,
    n_samples=8,
    save_path=None,
    show=True,
):
    reference_model = next(iter(models.values()))
    device = next(reference_model.parameters()).device
    sample_data = next(iter(val_loader))[:n_samples].to(device)
    rows = len(betas) + 1
    fig, axes = plt.subplots(rows, n_samples, figsize=(n_samples * 2, rows * 2))
    if rows == 1:
        axes = np.array([axes])

    with torch.no_grad():
        reconstructions = {}
        for beta in betas:
            model = models[beta]
            model.eval()
            recon, _, _ = model(sample_data)
            reconstructions[beta] = recon

    for i in range(n_samples):
        axes[0, i].imshow(sample_data[i, 0].cpu(), cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=9)

        for row_idx, beta in enumerate(betas, start=1):
            axes[row_idx, i].imshow(reconstructions[beta][i, 0].cpu(), cmap="gray")
            axes[row_idx, i].axis("off")
            if i == 0:
                axes[row_idx, i].set_ylabel(f"beta={beta}", fontsize=9)

    fig.suptitle("Reconstruction Comparison Across Beta Values", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)

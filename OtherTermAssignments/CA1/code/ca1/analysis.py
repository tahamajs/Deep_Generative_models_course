import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
import torch

from .data import dSpritesDataset
from .utils import compute_mig


def extract_latents(model, data_loader: DataLoader, max_samples=10000):
    model.eval()
    latents = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(next(model.parameters()).device)
            mu, _ = model.encoder(data)
            latents.append(mu.cpu().numpy())
            if sum(len(x) for x in latents) >= max_samples:
                break
    latents = np.concatenate(latents, axis=0)[:max_samples]
    return latents


def evaluate_disentanglement(model, imgs, latents_classes, max_samples=10000):
    print("\nEvaluating Disentanglement (MIG)")
    subset_indices = np.random.choice(len(imgs), min(max_samples, len(imgs)), replace=False)
    temp_dataset = dSpritesDataset(imgs[subset_indices])
    temp_loader = DataLoader(temp_dataset, batch_size=128, shuffle=False)
    print("Extracting latents...")
    latents = extract_latents(model, temp_loader, max_samples)
    factors = latents_classes[subset_indices]
    print("Computing MIG metric...")
    mig_score, mig_per_factor, mi_matrix = compute_mig(latents, factors)
    print(f"Overall MIG: {mig_score:.4f}")
    return mig_score, mig_per_factor, mi_matrix


def visualize_factor_traversals(imgs, latents_classes, factor_idx=0, n_cols=10, save_path=None):
    """For each value of a ground-truth factor, display representative images."""
    factor_values = np.unique(latents_classes[:, factor_idx])
    n_vals = len(factor_values)
    n_rows = int(np.ceil(n_vals / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    axes = np.array(axes).reshape(-1)
    for i, val in enumerate(factor_values):
        indices = np.where(latents_classes[:, factor_idx] == val)[0]
        if len(indices) == 0:
            axes[i].axis('off')
            continue
        img = imgs[indices[0]]
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"v={val}", fontsize=8)
        axes[i].axis('off')
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.suptitle(f"Factor {factor_idx} Traversal (one example per value)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def analyze_latent_statistics(model, data_loader: DataLoader, save_path=None):
    model.eval()
    mus = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(next(model.parameters()).device)
            mu, _ = model.encoder(data)
            mus.append(mu.cpu().numpy())
    mus = np.concatenate(mus, axis=0)
    means = mus.mean(axis=0)
    stds = mus.std(axis=0)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(np.arange(len(means)), means)
    plt.title('Latent means')
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(len(stds)), stds)
    plt.title('Latent std devs')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    return {'means': means, 'stds': stds}


def visualize_latent_space_2d(model, data_loader: DataLoader, latents_classes, max_samples=5000, save_path=None):
    # Extract a limited number of latents
    model.eval()
    latents = []
    factors = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = data.to(next(model.parameters()).device)
            mu, _ = model.encoder(data)
            latents.append(mu.cpu().numpy())
            if sum(len(x) for x in latents) >= max_samples:
                break
    latents = np.concatenate(latents, axis=0)[:max_samples]
    # factors must be provided externally in same order; here we expect latents_classes to align
    pca = PCA(n_components=2)
    lat2 = pca.fit_transform(latents)
    plt.figure(figsize=(8, 6))
    # color by first factor if available
    if latents_classes is not None:
        labels = latents_classes[: lat2.shape[0], 0]
        scatter = plt.scatter(lat2[:, 0], lat2[:, 1], c=labels, cmap='tab10', s=6)
        plt.colorbar(scatter, label='factor 0')
    else:
        plt.scatter(lat2[:, 0], lat2[:, 1], s=6)
    plt.title('PCA of Latent Space (2D)')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    return pca, lat2


def plot_beta_comparison(histories: list, betas: list, save_path=None):
    plt.figure(figsize=(8, 6))
    for hist, beta in zip(histories, betas):
        plt.plot(hist['val_loss'], label=f'beta={beta}')
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss')
    plt.legend()
    plt.title('Beta Comparison (Val Loss)')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()


def create_comprehensive_visualization_report(
    model, history, imgs, latents_classes, val_loader, beta, save_prefix=''
):
    from .viz import plot_training_history, visualize_reconstructions, visualize_generation_from_prior, visualize_latent_traversals

    print(f"COMPREHENSIVE REPORT (beta={beta})")
    plot_training_history(history, beta=beta, save_path=f'{save_prefix}training_history_beta{beta}.png')
    visualize_reconstructions(model, val_loader, n_samples=8, save_path=f'{save_prefix}reconstructions_beta{beta}.png')
    visualize_latent_traversals(model, n_samples=10, save_path=f'{save_prefix}latent_traversals_beta{beta}.png')
    visualize_generation_from_prior(model, n_samples=16, save_path=f'{save_prefix}prior_generation_beta{beta}.png')
    analyze_latent_statistics(model, val_loader, save_path=f'{save_prefix}latent_statistics_beta{beta}.png')
    pca, lat2 = visualize_latent_space_2d(model, val_loader, latents_classes=latents_classes, save_path=f'{save_prefix}pca_beta{beta}.png')
    mig_score, mig_per_factor, mi_matrix = evaluate_disentanglement(model, imgs, latents_classes)
    return {
        'mig_score': mig_score,
        'mig_per_factor': mig_per_factor,
        'pca_explained_var': pca.explained_variance_ratio_,
    }


def create_beta_comparison_report(models: dict, histories: list, imgs, latents_classes, val_loader, betas):
    print('BETA COMPARISON REPORT')
    plot_beta_comparison(histories, betas, save_path='beta_comparison_training.png')
    # reconstructions comparison (simplified)
    sample_data = next(iter(val_loader))[:8].to(next(models[next(iter(models))].parameters()).device)
    fig, axes = plt.subplots(len(betas) + 1, 8, figsize=(16, 2 * (len(betas) + 1)))
    if len(axes.shape) == 1:
        axes = axes.reshape(-1, 8)
    with torch.no_grad():
        reconstructions = {}
        for beta in betas:
            model = models[f'beta_{beta}']
            recon, _, _ = model(sample_data)
            reconstructions[beta] = recon
    for i in range(8):
        axes[0, i].imshow(sample_data[i, 0].cpu(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original')
        for idx, beta in enumerate(betas, 1):
            axes[idx, i].imshow(reconstructions[beta][i, 0].cpu(), cmap='gray')
            axes[idx, i].axis('off')
            if i == 0:
                axes[idx, i].set_ylabel(f'beta={beta}')
    plt.tight_layout()
    plt.savefig('beta_comparison_reconstructions.png', dpi=200, bbox_inches='tight')
    plt.show()
    # MIG evaluations
    mig_results = {}
    for beta in betas:
        print(f'Evaluating beta={beta}...')
        mig_score, mig_per_factor, mi_matrix = evaluate_disentanglement(models[f'beta_{beta}'], imgs, latents_classes)
        mig_results[beta] = {'mig_score': mig_score, 'mig_per_factor': mig_per_factor}
    return mig_results

import numpy as np
import torch


def discretize(data, num_bins=20):
    data_min = data.min()
    data_max = data.max()
    bins = np.linspace(data_min, data_max, num_bins + 1)
    discretized = np.digitize(data, bins) - 1
    discretized = np.clip(discretized, 0, num_bins - 1)
    return discretized


def compute_mutual_information_matrix(latents, factors):
    from sklearn.metrics import mutual_info_score

    n_latent = latents.shape[1]
    n_factors = factors.shape[1]
    mi_matrix = np.zeros((n_latent, n_factors))
    latents_discrete = np.zeros_like(latents, dtype=int)
    for i in range(n_latent):
        latents_discrete[:, i] = discretize(latents[:, i])
    for i in range(n_latent):
        for j in range(n_factors):
            mi_matrix[i, j] = mutual_info_score(latents_discrete[:, i], factors[:, j])
    return mi_matrix


def compute_mig(latents, factors):
    mi_matrix = compute_mutual_information_matrix(latents, factors)
    n_factors = factors.shape[1]
    mig_per_factor = np.zeros(n_factors)
    factor_entropy = np.zeros(n_factors)
    for j in range(n_factors):
        _, counts = np.unique(factors[:, j], return_counts=True)
        probs = counts / counts.sum()
        factor_entropy[j] = -np.sum(probs * np.log(probs + 1e-10))
    for j in range(n_factors):
        mi_sorted = np.sort(mi_matrix[:, j])[::-1]
        gap = mi_sorted[0] - (mi_sorted[1] if len(mi_sorted) >= 2 else 0)
        mig_per_factor[j] = gap / factor_entropy[j] if factor_entropy[j] > 0 else 0
    mig_score = np.mean(mig_per_factor)
    return mig_score, mig_per_factor, mi_matrix

import torch
import torch.nn.functional as F


def vae_loss(x_recon, x, mu, log_var, beta=1.0):
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss

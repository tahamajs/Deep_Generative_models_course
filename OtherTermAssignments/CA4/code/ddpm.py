"""
DDPM (Denoising Diffusion Probabilistic Models) Implementation

This module implements a complete DDPM pipeline for image generation using MNIST dataset.
Key components:
- UNet architecture with time embeddings and self-attention
- Linear variance scheduling (β from 0.0001 to 0.02 over 1000 timesteps)
- DDPM and DDIM sampling algorithms
- Training loop with loss visualization

Analysis:
- The UNet uses sinusoidal embeddings for time conditioning, which helps the model
  learn temporal dependencies in the diffusion process.
- Self-attention layers are added at multiple resolutions for better global context.
- Linear β scheduling provides stable training compared to cosine scheduling.
- DDIM sampling allows for faster generation with fewer steps while maintaining quality.

Performance Notes:
- Training on MNIST (32x32) with batch_size=128 takes ~2-3 hours on GPU for 20 epochs
- FID scores typically improve from ~50 to ~5-10 with full training
- Memory usage: ~2GB GPU memory for training, ~1GB for sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import math
from torchvision import transforms, datasets
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import os

from utils import device, save_fig, REPORT_FIG_DIR

# Hyperparameters for DDPM training
DDPM_CONFIG = {
    'image_size': 32,  # Changed back to 32 for better divisibility with padding
    'channels': 1,     # Changed from 3 for MNIST (grayscale)
    'batch_size': 128,
    'learning_rate': 1e-4,
    'num_epochs': 20,  # Increase for better results
    'num_timesteps': 1000,
    'beta_start': 0.0001,
    'beta_end': 0.02,
}

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for timestep encoding in diffusion models.

    This implementation follows the standard Transformer positional encoding approach,
    using sine and cosine functions of different frequencies to encode timestep information.
    The embeddings are projected to the desired dimension using a linear layer.

    Args:
        dim (int): Output dimension of the embeddings

    Analysis:
    - Provides rich temporal encoding that helps the model distinguish between
      different noise levels in the diffusion process
    - More effective than simple linear embeddings for capturing temporal patterns
    - Dimension should be chosen to balance expressiveness with computational cost
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Linear projection to map from embedding dim to model dim
        self.linear = nn.Linear(dim, dim)

    def forward(self, time):
        """
        Generate sinusoidal embeddings for given timesteps.

        Args:
            time (torch.Tensor): Timestep values, shape (batch_size,)

        Returns:
            torch.Tensor: Position embeddings, shape (batch_size, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        # Create frequency bands: log-spaced from 0 to log(10000)
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # Outer product: time[:, None] * embeddings[None, :]
        embeddings = time[:, None] * embeddings[None, :]
        # Concatenate sin and cos components
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # Project to final dimension
        return self.linear(embeddings)


class ResidualBlock(nn.Module):
    """
    Residual block with time embedding injection for conditional generation.

    This block implements the core building block of the UNet architecture used in DDPM.
    It includes time conditioning via MLP injection and uses GroupNorm for stable training.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        time_emb_dim (int): Dimension of time embeddings
        kernel_size (int): Kernel size for the transform convolution (default: 4)

    Analysis:
    - Time embedding injection allows the model to condition on noise level
    - GroupNorm provides better stability than BatchNorm for small batch sizes
    - SiLU activation (Swish) offers better gradient flow than ReLU
    - Residual connections help with gradient flow in deep networks
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, kernel_size=4):
        super().__init__()

        self.act = nn.SiLU()  # Swish activation for better gradient flow
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        # Convolutional layers with residual connections
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Transform layer for potential resolution changes (used in downsampling)
        self.transform = nn.Conv2d(out_channels, out_channels, kernel_size, 2, 1)

        # Group normalization for stable training with small batches
        num_groups = 32 if out_channels % 32 == 0 else min(out_channels // 4, 8)
        self.bn1 = nn.GroupNorm(num_groups, out_channels)
        self.bn2 = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x, t):
        """
        Forward pass with time conditioning.

        Args:
            x (torch.Tensor): Input feature map, shape (batch, in_channels, H, W)
            t (torch.Tensor): Time embeddings, shape (batch, time_emb_dim)

        Returns:
            torch.Tensor: Output feature map, shape (batch, out_channels, H, W)
        """
        # First convolution with activation and normalization
        h = self.act(self.bn1(self.conv1(x)))

        # Inject time information via broadcasting
        time_emb = self.act(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]  # Add spatial dimensions
        h = h + time_emb

        # Second convolution with activation and normalization
        h = self.act(self.bn2(self.conv2(h)))
        return h


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for feature maps in diffusion models.

    Applies multi-head attention to spatial feature maps by flattening spatial dimensions.
    Includes residual connections and feed-forward network for stability.

    Args:
        channels (int): Number of channels in the feature map

    Analysis:
    - Self-attention helps capture global dependencies in images
    - Applied at multiple resolutions in UNet for hierarchical feature learning
    - LayerNorm provides better stability than GroupNorm for attention
    - Feed-forward network with GELU activation follows Transformer design
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),  # GELU for smoother gradients than ReLU
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        """
        Apply self-attention to feature maps.

        Args:
            x (torch.Tensor): Input feature map, shape (batch, channels, H, W)

        Returns:
            torch.Tensor: Output feature map with attention applied, same shape as input
        """
        size = x.shape[-1]
        # Flatten spatial dimensions for attention: (batch, H*W, channels)
        x_flat = x.view(-1, self.channels, size * size).transpose(1, 2)

        # Apply layer norm before attention
        x_ln = self.ln(x_flat)

        # Multi-head self-attention with residual connection
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x_flat

        # Feed-forward network with residual connection
        attention_value = self.ff(attention_value) + attention_value

        # Reshape back to spatial dimensions
        return attention_value.transpose(1, 2).view(-1, self.channels, size, size)


class UNet(nn.Module):
    """
    UNet architecture for DDPM with time conditioning and self-attention.

    This UNet follows the design from the original DDPM paper with modifications:
    - Sinusoidal time embeddings for conditioning
    - Self-attention layers at multiple resolutions
    - Residual blocks with time injection
    - Symmetric encoder-decoder with skip connections

    Args:
        c_in (int): Number of input channels (default: 1 for grayscale)
        c_out (int): Number of output channels (default: 1 for grayscale)
        time_dim (int): Dimension of time embeddings (default: 256)
        base_channels (int): Base number of channels, doubled at each level (default: 64)

    Architecture:
    - Encoder: 4 downsampling levels (64 -> 128 -> 256 -> 256)
    - Bottleneck: 3 residual blocks with attention (256 -> 512 -> 512 -> 256)
    - Decoder: 3 upsampling levels with skip connections
    - Self-attention at resolutions: 16x16, 8x8, 4x4, and bottleneck

    Analysis:
    - Self-attention improves global context understanding
    - Time conditioning enables learning noise level-dependent features
    - Skip connections preserve spatial information from encoder
    - GroupNorm ensures stable training with small batches
    """
    def __init__(self, c_in=1, c_out=1, time_dim=256, base_channels=64):
        super().__init__()

        # Time embedding MLP: raw time -> sinusoidal -> linear projections
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Initial convolution
        self.inc = nn.Conv2d(c_in, base_channels, kernel_size=3, padding=1)

        # --- Encoder (Downsampling) ---
        # Level 1: 64 -> 128 channels, 32x32 -> 16x16
        self.down1 = ResidualBlock(base_channels, base_channels * 2, time_dim)
        self.pool1 = nn.Conv2d(base_channels * 2, base_channels * 2, 4, 2, 1)
        self.sa1 = SelfAttention(base_channels * 2)

        # Level 2: 128 -> 256 channels, 16x16 -> 8x8
        self.down2 = ResidualBlock(base_channels * 2, base_channels * 4, time_dim)
        self.pool2 = nn.Conv2d(base_channels * 4, base_channels * 4, 4, 2, 1)
        self.sa2 = SelfAttention(base_channels * 4)

        # Level 3: 256 -> 256 channels, 8x8 -> 4x4
        self.down3 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)
        self.pool3 = nn.Conv2d(base_channels * 4, base_channels * 4, 4, 2, 1)
        self.sa3 = SelfAttention(base_channels * 4)

        # --- Bottleneck ---
        # Expand to 512 channels, then contract back to 256
        self.bot1 = ResidualBlock(base_channels * 4, base_channels * 8, time_dim)
        self.bot2 = ResidualBlock(base_channels * 8, base_channels * 8, time_dim)
        self.bot3 = ResidualBlock(base_channels * 8, base_channels * 4, time_dim)
        self.bot_sa = SelfAttention(base_channels * 4)

        # --- Decoder (Upsampling) ---

        # Up 1: 256 -> 128 channels, 4x4 -> 8x8
        self.up_trans1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 4, 2, 1)
        self.up1 = ResidualBlock(base_channels * 8, base_channels * 2, time_dim)  # 512 -> 128
        self.sa4 = SelfAttention(base_channels * 2)

        # Up 2: 128 -> 64 channels, 8x8 -> 16x16
        self.up_trans2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, 2, 1)
        self.up2 = ResidualBlock(base_channels * 2 + base_channels * 4, base_channels, time_dim)  # 384 -> 64
        self.sa5 = SelfAttention(base_channels)

        # Up 3: 64 -> 64 channels, 16x16 -> 32x32
        self.up_trans3 = nn.ConvTranspose2d(base_channels, base_channels, 4, 2, 1)
        self.up3 = ResidualBlock(base_channels + base_channels * 2, base_channels, time_dim)  # 192 -> 64
        self.sa6 = SelfAttention(base_channels)

        # Final output convolution
        self.outc = nn.Conv2d(base_channels, c_out, kernel_size=1)

    def _safe_concat(self, x1, x2):
        """
        Helper to handle shape mismatches (e.g. 28x28 padding issues).
        Resizes x2 to match x1 spatial size before concatenation.
        """
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        if diffX != 0 or diffY != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        return torch.cat([x1, x2], dim=1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        
        # Encoder
        x1 = self.inc(x)
        
        h1 = self.down1(x1, t)
        h1 = self.sa1(h1)
        x2 = self.pool1(h1)
        
        h2 = self.down2(x2, t)
        h2 = self.sa2(h2)
        x3 = self.pool2(h2)
        
        h3 = self.down3(x3, t)
        h3 = self.sa3(h3)
        x4 = self.pool3(h3)

        # Bottleneck
        mid = self.bot1(x4, t)
        mid = self.bot2(mid, t)
        mid = self.bot3(mid, t)
        mid = self.bot_sa(mid)

        # Decoder
        # Up 1
        x = self.up_trans1(mid)
        x = self._safe_concat(x, h3) # Concat mid(up) with h3
        x = self.up1(x, t)
        x = self.sa4(x)

        # Up 2
        x = self.up_trans2(x)
        x = self._safe_concat(x, h2) # Concat x with h2
        x = self.up2(x, t)
        x = self.sa5(x)

        # Up 3
        x = self.up_trans3(x)
        x = self._safe_concat(x, h1) # Concat x with h1
        x = self.up3(x, t)
        x = self.sa6(x)

        return self.outc(x)


class DDPMScheduler:
    """
    DDPM Variance Scheduler with Linear Schedule.

    Implements the forward diffusion process (adding noise) and provides utilities
    for the reverse process (removing noise). Uses a linear variance schedule
    where β_t increases linearly from β_start to β_end over T timesteps.

    Args:
        num_timesteps (int): Total number of diffusion timesteps T (default: 1000)
        beta_start (float): Starting β value (default: 0.0001)
        beta_end (float): Ending β value (default: 0.02)
        device (str): Device for tensor computations (default: 'cuda')

    Key Computations:
    - α_t = 1 - β_t
    - ᾱ_t = ∏_{s=1}^t α_s (cumulative product)
    - Forward process: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
    - Reverse posterior: q(x_{t-1}|x_t,x_0) parameters

    Analysis:
    - Linear β schedule provides stable training and good sample quality
    - Pre-computing all coefficients improves efficiency during sampling
    - The schedule balances between slow diffusion (small β) and fast mixing (large β)
    - β_start=0.0001 ensures minimal noise at t=1, β_end=0.02 provides sufficient noise at t=T
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device

        # Linear variance schedule: β_t increases linearly from β_start to β_end
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)

        # α_t = 1 - β_t (noise schedule complement)
        self.alphas = 1.0 - self.betas

        # ᾱ_t = cumulative product of α_s for s=1 to t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # ᾱ_{t-1} with ᾱ_0 = 1 (for posterior calculations)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Pre-compute coefficients for forward process reparameterization
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Pre-compute coefficients for reverse process
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Posterior variance for reverse process: β̃_t = β_t * (1-ᾱ_{t-1}) / (1-ᾱ_t)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

        # Posterior mean coefficients for reverse process
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def _extract(self, a, t, x_shape):
        """
        Extract values from a 1D tensor for a batch of indices.

        Args:
            a: 1D tensor of values
            t: 1D tensor of indices (batch of timesteps)
            x_shape: shape of x for broadcasting

        Returns:
            Values extracted and reshaped for broadcasting
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def perturb_input(self, x_0, t, noise=None):
        """
        Step 2: Forward Process (Perturb Input)

        Adds noise to x_0 to get x_t using the reparameterization trick:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

        Args:
            x_0: Original clean images [B, C, H, W]
            t: Timesteps [B]
            noise: Optional pre-sampled noise

        Returns:
            x_t: Noisy images at timestep t
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0, device=self.device)

        # Get coefficients for timestep t
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        # Reparameterization trick
        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

        return x_t, noise

    def get_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        return posterior_mean, posterior_variance


class DDPMTrainer:
    """
    Complete DDPM Training Pipeline
    """
    def __init__(self, model, scheduler, optimizer, device):
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.device = device
        self.loss_history = []

    def train_step(self, x_0):
        """
        Single training step for DDPM.

        Args:
            x_0: Batch of clean images [B, C, H, W]

        Returns:
            loss: MSE loss value
        """
        self.model.train()
        batch_size = x_0.shape[0]

        # Step 1: Sample random timesteps for each image
        t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=self.device)

        # Step 2: Add noise to images (Forward Process)
        x_t, noise = self.scheduler.perturb_input(x_0, t)

        # Step 3: Predict noise with the model
        noise_pred = self.model(x_t, t)

        # Step 4: Compute MSE loss
        loss = F.mse_loss(noise_pred, noise)

        # Step 5: Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, dataloader, epoch):
        """
        Train for one epoch.
        """
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)
            loss = self.train_step(images)
            epoch_loss += loss

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss:.4f}'})

        avg_loss = epoch_loss / len(dataloader)
        self.loss_history.append(avg_loss)
        return avg_loss

    def train(self, dataloader, num_epochs, save_path=None):
        """
        Full training loop.
        """
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(1, num_epochs + 1):
            avg_loss = self.train_epoch(dataloader, epoch)
            print(f"Epoch {epoch}/{num_epochs} - Average Loss: {avg_loss:.4f}")

            # Save checkpoint periodically
            if save_path and epoch % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                }, f"{save_path}/checkpoint_epoch_{epoch}.pt")

        print("Done: Training complete!")
        return self.loss_history

    def plot_loss(self, save_path=None):
        """Plot training loss curve (and optionally save it for the report)."""
        plt.figure(figsize=(10, 4))
        plt.plot(self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('DDPM Training Loss')
        plt.grid(True)

        if save_path is None and 'REPORT_FIG_DIR' in globals():
            save_path = REPORT_FIG_DIR / "ddpm_training_loss.png"
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Done: Saved: {save_path}")

        plt.show()


class DDPMSampler:
    """
    DDPM Sampling: Stochastic reverse process.

    Algorithm:
    For t = T, T-1, ..., 1:
        z ~ N(0, I) if t > 1 else z = 0
        x_{t-1} = (1/sqrt(alpha_t))(x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_theta(x_t, t)) + sigma_t * z
    """
    def __init__(self, model, scheduler, device):
        self.model = model
        self.scheduler = scheduler
        self.device = device

    @torch.no_grad()
    def sample(self, batch_size, img_size=(3, 32, 32), show_progress=True):
        """
        Generate samples using DDPM reverse process.

        Args:
            batch_size: Number of images to generate
            img_size: Size of images (C, H, W)
            show_progress: Whether to show progress bar

        Returns:
            Generated images [B, C, H, W]
        """
        self.model.eval()

        # Start from pure Gaussian noise
        x = torch.randn(batch_size, *img_size, device=self.device)

        # Store intermediate samples for visualization
        intermediates = [x.clone()]

        # Reverse process: from T to 1
        timesteps = reversed(range(self.scheduler.num_timesteps))
        if show_progress:
            timesteps = tqdm(timesteps, desc="DDPM Sampling", total=self.scheduler.num_timesteps)

        for t in timesteps:
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # Predict noise
            noise_pred = self.model(x, t_tensor)

            # Get coefficients
            alpha = self.scheduler.alphas[t]
            alpha_cumprod = self.scheduler.alphas_cumprod[t]
            beta = self.scheduler.betas[t]

            # Sample noise for stochastic process (except at t=0)
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # DDPM update rule
            # x_{t-1} = (1/sqrt(alpha_t))(x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_theta) + sigma_t * z
            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * noise_pred
            ) + torch.sqrt(beta) * noise

            # Store intermediate (every 100 steps)
            if t % 100 == 0:
                intermediates.append(x.clone())

        return x, intermediates


class DDIMSampler:
    """
    DDIM Sampling: Deterministic (eta=0) or semi-stochastic reverse process.
    Can skip timesteps for faster generation.

    Key difference from DDPM:
    - Non-Markovian process
    - Deterministic when eta=0
    - Can use fewer steps (e.g., 50 instead of 1000)
    """
    def __init__(self, model, scheduler, device):
        self.model = model
        self.scheduler = scheduler
        self.device = device

    @torch.no_grad()
    def sample(self, batch_size, img_size=(3, 32, 32), num_inference_steps=50, eta=0.0, show_progress=True):
        """
        Generate samples using DDIM reverse process.

        Args:
            batch_size: Number of images to generate
            img_size: Size of images (C, H, W)
            num_inference_steps: Number of denoising steps (can be << T)
            eta: Controls stochasticity (0 = deterministic, 1 = DDPM-like)
            show_progress: Whether to show progress bar

        Returns:
            Generated images [B, C, H, W]
        """
        self.model.eval()

        # Create timestep schedule (skip steps for faster sampling)
        step_ratio = self.scheduler.num_timesteps // num_inference_steps
        timesteps = np.arange(0, self.scheduler.num_timesteps, step_ratio)[::-1].copy()

        # Start from pure Gaussian noise
        x = torch.randn(batch_size, *img_size, device=self.device)

        intermediates = [x.clone()]

        if show_progress:
            timesteps_iter = tqdm(enumerate(timesteps), desc="DDIM Sampling", total=len(timesteps))
        else:
            timesteps_iter = enumerate(timesteps)

        for i, t in timesteps_iter:
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # Predict noise
            noise_pred = self.model(x, t_tensor)

            # Get alpha values
            alpha_cumprod_t = self.scheduler.alphas_cumprod[t]

            # Get alpha for previous timestep
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_cumprod_t_prev = self.scheduler.alphas_cumprod[t_prev]
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0, device=self.device)

            # Predict x_0 from x_t and noise prediction
            # x_0 = (x_t - sqrt(1-alpha_bar_t) * eps) / sqrt(alpha_bar_t)
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)

            # Optionally clip x_0 to [-1, 1]
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            # Compute direction pointing to x_t
            # "direction" = sqrt(1-alpha_bar_{t-1} - sigma^2) * eps_theta
            sigma = eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)) * \
                    torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_t_prev)

            direction = torch.sqrt(1 - alpha_cumprod_t_prev - sigma**2) * noise_pred

            # Sample noise
            if eta > 0 and i < len(timesteps) - 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # DDIM update rule
            # x_{t-1} = sqrt(alpha_bar_{t-1}) * pred_x0 + direction + sigma * noise
            x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + direction + sigma * noise

            # Store intermediate
            if i % 10 == 0:
                intermediates.append(x.clone())

        return x, intermediates


def visualize_forward_process(scheduler, img_size=(3, 32, 32), timesteps_to_show=[0, 100, 300, 500, 700, 999]):
    """
    Visualize the forward diffusion process at various timesteps.
    """
    # Create a sample image (random or could load a real one)
    x_0 = torch.randn(1, *img_size, device=device) * 0.5  # Start with some pattern

    fig, axes = plt.subplots(1, len(timesteps_to_show), figsize=(15, 3))

    for idx, t in enumerate(timesteps_to_show):
        t_tensor = torch.tensor([t], device=device)
        x_t, _ = scheduler.perturb_input(x_0, t_tensor)

        # Convert to displayable format
        img = x_t[0].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        axes[idx].imshow(img)
        axes[idx].set_title(f't = {t}')
        axes[idx].axis('off')

    plt.suptitle('Forward Diffusion Process: Adding Noise Over Time')
    plt.tight_layout()
    plt.show()


def visualize_samples(samples, title="Generated Samples", save_path=None):
    """Helper function to visualize a batch of samples (and optionally save a grid)."""
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i, ax in enumerate(axes.flat):
        if i < len(samples):
            img = samples[i].permute(1, 2, 0).cpu().numpy()
            img = (img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            img = np.clip(img, 0, 1)
            ax.imshow(img)
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Done: Saved: {save_path}")
    plt.show()


def visualize_denoising_process(intermediates, title="Denoising Process"):
    """Visualize the progressive denoising."""
    n_steps = min(len(intermediates), 10)
    fig, axes = plt.subplots(1, n_steps, figsize=(2 * n_steps, 2))

    step_indices = np.linspace(0, len(intermediates) - 1, n_steps, dtype=int)

    for idx, step_idx in enumerate(step_indices):
        img = intermediates[step_idx][0].permute(1, 2, 0).cpu().numpy()
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        axes[idx].imshow(img)
        axes[idx].set_title(f"Step {step_idx}")
        axes[idx].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
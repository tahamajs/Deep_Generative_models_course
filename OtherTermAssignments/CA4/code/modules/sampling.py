"""Sampling utilities (DDPMScheduler, DDPMSampler, DDIMSampler).

These are translations of the notebook sampling helpers so the notebook
can import them instead of having long inline cells.
"""
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


class DDPMScheduler:
    """DDPM Variance Scheduler with Linear Schedule"""
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def perturb_input(self, x_0, t, noise=None):
        """Forward process: adds noise to x_0 using reparameterization trick."""
        if noise is None:
            noise = torch.randn_like(x_0, device=self.device)
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t, noise

    def get_posterior_mean_variance(self, x_0, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        return posterior_mean, posterior_variance


class DDPMSampler:
    def __init__(self, model, scheduler, device):
        self.model = model
        self.scheduler = scheduler
        self.device = device

    @torch.no_grad()
    def sample(self, batch_size, img_size=None, show_progress=True):
        self.model.eval()

        if img_size is None:
            # fallback values if notebook config not provided
            img_size = (1, 28, 28)

        x = torch.randn(batch_size, *img_size, device=self.device)

        intermediates = [x.clone()]
        timesteps = reversed(range(self.scheduler.num_timesteps))

        if show_progress:
            timesteps = tqdm(timesteps, desc="DDPM Sampling", total=self.scheduler.num_timesteps)

        for t in timesteps:
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            noise_pred = self.model(x, t_tensor)

            alpha = self.scheduler.alphas[t]
            alpha_cumprod = self.scheduler.alphas_cumprod[t]
            beta = self.scheduler.betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * noise_pred
            ) + torch.sqrt(beta) * noise

            if t % 100 == 0:
                intermediates.append(x.clone())

        return x, intermediates


class DDIMSampler:
    def __init__(self, model, scheduler, device):
        self.model = model
        self.scheduler = scheduler
        self.device = device

    @torch.no_grad()
    def sample(self, batch_size, img_size=None, num_inference_steps=50, eta=0.0, show_progress=True):
        self.model.eval()

        if img_size is None:
            img_size = (1, 28, 28)

        step_ratio = self.scheduler.num_timesteps // num_inference_steps
        timesteps = np.arange(0, self.scheduler.num_timesteps, step_ratio)[::-1].copy()

        x = torch.randn(batch_size, *img_size, device=self.device)

        intermediates = [x.clone()]

        if show_progress:
            timesteps_iter = tqdm(enumerate(timesteps), desc="DDIM Sampling", total=len(timesteps))
        else:
            timesteps_iter = enumerate(timesteps)

        for i, t in timesteps_iter:
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            noise_pred = self.model(x, t_tensor)

            alpha_cumprod_t = self.scheduler.alphas_cumprod[t]

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_cumprod_t_prev = self.scheduler.alphas_cumprod[t_prev]
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0, device=self.device)

            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            sigma = eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)) * \
                    torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_t_prev)

            direction = torch.sqrt(1 - alpha_cumprod_t_prev - sigma**2) * noise_pred

            if eta > 0 and i < len(timesteps) - 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + direction + sigma * noise

            if i % 10 == 0:
                intermediates.append(x.clone())

        return x, intermediates

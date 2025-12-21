"""MAF implementation (MADE + MAF blocks) and training helpers.
"""
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch import Tensor
from tqdm import tqdm
import torch.optim as optim


class MaskedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, mask: Tensor) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        masked_weight = self.linear.weight * self.mask
        return F.linear(x, masked_weight, self.linear.bias)


def create_masks(input_dim: int, hidden_dims: List[int], output_dim: int, seed: Optional[int] = 42):
    """Create autoregressive masks for MADE.
    This implementation follows a simple assignment of degrees (1..D) to units.
    """
    rng = np.random.RandomState(seed)
    D = input_dim

    # assign degrees for input (1..D)
    degrees = []
    degrees_input = np.arange(1, D + 1)
    degrees.append(degrees_input)

    # hidden degrees: sample uniformly from 1..D-1
    for h in hidden_dims:
        degrees_hidden = rng.randint(1, D + 1, size=h)
        degrees.append(degrees_hidden)

    # output degrees: for s/t outputs, we need degrees in 1..D
    degrees_output = np.arange(1, D + 1).repeat(2)  # for s and t
    degrees.append(degrees_output)

    masks = []
    # input -> first hidden
    in_deg = degrees[0]
    out_deg = degrees[1]
    mask = (out_deg[:, None] >= in_deg[None, :]).astype(np.float32)
    masks.append(torch.from_numpy(mask))

    # hidden -> hidden
    for i in range(1, len(degrees) - 2):
        in_deg = degrees[i]
        out_deg = degrees[i + 1]
        mask = (out_deg[:, None] >= in_deg[None, :]).astype(np.float32)
        masks.append(torch.from_numpy(mask))

    # last hidden -> output
    in_deg = degrees[-2]
    out_deg = degrees[-1]
    mask = (out_deg[:, None] > in_deg[None, :]).astype(np.float32)
    masks.append(torch.from_numpy(mask))

    return masks


class MADE(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, seed: Optional[int] = 42) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        masks = create_masks(input_dim, hidden_dims, output_dim, seed=seed)

        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(MaskedLinear(dims[i], dims[i + 1], masks[i]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # x is (batch, input_dim)
        return self.net(x)


class MAFBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 512], seed: Optional[int] = 42) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.made = MADE(input_dim, hidden_dims, 2 * input_dim, seed=seed)

    def forward(self, x: Tensor):
        # x: (batch, D)
        s_and_t = self.made(x)
        s, t = s_and_t.chunk(2, dim=1)
        s = torch.sigmoid(s + 2.0)
        z = (x - t) / (s + 1e-8)
        log_det_J = -torch.sum(torch.log(s + 1e-8), dim=1)
        return z, log_det_J

    def inverse(self, z: Tensor):
        # sequentially compute x from z
        batch_size = z.shape[0]
        x = torch.zeros_like(z)
        for i in range(self.input_dim):
            s_and_t = self.made(x)
            s, t = s_and_t.chunk(2, dim=1)
            s = torch.sigmoid(s + 2.0)
            x[:, i] = s[:, i] * z[:, i] + t[:, i]
        return x


class MAF(nn.Module):
    def __init__(self, input_dim: int, num_blocks: int = 7, hidden_dims: List[int] = [512, 512], seed: Optional[int] = 42) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.blocks = nn.ModuleList([MAFBlock(input_dim, hidden_dims, seed=seed + i) for i in range(num_blocks)])
        self.base_dist = torch.distributions.Normal(0, 1)

    def forward(self, x: Tensor):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        log_det_J = torch.zeros(batch_size, device=x.device)
        z = x_flat
        for block in self.blocks:
            z, ldj = block(z)
            log_det_J = log_det_J + ldj
        log_prob = self.base_dist.log_prob(z).sum(dim=1) + log_det_J
        return z, log_prob

    def generate(self, num_samples: int, device: str = "cpu") -> Tensor:
        z = self.base_dist.sample((num_samples, self.input_dim)).to(device)
        x = z
        for block in reversed(self.blocks):
            x = block.inverse(x)
        return x

    def calculate_nll(self, x: Tensor):
        _, log_prob = self.forward(x)
        return -log_prob


# Training helpers

def train_maf(model: MAF, train_loader: torch.utils.data.DataLoader, num_epochs: int = 100, lr: float = 1e-4, device: str = "cpu"):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            nll = model.calculate_nll(batch).mean()
            nll.backward()
            optimizer.step()
            epoch_loss += nll.item()
        avg = epoch_loss / len(train_loader)
        losses.append(avg)
        print(f"Epoch {epoch+1}, Avg NLL: {avg:.4f}")
    return losses


def generate_images_maf(model: MAF, num_images: int = 5, img_size: int = 128, device: str = "cpu"):
    model.eval()
    start = time.time()
    with torch.no_grad():
        samples = model.generate(num_images, device=device)
        samples = samples.view(num_images, 3, img_size, img_size)
        samples = torch.clamp(samples, -1, 1)
        samples = (samples + 1) / 2
    gen_time = time.time() - start
    return samples, gen_time

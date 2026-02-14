"""CycleGAN model, training and evaluation helpers."""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_nc: int = 3, output_nc: int = 3, ngf: int = 64, num_residual_blocks: int = 9) -> None:
        super().__init__()
        model = []
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]

        model += [
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True)
        ]

        model += [
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        ]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(ngf * 4)]

        model += [
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True)
        ]

        model += [
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc: int = 3, ndf: int = 64) -> None:
        super().__init__()
        model = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


import random


class ImagePool:
    def __init__(self, pool_size: int = 50) -> None:
        self.pool_size = pool_size
        self.images = []

    def query(self, images: torch.Tensor):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = image.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(image)
                return_images.append(image)
            else:
                if random.uniform(0, 1) > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    return_images.append(self.images[idx].clone())
                    self.images[idx] = image
                else:
                    return_images.append(image)
        return torch.cat(return_images, dim=0)


# Loss helpers

def adversarial_loss(prediction: torch.Tensor, is_real: bool):
    target = torch.ones_like(prediction) if is_real else torch.zeros_like(prediction)
    return F.mse_loss(prediction, target)


def cycle_consistency_loss(real_image: torch.Tensor, reconstructed_image: torch.Tensor):
    return F.l1_loss(reconstructed_image, real_image)


def identity_loss(generator: nn.Module, real_image: torch.Tensor):
    identity_image = generator(real_image)
    return F.l1_loss(identity_image, real_image)


def generator_loss(D: nn.Module, fake_image: torch.Tensor):
    pred_fake = D(fake_image)
    return adversarial_loss(pred_fake, True)


def discriminator_loss(D: nn.Module, real_image: torch.Tensor, fake_image: torch.Tensor):
    pred_real = D(real_image)
    pred_fake = D(fake_image.detach())
    loss_real = adversarial_loss(pred_real, True)
    loss_fake = adversarial_loss(pred_fake, False)
    return (loss_real + loss_fake) * 0.5


# Training loop

def train_cyclegan(G_AB: nn.Module, G_BA: nn.Module, D_A: nn.Module, D_B: nn.Module,
                   train_loader_A, train_loader_B, num_epochs: int = 20, lr: float = 0.0002,
                   beta1: float = 0.5, device: str = "cpu", checkpoint_dir: Optional[str] = None):
    G_AB = G_AB.to(device)
    G_BA = G_BA.to(device)
    D_A = D_A.to(device)
    D_B = D_B.to(device)

    optimizer_G = optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=lr, betas=(beta1, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(beta1, 0.999))

    fake_A_pool = ImagePool(pool_size=50)
    fake_B_pool = ImagePool(pool_size=50)

    history = {'G_loss': [], 'D_A_loss': [], 'D_B_loss': [], 'cycle_loss': [], 'identity_loss': []}

    for epoch in range(num_epochs):
        G_AB.train(); G_BA.train(); D_A.train(); D_B.train()
        epoch_G_loss = epoch_D_A_loss = epoch_D_B_loss = epoch_cycle_loss = epoch_identity_loss = 0.0

        data_iter_A = iter(train_loader_A)
        data_iter_B = iter(train_loader_B)
        num_batches = min(len(train_loader_A), len(train_loader_B))

        if num_batches == 0:
            print(f"Epoch {epoch+1}/{num_epochs} skipped: one of the dataloaders is empty.")
            history['G_loss'].append(0.0)
            history['D_A_loss'].append(0.0)
            history['D_B_loss'].append(0.0)
            history['cycle_loss'].append(0.0)
            history['identity_loss'].append(0.0)
            continue

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}")
        for i in pbar:
            try:
                real_A = next(data_iter_A).to(device)
                real_B = next(data_iter_B).to(device)
            except StopIteration:
                break

            batch_size = min(real_A.size(0), real_B.size(0))
            real_A = real_A[:batch_size]; real_B = real_B[:batch_size]

            # Generators
            optimizer_G.zero_grad()
            loss_identity_A = identity_loss(G_BA, real_A) * 10.0 * 0.5
            loss_identity_B = identity_loss(G_AB, real_B) * 10.0 * 0.5
            loss_identity_total = loss_identity_A + loss_identity_B

            fake_B = G_AB(real_A)
            loss_GAN_AB = generator_loss(D_B, fake_B)
            fake_A = G_BA(real_B)
            loss_GAN_BA = generator_loss(D_A, fake_A)

            recovered_A = G_BA(fake_B)
            loss_cycle_A = cycle_consistency_loss(real_A, recovered_A) * 10.0
            recovered_B = G_AB(fake_A)
            loss_cycle_B = cycle_consistency_loss(real_B, recovered_B) * 10.0
            loss_cycle_total = loss_cycle_A + loss_cycle_B

            loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_total + loss_identity_total
            loss_G.backward(); optimizer_G.step()

            # Discriminator A
            optimizer_D_A.zero_grad()
            fake_A_pooled = fake_A_pool.query(fake_A.detach())
            loss_D_A = discriminator_loss(D_A, real_A, fake_A_pooled)
            loss_D_A.backward(); optimizer_D_A.step()

            # Discriminator B
            optimizer_D_B.zero_grad()
            fake_B_pooled = fake_B_pool.query(fake_B.detach())
            loss_D_B = discriminator_loss(D_B, real_B, fake_B_pooled)
            loss_D_B.backward(); optimizer_D_B.step()

            epoch_G_loss += loss_G.item()
            epoch_D_A_loss += loss_D_A.item()
            epoch_D_B_loss += loss_D_B.item()
            epoch_cycle_loss += loss_cycle_total.item()
            epoch_identity_loss += loss_identity_total.item()

            pbar.set_postfix({'G': f"{loss_G.item():.3f}", 'D_A': f"{loss_D_A.item():.3f}", 'D_B': f"{loss_D_B.item():.3f}"})

        # Save checkpoints
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(G_AB.state_dict(), os.path.join(checkpoint_dir, f'G_AB_epoch_{epoch+1}.pth'))
            torch.save(G_BA.state_dict(), os.path.join(checkpoint_dir, f'G_BA_epoch_{epoch+1}.pth'))

        history['G_loss'].append(epoch_G_loss / max(1, num_batches))
        history['D_A_loss'].append(epoch_D_A_loss / max(1, num_batches))
        history['D_B_loss'].append(epoch_D_B_loss / max(1, num_batches))
        history['cycle_loss'].append(epoch_cycle_loss / max(1, num_batches))
        history['identity_loss'].append(epoch_identity_loss / max(1, num_batches))

        print(f"Epoch {epoch+1} - G: {history['G_loss'][-1]:.4f}, D_A: {history['D_A_loss'][-1]:.4f}, D_B: {history['D_B_loss'][-1]:.4f}")

    return history


def test_cyclegan(G_AB: nn.Module, G_BA: nn.Module, test_loader_A, test_loader_B, device: str = 'cpu', num_samples: int = 5):
    G_AB = G_AB.to(device); G_BA = G_BA.to(device)
    G_AB.eval(); G_BA.eval()
    with torch.no_grad():
        real_A = next(iter(test_loader_A))[:num_samples].to(device)
        fake_B = G_AB(real_A); recovered_A = G_BA(fake_B)
        real_B = next(iter(test_loader_B))[:num_samples].to(device)
        fake_A = G_BA(real_B); recovered_B = G_AB(fake_A)
    return (real_A, fake_B, recovered_A), (real_B, fake_A, recovered_B)

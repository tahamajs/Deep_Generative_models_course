#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple test to isolate the autograd issue
def test_simple_gan():
    # Create simple networks
    class SimpleGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 784),
            )

        def forward(self, z):
            return self.net(z).view(-1, 1, 28, 28)

    class SimpleDiscriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.net(x)

    g = SimpleGenerator()
    d = SimpleDiscriminator()

    # Test data
    batch_size = 4
    z = torch.randn(batch_size, 64)
    x_real = torch.randn(batch_size, 1, 28, 28)

    # Test forward passes
    x_fake = g(z)
    d_real = d(x_real)
    d_fake = d(x_fake)

    print("Forward passes work fine")

    # Test losses
    d_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
             F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))

    # Fresh forward for generator loss
    z_g = torch.randn(batch_size, 64)
    x_fake_g = g(z_g)
    d_fake_g = d(x_fake_g)
    g_loss = F.binary_cross_entropy_with_logits(d_fake_g, torch.ones_like(d_fake_g))

    print("Losses computed successfully")

    # Test backward passes
    d_optimizer = torch.optim.Adam(d.parameters(), lr=1e-3)
    g_optimizer = torch.optim.Adam(g.parameters(), lr=1e-3)

    d_optimizer.zero_grad()
    d_loss.backward(retain_graph=True)
    d_optimizer.step()

    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    print("Backward passes work fine!")
    print("Test passed - the issue is not with basic GAN operations")

if __name__ == "__main__":
    test_simple_gan()
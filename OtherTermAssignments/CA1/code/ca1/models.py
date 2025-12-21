import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, h_dim=256):
        super(Encoder, self).__init__()
        self.h_dim = h_dim
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(128 * 8 * 8, h_dim * 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        h = self.fc(x)
        mu, log_var = torch.chunk(h, 2, dim=1)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, h_dim=256):
        super(Decoder, self).__init__()
        self.h_dim = h_dim
        self.fc = nn.Linear(h_dim, 128 * 8 * 8)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = F.relu(x)
        x = x.view(x.size(0), 128, 8, 8)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x


class VAE(nn.Module):
    def __init__(self, h_dim=256):
        super(VAE, self).__init__()
        self.h_dim = h_dim
        self.encoder = Encoder(h_dim)
        self.decoder = Decoder(h_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.h_dim).to(device)
        samples = self.decoder(z)
        return samples

import argparse
import numpy as np
import torch
import torch.utils.data
from codebase import utils as ut
from codebase.models import nns
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class FSVAE(nn.Module):
    def __init__(self, nn='v2', name='fsvae'):
        super().__init__()
        self.name = name
        self.z_dim = 10
        self.y_dim = 10
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.y_dim)
        self.dec = nn.Decoder(self.z_dim, self.y_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x, y):
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that we are interested in the ELBO of ln p(x | y)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################

        # Encode x,y to get q(z|x,y) parameters
        qm, qv = self.enc.encode(x, y)

        # Sample z from q(z|x,y)
        z = ut.sample_gaussian(qm, qv)

        # Decode to get mean of p(x|z,y)
        mu = self.dec.decode(z, y)  # (batch, dim)

        # Reconstruction loss: -log p(x|z,y)
        # p(x|z,y) = N(x|mu, sigma^2 I) with sigma^2 = 1/10
        sigma_squared = 1.0 / 10.0
        # log p(x|z,y) = -0.5 * dim * log(2π * σ^2) - 0.5/σ^2 * ||x - mu||^2
        dim = x.size(1)
        log_px_zy = -0.5 * dim * torch.log(2 * torch.pi * sigma_squared) - 0.5 / sigma_squared * ((x - mu) ** 2).sum(1)
        rec = -log_px_zy.mean()  # Average over batch

        # KL divergence: DKL(q(z|x,y) || p(z))
        kl_z = ut.kl_normal(qm, qv, self.z_prior_m.expand_as(qm), self.z_prior_v.expand_as(qv))
        kl_z = kl_z.mean()  # Average over batch

        # Negative ELBO = KL + Reconstruction
        nelbo = kl_z + rec

        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl_z, rec

    def loss(self, x, y):
        nelbo, kl_z, rec = self.negative_elbo_bound(x, y)
        loss = nelbo

        summaries = dict((
            ('train/loss', loss),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl_z),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def compute_mean_given(self, z, y):
        return self.dec.decode(z, y)

    def sample_z(self, batch):
        return ut.sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                  self.z_prior[1].expand(batch, self.z_dim))

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

class SSVAE(nn.Module):
    def __init__(self, nn='v1', name='ssvae', gen_weight=1, class_weight=100):
        super().__init__()
        self.name = name
        self.z_dim = 64
        self.y_dim = 10
        self.gen_weight = gen_weight
        self.class_weight = class_weight
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.y_dim)
        self.dec = nn.Decoder(self.z_dim, self.y_dim)
        self.cls = nn.Classifier(self.y_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl_z: tensor: (): ELBO KL divergence for z
            kl_y: tensor: (): ELBO KL divergence for y
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL_Z, KL_Y and Rec decomposition
        #
        # To assist you in the vectorization of the summation over y, we have
        # the computation of q(y | x) and some tensor tiling code for you.
        #
        # Note that nelbo = kl_z + kl_y + rec
        #
        # Outputs should all be scalar
        ################################################################################
        y_logits = self.cls.classify(x)
        y_logprob = F.log_softmax(y_logits, dim=1)
        y_prob = torch.softmax(y_logprob, dim=1) # (batch, y_dim)

        # Duplicate y based on x's batch size. Then duplicate x
        # This enumerates all possible combination of x with labels (0, 1, ..., 9)
        y = np.repeat(np.arange(self.y_dim), x.size(0))
        y = x.new(np.eye(self.y_dim)[y])
        x = ut.duplicate(x, self.y_dim)

        # Encode to get q(z|x,y) parameters for each (x,y) pair
        qm, qv = self.enc.encode(x, y)  # (batch * y_dim, z_dim)

        # Sample z from q(z|x,y) for each (x,y) pair
        z = ut.sample_gaussian(qm, qv)  # (batch * y_dim, z_dim)

        # Decode to get reconstruction logits p(x|z,y)
        logits = self.dec.decode(z, y)  # (batch * y_dim, dim)

        # Compute reconstruction loss: -log p(x|z,y)
        # Reshape for proper computation: (batch, y_dim, dim)
        x_reshaped = x.view(x.size(0) // self.y_dim, self.y_dim, -1)
        logits_reshaped = logits.view(x.size(0) // self.y_dim, self.y_dim, -1)

        # Compute log p(x|z,y) for each (x,y,z) triple
        log_px_zy = ut.log_bernoulli_with_logits(x_reshaped, logits_reshaped)  # (batch, y_dim)

        # Weight by q(y|x) and sum over y: E_{q(y|x)}[-log p(x|z,y)]
        rec = -(y_prob * log_px_zy).sum(1).mean()  # Average over batch

        # KL for y: DKL(q(y|x) || p(y)) where p(y) is uniform
        # p(y) has uniform probability 1/y_dim
        log_py = torch.log(torch.tensor(1.0 / self.y_dim))
        kl_y = ut.kl_cat(y_prob, y_logprob, log_py.expand_as(y_prob))
        kl_y = kl_y.mean()  # Average over batch

        # KL for z: E_{q(y|x)}[DKL(q(z|x,y) || p(z))]
        # Reshape q parameters: (batch, y_dim, z_dim)
        qm_reshaped = qm.view(x.size(0) // self.y_dim, self.y_dim, -1)
        qv_reshaped = qv.view(x.size(0) // self.y_dim, self.y_dim, -1)

        # Sample one z per (x,y) pair for Monte Carlo
        z_reshaped = z.view(x.size(0) // self.y_dim, self.y_dim, -1)  # (batch, y_dim, z_dim)

        # Compute log q(z|x,y) and log p(z)
        log_qz_xy = ut.log_normal(z_reshaped, qm_reshaped, qv_reshaped)  # (batch, y_dim)
        log_pz = ut.log_normal(z_reshaped, self.z_prior_m.expand_as(z_reshaped), self.z_prior_v.expand_as(z_reshaped))  # (batch, y_dim)

        # KL_z = E_{q(y|x)}[log q(z|x,y) - log p(z)]
        kl_z_per_y = log_qz_xy - log_pz  # (batch, y_dim)
        kl_z = (y_prob * kl_z_per_y).sum(1).mean()  # Average over batch

        # Negative ELBO = KL_y + KL_z + Reconstruction
        nelbo = kl_y + kl_z + rec
        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl_z, kl_y, rec

    def classification_cross_entropy(self, x, y):
        y_logits = self.cls.classify(x)
        return F.cross_entropy(y_logits, y.argmax(1))

    def loss(self, x, xl, yl):
        if self.gen_weight > 0:
            nelbo, kl_z, kl_y, rec = self.negative_elbo_bound(x)
        else:
            nelbo, kl_z, kl_y, rec = [0] * 4
        ce = self.classification_cross_entropy(xl, yl)
        loss = self.gen_weight * nelbo + self.class_weight * ce

        summaries = dict((
            ('train/loss', loss),
            ('class/ce', ce),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl_z),
            ('gen/kl_y', kl_y),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def compute_sigmoid_given(self, z, y):
        logits = self.dec.decode(z, y)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                  self.z_prior[1].expand(batch, self.z_dim))

    def sample_x_given(self, z, y):
        return torch.bernoulli(self.compute_sigmoid_given(z, y))

import numpy as np
import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class GMVAE(nn.Module):
    def __init__(self, nn='v1', z_dim=2, k=500, name='gmvae'):
        super().__init__()
        self.name = name
        self.k = k
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # To help you start, we have computed the mixture of Gaussians prior
        # prior = (m_mixture, v_mixture) for you, where
        # m_mixture and v_mixture each have shape (1, self.k, self.z_dim)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        # Compute the mixture of Gaussian prior
        prior = ut.gaussian_parameters(self.z_pre, dim=1)
        m_mixture, v_mixture = prior

        # Encode x to get q(z|x) parameters
        qm, qv = self.enc.encode(x)

        # Sample z from q(z|x)
        z = ut.sample_gaussian(qm, qv)

        # Decode z to get reconstruction logits
        logits = self.dec.decode(z)

        # Compute reconstruction loss: -log p(x|z)
        rec = -ut.log_bernoulli_with_logits(x, logits)
        rec = rec.mean()  # Average over batch

        # Compute KL divergence: KL(q(z|x) || p(z))
        # Since p(z) is a mixture, we use Monte Carlo: KL ≈ log q(z|x) - log p(z)
        log_qz_x = ut.log_normal(z, qm, qv)

        # Expand mixture parameters to match batch size
        m_mix_expanded = m_mixture.expand(x.size(0), -1, -1)  # (batch, k, z_dim)
        v_mix_expanded = v_mixture.expand(x.size(0), -1, -1)  # (batch, k, z_dim)

        log_pz = ut.log_normal_mixture(z, m_mix_expanded, v_mix_expanded)

        kl = log_qz_x - log_pz
        kl = kl.mean()  # Average over batch

        # Negative ELBO = KL + Reconstruction
        nelbo = kl + rec
        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################
        # Compute the mixture of Gaussian prior
        prior = ut.gaussian_parameters(self.z_pre, dim=1)
        m_mixture, v_mixture = prior

        # Encode x to get q(z|x) parameters
        qm, qv = self.enc.encode(x)

        # Duplicate x and parameters for iw samples
        x_dup = ut.duplicate(x, iw)  # (batch * iw, dim)
        qm_dup = ut.duplicate(qm, iw)  # (batch * iw, z_dim)
        qv_dup = ut.duplicate(qv, iw)  # (batch * iw, z_dim)

        # Sample z from q(z|x) - iw samples per batch element
        z = ut.sample_gaussian(qm_dup, qv_dup)  # (batch * iw, z_dim)

        # Decode z to get reconstruction logits
        logits = self.dec.decode(z)  # (batch * iw, dim)

        # Compute log p(x|z) for each sample
        log_px_z = ut.log_bernoulli_with_logits(x_dup, logits)  # (batch * iw,)

        # Compute log p(z) for each sample using mixture prior
        # Expand mixture parameters to match batch*iw size
        batch_iw = x.size(0) * iw
        m_mix_expanded = m_mixture.expand(batch_iw, -1, -1)  # (batch*iw, k, z_dim)
        v_mix_expanded = v_mixture.expand(batch_iw, -1, -1)  # (batch*iw, k, z_dim)

        log_pz = ut.log_normal_mixture(z, m_mix_expanded, v_mix_expanded)  # (batch*iw,)

        # Compute log q(z|x) for each sample
        log_qz_x = ut.log_normal(z, qm_dup, qv_dup)  # (batch * iw,)

        # Compute log p(x,z) = log p(x|z) + log p(z)
        log_pxz = log_px_z + log_pz  # (batch * iw,)

        # Compute log weights: log p(x,z) - log q(z|x)
        log_weights = log_pxz - log_qz_x  # (batch * iw,)

        # Reshape to (batch, iw) for log_mean_exp
        log_weights = log_weights.view(x.size(0), iw)

        # Compute IWAE bound
        iwae = ut.log_mean_exp(log_weights, dim=1)  # (batch,)
        niwae = -iwae.mean()  # Negative IWAE, averaged over batch

        # For KL and Rec, use single sample (ELBO decomposition)
        z_single = z[:x.size(0)]  # Take first sample per batch element
        logits_single = self.dec.decode(z_single)

        rec = -ut.log_bernoulli_with_logits(x, logits_single)
        rec = rec.mean()

        # Compute KL using Monte Carlo
        log_qz_x_single = ut.log_normal(z_single, qm, qv)
        m_mix_single = m_mixture.expand(x.size(0), -1, -1)
        v_mix_single = v_mixture.expand(x.size(0), -1, -1)
        log_pz_single = ut.log_normal_mixture(z_single, m_mix_single, v_mix_single)

        kl = log_qz_x_single - log_pz_single
        kl = kl.mean()

        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        m, v = ut.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return ut.sample_gaussian(m, v)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))

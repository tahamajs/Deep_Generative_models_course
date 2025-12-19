import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, nn="v1", name="vae", z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

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
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################

        # Encode x to get q(z|x) parameters
        qm, qv = self.enc.encode(x)

        # Sample z from q(z|x)
        z = ut.sample_gaussian(qm, qv)

        # Decode z to get reconstruction logits
        logits = self.dec.decode(z)

        # Compute reconstruction loss: -log p(x|z) ≈ -log Bern(x|σ(logits))
        # Since x is binary, this is -log_bernoulli_with_logits
        rec = -ut.log_bernoulli_with_logits(x, logits)
        rec = rec.mean()  # Average over batch

        # Compute KL divergence: KL(q(z|x) || p(z))
        kl = ut.kl_normal(
            qm, qv, self.z_prior_m.expand_as(qm), self.z_prior_v.expand_as(qv)
        )
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

        # Compute log p(z) for each sample
        log_pz = ut.log_normal(
            z, self.z_prior_m.expand_as(z), self.z_prior_v.expand_as(z)
        )  # (batch * iw,)

        # Compute log q(z|x) for each sample
        log_qz_x = ut.log_normal(z, qm_dup, qv_dup)  # (batch * iw,)

        # Compute log p(x,z) = log p(x|z) + log p(z)
        log_pxz = log_px_z + log_pz  # (batch * iw,)

        # Compute log weights: log p(x,z) - log q(z|x)
        log_weights = log_pxz - log_qz_x  # (batch * iw,)

        # Reshape to (batch, iw) for log_mean_exp
        log_weights = log_weights.view(x.size(0), iw)

        # Compute IWAE bound: log(1/iw * sum_i exp(log p(x,z^(i)) - log q(z^(i)|x)))
        # = log_mean_exp(log p(x,z^(i)) - log q(z^(i)|x))
        iwae = ut.log_mean_exp(log_weights, dim=1)  # (batch,)
        niwae = -iwae.mean()  # Negative IWAE, averaged over batch

        # For KL and Rec, use single sample (ELBO decomposition)
        z_single = z[: x.size(0)]  # Take first sample per batch element
        logits_single = self.dec.decode(z_single)

        rec = -ut.log_bernoulli_with_logits(x, logits_single)
        rec = rec.mean()

        kl = ut.kl_normal(
            qm, qv, self.z_prior_m.expand_as(qm), self.z_prior_v.expand_as(qv)
        )
        kl = kl.mean()

        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict(
            (
                ("train/loss", nelbo),
                ("gen/elbo", -nelbo),
                ("gen/kl_z", kl),
                ("gen/rec", rec),
            )
        )

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim),
        )

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))

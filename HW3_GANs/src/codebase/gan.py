import torch
from torch.nn import functional as F

def loss_nonsaturating(g, d, x_real, *, device):
    '''
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating discriminator loss
    - g_loss (torch.Tensor): nonsaturating generator loss
    '''
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # YOUR CODE STARTS HERE
    # You may find some or all of the below useful:
    #   - F.binary_cross_entropy_with_logits
    #   - F.logsigmoid
    raise NotImplementedError
    # YOUR CODE ENDS HERE

    return d_loss, g_loss

def conditional_loss_nonsaturating(g, d, x_real, y_real, *, device):
    '''
    Arguments:
    - g (codebase.network.ConditionalGenerator): The generator network
    - d (codebase.network.ConditionalDiscriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - y_real (torch.Tensor): training data labels (64)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating conditional discriminator loss
    - g_loss (torch.Tensor): nonsaturating conditional generator loss
    '''
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    y_fake = y_real  # use the real labels as the fake labels as well

    # YOUR CODE STARTS HERE
    raise NotImplementedError
    # YOUR CODE ENDS HERE

    return d_loss, g_loss

def loss_wasserstein_gp(g, d, x_real, *, device):
    '''
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs value of discriminator
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): wasserstein discriminator loss
    - g_loss (torch.Tensor): wasserstein generator loss
    '''
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # YOUR CODE STARTS HERE
    # You may find some or all of the below useful:
    #   - torch.rand
    #   - torch.autograd.grad(..., create_graph=True)
    raise NotImplementedError
    # YOUR CODE ENDS HERE

    return d_loss, g_loss

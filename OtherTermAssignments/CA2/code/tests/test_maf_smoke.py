import os
import torch
from torch.utils.data import DataLoader
import numpy as np

from maf import MAF, train_maf, generate_images_maf


class RandomVecDataset(torch.utils.data.Dataset):
    def __init__(self, n=20, dim=8):
        self.n = n
        self.dim = dim

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.randn(self.dim)


def test_maf_train_and_generate(tmp_path):
    device = 'cpu'
    input_dim = 8
    model = MAF(input_dim=input_dim, num_blocks=1, hidden_dims=[16])

    ds = RandomVecDataset(n=10, dim=input_dim)
    loader = DataLoader(ds, batch_size=2)

    ckpt_dir = os.path.join(tmp_path, 'ckpts')
    losses = train_maf(model, loader, num_epochs=1, lr=1e-3, device=device, checkpoint_dir=ckpt_dir, save_every=1)
    assert len(losses) == 1
    # checkpoint should be saved
    files = os.listdir(ckpt_dir)
    assert any('maf_epoch' in f for f in files)

    # generation
    samples, gen_time = generate_images_maf(model, num_images=2, img_size=1, device=device)
    assert samples.shape[0] == 2

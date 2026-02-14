import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import CONFIG


class dSpritesDataset(Dataset):
    def __init__(self, imgs: np.ndarray, transform=None):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        img = torch.FloatTensor(img).unsqueeze(0)
        if self.transform:
            img = self.transform(img)
        return img


def load_dsprites(path: str = None) -> Tuple:
    path = path or CONFIG["data_path"]
    url = (
        "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    )
    dst_dir = os.path.dirname(path) or "."
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.exists(path):
        print(f"Dataset not found at '{path}'. Downloading from {url}...")
        try:
            import urllib.request

            urllib.request.urlretrieve(url, path)
            print("Download completed.")
        except Exception as e:
            print("Automatic download failed:", e)
            return None, None, None, None

    try:
        data = np.load(path, allow_pickle=True, encoding="latin1")
        imgs = data["imgs"]
        latents_values = data["latents_values"]
        latents_classes = data["latents_classes"]
        metadata = data.get("metadata", None)
        return imgs, latents_values, latents_classes, metadata
    except Exception as e:
        print("Failed to load the dataset file:", e)
        return None, None, None, None


def seed_worker(worker_id: int) -> None:
    worker_seed = (CONFIG["seed"] + worker_id) % 2 ** 32
    np.random.seed(worker_seed)


def create_dataloaders(
    imgs,
    batch_size=128,
    train_split=0.9,
    generator=None,
    return_indices: bool = False,
    seed: int = None,
):
    n_train = int(len(imgs) * train_split)
    rng = np.random.default_rng(CONFIG["seed"] if seed is None else seed)
    indices = rng.permutation(len(imgs))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    train_dataset = dSpritesDataset(imgs[train_indices])
    val_dataset = dSpritesDataset(imgs[val_indices])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if return_indices:
        return train_loader, val_loader, train_indices, val_indices
    return train_loader, val_loader

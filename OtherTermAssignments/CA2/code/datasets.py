"""Datasets for MVTec Capsule and CycleGAN experiments."""
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch


class CapsuleDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, img_size: int = 128) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        self.images = []

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Dataset folder not found: {root_dir}")

        for fname in sorted(os.listdir(root_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                self.images.append(os.path.join(root_dir, fname))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


class ImageDataset(Dataset):
    def __init__(self, root_dir: str, transform=None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.images = []

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Dataset folder not found: {root_dir}")

        for fname in sorted(os.listdir(root_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                self.images.append(os.path.join(root_dir, fname))

        print(f"Loaded {len(self.images)} images from {root_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx % len(self.images)]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


# Common transforms
capsule_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

cyclegan_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

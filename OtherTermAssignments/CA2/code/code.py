# This module has been refactored. Use `run.py` to run experiments.
# Examples:
#   python run.py maf --mode train --epochs 2 --quick
#   python run.py cyclegan --mode train --epochs 2 --quick

print("This file is kept for compatibility. Use `run.py` to run experiments.")


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer('mask', mask)

    def forward(self, x):
        masked_weight = self.linear.weight * self.mask
        return F.linear(x, masked_weight, self.linear.bias)

def create_masks(input_dim, hidden_dims, output_dim):
    masks = []
    m_input = torch.arange(1, input_dim + 1)

    m_hidden = []
    for hidden_dim in hidden_dims:
        m_h = torch.randint(low=1, high=input_dim, size=(hidden_dim,))
        m_hidden.append(m_h)

    m_output = torch.arange(1, output_dim // 2 + 1).repeat(2)

    mask = (m_input.unsqueeze(1) <= m_hidden[0].unsqueeze(0)).float()
    masks.append(mask)

    for i in range(len(hidden_dims) - 1):
        mask = (m_hidden[i].unsqueeze(1) <= m_hidden[i+1].unsqueeze(0)).float()
        masks.append(mask)

    mask = (m_hidden[-1].unsqueeze(1) < m_output.unsqueeze(0)).float()
    masks.append(mask)

    return masks

class MADE(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim) -> None:
        super().__init__()
        self.input_dim: Any = input_dim
        self.output_dim: Any = output_dim

        masks = create_masks(input_dim, hidden_dims, output_dim)

        self.layers = nn.ModuleList()
        dims: list[Any] = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            self.layers.append(MaskedLinear(dims[i], dims[i+1], masks[i]))

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))

        x = self.layers[-1](x)

        return x


class MAFBlock(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 512]) -> None:
        super().__init__()
        self.input_dim: Any = input_dim
        self.made: MADE = MADE(input_dim, hidden_dims, 2 * input_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)

        s_and_t = self.made(x_flat)
        s, t = s_and_t.chunk(2, dim=1)

        s = torch.sigmoid(s + 2.0)

        z = (x_flat - t) / (s + 1e-8)

        log_det_J = -torch.sum(torch.log(s + 1e-8), dim=1)

        return z, log_det_J

    def inverse(self, z):
        batch_size = z.shape[0]
        x = torch.zeros_like(z)

        for i in range(self.input_dim):
            s_and_t = self.made(x)
            s, t = s_and_t.chunk(2, dim=1)
            s = torch.sigmoid(s + 2.0)

            x[:, i] = s[:, i] * z[:, i] + t[:, i]

        return x


class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer('mask', mask)

    def forward(self, x):
        masked_weight = self.linear.weight * self.mask
        return F.linear(x, masked_weight, self.linear.bias)

class MADE(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim) -> None:
        super().__init__()
        self.input_dim: Any = input_dim

        # Create masks for autoregressive property
        self.masks = self.create_masks(input_dim, hidden_dims, output_dim)

        layers = []
        in_dim: Any = input_dim
        mask_idx = 0
        for h_dim in hidden_dims:
            layers.append(MaskedLinear(in_dim, h_dim, self.masks[mask_idx]))
            layers.append(nn.ReLU())
            mask_idx += 1
            in_dim = h_dim

        layers.append(MaskedLinear(in_dim, output_dim, self.masks[mask_idx]))

        self.net = nn.Sequential(*layers)

    def create_masks(self, input_dim, hidden_dims, output_dim):
        masks = []

        # Input to first hidden
        in_dim = input_dim
        out_dim = hidden_dims[0]
        m = torch.zeros(out_dim, in_dim)
        for d in range(out_dim):
            m[d, :min(d+1, in_dim)] = 1
        masks.append(m)

        # Hidden to hidden
        for i in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i+1]
            m = torch.zeros(out_dim, in_dim)
            for d in range(out_dim):
                m[d, :min(d+1, in_dim)] = 1
            masks.append(m)

        # Last hidden to output
        in_dim = hidden_dims[-1]
        out_dim = output_dim
        m = torch.zeros(out_dim, in_dim)
        for d in range(out_dim):
            m[d, :min(d+1, in_dim)] = 1
        masks.append(m)

        return masks

    def forward(self, x):
        return self.net(x)

class MAF(nn.Module):
    def __init__(self, input_dim, num_blocks=7, hidden_dims=[512, 512]) -> None:
        super().__init__()
        self.input_dim: Any = input_dim
        self.num_blocks: int = num_blocks

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(MAFBlock(input_dim, hidden_dims))

        self.base_dist = torch.distributions.Normal(0, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)

        log_det_J = 0
        z = x_flat

        for block in self.blocks:
            z, ldj = block(z)
            log_det_J += ldj

        log_prob = self.base_dist.log_prob(z).sum(dim=1) + log_det_J

        return z, log_prob

    def generate(self, num_samples, device='cpu'):
        z = self.base_dist.sample((num_samples, self.input_dim)).to(device)

        x = z
        for block in reversed(self.blocks):
            x = block.inverse(x)

        return x

    def calculate_nll(self, x):
        _, log_prob = self.forward(x)
        return -log_prob.mean()


class CapsuleDataset(Dataset):
    def __init__(self, root_dir, transform=None, img_size=128) -> None:
        self.root_dir: Any = root_dir
        self.transform = transform
        self.img_size: int = img_size
        self.images = []

        for fname: str in os.listdir(root_dir):
            if fname.endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(root_dir, fname))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image: Image.Image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

def train_maf(model, train_loader, num_epochs=100, lr=0.0001, device='cpu'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in pbar:
            batch = batch.to(device)

            optimizer.zero_grad()
            nll = model.calculate_nll(batch)

            nll.backward()
            optimizer.step()

            epoch_loss += nll.item()
            pbar.set_postfix({'NLL': nll.item()})

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}, Average NLL: {avg_loss:.4f}')

    return losses

def generate_images_maf(model, num_images=5, img_size=128, device='cpu'):
    model.eval()

    start_time: float = time.time()

    with torch.no_grad():
        samples = model.generate(num_images, device=device)
        samples = samples.view(num_images, 3, img_size, img_size)
        samples = torch.clamp(samples, -1, 1)
        samples = (samples + 1) / 2

    generation_time: float = time.time() - start_time

    return samples, generation_time

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


lambda_A = 10.0
lambda_B = 10.0
lambda_identity = 0.5

def adversarial_loss(prediction, is_real):
    if is_real:
        target = torch.ones_like(prediction)
    else:
        target = torch.zeros_like(prediction)
    return F.mse_loss(prediction, target)

def cycle_consistency_loss(real_image, reconstructed_image):
    return F.l1_loss(reconstructed_image, real_image)

def identity_loss(generator, real_image):
    identity_image = generator(real_image)
    return F.l1_loss(identity_image, real_image)

def generator_loss(D, fake_image):
    pred_fake = D(fake_image)
    return adversarial_loss(pred_fake, True)

def discriminator_loss(D, real_image, fake_image):
    pred_real = D(real_image)
    pred_fake = D(fake_image.detach())

    loss_real = adversarial_loss(pred_real, True)
    loss_fake = adversarial_loss(pred_fake, False)

    return (loss_real + loss_fake) * 0.5


class ResidualBlock(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, num_residual_blocks=9) -> None:
        super().__init__()

        model = []

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]

        model += [
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True)
        ]

        model += [
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        ]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(ngf * 4)]

        model += [
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True)
        ]

        model += [
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64) -> None:
        super().__init__()

        model = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


import random

class ImagePool:
    def __init__(self, pool_size=50) -> None:
        self.pool_size: int = pool_size
        self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images

        return_images = []

        for image in images:
            image = image.unsqueeze(0)

            if len(self.images) < self.pool_size:
                self.images.append(image)
                return_images.append(image)
            else:
                if random.uniform(0, 1) > 0.5:
                    random_id: int = random.randint(0, self.pool_size - 1)
                    return_images.append(self.images[random_id].clone())
                    self.images[random_id] = image
                else:
                    return_images.append(image)

        return torch.cat(return_images, dim=0)


def train_cyclegan(G_AB, G_BA, D_A, D_B, train_loader_A, train_loader_B,
                   num_epochs=20, lr=0.0002, beta1=0.5, device='cpu'):

    G_AB = G_AB.to(device)
    G_BA = G_BA.to(device)
    D_A = D_A.to(device)
    D_B = D_B.to(device)

    optimizer_G = optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=lr, betas=(beta1, 0.999)
    )
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(beta1, 0.999))

    scheduler_G = optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=lambda epoch: 1.0 - max(0, epoch - num_epochs // 2) / (num_epochs // 2)
    )

    scheduler_D_A = optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=lambda epoch: 1.0 - max(0, epoch - num_epochs // 2) / (num_epochs // 2)
    )

    scheduler_D_B = optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=lambda epoch: 1.0 - max(0, epoch - num_epochs // 2) / (num_epochs // 2)
    )

    fake_A_pool = ImagePool(pool_size=50)
    fake_B_pool = ImagePool(pool_size=50)

    history = {
        'G_loss': [],
        'D_A_loss': [],
        'D_B_loss': [],
        'cycle_loss': [],
        'identity_loss': []
    }

    for epoch in range(num_epochs):
        G_AB.train()
        G_BA.train()
        D_A.train()
        D_B.train()

        epoch_G_loss = 0
        epoch_D_A_loss = 0
        epoch_D_B_loss = 0
        epoch_cycle_loss = 0
        epoch_identity_loss = 0

        data_iter_A = iter(train_loader_A)
        data_iter_B = iter(train_loader_B)

        num_batches: int = min(len(train_loader_A), len(train_loader_B))
        pbar: tqdm[int] = tqdm(range(num_batches), desc=f'Epoch {epoch+1}/{num_epochs}')

        for i in pbar:
            try:
                real_A = next(data_iter_A).to(device)
                real_B = next(data_iter_B).to(device)
            except StopIteration:
                break

            batch_size = min(real_A.size(0), real_B.size(0))
            real_A = real_A[:batch_size]
            real_B = real_B[:batch_size]

            optimizer_G.zero_grad()

            loss_identity_A = identity_loss(G_BA, real_A) * lambda_A * lambda_identity
            loss_identity_B = identity_loss(G_AB, real_B) * lambda_B * lambda_identity
            loss_identity_total = loss_identity_A + loss_identity_B

            fake_B = G_AB(real_A)
            loss_GAN_AB = generator_loss(D_B, fake_B)

            fake_A = G_BA(real_B)
            loss_GAN_BA = generator_loss(D_A, fake_A)

            recovered_A = G_BA(fake_B)
            loss_cycle_A = cycle_consistency_loss(real_A, recovered_A) * lambda_A

            recovered_B = G_AB(fake_A)
            loss_cycle_B = cycle_consistency_loss(real_B, recovered_B) * lambda_B

            loss_cycle_total = loss_cycle_A + loss_cycle_B

            loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_total + loss_identity_total

            loss_G.backward()
            optimizer_G.step()

            optimizer_D_A.zero_grad()

            fake_A_pooled = fake_A_pool.query(fake_A.detach())
            loss_D_A = discriminator_loss(D_A, real_A, fake_A_pooled)

            loss_D_A.backward()
            optimizer_D_A.step()

            optimizer_D_B.zero_grad()

            fake_B_pooled = fake_B_pool.query(fake_B.detach())
            loss_D_B = discriminator_loss(D_B, real_B, fake_B_pooled)

            loss_D_B.backward()
            optimizer_D_B.step()

            epoch_G_loss += loss_G.item()
            epoch_D_A_loss += loss_D_A.item()
            epoch_D_B_loss += loss_D_B.item()
            epoch_cycle_loss += loss_cycle_total.item()
            epoch_identity_loss += loss_identity_total.item()

            pbar.set_postfix({
                'G': f'{loss_G.item():.3f}',
                'D_A': f'{loss_D_A.item():.3f}',
                'D_B': f'{loss_D_B.item():.3f}'
            })

        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        history['G_loss'].append(epoch_G_loss / num_batches)
        history['D_A_loss'].append(epoch_D_A_loss / num_batches)
        history['D_B_loss'].append(epoch_D_B_loss / num_batches)
        history['cycle_loss'].append(epoch_cycle_loss / num_batches)
        history['identity_loss'].append(epoch_identity_loss / num_batches)

        print(f'\nEpoch {epoch+1} - G: {history["G_loss"][-1]:.4f}, '
              f'D_A: {history["D_A_loss"][-1]:.4f}, D_B: {history["D_B_loss"][-1]:.4f}, '
              f'Cycle: {history["cycle_loss"][-1]:.4f}, Identity: {history["identity_loss"][-1]:.4f}')

    return history


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None) -> None:
        self.root_dir: Any = root_dir
        self.transform = transform
        self.images = []

        for fname in os.listdir(root_dir):
            if fname.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.images.append(os.path.join(root_dir, fname))

        print(f"Loaded {len(self.images)} images from {root_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx % len(self.images)]
        image: Image.Image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

cyclegan_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def visualize_samples(images, titles=None, figsize=(15, 5), fig=None, axes=None) -> None:
    n: int = len(images)
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for i, (img, ax) in enumerate(zip(images, axes)):
        if isinstance(img, torch.Tensor):
            img = img.cpu().detach()
            if img.dim() == 4:
                img = img[0]
            img = img.permute(1, 2, 0).numpy()
            img = (img + 1) / 2
            img = np.clip(img, 0, 1)

        ax.imshow(img)
        ax.axis('off')
        if titles and i < len(titles):
            ax.set_title(titles[i])

    plt.tight_layout()


def test_cyclegan(G_AB, G_BA, test_loader_A, test_loader_B, device='cpu', num_samples=5) -> None:
    G_AB.eval()
    G_BA.eval()

    with torch.no_grad():
        real_A = next(iter(test_loader_A))[:num_samples].to(device)
        fake_B = G_AB(real_A)
        recovered_A = G_BA(fake_B)

        real_B = next(iter(test_loader_B))[:num_samples].to(device)
        fake_A = G_BA(real_B)
        recovered_B = G_AB(fake_A)

    for i in range(num_samples):
        visualize_samples(
            [real_A[i], fake_B[i], recovered_A[i]],
            titles=['Real A', 'Fake B', 'Recovered A']
        )

    for i in range(num_samples):
        visualize_samples(
            [real_B[i], fake_A[i], recovered_B[i]],
            titles=['Real B', 'Fake A', 'Recovered B']
        )

def plot_training_history(history) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(history['G_loss'])
    axes[0, 0].set_title('Generator Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)

    axes[0, 1].plot(history['D_A_loss'], label='D_A')
    axes[0, 1].plot(history['D_B_loss'], label='D_B')
    axes[0, 1].set_title('Discriminator Losses')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(history['cycle_loss'])
    axes[1, 0].set_title('Cycle Consistency Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)

    axes[1, 1].plot(history['identity_loss'])
    axes[1, 1].set_title('Identity Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def evaluate_anomaly_detection(normal_scores, anomaly_scores) -> tuple[float | floating[_16Bit] | floating[_32Bit] | float64, ndarray[tuple[Any, ...], dtype[Any]], ndarray[tuple[Any, ...], dtype[Any]]]:
    y_true: np.ndarray[tuple[Any, ...], np.dtype[np.float64]] = np.concatenate([
        np.zeros(len(normal_scores)),
        np.ones(len(anomaly_scores))
    ])

    y_scores = np.concatenate([normal_scores, anomaly_scores])

    auroc: float | np.floating[np._16Bit] | np.floating[np._32Bit] | np.float64 = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    return auroc, fpr, tpr

def plot_roc_curve(fpr, tpr, auroc) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUROC = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Anomaly Detection')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_score_distributions(normal_scores, anomaly_scores) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(normal_scores, bins=50, alpha=0.6, label='Normal', density=True)
    plt.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', density=True)
    plt.xlabel('Anomaly Score (NLL)')
    plt.ylabel('Density')
    plt.title('Distribution of Anomaly Scores')
    plt.legend()
    plt.grid(True)
    plt.show()


from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def calculate_anomaly_scores(model, test_loader, device='cpu'):
    model.eval()
    anomaly_scores = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Computing anomaly scores'):
            batch = batch.to(device)
            _, log_prob = model.forward(batch)
            nll = -log_prob
            anomaly_scores.extend(nll.cpu().numpy())

    return np.array(anomaly_scores)

def evaluate_anomaly_detection(normal_scores, anomaly_scores) -> tuple[float | floating[_16Bit] | floating[_32Bit] | float64, ndarray[tuple[Any, ...], dtype[Any]], ndarray[tuple[Any, ...], dtype[Any]]]:
    y_true: np.ndarray[tuple[Any, ...], np.dtype[np.float64]] = np.concatenate([
        np.zeros(len(normal_scores)),
        np.ones(len(anomaly_scores))
    ])

    y_scores = np.concatenate([normal_scores, anomaly_scores])

    auroc: float | np.floating[np._16Bit] | np.floating[np._32Bit] | np.float64 = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    return auroc, fpr, tpr

def plot_roc_curve(fpr, tpr, auroc) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUROC = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Anomaly Detection')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_score_distributions(normal_scores, anomaly_scores) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(normal_scores, bins=50, alpha=0.6, label='Normal', density=True)
    plt.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', density=True)
    plt.xlabel('Anomaly Score (NLL)')
    plt.ylabel('Density')
    plt.title('Distribution of Anomaly Scores')
    plt.legend()
    plt.grid(True)
    plt.show()


from torch.utils.data import Dataset
from PIL import Image
import os

class CapsuleDataset(Dataset):
    def __init__(self, root_dir, transform=None, img_size=128) -> None:
        self.root_dir: Any = root_dir
        self.transform = transform
        self.img_size: int = img_size
        self.images = []

        for fname: str in os.listdir(root_dir):
            if fname.endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(root_dir, fname))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image: Image.Image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image.view(-1)


dataset_path = 'capsule/train/good'

if not os.path.isdir(dataset_path):
    print("Downloading MVTec Capsule dataset using kagglehub...")

    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub...")
        import sys
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'kagglehub', '-q'], check=True)
        import kagglehub

    download_path = kagglehub.dataset_download("ipythonx/mvtec-ad")
    print(f"✓ Dataset downloaded to: {download_path}")

    capsule_source: str = os.path.join(download_path, 'capsule')
    if os.path.exists(capsule_source):
        shutil.copytree(capsule_source, 'capsule')
        print("✓ Capsule dataset copied to working directory")
    else:
        print(f"Searching for capsule folder in {download_path}...")
        for root, dirs, files in os.walk(download_path):
            if 'capsule' in dirs:
                capsule_path: str = os.path.join(root, 'capsule')
                shutil.copytree(capsule_path, 'capsule')
                print(f"✓ Found and copied capsule dataset from {capsule_path}")
                break

    if os.path.isdir(dataset_path):
        print("✓ Dataset ready at capsule/train/good")
    else:
        print(f"⚠ Please manually copy capsule folder from {download_path} to current directory")
else:
    print("✓ Dataset already exists at capsule/train/good")



transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


train_dataset: CapsuleDataset = CapsuleDataset(
    root_dir='capsule/train/good',
    transform=transform,
    img_size=128
)

train_loader = DataLoader(
    train_dataset,
    batch_size=3,
    shuffle=True,
    num_workers=2
)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = 128 * 128 * 3
maf_model: MAF = MAF(
    input_dim=input_dim,
    num_blocks=7,
    hidden_dims=[512, 512]
)

print(f"Model parameters: {sum(p.numel() for p in maf_model.parameters()):,}")



losses = train_maf(
    model=maf_model,
    train_loader=train_loader,
    num_epochs=100,
    lr=0.0001,
    device=device
)

plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Negative Log Likelihood')
plt.title('MAF Training Loss')
plt.grid(True)
plt.show()



generated_images, gen_time = generate_images_maf(
    model=maf_model,
    num_images=5,
    img_size=128,
    device=device
)

print(f"Generation time: {gen_time:.2f} seconds ({gen_time/5:.2f} sec per image)")



fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, ax in enumerate(axes):
    img = generated_images[i].permute(1, 2, 0).cpu().numpy()
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.show()


def calculate_anomaly_scores(model, data_loader, device):
    """Calculate anomaly scores (negative log-likelihood) for a dataset."""
    model.eval()
    scores = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            _, log_prob = model.forward(batch)
            nll = -log_prob
            scores.extend(nll.cpu().numpy())

    return np.array(scores)


def evaluate_anomaly_detection(normal_scores, anomaly_scores) -> tuple[float | floating[_16Bit] | floating[_32Bit] | float64, ndarray[tuple[Any, ...], dtype[Any]], ndarray[tuple[Any, ...], dtype[Any]]]:
    """Evaluate anomaly detection performance using AUROC."""
    from sklearn.metrics import roc_auc_score, roc_curve

    y_true: np.ndarray[tuple[Any, ...], np.dtype[np.float64]] = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
    y_scores = np.concatenate([normal_scores, anomaly_scores])

    auroc: float | np.floating[np._16Bit] | np.floating[np._32Bit] | np.float64 = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    return auroc, fpr, tpr


def plot_roc_curve(fpr, tpr, auroc) -> None:
    """Plot ROC curve for anomaly detection evaluation."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUROC = {auroc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve for Anomaly Detection', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_score_distributions(normal_scores, anomaly_scores) -> None:
    """Plot distributions of anomaly scores for normal vs anomalous samples."""
    plt.figure(figsize=(10, 6))
    plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal Samples',
             density=True, color='blue', edgecolor='black')
    plt.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomalous Samples',
             density=True, color='red', edgecolor='black')
    plt.xlabel('Anomaly Score (Negative Log-Likelihood)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Anomaly Scores', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_samples(images, titles=None, fig=None, axes=None) -> None:
    """Visualize a batch of images."""
    if fig is None or axes is None:
        n_images: int = len(images)
        cols: int = min(5, n_images)
        rows: int = (n_images + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))

        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

    images = images.cpu() if hasattr(images, 'cpu') else images

    for i, img in enumerate(images):
        if i >= len(axes.flat):
            break

        row, col = divmod(i, axes.shape[1]) if len(axes.shape) > 1 else (0, i)

        if isinstance(img, torch.Tensor):
            if img.dim() == 4:
                img = img[0]
            if img.shape[0] == 1:
                img = img.squeeze(0)
                cmap = 'gray'
            elif img.shape[0] == 3:
                img = img.permute(1, 2, 0)
                img = (img + 1) / 2
                cmap = None
            else:
                img = img.permute(1, 2, 0) if img.dim() == 3 else img
                cmap = None
        else:
            img = img.squeeze() if img.ndim > 2 else img
            cmap: None | str = 'gray' if img.ndim == 2 else None

        axes[row, col].imshow(img, cmap=cmap)
        axes[row, col].axis('off')

        if titles and i < len(titles):
            axes[row, col].set_title(titles[i], fontsize=10)

    for i in range(len(images), len(axes.flat)):
        row, col = divmod(i, axes.shape[1]) if len(axes.shape) > 1 else (0, i)
        axes[row, col].axis('off')

    plt.tight_layout()
    if fig is None:
        plt.show()


test_normal_dataset: CapsuleDataset = CapsuleDataset(
    root_dir='capsule/test/good',
    transform=transform,
    img_size=128
)

test_anomaly_dataset: CapsuleDataset = CapsuleDataset(
    root_dir='capsule/test/crack',
    transform=transform,
    img_size=128
)

test_normal_loader = DataLoader(test_normal_dataset, batch_size=8, shuffle=False)
test_anomaly_loader = DataLoader(test_anomaly_dataset, batch_size=8, shuffle=False)

print("Calculating anomaly scores for normal test images...")
normal_scores = calculate_anomaly_scores(maf_model, test_normal_loader, device)

print("Calculating anomaly scores for anomalous test images...")
anomaly_scores = calculate_anomaly_scores(maf_model, test_anomaly_loader, device)

auroc, fpr, tpr = evaluate_anomaly_detection(normal_scores, anomaly_scores)
print(f"AUROC: {auroc:.4f}")

plot_roc_curve(fpr, tpr, auroc)
plot_score_distributions(normal_scores, anomaly_scores)

print(f"Normal images - Mean score: {normal_scores.mean():.4f}, Std: {normal_scores.std():.4f}")
print(f"Anomaly images - Mean score: {anomaly_scores.mean():.4f}, Std: {anomaly_scores.std():.4f}")




!wget https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip
!unzip horse2zebra.zip

train_dataset_A: ImageDataset = ImageDataset(
    root_dir='horse2zebra/trainA',
    transform=cyclegan_transform
)

train_dataset_B: ImageDataset = ImageDataset(
    root_dir='horse2zebra/trainB',
    transform=cyclegan_transform
)

test_dataset_A: ImageDataset = ImageDataset(
    root_dir='horse2zebra/testA',
    transform=cyclegan_transform
)

test_dataset_B: ImageDataset = ImageDataset(
    root_dir='horse2zebra/testB',
    transform=cyclegan_transform
)



train_loader_A = DataLoader(train_dataset_A, batch_size=16, shuffle=True, num_workers=2)
train_loader_B = DataLoader(train_dataset_B, batch_size=16, shuffle=True, num_workers=2)
test_loader_A = DataLoader(test_dataset_A, batch_size=5, shuffle=False)
test_loader_B = DataLoader(test_dataset_B, batch_size=5, shuffle=False)



print("Sample images from domain A:")
sample_A = next(iter(test_loader_A))
visualize_samples(sample_A[:5])

print("Sample images from domain B:")
sample_B = next(iter(test_loader_B))
visualize_samples(sample_B[:5])



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G_AB = Generator(input_nc=3, output_nc=3, ngf=64, num_residual_blocks=9)
G_BA = Generator(input_nc=3, output_nc=3, ngf=64, num_residual_blocks=9)
D_A = Discriminator(input_nc=3, ndf=64)
D_B = Discriminator(input_nc=3, ndf=64)

print(f"G_AB parameters: {sum(p.numel() for p in G_AB.parameters()):,}")
print(f"G_BA parameters: {sum(p.numel() for p in G_BA.parameters()):,}")
print(f"D_A parameters: {sum(p.numel() for p in D_A.parameters()):,}")
print(f"D_B parameters: {sum(p.numel() for p in D_B.parameters()):,}")

history = train_cyclegan(
    G_AB=G_AB,
    G_BA=G_BA,
    D_A=D_A,
    D_B=D_B,
    train_loader_A=train_loader_A,
    train_loader_B=train_loader_B,
    num_epochs=20,
    lr=0.0002,
    beta1=0.5,
    device=device
)

plot_training_history(history)

test_cyclegan(G_AB, G_BA, test_loader_A, test_loader_B, device)

torch.save(G_AB.state_dict(), 'G_AB_final.pth')
torch.save(G_BA.state_dict(), 'G_BA_final.pth')
torch.save(D_A.state_dict(), 'D_A_final.pth')
torch.save(D_B.state_dict(), 'D_B_final.pth')

import torch
import matplotlib.pyplot as plt
import os

def visualize_training_progression(epochs_to_check, dataloader_A, dataloader_B,
                                   device, checkpoint_dir='checkpoints') -> None:
    """
    Visualizes generation quality at specific epochs (Early, Mid, Late).

    Args:
        epochs_to_check (list): List of epoch numbers to visualize (e.g., [1, 50, 100]).
        dataloader_A: DataLoader for Domain A (e.g., Horse).
        dataloader_B: DataLoader for Domain B (e.g., Zebra).
        checkpoint_dir: Directory where models are saved.
    """
    real_A = next(iter(dataloader_A)).to(device)
    real_B = next(iter(dataloader_B)).to(device)

    real_A = real_A[0:1]
    real_B = real_B[0:1]

    plt.figure(figsize=(12, 4 * len(epochs_to_check)))

    for i, epoch in enumerate(epochs_to_check):
        netG_A = Generator(input_nc=3, output_nc=3).to(device)
        netG_B = Generator(input_nc=3, output_nc=3).to(device)

        try:
            netG_A.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'netG_A_epoch_{epoch}.pth')))
            netG_B.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'netG_B_epoch_{epoch}.pth')))
        except FileNotFoundError:
            print(f"Checkpoint for epoch {epoch} not found. Skipping.")
            continue

        netG_A.eval()
        netG_B.eval()

        with torch.no_grad():
            fake_B = netG_A(real_A)
            fake_A = netG_B(real_B)

        def denorm(img):
            return (img.squeeze().permute(1, 2, 0).cpu().numpy() + 1) / 2

        plt.subplot(len(epochs_to_check), 4, i*4 + 1)
        plt.imshow(denorm(real_A))
        plt.title(f'Epoch {epoch}\nReal A (Source)')
        plt.axis('off')

        plt.subplot(len(epochs_to_check), 4, i*4 + 2)
        plt.imshow(denorm(fake_B))
        plt.title(f'Epoch {epoch}\nGenerated B (Transformed)')
        plt.axis('off')

        plt.subplot(len(epochs_to_check), 4, i*4 + 3)
        plt.imshow(denorm(real_B))
        plt.title(f'Epoch {epoch}\nReal B (Source)')
        plt.axis('off')

        plt.subplot(len(epochs_to_check), 4, i*4 + 4)
        plt.imshow(denorm(fake_A))
        plt.title(f'Epoch {epoch}\nGenerated A (Transformed)')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

epochs_list: list[int] = [1, 10, 20]
visualize_training_progression(epochs_list, train_loader_A, train_loader_B, device)

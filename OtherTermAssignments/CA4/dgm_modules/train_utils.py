"""Training helpers: loop, checkpointing, simple logging."""
import torch
from pathlib import Path

def train_loop(model, dataloader, optimizer, device, epochs=1, save_path=None):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for i, batch in enumerate(dataloader):
            x = batch[0].to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = ((out - x)**2).mean()
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"Epoch {epoch+1}/{epochs} — loss: {running/len(dataloader):.4f}")
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(p))

def visualize_denoising_process(intermediates, title="Denoising Process"):
    """Visualize the progressive denoising."""
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    n_steps = min(len(intermediates), 10)
    fig, axes = plt.subplots(1, n_steps, figsize=(2 * n_steps, 2))

    step_indices = np.linspace(0, len(intermediates) - 1, n_steps, dtype=int)

    for idx, step_idx in enumerate(step_indices):
        img = intermediates[step_idx][0].permute(1, 2, 0).cpu().numpy()
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(f'Step {step_idx}')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualize_samples(samples, title="Generated Samples", save_path=None):
    """Helper to visualize a batch of image samples (expects [B,C,H,W] tensors)."""
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    fig, axes = plt.subplots(2, min(8, samples.shape[0]//2 or 1), figsize=(16, 4))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < len(samples):
            img = samples[i].permute(1, 2, 0).cpu().numpy()
            img = (img + 1) / 2
            img = np.clip(img, 0, 1)
            ax.imshow(img)
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


class DDPMTrainer:
    """DDPM training pipeline helper copied from the notebook.

    Usage:
        trainer = DDPMTrainer(model, scheduler, optimizer, device)
        trainer.train(dataloader, num_epochs, save_path)
    """
    def __init__(self, model, scheduler, optimizer, device):
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.device = device
        self.loss_history = []

    def train_step(self, x_0):
        """Single training step for DDPM. Expects `x_0` on the trainer device."""
        import torch.nn.functional as F

        self.model.train()
        batch_size = x_0.shape[0]

        t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=self.device)
        x_t, noise = self.scheduler.perturb_input(x_0, t)
        noise_pred = self.model(x_t, t)
        loss = F.mse_loss(noise_pred, noise)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, dataloader, epoch):
        epoch_loss = 0
        from tqdm.auto import tqdm
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)
            loss = self.train_step(images)
            epoch_loss += loss
            pbar.set_postfix({'loss': f'{loss:.4f}'})

        avg_loss = epoch_loss / len(dataloader)
        self.loss_history.append(avg_loss)
        return avg_loss

    def train(self, dataloader, num_epochs, save_path=None):
        print(f"Starting training for {num_epochs} epochs...")
        for epoch in range(1, num_epochs + 1):
            avg_loss = self.train_epoch(dataloader, epoch)
            print(f"Epoch {epoch}/{num_epochs} - Average Loss: {avg_loss:.4f}")
            if save_path and epoch % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                }, f"{save_path}/checkpoint_epoch_{epoch}.pt")
        print("Done: Training complete!")
        return self.loss_history

    def plot_loss(self, save_path=None):
        import matplotlib.pyplot as plt
        from pathlib import Path
        plt.figure(figsize=(10, 4))
        plt.plot(self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('DDPM Training Loss')
        plt.grid(True)
        if save_path is None and 'REPORT_FIG_DIR' in globals():
            save_path = REPORT_FIG_DIR / "ddpm_training_loss.png"
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Done: Saved: {save_path}")
        plt.show()

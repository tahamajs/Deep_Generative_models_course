import torch
from tqdm import tqdm
from typing import Tuple

from .losses import vae_loss


def train_epoch(model, train_loader, optimizer, device, beta=1.0):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    pbar = tqdm(train_loader, desc="Training")
    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        x_recon, mu, log_var = model(data)
        loss, recon_loss, kl_loss = vae_loss(x_recon, data, mu, log_var, beta)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        pbar.set_postfix({
            "loss": loss.item() / len(data),
            "recon": recon_loss.item() / len(data),
            "kl": kl_loss.item() / len(data),
        })
    n_samples = len(train_loader.dataset)
    return total_loss / n_samples, total_recon / n_samples, total_kl / n_samples


def validate(model, val_loader, device, beta=1.0):
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            x_recon, mu, log_var = model(data)
            loss, recon_loss, kl_loss = vae_loss(x_recon, data, mu, log_var, beta)
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
    n_samples = len(val_loader.dataset)
    return total_loss / n_samples, total_recon / n_samples, total_kl / n_samples


def train_vae(
    model,
    train_loader,
    val_loader,
    device,
    epochs=50,
    lr=0.001,
    beta=1.0,
    save_path="vae_model.pth",
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    history = {
        "train_loss": [],
        "train_recon": [],
        "train_kl": [],
        "val_loss": [],
        "val_recon": [],
        "val_kl": [],
    }
    best_val_loss = float("inf")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device, beta
        )
        val_loss, val_recon, val_kl = validate(model, val_loader, device, beta)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["train_recon"].append(train_recon)
        history["train_kl"].append(train_kl)
        history["val_loss"].append(val_loss)
        history["val_recon"].append(val_recon)
        history["val_kl"].append(val_kl)
        print(f"  Train - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, KL: {val_kl:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "beta": beta,
                },
                save_path,
            )
            print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
    return history

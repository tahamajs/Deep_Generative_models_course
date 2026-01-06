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

def save_checkpoint(model, path):
    import torch
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

#!/usr/bin/env python3
"""DDPM/DDIM execution script with quick/full presets and report outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Run DDPM/DDIM training and save report figures.")
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--ddim-steps", type=int, default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--save-model", action="store_true")
    return parser.parse_args()


def save_schedule_plot(scheduler, out_path: Path, show: bool):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(scheduler.betas.cpu().numpy())
    plt.title("beta_t (Linear Schedule)")
    plt.xlabel("t")
    plt.ylabel("beta_t")

    plt.subplot(1, 3, 2)
    plt.plot(scheduler.alphas_cumprod.cpu().numpy())
    plt.title("alpha_bar_t")
    plt.xlabel("t")
    plt.ylabel("alpha_bar_t")

    plt.subplot(1, 3, 3)
    plt.plot(scheduler.sqrt_one_minus_alphas_cumprod.cpu().numpy())
    plt.title("sqrt(1 - alpha_bar_t)")
    plt.xlabel("t")
    plt.ylabel("noise level")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def save_ddpm_ddim_comparison(ddpm_samples, ddim_samples, out_path: Path, show: bool):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for idx, (samples, title) in enumerate(
        [(ddpm_samples, "DDPM"), (ddim_samples, "DDIM")]
    ):
        grid = samples[:16].detach().cpu()
        rows = []
        for r in range(4):
            row_imgs = []
            for c in range(4):
                img = grid[r * 4 + c].permute(1, 2, 0).numpy()
                img = np.clip((img + 1) / 2, 0, 1)
                row_imgs.append(img)
            rows.append(np.concatenate(row_imgs, axis=1))
        axes[idx].imshow(np.concatenate(rows, axis=0))
        axes[idx].set_title(title)
        axes[idx].axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    args = parse_args()
    if not args.show_plots:
        matplotlib.use("Agg")

    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    from ddpm import (
        DDIMSampler,
        DDPMTrainer,
        DDPM_CONFIG,
        DDPMSampler,
        DDPMScheduler,
        UNet,
        device,
        visualize_denoising_process,
        visualize_forward_process,
        visualize_samples,
    )
    from utils import REPORT_FIG_DIR, set_report_fig_dir

    out_dir = args.output_dir.resolve() if args.output_dir else REPORT_FIG_DIR
    set_report_fig_dir(out_dir)

    epochs = args.epochs
    if epochs is None:
        epochs = 1 if args.preset == "quick" else DDPM_CONFIG["num_epochs"]
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 16 if args.preset == "quick" else DDPM_CONFIG["batch_size"]
    ddim_steps = args.ddim_steps
    if ddim_steps is None:
        ddim_steps = 25 if args.preset == "quick" else 50

    cfg = dict(DDPM_CONFIG)
    cfg["batch_size"] = batch_size
    if args.preset == "quick":
        cfg["num_timesteps"] = 200

    transform = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    try:
        dataset = datasets.MNIST(
            root="./data",
            train=True,
            download=args.allow_download,
            transform=transform,
        )
    except RuntimeError as exc:
        if args.allow_download:
            raise
        print("MNIST not found locally; using torchvision FakeData fallback for quick run.")
        dataset = datasets.FakeData(
            size=cfg["batch_size"] * (8 if args.preset == "quick" else 64),
            image_size=(1, cfg["image_size"], cfg["image_size"]),
            num_classes=10,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
        )
    if args.preset == "quick":
        max_items = min(len(dataset), cfg["batch_size"] * 8)
        dataset = Subset(dataset, range(max_items))
    train_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    scheduler = DDPMScheduler(
        num_timesteps=cfg["num_timesteps"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
        device=device,
    )
    save_schedule_plot(scheduler, out_dir / "forward_diffusion_viz.png", args.show_plots)
    viz_steps = np.linspace(0, cfg["num_timesteps"] - 1, 6, dtype=int).tolist()
    visualize_forward_process(
        scheduler,
        img_size=(cfg["channels"], cfg["image_size"], cfg["image_size"]),
        timesteps_to_show=viz_steps,
        save_path=out_dir / "main_1_3_Visualize_Forward_Diffusion_Process_cell013_out02.png",
        show=args.show_plots,
    )

    model = UNet(
        c_in=cfg["channels"],
        c_out=cfg["channels"],
        time_dim=256,
        base_channels=32 if args.preset == "quick" else 64,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    trainer = DDPMTrainer(model, scheduler, optimizer, device)

    print(f"Training DDPM for {epochs} epoch(s), batch_size={cfg['batch_size']}")
    trainer.train(train_loader, num_epochs=epochs)
    trainer.plot_loss(save_path=out_dir / "ddpm_training_loss.png", show=args.show_plots)

    ddpm_sampler = DDPMSampler(model, scheduler, device)
    ddim_sampler = DDIMSampler(model, scheduler, device)
    img_size = (cfg["channels"], cfg["image_size"], cfg["image_size"])

    ddpm_samples, ddpm_intermediates = ddpm_sampler.sample(
        batch_size=args.num_samples,
        img_size=img_size,
        show_progress=not args.show_plots,
    )
    ddim_samples, ddim_intermediates = ddim_sampler.sample(
        batch_size=args.num_samples,
        img_size=img_size,
        num_inference_steps=ddim_steps,
        eta=0.0,
        show_progress=not args.show_plots,
    )

    visualize_samples(
        ddpm_samples,
        title="DDPM Samples",
        save_path=out_dir / "ddpm_samples_grid.png",
        show=args.show_plots,
    )
    visualize_samples(
        ddim_samples,
        title=f"DDIM Samples ({ddim_steps} steps)",
        save_path=out_dir / "ddim_samples_grid.png",
        show=args.show_plots,
    )
    save_ddpm_ddim_comparison(ddpm_samples, ddim_samples, out_dir / "ddpm_ddim_samples.png", args.show_plots)
    visualize_denoising_process(ddpm_intermediates, "DDPM Denoising", show=args.show_plots)
    visualize_denoising_process(ddim_intermediates, "DDIM Denoising", show=args.show_plots)

    if args.save_model:
        model_path = out_dir / "ddpm_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_history": trainer.loss_history,
                "config": cfg,
            },
            model_path,
        )
        print(f"Saved model: {model_path}")

    print("DDPM pipeline complete")


if __name__ == "__main__":
    main()

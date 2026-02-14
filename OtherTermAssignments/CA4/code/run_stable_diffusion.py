#!/usr/bin/env python3
"""Optional DreamBooth stage with graceful skip behavior."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib


def parse_args():
    parser = argparse.ArgumentParser(description="Run optional DreamBooth stage.")
    parser.add_argument("--mode", choices=["auto", "run", "skip"], default="auto")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--show-plots", action="store_true")
    return parser.parse_args()


def find_instance_dir(default_path: Path):
    candidates = [
        default_path,
        Path(__file__).resolve().parent / "notebook" / "sdrf" / "instance_data",
    ]
    for cand in candidates:
        if cand.exists() and any(cand.glob("*.jpg")):
            return cand
    return None


def export_instance_preview(instance_dir: Path, output_dir: Path):
    from PIL import Image

    images = sorted(
        list(instance_dir.glob("*.png"))
        + list(instance_dir.glob("*.jpg"))
        + list(instance_dir.glob("*.jpeg"))
        + list(instance_dir.glob("*.webp"))
    )
    if not images:
        return
    img = Image.open(images[0]).convert("RGB")
    out = output_dir / "dreambooth_instance_sample.png"
    img.save(out)
    print(f"Saved figure: {out}")


def save_generated_grid(images, output_path: Path):
    import math
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(images)
    cols = min(4, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            ax.imshow(images[i])
        ax.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    plt.close(fig)


def main():
    args = parse_args()
    if not args.show_plots:
        matplotlib.use("Agg")

    from stable_diffusion import (
        DIFFUSERS_AVAILABLE,
        DREAMBOOTH_CONFIG,
        DreamBoothDataset,
        DreamBoothInference,
        DreamBoothTrainer,
        collate_fn,
        generate_class_images,
    )
    import torch
    from torch.utils.data import DataLoader
    from utils import REPORT_FIG_DIR, set_report_fig_dir

    output_dir = args.output_dir.resolve() if args.output_dir else REPORT_FIG_DIR
    set_report_fig_dir(output_dir)

    if args.mode == "skip":
        print("DreamBooth stage skipped by user option.")
        return

    if not DIFFUSERS_AVAILABLE:
        msg = "DreamBooth dependencies are unavailable."
        if args.mode == "run":
            raise RuntimeError(msg)
        print(f"{msg} Skipping DreamBooth stage.")
        return

    instance_dir = find_instance_dir(Path(DREAMBOOTH_CONFIG["instance_data_root"]).resolve())
    if instance_dir is None:
        msg = "No instance images found for DreamBooth."
        if args.mode == "run":
            raise RuntimeError(msg)
        print(f"{msg} Skipping DreamBooth stage.")
        return

    class_dir = Path(DREAMBOOTH_CONFIG["class_data_root"]).resolve()
    class_dir.mkdir(parents=True, exist_ok=True)
    export_instance_preview(instance_dir, output_dir)

    try:
        generate_class_images(
            class_prompt=DREAMBOOTH_CONFIG["class_prompt"],
            class_data_root=class_dir,
            num_class_images=DREAMBOOTH_CONFIG["num_class_images"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    except Exception as exc:
        if args.mode == "run":
            raise
        print(f"Class image generation failed ({exc}); skipping DreamBooth stage.")
        return

    try:
        trainer = DreamBoothTrainer(
            lora_rank=DREAMBOOTH_CONFIG["lora_rank"],
            learning_rate=DREAMBOOTH_CONFIG["learning_rate"],
            prior_preservation=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        dataset = DreamBoothDataset(
            instance_data_root=instance_dir,
            instance_prompt=DREAMBOOTH_CONFIG["instance_prompt"],
            tokenizer=trainer.tokenizer,
            class_data_root=class_dir,
            class_prompt=DREAMBOOTH_CONFIG["class_prompt"],
            size=512,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda x: collate_fn(x, with_prior_preservation=True),
        )
        trainer.train(dataloader, num_epochs=args.num_epochs)

        lora_dir = output_dir / "dreambooth_lora"
        trainer.save_lora_weights(lora_dir)

        inference = DreamBoothInference(lora_weights_path=lora_dir)
        prompts = [
            "a photo of sks dog on the moon",
            "a photo of sks dog in a bucket",
            "a photo of sks dog wearing a hat",
            "a painting of sks dog in the style of Van Gogh",
        ]
        images = []
        for prompt in prompts:
            images.extend(inference.generate(prompt, num_images=1))
        if images:
            save_generated_grid(images, output_dir / "dreambooth_generated_grid.png")
        print("DreamBooth stage complete")
    except Exception as exc:
        if args.mode == "run":
            raise
        print(f"DreamBooth stage failed ({exc}); skipped in auto mode.")


if __name__ == "__main__":
    main()

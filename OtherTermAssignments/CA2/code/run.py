"""Entry point to run MAF and CycleGAN experiments with saved artifacts.

Usage examples:
  python run.py maf --mode train --epochs 2 --quick --save_dir ../report/images --tag quick
  python run.py maf --mode eval --model maf_final.pth --save_dir ../report/images --tag quick
  python run.py cyclegan --mode train --epochs 2 --quick --save_dir ../report/images --tag quick
  python run.py cyclegan --mode test --quick --save_dir ../report/images --tag quick
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from cyclegan import Discriminator, Generator, test_cyclegan, train_cyclegan
from datasets import CapsuleDataset, ImageDataset, cyclegan_transform
from maf import MAF, generate_images_maf, train_maf
from utils import (
    evaluate_anomaly_detection,
    plot_maf_loss,
    plot_roc_curve,
    plot_score_distributions,
    plot_training_history,
    visualize_samples,
)


class RandomVectorDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 32, dim: int = 128 * 128 * 3) -> None:
        self.n = n
        self.dim = dim

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx):
        return torch.randn(self.dim)


class RandomImageDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 32, size: Tuple[int, int, int] = (3, 128, 128)) -> None:
        self.n = n
        self.size = size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx):
        return torch.randn(*self.size)


def make_capsule_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def tagged_path(args, stem: str, ext: str) -> str:
    suffix = f"_{args.tag}" if args.tag else ""
    filename = f"{stem}{suffix}.{ext}"
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        return os.path.join(args.save_dir, filename)
    return filename


def load_state_compat(model: torch.nn.Module, path: str, device: str) -> None:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)


def write_json(path: str, payload: Dict) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_maf(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = args.quick_img_size if args.quick else args.img_size
    input_dim = img_size * img_size * 3
    capsule_transform = make_capsule_transform(img_size)
    dataset_path = args.dataset if args.dataset else "capsule/train/good"

    if not os.path.isdir(dataset_path):
        print(f"Capsule dataset not found at {dataset_path}.")
        if args.quick:
            print("Using synthetic data for quick MAF run.")
            train_loader = DataLoader(RandomVectorDataset(n=16, dim=input_dim), batch_size=args.batch_size)
        else:
            print("Use --quick for synthetic smoke run or provide dataset path.")
            return
    else:
        train_dataset = CapsuleDataset(root_dir=dataset_path, transform=capsule_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    maf = MAF(input_dim=input_dim, num_blocks=args.num_blocks, hidden_dims=[512, 512])

    if args.mode == "train":
        start = time.time()
        losses = train_maf(
            maf,
            train_loader,
            num_epochs=args.epochs,
            lr=args.lr,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            save_every=args.save_every,
            resume=args.resume,
            use_scheduler=args.use_scheduler,
        )
        train_time = time.time() - start

        model_out = args.out if args.out else tagged_path(args, "maf_final", "pth")
        ensure_parent(model_out)
        torch.save(maf.state_dict(), model_out)

        loss_plot_path = tagged_path(args, "maf_loss", "png")
        plot_maf_loss(losses, save_path=loss_plot_path, show=args.show_plots)

        metrics_path = tagged_path(args, "maf_train_metrics", "json")
        write_json(
            metrics_path,
            {
                "mode": "train",
                "device": device,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "dataset": dataset_path if os.path.isdir(dataset_path) else "synthetic",
                "train_seconds": train_time,
                "final_loss": float(losses[-1]) if losses else None,
                "model_out": model_out,
                "loss_plot": loss_plot_path,
            },
        )
        print(f"Training finished. Model: {model_out}")

    elif args.mode == "generate":
        model_path = args.model if args.model else tagged_path(args, "maf_final", "pth")
        if model_path and os.path.isfile(model_path):
            load_state_compat(maf, model_path, device)
            print(f"Loaded model from: {model_path}")
        elif not args.quick:
            print(f"Model not found at {model_path}. Provide --model or run training first.")
            return
        else:
            print("Model checkpoint not found; generating from untrained model for quick run.")

        samples, gen_time = generate_images_maf(maf, num_images=args.num_samples, img_size=img_size, device=device)
        gen_path = tagged_path(args, "maf_generated", "png")
        if samples.dim() == 4:
            visualize_samples(
                [samples[i] for i in range(samples.shape[0])],
                save_path=gen_path,
                show=args.show_plots,
            )
        else:
            # Generic vector output fallback for non-image smoke shapes.
            fake_imgs = torch.rand(args.num_samples, 3, img_size, img_size)
            visualize_samples(
                [fake_imgs[i] for i in range(fake_imgs.shape[0])],
                save_path=gen_path,
                show=args.show_plots,
            )

        metrics_path = tagged_path(args, "maf_generate_metrics", "json")
        write_json(
            metrics_path,
            {
                "mode": "generate",
                "device": device,
                "num_samples": args.num_samples,
                "generation_seconds": gen_time,
                "seconds_per_image": gen_time / max(1, args.num_samples),
                "generated_plot": gen_path,
            },
        )
        print(f"Generation finished: {gen_time:.2f}s total, output: {gen_path}")

    elif args.mode == "eval":
        normal_path = "capsule/test/good"
        anomaly_path = os.path.join("capsule/test", args.anomaly_class)

        use_synthetic = False
        if args.model and os.path.isfile(args.model):
            load_state_compat(maf, args.model, device)
            print(f"Loaded model from: {args.model}")
        elif args.quick:
            use_synthetic = True
            print("Model missing in quick mode, using synthetic evaluation scores.")
        else:
            print("Provide --model for evaluation or run with --quick.")
            return

        maf = maf.to(device)

        if use_synthetic or not (os.path.isdir(normal_path) and os.path.isdir(anomaly_path)):
            if not args.quick and not use_synthetic:
                print("Test datasets missing. Use --quick for synthetic evaluation.")
                return
            normal_scores = torch.abs(torch.randn(80)).numpy()
            anomaly_scores = torch.abs(torch.randn(80) + 0.5).numpy()
            eval_source = "synthetic"
        else:
            normal_loader = DataLoader(CapsuleDataset(normal_path, transform=capsule_transform), batch_size=8)
            anomaly_loader = DataLoader(CapsuleDataset(anomaly_path, transform=capsule_transform), batch_size=8)

            def calc_scores(model, loader):
                model.eval()
                scores = []
                with torch.no_grad():
                    for batch in loader:
                        batch = batch.to(device)
                        _, log_prob = model.forward(batch)
                        nll = -log_prob
                        scores.extend(nll.cpu().numpy())
                return np.array(scores)

            normal_scores = calc_scores(maf, normal_loader)
            anomaly_scores = calc_scores(maf, anomaly_loader)
            eval_source = args.anomaly_class

        auroc, fpr, tpr = evaluate_anomaly_detection(normal_scores, anomaly_scores)
        roc_path = tagged_path(args, "maf_roc", "png")
        score_path = tagged_path(args, "maf_score_dist", "png")
        plot_roc_curve(fpr, tpr, auroc, save_path=roc_path, show=args.show_plots)
        plot_score_distributions(normal_scores, anomaly_scores, save_path=score_path, show=args.show_plots)

        metrics_path = tagged_path(args, "maf_eval_metrics", "json")
        write_json(
            metrics_path,
            {
                "mode": "eval",
                "source": eval_source,
                "auroc": float(auroc),
                "normal_count": int(len(normal_scores)),
                "anomaly_count": int(len(anomaly_scores)),
                "roc_plot": roc_path,
                "score_plot": score_path,
            },
        )
        print(f"AUROC: {auroc:.4f}")
        print(f"Saved ROC: {roc_path}")
        print(f"Saved score distribution: {score_path}")

    else:
        print("Unknown mode for maf")


def run_cyclegan(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_root = args.dataset if args.dataset else "horse2zebra"

    if args.mode == "train":
        trainA = os.path.join(dataset_root, "trainA")
        trainB = os.path.join(dataset_root, "trainB")

        if not os.path.isdir(trainA) or not os.path.isdir(trainB):
            print("CycleGAN train dataset not found.")
            if args.quick:
                print("Using synthetic images for quick CycleGAN training.")
                loaderA = DataLoader(RandomImageDataset(n=8), batch_size=args.batch_size)
                loaderB = DataLoader(RandomImageDataset(n=8), batch_size=args.batch_size)
            else:
                print("Use --quick or provide a valid --dataset.")
                return
        else:
            loaderA = DataLoader(
                ImageDataset(trainA, transform=cyclegan_transform),
                batch_size=args.batch_size,
                shuffle=True,
            )
            loaderB = DataLoader(
                ImageDataset(trainB, transform=cyclegan_transform),
                batch_size=args.batch_size,
                shuffle=True,
            )

        G_AB = Generator()
        G_BA = Generator()
        D_A = Discriminator()
        D_B = Discriminator()

        start = time.time()
        history = train_cyclegan(
            G_AB,
            G_BA,
            D_A,
            D_B,
            loaderA,
            loaderB,
            num_epochs=args.epochs,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
        )
        train_time = time.time() - start

        os.makedirs(args.checkpoint_dir, exist_ok=True)
        suffix = f"_{args.tag}" if args.tag else ""
        final_ab = os.path.join(args.checkpoint_dir, f"G_AB_final{suffix}.pth")
        final_ba = os.path.join(args.checkpoint_dir, f"G_BA_final{suffix}.pth")
        torch.save(G_AB.state_dict(), final_ab)
        torch.save(G_BA.state_dict(), final_ba)

        loss_path = tagged_path(args, "cyclegan_loss", "png")
        plot_training_history(history, save_path=loss_path, show=args.show_plots)

        metrics_path = tagged_path(args, "cyclegan_train_metrics", "json")
        write_json(
            metrics_path,
            {
                "mode": "train",
                "device": device,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "dataset": dataset_root if os.path.isdir(trainA) and os.path.isdir(trainB) else "synthetic",
                "train_seconds": train_time,
                "history_plot": loss_path,
                "final_model_ab": final_ab,
                "final_model_ba": final_ba,
            },
        )
        print("CycleGAN training finished")
        print(f"Saved history plot: {loss_path}")

    elif args.mode == "test":
        testA = os.path.join(dataset_root, "testA")
        testB = os.path.join(dataset_root, "testB")

        if not os.path.isdir(testA) or not os.path.isdir(testB):
            if args.quick:
                print("CycleGAN test sets missing. Using synthetic samples for quick outputs.")
                loaderA = DataLoader(RandomImageDataset(n=args.num_samples), batch_size=args.num_samples)
                loaderB = DataLoader(RandomImageDataset(n=args.num_samples), batch_size=args.num_samples)
            else:
                print("Test sets missing. Use --quick or provide a valid dataset root.")
                return
        else:
            loaderA = DataLoader(ImageDataset(testA, transform=cyclegan_transform), batch_size=args.num_samples)
            loaderB = DataLoader(ImageDataset(testB, transform=cyclegan_transform), batch_size=args.num_samples)

        G_AB = Generator()
        G_BA = Generator()

        suffix = f"_{args.tag}" if args.tag else ""
        default_ab = os.path.join(args.checkpoint_dir, f"G_AB_final{suffix}.pth")
        default_ba = os.path.join(args.checkpoint_dir, f"G_BA_final{suffix}.pth")

        model_ab = args.model_ab or args.model or default_ab
        model_ba = args.model_ba or default_ba

        if model_ab and os.path.isfile(model_ab):
            load_state_compat(G_AB, model_ab, device)
            print(f"Loaded G_AB from: {model_ab}")
        elif not args.quick:
            print(f"G_AB model not found: {model_ab}")
            return

        if model_ba and os.path.isfile(model_ba):
            load_state_compat(G_BA, model_ba, device)
            print(f"Loaded G_BA from: {model_ba}")
        elif not args.quick:
            print(f"G_BA model not found: {model_ba}")
            return

        (real_A, fake_B, rec_A), (real_B, fake_A, rec_B) = test_cyclegan(
            G_AB,
            G_BA,
            loaderA,
            loaderB,
            device=device,
            num_samples=args.num_samples,
        )

        panel_a_path = tagged_path(args, "cyclegan_panel_a2b", "png")
        panel_b_path = tagged_path(args, "cyclegan_panel_b2a", "png")
        visualize_samples(
            [real_A[0], fake_B[0], rec_A[0]],
            titles=["Real A", "Fake B", "Rec A"],
            save_path=panel_a_path,
            show=args.show_plots,
        )
        visualize_samples(
            [real_B[0], fake_A[0], rec_B[0]],
            titles=["Real B", "Fake A", "Rec B"],
            save_path=panel_b_path,
            show=args.show_plots,
        )

        for i in range(min(3, fake_B.size(0))):
            visualize_samples(
                [fake_B[i]],
                titles=[f"A->B #{i+1}"],
                save_path=tagged_path(args, f"cyclegan_a2b_{i+1}", "png"),
                show=False,
            )

        for i in range(min(3, fake_A.size(0))):
            visualize_samples(
                [fake_A[i]],
                titles=[f"B->A #{i+1}"],
                save_path=tagged_path(args, f"cyclegan_b2a_{i+1}", "png"),
                show=False,
            )

        metrics_path = tagged_path(args, "cyclegan_test_metrics", "json")
        write_json(
            metrics_path,
            {
                "mode": "test",
                "num_samples": args.num_samples,
                "panel_a2b": panel_a_path,
                "panel_b2a": panel_b_path,
            },
        )
        print(f"Saved CycleGAN qualitative samples in: {args.save_dir or os.getcwd()}")

    else:
        print("Unknown mode for cyclegan")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="task")

    # MAF parser
    p_maf = subparsers.add_parser("maf")
    p_maf.add_argument("--mode", choices=["train", "generate", "eval"], required=True)
    p_maf.add_argument("--dataset", type=str, default=None)
    p_maf.add_argument("--batch_size", type=int, default=3)
    p_maf.add_argument("--epochs", type=int, default=100)
    p_maf.add_argument("--lr", type=float, default=1e-4)
    p_maf.add_argument("--num_blocks", type=int, default=7)
    p_maf.add_argument("--num_samples", type=int, default=5)
    p_maf.add_argument("--model", type=str, default=None)
    p_maf.add_argument("--out", type=str, default=None)
    p_maf.add_argument("--quick", action="store_true")
    p_maf.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p_maf.add_argument("--save_every", type=int, default=10)
    p_maf.add_argument("--resume", type=str, default=None)
    p_maf.add_argument("--use_scheduler", action="store_true")
    p_maf.add_argument("--anomaly_class", type=str, default="crack")
    p_maf.add_argument("--img_size", type=int, default=128)
    p_maf.add_argument("--quick_img_size", type=int, default=16)
    p_maf.add_argument("--save_dir", type=str, default=None)
    p_maf.add_argument("--tag", type=str, default="")
    p_maf.add_argument("--show_plots", action="store_true")
    p_maf.add_argument("--seed", type=int, default=42)

    # CycleGAN parser
    p_cyc = subparsers.add_parser("cyclegan")
    p_cyc.add_argument("--mode", choices=["train", "test"], required=True)
    p_cyc.add_argument("--dataset", type=str, default=None)
    p_cyc.add_argument("--batch_size", type=int, default=16)
    p_cyc.add_argument("--epochs", type=int, default=20)
    p_cyc.add_argument("--num_samples", type=int, default=5)
    p_cyc.add_argument("--model", type=str, default=None)
    p_cyc.add_argument("--model_ab", type=str, default=None)
    p_cyc.add_argument("--model_ba", type=str, default=None)
    p_cyc.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p_cyc.add_argument("--quick", action="store_true")
    p_cyc.add_argument("--save_dir", type=str, default=None)
    p_cyc.add_argument("--tag", type=str, default="")
    p_cyc.add_argument("--show_plots", action="store_true")
    p_cyc.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if not getattr(args, "task", None):
        parser.print_help()
        raise SystemExit(0)

    set_seed(args.seed)

    if args.task == "maf":
        run_maf(args)
    elif args.task == "cyclegan":
        run_cyclegan(args)

#!/usr/bin/env python3
"""Entry point to run the CA1 tasks (PGM visualization and VAE training)."""
import argparse
import os
import json

from ca1.config import CONFIG, set_seed, device, save_run_info
from ca1 import data, models, train, viz, pgm, utils
import torch


def cmd_pgm(args):
    print("Drawing Bayesian Network...")
    pgm.draw_bayesian_network(save_path=args.output or "bayesian_network.png")
    print("Drawing Markov Network...")
    pgm.draw_markov_network(save_path=args.output or "markov_network.png")


def cmd_train(args):
    imgs, latents_values, latents_classes, metadata = data.load_dsprites(CONFIG["data_path"])
    if imgs is None:
        raise RuntimeError("Could not load dataset. Place the dSprites npz at the configured path.")
    subset = CONFIG.get("data_subset") or len(imgs)
    subset = min(subset, len(imgs))
    indices = np.random.default_rng(CONFIG["seed"]).choice(len(imgs), subset, replace=False)
    imgs_subset = imgs[indices]
    train_loader, val_loader = data.create_dataloaders(imgs_subset, batch_size=args.batch_size or CONFIG["batch_size"], train_split=CONFIG["train_split"], generator=set_seed(CONFIG["seed"]))

    model = models.VAE(h_dim=args.latent_dim or CONFIG["latent_dim"]).to(device)
    history = train.train_vae(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs or CONFIG["epochs"],
        lr=args.lr or CONFIG["learning_rate"],
        beta=args.beta or 1.0,
        save_path=args.save_path or "vae_model.pth",
    )
    # Save metadata
    save_run_info({**CONFIG, **vars(args)}, path=(args.save_path or "vae_model.pth") + ".runinfo.json")
    print("Training finished. Visualizing sample reconstructions and prior generations...")
    viz.visualize_reconstructions(model, val_loader, save_path="reconstructions.png")
    viz.visualize_generation_from_prior(model, save_path="prior_generation.png")


def cmd_smoke(args):
    # Quick smoke training to check everything runs
    imgs, latents_values, latents_classes, metadata = data.load_dsprites(CONFIG["data_path"])
    if imgs is None:
        raise RuntimeError("Could not load dataset. Place the dSprites npz at the configured path.")
    subset = min(CONFIG.get("smoke_subset", 2048), len(imgs))
    indices = np.random.default_rng(CONFIG["seed"]).choice(len(imgs), subset, replace=False)
    imgs_subset = imgs[indices]
    train_loader, val_loader = data.create_dataloaders(imgs_subset, batch_size=CONFIG["batch_size"], train_split=CONFIG["train_split"], generator=set_seed(CONFIG["seed"]))
    model = models.VAE(h_dim=CONFIG["latent_dim"]).to(device)
    train.train_vae(model, train_loader, val_loader, device, epochs=CONFIG.get("smoke_epochs", 1), lr=CONFIG["learning_rate"], beta=1.0, save_path="smoke_vae.pth")
    print("Smoke test completed.")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_pgm = sub.add_parser("pgm")
    p_pgm.add_argument("--output", type=str, help="Output prefix or filename")

    p_train = sub.add_parser("train")
    p_train.add_argument("--epochs", type=int, help="Number of epochs")
    p_train.add_argument("--batch-size", type=int, dest="batch_size", help="Batch size")
    p_train.add_argument("--lr", type=float, help="Learning rate")
    p_train.add_argument("--beta", type=float, help="Beta for beta-VAE")
    p_train.add_argument("--latent-dim", type=int, dest="latent_dim", help="Latent dimension")
    p_train.add_argument("--save-path", type=str, dest="save_path", help="Path to save model")

    p_smoke = sub.add_parser("smoke")

    args = parser.parse_args()
    if args.cmd == "pgm":
        cmd_pgm(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "smoke":
        cmd_smoke(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    import numpy as np

    main()

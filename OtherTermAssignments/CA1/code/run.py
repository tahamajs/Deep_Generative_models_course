#!/usr/bin/env python3
"""Entry point to run the CA1 tasks (PGM visualization and VAE training)."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from ca1.config import CONFIG, set_seed, device, save_run_info
from ca1 import analysis, data, models, pgm, train, viz


def _default_report_output_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "report" / "DGM_Report_Template" / "figures"


def _parse_betas(raw_betas: str):
    betas = [float(x.strip()) for x in raw_betas.split(",") if x.strip()]
    if not betas:
        raise ValueError("At least one beta value must be provided.")
    return betas


def _format_beta(beta: float) -> str:
    return str(beta).replace(".", "p")


def cmd_pgm(args):
    print("Drawing Bayesian Network...")
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        bayes_path = out / "bayesian_network.png"
        markov_path = out / "complex_network.png"
    else:
        bayes_path = Path(args.output or "bayesian_network.png")
        markov_path = Path("complex_network.png")
    pgm.draw_bayesian_network(save_path=str(bayes_path))
    print("Drawing Complex/Markov Network...")
    pgm.draw_markov_network(save_path=str(markov_path))


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
    viz.visualize_reconstructions(model, val_loader, save_path="reconstructions.png", show=False)
    viz.visualize_generation_from_prior(model, save_path="prior_generation.png", show=False)


def cmd_smoke(args):
    # Quick smoke training to check everything runs
    imgs, latents_values, latents_classes, metadata = data.load_dsprites(CONFIG["data_path"])
    if imgs is None:
        raise RuntimeError("Could not load dataset. Place the dSprites npz at the configured path.")
    subset = min(CONFIG.get("smoke_subset", 2048), len(imgs))
    indices = np.random.default_rng(CONFIG["seed"]).choice(len(imgs), subset, replace=False)
    imgs_subset = imgs[indices]
    train_loader, val_loader = data.create_dataloaders(
        imgs_subset,
        batch_size=CONFIG["batch_size"],
        train_split=CONFIG["train_split"],
        generator=set_seed(CONFIG["seed"]),
        seed=CONFIG["seed"],
    )
    model = models.VAE(h_dim=CONFIG["latent_dim"]).to(device)
    train.train_vae(
        model,
        train_loader,
        val_loader,
        device,
        epochs=CONFIG.get("smoke_epochs", 1),
        lr=CONFIG["learning_rate"],
        beta=1.0,
        save_path="smoke_vae.pth",
    )
    print("Smoke test completed.")


def cmd_report(args):
    seed = args.seed
    generator = set_seed(seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    betas = _parse_betas(args.betas)
    print(f"Generating report figures in: {output_dir}")
    print(f"Using betas={betas}, epochs={args.epochs}, subset={args.subset}, device={device}")

    imgs, latents_values, latents_classes, metadata = data.load_dsprites(args.data_path)
    if imgs is None:
        raise RuntimeError("Could not load dataset. Place the dSprites npz at the configured path.")

    subset = min(args.subset, len(imgs))
    subset_indices = np.random.default_rng(seed).choice(len(imgs), subset, replace=False)
    imgs_subset = imgs[subset_indices]
    latents_subset = latents_classes[subset_indices]

    train_loader, val_loader, _, val_indices = data.create_dataloaders(
        imgs_subset,
        batch_size=args.batch_size,
        train_split=args.train_split,
        generator=generator,
        return_indices=True,
        seed=seed,
    )
    val_latents_classes = latents_subset[val_indices]

    print("Drawing PGM figures...")
    pgm.draw_bayesian_network(save_path=str(output_dir / "bayesian_network.png"))
    pgm.draw_markov_network(save_path=str(output_dir / "complex_network.png"))

    print("Saving dataset sample figure...")
    viz.plot_dsprites_samples(
        imgs_subset,
        n_samples=16,
        save_path=str(output_dir / "dsprites_samples.png"),
        show=False,
        seed=seed,
    )

    models_by_beta = {}
    histories_by_beta = {}
    for beta in betas:
        print(f"\nTraining model for beta={beta}")
        model = models.VAE(h_dim=args.latent_dim).to(device)
        checkpoint_path = checkpoints_dir / f"vae_beta{_format_beta(beta)}.pth"
        history = train.train_vae(
            model,
            train_loader,
            val_loader,
            device,
            epochs=args.epochs,
            lr=args.lr,
            beta=beta,
            save_path=str(checkpoint_path),
        )
        models_by_beta[beta] = model
        histories_by_beta[beta] = history
        save_run_info(
            {
                **CONFIG,
                **vars(args),
                "beta": beta,
            },
            path=str(checkpoint_path) + ".runinfo.json",
        )

    base_beta = betas[0]
    base_model = models_by_beta[base_beta]
    base_history = histories_by_beta[base_beta]

    print("\nGenerating report figure set...")
    viz.plot_training_history(
        base_history,
        beta=base_beta,
        save_path=str(output_dir / "training_loss.png"),
        show=False,
    )
    viz.plot_recon_kl_history(
        base_history,
        beta=base_beta,
        save_path=str(output_dir / "recon_kl_loss.png"),
        show=False,
    )
    viz.visualize_reconstructions(
        base_model,
        val_loader,
        n_samples=8,
        save_path=str(output_dir / "reconstructions.png"),
        show=False,
    )
    analysis.visualize_latent_space_2d(
        base_model,
        val_loader,
        latents_classes=val_latents_classes,
        max_samples=min(5000, len(val_latents_classes)),
        save_path=str(output_dir / "latent_space_pca.png"),
        color_factor_idx=1,
        show=False,
    )
    viz.plot_beta_reconstruction_comparison(
        models_by_beta,
        val_loader,
        betas,
        n_samples=8,
        save_path=str(output_dir / "beta_comparison.png"),
        show=False,
    )
    viz.visualize_latent_traversal_grid(
        base_model,
        n_dims=min(8, args.latent_dim),
        n_steps=10,
        save_path=str(output_dir / "latent_traversal.png"),
        show=False,
        seed=seed,
    )

    summary = {
        "output_dir": str(output_dir),
        "betas": betas,
        "epochs": args.epochs,
        "subset": subset,
        "base_beta": base_beta,
        "final_metrics": {
            str(beta): {
                "val_loss": histories_by_beta[beta]["val_loss"][-1],
                "val_recon": histories_by_beta[beta]["val_recon"][-1],
                "val_kl": histories_by_beta[beta]["val_kl"][-1],
            }
            for beta in betas
        },
    }
    with open(output_dir / "report_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nReport figures generated successfully:")
    for name in [
        "bayesian_network.png",
        "complex_network.png",
        "dsprites_samples.png",
        "training_loss.png",
        "recon_kl_loss.png",
        "reconstructions.png",
        "latent_space_pca.png",
        "beta_comparison.png",
        "latent_traversal.png",
    ]:
        print(f"  - {output_dir / name}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_pgm = sub.add_parser("pgm")
    p_pgm.add_argument("--output", type=str, help="Output prefix or filename")
    p_pgm.add_argument("--output-dir", type=str, help="Directory to save PGM figures")

    p_train = sub.add_parser("train")
    p_train.add_argument("--epochs", type=int, help="Number of epochs")
    p_train.add_argument("--batch-size", type=int, dest="batch_size", help="Batch size")
    p_train.add_argument("--lr", type=float, help="Learning rate")
    p_train.add_argument("--beta", type=float, help="Beta for beta-VAE")
    p_train.add_argument("--latent-dim", type=int, dest="latent_dim", help="Latent dimension")
    p_train.add_argument("--save-path", type=str, dest="save_path", help="Path to save model")

    p_smoke = sub.add_parser("smoke")

    p_report = sub.add_parser("report")
    p_report.add_argument("--output-dir", type=str, default=str(_default_report_output_dir()))
    p_report.add_argument("--data-path", type=str, default=CONFIG["data_path"])
    p_report.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    p_report.add_argument("--batch-size", type=int, dest="batch_size", default=CONFIG["batch_size"])
    p_report.add_argument("--lr", type=float, default=CONFIG["learning_rate"])
    p_report.add_argument("--latent-dim", type=int, dest="latent_dim", default=CONFIG["latent_dim"])
    p_report.add_argument("--subset", type=int, default=CONFIG["data_subset"])
    p_report.add_argument("--train-split", type=float, default=CONFIG["train_split"])
    p_report.add_argument("--betas", type=str, default=",".join(str(b) for b in CONFIG["betas"]))
    p_report.add_argument("--seed", type=int, default=CONFIG["seed"])

    args = parser.parse_args()
    if args.cmd == "pgm":
        cmd_pgm(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "smoke":
        cmd_smoke(args)
    elif args.cmd == "report":
        cmd_report(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

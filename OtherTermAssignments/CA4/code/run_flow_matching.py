#!/usr/bin/env python3
"""Flow Matching execution script with quick/full presets and report outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Run Flow Matching and save report figures.")
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--data-source", choices=["synthetic", "spy", "auto"], default="auto")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--solver", choices=["euler", "heun", "rk4"], default="heun")
    parser.add_argument("--solver-steps", type=int, default=None)
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def make_synthetic_returns(n_points=4000, seed=42):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.01, n_points).astype(np.float32)
    return returns


def pick_real_reference(dataset, n_samples):
    seqs = dataset.sequences[:n_samples]
    if torch.is_tensor(seqs):
        return seqs.cpu().numpy()
    return np.asarray(seqs)


def main():
    args = parse_args()
    if not args.show_plots:
        matplotlib.use("Agg")

    from flow_matching import (
        FM_CONFIG,
        FlowMatchingSampler,
        FlowMatchingTrainer,
        TimeSeriesDataset,
        VectorFieldTransformer,
        evaluate_generated_samples,
        load_spy_data,
        visualize_flow_matching_results,
    )
    from utils import REPORT_FIG_DIR, set_report_fig_dir

    out_dir = args.output_dir.resolve() if args.output_dir else REPORT_FIG_DIR
    set_report_fig_dir(out_dir)

    epochs = args.epochs
    if epochs is None:
        epochs = 8 if args.preset == "quick" else FM_CONFIG["num_epochs"]
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 64 if args.preset == "quick" else FM_CONFIG["batch_size"]
    solver_steps = args.solver_steps
    if solver_steps is None:
        solver_steps = 40 if args.preset == "quick" else 100

    train_dataset = None
    test_dataset = None
    if args.data_source in {"spy", "auto"}:
        try:
            train_dataset, test_dataset, _ = load_spy_data(seq_len=args.seq_len)
        except Exception as exc:
            if args.data_source == "spy":
                raise
            print(f"SPY path unavailable ({exc}), switching to synthetic data.")

    if train_dataset is None or test_dataset is None:
        synthetic = make_synthetic_returns()
        split = int(0.9 * len(synthetic))
        train_dataset = TimeSeriesDataset(synthetic[:split], seq_len=args.seq_len)
        test_dataset = TimeSeriesDataset(synthetic[split:], seq_len=args.seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = VectorFieldTransformer(
        seq_len=train_dataset.sequences.shape[1],
        feature_dim=train_dataset.sequences.shape[2],
        hidden_dim=128 if args.preset == "quick" else FM_CONFIG["hidden_dim"],
        num_heads=4,
        num_layers=3 if args.preset == "quick" else FM_CONFIG["num_layers"],
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    optimizer = torch.optim.Adam(model.parameters(), lr=FM_CONFIG["learning_rate"])
    trainer = FlowMatchingTrainer(model, optimizer, model.pos_embed.device)

    print(
        f"Training Flow Matching for {epochs} epoch(s), "
        f"batch_size={batch_size}, seq_len={train_dataset.sequences.shape[1]}"
    )
    trainer.train(train_loader, test_loader, num_epochs=epochs)

    sampler = FlowMatchingSampler(model, model.pos_embed.device)
    generated = sampler.sample(
        n_samples=args.sample_count,
        seq_len=train_dataset.sequences.shape[1],
        feature_dim=train_dataset.sequences.shape[2],
        n_steps=solver_steps,
        solver=args.solver,
    )

    real_ref = pick_real_reference(test_dataset, args.sample_count)
    evaluate_generated_samples(real_ref, generated)
    visualize_flow_matching_results(
        real_ref,
        generated,
        loss_history=trainer.train_loss_history,
        show=args.show_plots,
    )

    print("Flow Matching pipeline complete")


if __name__ == "__main__":
    main()

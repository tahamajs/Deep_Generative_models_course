#!/usr/bin/env python3
"""
Flow Matching Runner Script

This script demonstrates the Flow Matching implementation for time series generation.
"""

from flow_matching import *

if __name__ == "__main__":
    # Complete Flow Matching training example

    # Step 1: Generate synthetic time series data
    print("Generating synthetic time series data...")
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate synthetic data (you can replace with real data)
    num_samples = 1000
    seq_length = 100
    feature_dim = 2

    # Create synthetic time series (e.g., sine waves with noise)
    t = np.linspace(0, 4*np.pi, seq_length)
    data = []
    for _ in range(num_samples):
        freq1 = np.random.uniform(0.5, 2.0)
        freq2 = np.random.uniform(0.5, 2.0)
        phase1 = np.random.uniform(0, 2*np.pi)
        phase2 = np.random.uniform(0, 2*np.pi)
        noise = np.random.normal(0, 0.1, (seq_length, feature_dim))

        series = np.column_stack([
            np.sin(freq1 * t + phase1),
            np.cos(freq2 * t + phase2)
        ]) + noise

        data.append(series)

    data = np.array(data)  # Shape: (num_samples, seq_length, feature_dim)

    # Step 2: Create dataset
    dataset = TimeSeriesDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Step 3: Initialize trainer and train
    print("Training Flow Matching model...")
    trainer = FlowMatchingTrainer(
        feature_dim=feature_dim,
        hidden_dim=128,
        num_layers=3,
        learning_rate=1e-3
    )

    trainer.train(dataloader, num_epochs=50)

    # Step 4: Generate samples
    print("Generating samples...")
    sampler = FlowMatchingSampler(trainer.vector_field)
    samples = sampler.sample(num_samples=10, seq_length=seq_length)

    # Step 5: Evaluate
    print("Evaluating generated samples...")
    evaluate_generated_samples(data, samples)

    # Step 6: Visualize
    print("Visualizing results...")
    visualize_flow_matching_results(data, samples, trainer.loss_history)

    print("Flow Matching training and evaluation complete!")
#!/usr/bin/env python3
"""
DDPM & DDIM Runner Script

This script demonstrates the complete DDPM and DDIM implementation.
"""

from ddpm import *

if __name__ == "__main__":
    # Initialize scheduler and test
    scheduler = DDPMScheduler(num_timesteps=1000, device=device)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(scheduler.betas.cpu().numpy())
    plt.title('β_t (Linear Schedule)')
    plt.xlabel('Timestep t')
    plt.ylabel('β_t')

    plt.subplot(1, 3, 2)
    plt.plot(scheduler.alphas_cumprod.cpu().numpy())
    plt.title('ᾱ_t (Cumulative Product)')
    plt.xlabel('Timestep t')
    plt.ylabel('ᾱ_t')

    plt.subplot(1, 3, 3)
    plt.plot(scheduler.sqrt_one_minus_alphas_cumprod.cpu().numpy())
    plt.title('√(1 - ᾱ_t)')
    plt.xlabel('Timestep t')
    plt.ylabel('Noise Level')

    plt.tight_layout()
    plt.show()

    # Visualize
    visualize_forward_process(scheduler)
    print("✅ Forward process visualization complete!")

    # Image transformations
    transform = transforms.Compose([
        transforms.Pad(2),  # Pad 28x28 MNIST to 32x32
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])

    # Load MNIST dataset (simpler than CIFAR-10)
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=DDPM_CONFIG['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    sample_batch, labels = next(iter(train_loader))
    print(f"Batch shape: {sample_batch.shape}")

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i, ax in enumerate(axes.flat):
        img = sample_batch[i].permute(1, 2, 0).numpy()
        img = (img + 1) / 2
        ax.imshow(img)
        ax.axis('off')
    plt.suptitle('Sample Training Images (MNIST)')
    plt.tight_layout()
    plt.show()

    ddpm_model = UNet(
        c_in=DDPM_CONFIG['channels'],
        c_out=DDPM_CONFIG['channels'],
        time_dim=256,
        base_channels=64
    ).to(device)

    ddpm_scheduler = DDPMScheduler(
        num_timesteps=DDPM_CONFIG['num_timesteps'],
        beta_start=DDPM_CONFIG['beta_start'],
        beta_end=DDPM_CONFIG['beta_end'],
        device=device
    )

    optimizer = torch.optim.AdamW(ddpm_model.parameters(), lr=DDPM_CONFIG['learning_rate'])

    trainer = DDPMTrainer(ddpm_model, ddpm_scheduler, optimizer, device)

    print(f"Model parameters: {sum(p.numel() for p in ddpm_model.parameters()):,}")

    DEMO_EPOCHS = 10  # Change to DDPM_CONFIG['num_epochs'] for full training

    loss_history = trainer.train(train_loader, num_epochs=DEMO_EPOCHS)
    trainer.plot_loss(save_path=REPORT_FIG_DIR / "ddpm_training_loss.png")

    ddpm_sampler = DDPMSampler(ddpm_model, ddpm_scheduler, device)
    ddim_sampler = DDIMSampler(ddpm_model, ddpm_scheduler, device)

    print("Generating samples with DDPM (1000 steps)...")
    ddpm_samples, ddpm_intermediates = ddpm_sampler.sample(batch_size=16)

    print("\nGenerating samples with DDIM (50 steps)...")
    ddim_samples, ddim_intermediates = ddim_sampler.sample(batch_size=16, num_inference_steps=50, eta=0.0)

    visualize_samples(ddpm_samples, "DDPM Samples (1000 steps)", save_path=REPORT_FIG_DIR / "ddpm_samples_grid.png")
    visualize_samples(ddim_samples, "DDIM Samples (50 steps, η=0)", save_path=REPORT_FIG_DIR / "ddim_samples_grid.png")

    visualize_denoising_process(ddpm_intermediates, "DDPM Denoising Process")
    visualize_denoising_process(ddim_intermediates, "DDIM Denoising Process")

    # Save model
    if len(trainer.loss_history) > 0:
        torch.save({
            'model_state_dict': ddpm_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': trainer.loss_history,
            'config': DDPM_CONFIG,
        }, 'ddpm_model.pt')
        print("DDPM model saved to 'ddpm_model.pt'")
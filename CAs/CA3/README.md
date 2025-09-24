# CA3 - Diffusion Models and Score-based Generative Models

This folder contains an educational project for the "Deep Generative Models" course. The code implements two cutting-edge approaches to generative modeling:

- **Diffusion Models**: Denoising Diffusion Probabilistic Models (DDPM) and Denoising Diffusion Implicit Models (DDIM) for high-quality image generation.
- **Score-based Generative Models**: Learning the score function (gradient of log-density) for sampling from complex distributions using Langevin dynamics.

This README documents the notebook structure, configuration, how to run experiments (without running code here), reproducibility notes, and troubleshooting tips.

## Repository layout

- `codes/Diffusion_Models (1).ipynb` — Implementation of DDPM and DDIM on the Sprites dataset. Includes U-Net architecture, forward/reverse diffusion, and FID evaluation.
- `codes/score_based_models.ipynb` — Score-based generative modeling on synthetic 2D Gaussian mixture data. Covers score matching, annealed Langevin sampling, and trajectory visualization.
- `description/DGM_HW3.pdf` — Assignment description and requirements.
- `report/DGM_CA3.pdf` — Detailed report on experiments, results, and analysis.

## Notebook Overview

### Diffusion_Models (1).ipynb

1. **Setup and Configuration**: Device setup, hyperparameters, and utility functions for image normalization and plotting.
2. **Custom Dataset Class**: PyTorch Dataset for loading and preprocessing the Sprites dataset with conditional generation support.
3. **U-Net Architecture**: Complete U-Net with residual blocks, time and context embeddings for conditional diffusion.
4. **Diffusion Schedule**: Computation of beta, alpha, and alpha_bar for the noise schedule.
5. **Forward Diffusion**: Function to add noise to images according to the diffusion process.
6. **Sampling Functions**: DDPM and DDIM sampling implementations for generating new images.
7. **FID Computation**: Fréchet Inception Distance calculation for quantitative evaluation.
8. **Training Loop**: Full training with loss monitoring, sample generation, and FID tracking.
9. **Results Visualization**: Plots of training loss, FID scores, and generated image grids.

### score_based_models.ipynb

1. **Data Generation**: Synthetic 2D Gaussian mixture data generation with visualization.
2. **Score Field Visualization**: Functions to plot learned score fields at different noise levels.
3. **Noise Addition and Loss**: Score matching objective implementation.
4. **Score Network Architecture**: MLP for predicting scores conditioned on noise level.
5. **Training Function**: Score network training with early stopping and progress monitoring.
6. **Sampling Methods**: Annealed Langevin, standard Langevin, and deterministic sampling.
7. **Visualization**: Sampling results comparison and trajectory plots.
8. **Experiments**: Fixed sigma training and varying sigma training comparisons.

## How to run (local environment instructions)

1. Ensure you have the environment set up as described in the main project README (venv/conda with PyTorch, torchvision, etc.).

2. For Diffusion Models:
   - Download the Sprites dataset (automatically handled in notebook).
   - Review hyperparameters: `timesteps=1000`, `n_feat=64`, `batch_size=100`, `n_epoch=40`.
   - Run training cells to train the U-Net on diffusion.
   - Generate samples using DDPM or DDIM sampling.
   - Evaluate with FID scores.

3. For Score-based Models:
   - The notebook uses synthetic data, so no external downloads needed.
   - Train score networks with fixed or varying sigma.
   - Visualize score fields and sampling trajectories.
   - Compare different sampling methods.

## Reproducibility Notes

- Set random seeds at the beginning of each notebook for reproducible results.
- The diffusion models use a fixed noise schedule; FID evaluation requires consistent real/fake image preprocessing.
- Score-based models work on 2D synthetic data, making them fast to train and easy to visualize.

## Dependencies

- PyTorch, torchvision
- NumPy, Matplotlib
- tqdm, pytorch-fid
- For diffusion: Custom U-Net implementation
- For score-based: MLP with noise conditioning

## Key Concepts Covered

### Diffusion Models
- Forward process: Progressive noise addition
- Reverse process: Learned denoising
- DDPM vs DDIM: Trade-offs between quality and speed
- Conditional generation with context embeddings

### Score-based Models
- Score matching objective
- Noise-conditioned score prediction
- Langevin dynamics for sampling
- Annealed sampling for improved results

## Expected Results

- **Diffusion Models**: High-quality 16x16 sprite images, FID scores improving over training epochs.
- **Score-based Models**: Accurate score field learning, effective sampling from 2D Gaussian mixtures.

## Troubleshooting

- **CUDA Memory**: Reduce batch_size or image size for diffusion models.
- **FID Errors**: Ensure pytorch-fid is installed and images are properly normalized.
- **Slow Training**: Diffusion models are computationally intensive; consider smaller timesteps for experimentation.

## References

- Ho, J., et al. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
- Song, Y., et al. "Score-Based Generative Modeling through Stochastic Differential Equations." ICLR 2021.
- Dhariwal, P. and Nichol, A. "Diffusion Models Beat GANs on Image Synthesis." NeurIPS 2021.

The notebooks include detailed Markdown explanations for each code cell, making them suitable for educational purposes and portfolio presentation.
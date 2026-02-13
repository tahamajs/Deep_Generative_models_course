# CA3: Score-Based Generative Modeling (Langevin and Annealed Sampling)

This project implements score-based generative modeling on a synthetic 2D Gaussian mixture. The model learns score fields `s_theta(x, sigma)` and samples using deterministic, Langevin, and annealed Langevin dynamics.

## Project Structure

- `codes/CA3_Score_Based_Models.ipynb`: main implementation notebook.
- `codes/CA3_Diffusion_Models_overview.ipynb`: conceptual/overview notebook.
- `description/DGM_HW3.pdf`: assignment statement.
- `images/`: exported plots and sampling visualizations.
- `report/`: final report and LaTeX files.

## How to Run

```bash
cd CA3_Diffusion_Models/codes
jupyter lab CA3_Score_Based_Models.ipynb
```

Run cells in order; the notebook is sequential and phase-based.

## Methods Used (Complete Map)

### 1) Data Generation and Density Methods

- `generate_mixture_gaussians(n_samples=5000, train_split=0.8)`:
  - Creates synthetic bimodal 2D dataset.
  - Randomizes mixture weights, means, and variances.
  - Returns train/test splits plus true distribution parameters.
- `gaussian_pdf(x, mean, cov)`:
  - Computes multivariate Gaussian density.
- `p_x(points, params)`:
  - Computes full mixture density from component PDFs.

### 2) Noise and Score-Matching Methods

- `add_noise(x, sigma)`:
  - Adds Gaussian noise `epsilon ~ N(0, sigma^2 I)` to samples.
- `score_matching_loss(x, x_noisy, epsilon, sigma, outputs)`:
  - Denoising score matching loss in noisy space.

### 3) Model Method

- `class ScoreNet(nn.Module)`:
  - MLP with BatchNorm + LeakyReLU.
  - Input: concatenated `[x, sigma]`.
  - Output: 2D score estimate.
- `ScoreNet.forward(x, sigma)`:
  - Predicts `grad_x log p_sigma(x)`.

### 4) Training Method

- `train_score_model(...)`:
  - Full training loop with Adam and StepLR scheduler.
  - Supports two modes:
    - fixed-sigma training (`fixed_sigma` set)
    - varying-sigma training (`sigma` randomly sampled in `[1, 20)`).
  - Optional gradient clipping.
  - Saves best checkpoint by train loss.

### 5) Sampling Methods

- `deterministic_sampling(model, start_points, num_steps, step_size)`:
  - Gradient ascent without noise.
  - Fast and stable, lower diversity.
- `langevin_sampling(model, start_points, num_steps, step_size)`:
  - Gradient step + Gaussian noise each step.
  - Better exploration/mode coverage.
- `annealed_sampling(model, start_points, sigma_start, sigma_end, steps_per_sigma, num_sigmas, step_size)`:
  - Multi-scale Langevin, gradually reducing sigma.
  - Best quality mode coverage in this project.

### 6) Experiment Driver Methods

- `perform_sampling_and_plot(model, epoch, num_points=1000)`:
  - Runs deterministic/Langevin/annealed sampling and plots comparison.
- Phase setup in notebook:
  - Fixed-sigma experiments for `sigma = [1, 3, 7]`.
  - Trajectory experiments from predefined start points.
  - Varying-sigma training for 1000 epochs.

### 7) Visualization and Analysis Methods

### Data and density visualization

- `plot_heatmap(...)`
- `plot_train_test_samples(...)`
- `visualize_noise_effect_on_data(...)`
- `visualize_density_vs_score_concept(...)`

### Score-field and training-step visualization

- `plot_score_field_individual(...)`
- `demonstrate_score_matching_step(...)`
- `visualize_training_step_example(...)`
- `plot_score_magnitude_field(...)`

### Sampling visualization

- `plot_sampling_results(...)`
- `plot_sampling_trajectories_with_start_points(...)`
- `plot_annealed_sampling_steps(...)`
- `visualize_sampling_concepts(...)`

### Model comparison visualization

- `plot_loss_comparison(...)`
- `compare_fixed_vs_varying_sigma(...)`
- `visualize_distribution_overlap(...)`
- `create_project_roadmap(...)`

## Training Settings Used in Notebook

### Fixed-sigma phases

- `sigma_values = [1, 3, 7]`
- `epochs = 100` each
- `batch_size = 256`
- `learning_rate = 1e-3`
- `scheduler_step = 100`, `scheduler_gamma = 0.5`
- `clip_grad = 1.0`

### Varying-sigma phase

- `epochs = 1000`
- same optimizer family and batch settings
- sigma sampled per batch from integer range `[1, 20)`

## What This Project Demonstrates

- How score matching learns gradient fields instead of explicit density.
- Why fixed-sigma models specialize at one noise scale.
- Why varying-sigma models are more general for annealed sampling.
- Practical tradeoff between deterministic speed and stochastic coverage.

## Output Artifacts

- Notebook-generated figures are exported into `images/`.
- Report-ready results and analysis are in `report/`.

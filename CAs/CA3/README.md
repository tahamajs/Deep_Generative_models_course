# CA3 - Diffusion Models and Score-based Generative Models

## Overview

This assignment implements two advanced generative modeling techniques: **Diffusion Models** (DDPM and DDIM) and **Score-based Generative Models**. These methods represent state-of-the-art approaches in deep generative modeling, enabling high-quality sample generation from complex distributions.

### Assignment Objectives

- Implement Denoising Diffusion Probabilistic Models (DDPM) and Denoising Diffusion Implicit Models (DDIM) for image generation
- Develop score-based generative models using Langevin dynamics
- Compare different sampling strategies and evaluate generation quality
- Understand the mathematical foundations of diffusion processes and score matching

## Prerequisites

### Required Knowledge

- **Probability Theory**: Understanding of Markov chains, Gaussian distributions, stochastic processes
- **Deep Learning**: PyTorch proficiency, neural network architectures (CNNs, MLPs)
- **Generative Modeling**: Basic concepts of VAEs, GANs, and flow-based models
- **Optimization**: Gradient-based optimization, loss functions

### Technical Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA-compatible GPU (recommended for diffusion models)
- Libraries: `torchvision`, `numpy`, `matplotlib`, `tqdm`, `pytorch-fid`

### Environment Setup

```bash
# Create virtual environment
python -m venv dgm_env
source dgm_env/bin/activate  # On Windows: dgm_env\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy matplotlib tqdm seaborn
pip install pytorch-fid
```

## Core Concepts Explained

### Diffusion Models

Diffusion models are a class of generative models that learn to reverse a gradual noising process. They consist of two main processes:

#### Forward Diffusion Process (Adding Noise)

The forward process gradually adds Gaussian noise to data over T timesteps:

**Mathematical Foundation:**

- Start with clean data: \( x_0 \sim q(x_0) \)
- At each timestep t: \( q(x*t | x*{t-1}) = \mathcal{N}(x*t; \sqrt{1-\beta_t} x*{t-1}, \beta_t I) \)
- Variance schedule: \( \beta_t \) increases from \( \beta_1 = 10^{-4} \) to \( \beta_T = 0.02 \)
- Closed-form solution: \( q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}\_t} x_0, (1-\bar{\alpha}\_t) I) \)
- Where \( \bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s) \)

#### Reverse Diffusion Process (Denoising)

The model learns to predict and remove noise:

- \( p*\theta(x*{t-1} | x*t) = \mathcal{N}(x*{t-1}; \mu\_\theta(x_t, t), \sigma_t^2 I) \)
- Noise prediction: \( \epsilon\_\theta(x_t, t) \approx \epsilon \)
- Simplified objective: \( L = ||\epsilon - \epsilon\_\theta(x_t, t)||^2 \)

#### DDPM vs DDIM

- **DDPM**: Uses all timesteps, slower but potentially higher quality
- **DDIM**: Uses fewer steps via deterministic sampling, faster inference
- DDIM sampling: \( x*{t-1} = \sqrt{\bar{\alpha}*{t-1}} (\frac{x*t - \sqrt{1-\bar{\alpha}\_t} \epsilon*\theta}{\sqrt{\bar{\alpha}_t}}) + \sqrt{1-\bar{\alpha}_{t-1} - \sigma*t^2} \epsilon*\theta \)

#### U-Net Architecture

The denoising network uses a U-Net with:

- Encoder-decoder structure with skip connections
- Time embeddings: sinusoidal positional encoding
- Conditional embeddings for class-specific generation
- Residual blocks with attention mechanisms

### Score-based Generative Models

Score-based models learn the gradient of the log-density (score function) to enable sampling via Langevin dynamics.

#### Score Function

- Score: \( s*\theta(x, \sigma) = \nabla_x \log p*\sigma(x) \)
- For noisy data: \( p*\sigma(x) = \int p*{data}(y) \mathcal{N}(x; y, \sigma^2 I) dy \)
- Score matching objective: \( \mathbb{E}_{p_\sigma(x)} [||s_\theta(x, \sigma) - \nabla_x \log p_\sigma(x)||^2] \)

#### Denoising Score Matching

- Practical objective: \( L(\theta, \sigma) = \mathbb{E}_{x \sim p_{data}} \mathbb{E}_{\tilde{x} \sim \mathcal{N}(x, \sigma^2 I)} [||s_\theta(\tilde{x}, \sigma) - (\tilde{x} - x)/\sigma^2||^2] \)
- Multiple noise scales: \( \sigma_1 < \sigma_2 < \dots < \sigma_L \)

#### Langevin Dynamics Sampling

- Stochastic sampling: \( x*{k+1} = x_k + \frac{\epsilon}{2} s*\theta(x_k, \sigma) + \sqrt{\epsilon} z \)
- Annealed sampling: Gradually reduce noise scale during sampling
- Deterministic sampling: Remove stochastic term for faster inference

## Data Preparation

### Diffusion Models (Sprites Dataset)

- **Dataset**: 16x16 sprite images with categorical labels
- **Preprocessing**: Normalization to [-1, 1], conditional generation support
- **Custom Dataset Class**: Handles loading, transformation, and context embedding

### Score-based Models (Synthetic Data)

- **Data Generation**: 2D Gaussian mixture distributions
- **Mixture Components**: Randomly positioned Gaussians with varying weights
- **Visualization**: Contour plots and scatter plots for analysis

## Model Architecture

### Diffusion Model Components

```python
# U-Net Architecture
- Encoder: Downsampling with residual blocks
- Bottleneck: Time and context embeddings
- Decoder: Upsampling with skip connections
- Output: Noise prediction ε_θ(x_t, t, c)
```

### Score Network

```python
# MLP Architecture
- Input: Noisy sample x_t and noise level σ
- Hidden layers: Fully connected with ReLU activation
- Output: Score vector ∇_x log p_σ(x)
```

## Training

### Diffusion Model Training

1. **Data Loading**: Batch sprites with random timesteps
2. **Noise Addition**: Sample t, add noise according to q(x_t|x_0)
3. **Model Prediction**: Predict noise ε_θ(x_t, t, c)
4. **Loss Computation**: MSE between predicted and actual noise
5. **Optimization**: Adam optimizer with learning rate scheduling

### Score-based Model Training

1. **Noise Perturbation**: Add noise at multiple scales σ
2. **Score Prediction**: Train network to predict ∇*x log p*σ(x)
3. **Loss Function**: Denoising score matching objective
4. **Early Stopping**: Monitor validation loss for convergence

## Evaluation

### Quantitative Metrics

- **FID Score**: Fréchet Inception Distance for image quality assessment
- **Sample Quality**: Visual inspection of generated samples
- **Diversity**: Coverage of data distribution

### Qualitative Analysis

- **Diffusion Models**: Generated image grids, conditional generation samples
- **Score-based Models**: Sampling trajectories, score field visualization

## Results and Analysis

### Expected Outcomes

- **Diffusion Models**: High-quality 16x16 sprite generation, FID < 50 after training
- **Score-based Models**: Accurate density estimation, effective sampling from 2D mixtures

### Hyperparameter Sensitivity

- **Timesteps**: More steps improve quality but increase computation
- **Noise Schedule**: β values affect training stability
- **Network Capacity**: Larger models capture more complex distributions

## Troubleshooting

### Common Issues

- **CUDA Out of Memory**: Reduce batch_size or image resolution
- **Training Instability**: Adjust learning rate or noise schedule
- **Poor Sample Quality**: Increase model capacity or training epochs
- **FID Computation Errors**: Ensure proper image normalization

### Performance Optimization

- Use mixed precision training (FP16) for faster convergence
- Implement gradient accumulation for larger effective batch sizes
- Utilize data parallelism across multiple GPUs

## Reproducibility

### Random Seeds

```python
student_number = 810101504
np.random.seed(student_number)
torch.manual_seed(student_number)
torch.cuda.manual_seed_all(student_number)
```

### Hyperparameters

- Diffusion: `timesteps=1000`, `n_feat=64`, `batch_size=100`, `n_epoch=40`
- Score-based: Varying sigma schedules, MLP with 3 hidden layers

### Environment

- PyTorch version: 1.12+
- CUDA version: 11.6+
- Hardware: NVIDIA GPU with 8GB+ VRAM

## Dependencies

```
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.64.0
pytorch-fid>=0.10.0
seaborn>=0.11.0
```

## References

1. **Diffusion Models**:

   - Ho, J., et al. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
   - Song, Y., et al. "Denoising Diffusion Implicit Models." ICLR 2021.
   - Dhariwal, P. and Nichol, A. "Diffusion Models Beat GANs on Image Synthesis." NeurIPS 2021.

2. **Score-based Models**:

   - Song, Y., et al. "Score-Based Generative Modeling through Stochastic Differential Equations." ICLR 2021.
   - Song, Y., et al. "Estimating the Optimal Covariance with Imperfect Means." ICML 2020.

3. **Implementation References**:
   - U-Net: Ronneberger, O., et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015.
   - Langevin Dynamics: Welling, M. and Teh, Y. W. "Bayesian Learning via Stochastic Gradient Langevin Dynamics." ICML 2011.

## File Structure

```
CA3/
├── codes/
│   ├── Diffusion_Models.ipynb    # DDPM/DDIM implementation
│   └── score_based_models.ipynb  # Score-based generative models
├── description/
│   └── DGM_HW3.pdf              # Assignment description
└── report/
    ├── DGM_CA3.pdf             # Implementation report
    └── DGM_CA3_EN_final.pdf    # Final English report
```

## Usage Instructions

### Running Diffusion Models

1. Open `Diffusion_Models.ipynb` in Jupyter/Colab
2. Execute cells sequentially (imports → config → data → model → training)
3. Monitor training progress and FID scores
4. Generate samples using DDPM or DDIM sampling

### Running Score-based Models

1. Open `score_based_models.ipynb` in Jupyter/Colab
2. Run data generation and visualization cells
3. Train score networks with different sigma schedules
4. Compare sampling methods and visualize results

### Colab Execution

- Upload notebooks to Google Colab
- Enable GPU runtime for faster training
- Install dependencies: `!pip install pytorch-fid`
- Download datasets automatically (Sprites) or generate synthetically

This comprehensive implementation covers both theoretical foundations and practical considerations for advanced generative modeling techniques.

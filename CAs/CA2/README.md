# CA2 - GANs and Normalizing Flows

## Overview

This assignment implements two powerful generative modeling approaches: **Generative Adversarial Networks (GANs)** and **Normalizing Flows**. The focus is on the FashionMNIST dataset, demonstrating both pixel-level and latent-space generative modeling with quantitative evaluation using Fréchet Inception Distance (FID).

### Assignment Objectives

- Implement DCGAN-style GANs for high-quality image generation
- Develop RealNVP normalizing flows for density estimation and sampling
- Compare performance in pixel space vs. learned latent representations
- Evaluate generative quality using FID scores
- Understand adversarial training and invertible transformations

## Prerequisites

### Required Knowledge

- **Deep Learning**: CNN architectures, adversarial training, invertible functions
- **Probability Theory**: Change of variables, log-likelihood maximization
- **Optimization**: Min-max games, gradient-based optimization
- **Computer Vision**: Image generation, evaluation metrics

### Technical Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA-compatible GPU (recommended)
- Libraries: `torchvision`, `numpy`, `matplotlib`, `scipy`, `pytorch-fid`

### Environment Setup

```bash
# Create virtual environment
python -m venv dgm_env
source dgm_env/bin/activate  # On Windows: dgm_env\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy matplotlib scipy
pip install pytorch-fid
```

## Core Concepts Explained

### Generative Adversarial Networks (GANs)

GANs consist of two neural networks competing in a zero-sum game:

#### Adversarial Framework

- **Generator (G)**: Maps random noise z to data distribution: \( G: \mathcal{Z} \rightarrow \mathcal{X} \)
- **Discriminator (D)**: Binary classifier distinguishing real vs. fake: \( D: \mathcal{X} \rightarrow [0,1] \)
- **Objective**: \( \min*G \max_D \mathbb{E}*{x \sim p*{data}} [\log D(x)] + \mathbb{E}*{z \sim p_z} [\log (1 - D(G(z)))] \)

#### DCGAN Architecture

- **Generator**: Transposed convolutions with batch normalization and ReLU
- **Discriminator**: Convolutions with LeakyReLU and sigmoid output
- **Training**: Alternate updates, non-saturating loss for G, standard BCE for D

#### Training Dynamics

- **Mode Collapse**: Generator produces limited variety
- **Convergence Issues**: Gradient vanishing, oscillation
- **Stabilization**: Batch normalization, learning rate scheduling, noise injection

### Normalizing Flows

Normalizing flows transform simple distributions into complex ones via invertible transformations.

#### Change of Variables

For invertible function \( f: \mathcal{X} \rightarrow \mathcal{Z} \):

- \( p_X(x) = p_Z(f(x)) |\det \frac{\partial f}{\partial x}| \)
- Log-likelihood: \( \log p_X(x) = \log p_Z(z) + \log |\det J_f(x)| \)

#### RealNVP (Real-valued Non-Volume Preserving)

- **Affine Coupling**: \( y*{1:d} = x*{1:d} \), \( y*{d+1:D} = x*{d+1:D} \odot \exp(s(x*{1:d})) + t(x*{1:d}) \)
- **Scale and Translation**: Neural networks predict s and t
- **Invertibility**: Easy forward and inverse passes
- **Volume Preservation**: Jacobian determinant is 1 for coupling layers

#### Multi-Scale Architecture

- **Squeeze Operation**: Rearranges spatial dimensions
- **Checkerboard Masking**: Alternating coupling layers
- **Channel-wise Masking**: Split along feature dimension

### Fréchet Inception Distance (FID)

FID measures distribution similarity using Inception features:

#### Mathematical Foundation

- **Inception Features**: Pre-trained Inception network extracts features
- **Statistics**: Mean μ and covariance Σ for real and generated distributions
- **Distance**: \( d^2((\mu_r, \Sigma_r), (\mu_g, \Sigma_g)) = ||\mu_r - \mu_g||^2 + \Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}) \)

#### Interpretation

- Lower FID indicates better generative quality
- Sensitive to mode collapse and diversity
- Computationally expensive but reliable

## Data Preparation

### FashionMNIST Dataset

- **Resolution**: 28x28 grayscale images
- **Classes**: 10 fashion categories (T-shirt, trouser, etc.)
- **Preprocessing**: Resize to 64x64 for GANs, normalize to [-1, 1]
- **Latent Encoding**: Autoencoder compresses to 50D for flows

### Autoencoder for Latent Space

- **Encoder**: CNN compressing 28x28 → 50D
- **Decoder**: Transposed CNN reconstructing images
- **Training**: MSE reconstruction loss
- **Purpose**: Enable efficient normalizing flows in latent space

## Model Architecture

### GAN Components

```python
# Generator Architecture
- Input: 100D noise vector
- Layers: 5 transposed conv blocks
- Output: 64x64 grayscale image
- Activations: ReLU + Tanh

# Discriminator Architecture
- Input: 64x64 grayscale image
- Layers: 5 conv blocks
- Output: Scalar probability
- Activations: LeakyReLU + Sigmoid
```

### Normalizing Flow Components

```python
# RealNVP Block
- Coupling layers with affine transformations
- Scale/translation networks: MLPs
- Masking patterns: Checkerboard/channel-wise
- Squeeze operations for multi-scale
```

## Training

### GAN Training

1. **Data Loading**: Batched FashionMNIST images
2. **Discriminator Update**: Real/fake classification
3. **Generator Update**: Fool discriminator with fake samples
4. **Monitoring**: Loss curves, sample generation
5. **Evaluation**: FID computation per epoch

### Flow Training

1. **Maximum Likelihood**: Minimize negative log-likelihood
2. **Latent Space**: Train on autoencoder representations
3. **Pixel Space**: Direct training on images (challenging)
4. **Regularization**: Weight decay, early stopping

## Evaluation

### Quantitative Metrics

- **FID Score**: Distribution similarity (lower is better)
- **Log-Likelihood**: For normalizing flows (higher is better)
- **Out-of-Distribution**: Detection using MNIST/KMNIST

### Qualitative Analysis

- **Sample Quality**: Visual inspection of generated images
- **Diversity**: Coverage of fashion categories
- **Mode Collapse**: Check for repetitive patterns

## Results and Analysis

### Expected Outcomes

- **GANs**: FID < 50 after training, diverse fashion images
- **Flows**: Better likelihood in latent space vs. pixel space
- **Comparison**: GANs excel at quality, flows at density estimation

### Hyperparameter Sensitivity

- **Learning Rate**: Critical for stable training (0.0002 typical)
- **Batch Size**: Larger batches improve stability
- **Architecture Depth**: Deeper models capture more complexity

## Troubleshooting

### Common Issues

- **GAN Training Instability**: Adjust learning rates, add noise
- **Mode Collapse**: Use experience replay, different architectures
- **Flow Training**: Gradient explosion in pixel space
- **FID Computation**: Ensure sufficient sample size (1000+ images)

### Performance Optimization

- Use mixed precision training (FP16)
- Implement gradient penalties (WGAN-GP)
- Utilize progressive growing for high resolution

## Reproducibility

### Random Seeds

```python
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
```

### Hyperparameters

- GAN: `latent_dim=100`, `lr=0.0002`, `batch_size=64`, `epochs=10`
- Flow: `hidden_dim=512`, `num_layers=8`, `batch_size=128`

### Environment

- PyTorch version: 1.12+
- CUDA version: 11.6+
- Hardware: GPU with 4GB+ VRAM

## Dependencies

```
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
pytorch-fid>=0.10.0
```

## References

1. **GANs**:

   - Goodfellow, I., et al. "Generative Adversarial Nets." NIPS 2014.
   - Radford, A., et al. "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks." ICLR 2016.
   - Heusel, M., et al. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." NIPS 2017.

2. **Normalizing Flows**:

   - Dinh, L., et al. "Density estimation using Real NVP." ICLR 2017.
   - Kingma, D. P. and Dhariwal, P. "Glow: Generative Flow with Invertible 1x1 Convolutions." NeurIPS 2018.

3. **Evaluation**:
   - Salimans, T., et al. "Improved Techniques for Training GANs." NIPS 2016.

## File Structure

```
CA2/
├── code/
│   ├── CA2_DGM.ipynb          # Main implementation notebook
│   └── Q2_final_res.ipynb     # GAN training with FID evaluation
├── description/
│   └── [Assignment PDF]       # Original problem statement
└── report/
    └── [Report PDF]          # Implementation details and results
```

## Usage Instructions

### Running GAN Training

1. Open `Q2_final_res.ipynb` in Jupyter/Colab
2. Execute cells sequentially (imports → config → data → model → training)
3. Monitor FID scores and generated samples
4. Adjust hyperparameters for better performance

### Running Normalizing Flows

1. Open `CA2_DGM.ipynb` sections on RealNVP
2. Train autoencoder first for latent representations
3. Train flows in pixel or latent space
4. Evaluate log-likelihood and sample quality

### Colab Execution

- Upload notebooks to Google Colab
- Enable GPU runtime for faster training
- Install dependencies: `!pip install pytorch-fid`
- Monitor training with generated image grids

This comprehensive implementation explores the trade-offs between adversarial and flow-based generative modeling approaches.

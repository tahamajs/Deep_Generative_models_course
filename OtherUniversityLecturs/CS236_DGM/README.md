![Hits](https://hitcounter.pythonanywhere.com/count/tag.svg?url=https%3A%2F%2Fgithub.com%2FSKKSaikia%2FCS236_DGM)

# CS236: Deep Generative Models

<img src="https://github.com/SKKSaikia/CS236_DGM/blob/master/cs236_c.jpg">

**Stanford CS236 - Deep Generative Models** - Complete course materials, assignments, and resources for the Spring 2019 offering of Stanford's CS236 course on Deep Generative Models.

## Table of Contents

1. [Course Overview](#course-overview)
2. [Repository Structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [Homeworks](#homeworks)
5. [Installation & Setup](#installation--setup)
6. [Running the Assignments](#running-the-assignments)
7. [Submission Instructions](#submission-instructions)
8. [Detailed Assignment Breakdown](#detailed-assignment-breakdown)
9. [Resources & Documentation](#resources--documentation)
10. [Final Project](#final-project)
11. [Troubleshooting](#troubleshooting)
12. [Contributing](#contributing)

---

## Course Overview

**Stanford CS236: Deep Generative Models** explores the theory and practice of generative modeling using deep neural networks. The course covers fundamental concepts and recent advances in generative models, including:

- **Variational Autoencoders (VAEs)** - Learning latent variable models through variational inference
- **Generative Adversarial Networks (GANs)** - Training generative models through adversarial optimization
- **Normalizing Flows** - Constructing invertible transformations for density estimation
- **Autoregressive Models** - Sequential generation using conditional distributions

The course emphasizes both theoretical foundations and practical implementation, covering applications in computer vision, natural language processing, and other domains.

**Course Information:**

- **Institution**: Stanford University
- **Department**: Computer Science
- **Course Code**: CS236
- **Semester**: Spring 2019
- **Instructor**: [Course Staff]
- **Grading**: Homeworks (15% × 3 = 45%) + Midterm (15%) + Final Project (40%)

---

## Repository Structure

This repository contains all course materials organized as follows:

```
CS236_DGM/
├── README.md                          # This comprehensive guide
├── cs236_c.jpg                        # Course banner image
├──
├── HW1_Basic_VAE/                     # Homework 1: Basic Variational Autoencoders
│   ├── description/                   # Assignment PDF and solutions
│   ├── src/                           # Source code (main.py, model.py, dataset.py)
│   ├── data/                          # Training data (papers.csv)
│   ├── checkpoints/                   # Model checkpoints (*.pth, *.pkl)
│   ├── configs/                       # Configuration files (config.yml)
│   ├── requirements/                  # Dependencies (requirements.txt)
│   ├── scripts/                       # Submission script (make_submission.sh)
│   ├── docs/                          # Additional documentation
│   ├── report/                        # LaTeX report template (nips_2018.*)
│   ├── logs/                          # Training logs
│   └── *.png, *.pkl                   # Generated outputs and data
│
├── HW2_Advanced_VAEs/                 # Homework 2: Advanced VAEs
│   ├── description/                   # Assignment PDF and solutions
│   ├── src/
│   │   ├── codebase/                  # Core implementation
│   │   │   ├── utils.py              # Utility functions
│   │   │   ├── models/               # Model implementations
│   │   │   │   ├── vae.py            # Basic VAE
│   │   │   │   ├── gmvae.py          # Gaussian Mixture VAE
│   │   │   │   ├── ssvae.py          # Semi-supervised VAE
│   │   │   │   └── fsvae.py          # Factorized Split VAE (bonus)
│   │   │   └── nns/                  # Neural network components
│   │   └── run_*.py                  # Training scripts for each model
│   ├── requirements/                  # Dependencies
│   ├── scripts/                       # Submission script
│   ├── docs/                          # Detailed README
│   └── report/                        # LaTeX template
│
├── HW3_GANs/                         # Homework 3: Generative Adversarial Networks
│   ├── description/                   # Assignment PDF and solutions
│   ├── src/
│   │   ├── codebase/
│   │   │   ├── gan.py                # GAN implementation
│   │   │   └── network.py            # Network architectures
│   │   ├── data/                     # FashionMNIST dataset
│   │   ├── out_*/                     # Generated outputs by loss type
│   │   │   ├── fake_*.png            # Generated images at checkpoints
│   │   │   ├── real.png              # Real data samples
│   │   │   └── model_*.pt            # Model checkpoints
│   │   ├── run_gan.py                # Standard GAN training
│   │   ├── run_conditional_gan.py    # Conditional GAN training
│   │   └── test_gan.py               # Testing/debugging script
│   ├── scripts/                       # Submission script
│   └── report/                        # LaTeX template
│
├── doc/                              # Documentation and papers
│   ├── CS236Project*.pdf             # Project guidelines and examples
│   ├── Deep Learning Book - Ian Goodfellow.pdf
│   ├── paper-Auto-Encoding Variational Bayes.pdf
│   ├── Improving Variational Inference with Inverse Autoregressive Flow.pdf
│   └── Glow- Generative Flow with Invertible 1x1 Convolutions.pdf
│
├── exam/                            # Course examinations
│   └── CS236_mid_term_soln.pdf      # Midterm solutions
│
└── Template/                        # Report templates
    ├── nips_style_files.zip         # Complete LaTeX template package
    └── nips_style_files/            # Extracted template files
        ├── nips_2018.sty
        ├── nips_2018.tex
        └── nips_2018.pdf
```

---

## Prerequisites

### Technical Requirements

- **Python**: 3.6+ (tested with 3.6-3.7)
- **PyTorch**: 0.4.1.post2 (specific version required)
- **CUDA**: Optional but recommended for GPU acceleration
- **Operating System**: Linux/macOS/Windows (Linux recommended)

### Mathematical Background

- Probability theory and statistics
- Linear algebra and multivariate calculus
- Basic machine learning concepts
- Deep learning fundamentals (neural networks, backpropagation)

### Programming Skills

- Python programming
- PyTorch/TensorFlow experience (PyTorch preferred)
- Jupyter notebooks for experimentation
- Command-line interface usage

---

## Homeworks

### HW1: Basic Variational Autoencoders (15%)

**Focus**: Implementation of fundamental VAE concepts on text data

- **Dataset**: Research paper abstracts (text generation)
- **Key Concepts**: ELBO, reparameterization trick, KL divergence
- **Deliverables**: Model code, generated samples, analysis

### HW2: Advanced VAEs (15%)

**Focus**: Extensions and improvements to basic VAEs

- **Models**: Gaussian Mixture VAE, Semi-supervised VAE, IWAE
- **Dataset**: MNIST (image generation)
- **Key Concepts**: Importance weighting, semi-supervised learning, mixture models
- **Bonus**: Factorized Split VAE implementation

### HW3: Generative Adversarial Networks (15%)

**Focus**: GAN training and conditional generation

- **Dataset**: FashionMNIST
- **Key Concepts**: Adversarial training, different loss functions
- **Variants**: Standard GAN, Conditional GAN, Non-saturating loss, Wasserstein GAN

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SKKSaikia/CS236_DGM.git
cd CS236_DGM
```

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv cs236_env
source cs236_env/bin/activate  # On Windows: cs236_env\Scripts\activate

# Or using conda
conda create -n cs236 python=3.6
conda activate cs236
```

### 3. Install Dependencies per Assignment

#### HW1 Dependencies

```bash
cd HW1_Basic_VAE
pip install -r requirements/requirements.txt
# Installs: matplotlib, numpy, pyyaml, torch
```

#### HW2 Dependencies

```bash
cd ../HW2_Advanced_VAEs
pip install -r requirements/requirements.txt
# Installs: tqdm==4.20.0, numpy==1.15.2, torchvision==0.2.1, torch==0.4.1.post2
```

#### HW3 Dependencies

```bash
cd ../HW3_GANs
pip install -r ../HW2_Advanced_VAEs/requirements/requirements.txt
# Same as HW2: torch, torchvision, numpy, tqdm
```

### 4. Verify Installation

```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## Running the Assignments

### HW1: Basic VAE

```bash
cd HW1_Basic_VAE/src

# Run the complete training pipeline
python main.py

# This will:
# 1. Load configuration from ../configs/config.yml
# 2. Train VAE on research paper abstracts
# 3. Generate samples for different datasets
# 4. Save outputs: shakespeare.png, random.png, nips.png
# 5. Create checkpoint: ../checkpoints/checkpoint.pth
```

**Expected Outputs:**

- `shakespeare.pkl/png`: Samples conditioned on Shakespeare text
- `random.pkl/png`: Random samples from latent space
- `nips.pkl/png`: Samples conditioned on NIPS abstracts
- `answers.pkl`: Model answers/results
- `samples.txt`: Text samples

### HW2: Advanced VAEs

```bash
cd HW2_Advanced_VAEs/src

# Train individual models (run from repository root)
python -m HW2_Advanced_VAEs.src.run_vae        # Basic VAE
python -m HW2_Advanced_VAEs.src.run_gmvae      # Gaussian Mixture VAE
python -m HW2_Advanced_VAEs.src.run_ssvae      # Semi-supervised VAE
python -m HW2_Advanced_VAEs.src.run_fsvae      # Factorized Split VAE (bonus)

# Each script will:
# - Load MNIST dataset automatically
# - Train for specified epochs (see code for defaults)
# - Save periodic checkpoints
# - Generate sample images
```

**Performance Notes:**

- **Basic VAE**: ~5 minutes on CPU, ~1 minute on GPU
- **GMVAE**: ~5 minutes on CPU, ~1 minute on GPU
- **SSVAE**: ~50 minutes on CPU, ~10 minutes on GPU
- **FSVAE**: Hours on CPU, ~30 minutes on GPU (stop when samples look reasonable)

### HW3: GANs

```bash
cd HW3_GANs/src

# Train standard GAN with non-saturating loss
python run_gan.py --device cpu --num_epochs 10 --loss_type nonsaturating

# Train with Wasserstein GAN + gradient penalty
python run_gan.py --device cuda --num_epochs 100 --loss_type wasserstein_gp

# Train conditional GAN
python run_conditional_gan.py --device cuda --num_epochs 50

# Test basic GAN functionality (debugging)
python test_gan.py
```

**Command Line Arguments:**

- `--device`: 'cpu' or 'cuda' (GPU recommended for faster training)
- `--num_epochs`: Number of training epochs (default: 1 for testing)
- `--loss_type`: 'nonsaturating' or 'wasserstein_gp'

**Output Directories:**

- `out_nonsaturating/`: Results from non-saturating GAN loss
- `out_wasserstein_gp/`: Results from WGAN-GP
- `out_nonsaturating_conditional/`: Conditional GAN results

---

## Submission Instructions

### HW1 Submission

```bash
cd HW1_Basic_VAE
bash scripts/make_submission.sh
# Creates: hw1.zip containing:
# - answers.pkl, samples.txt
# - Generated images: shakespeare.png, random.png, nips.png
# - Raw data: shakespeare_raw.pkl, random_raw.pkl, nips_raw.pkl
# - Source code: main.py, model.py
```

### HW2 Submission

```bash
cd HW2_Advanced_VAEs
bash scripts/make_submission.sh
# Creates: hw2.zip containing:
# - Modified source files:
#   * codebase/utils.py
#   * codebase/models/vae.py
#   * codebase/models/gmvae.py
#   * codebase/models/ssvae.py
#   * codebase/models/fsvae.py (bonus)
```

### HW3 Submission

```bash
cd HW3_GANs
bash scripts/make_submission.sh
# Creates: hw3.zip containing:
# - codebase/gan.py (your implementation)
# - out*/fake_0900.png (generated images at epoch 900)
```

**Important Notes:**

- Only submit modified files as specified
- Do not change hyperparameters unless instructed
- Ensure code runs without errors before submission
- Test submissions by extracting and running in a fresh environment

---

## Detailed Assignment Breakdown

### HW1: Basic VAE - Implementation Details

**Files to Modify:**

- `src/main.py`: Main training script
- `src/model.py`: VAE architecture
- `src/dataset.py`: Data loading and preprocessing

**Key Components:**

1. **Encoder Network**: Maps input text to latent distribution parameters (μ, σ)
2. **Decoder Network**: Reconstructs text from latent samples
3. **Reparameterization Trick**: Enables gradient flow through stochastic sampling
4. **ELBO Loss**: Evidence Lower Bound optimization objective

**Datasets Used:**

- **Shakespeare**: Character-level text generation
- **Random**: Unconditional generation from prior
- **NIPS**: Scientific abstract generation

### HW2: Advanced VAEs - Implementation Checklist

**Required Functions to Implement (in order):**

1. `sample_gaussian` in `utils.py`
2. `negative_elbo_bound` in `vae.py`
3. `log_normal` in `utils.py`
4. `log_normal_mixture` in `utils.py`
5. `negative_elbo_bound` in `gmvae.py`
6. `negative_iwae_bound` in `vae.py`
7. `negative_iwae_bound` in `gmvae.py`
8. `negative_elbo_bound` in `ssvae.py`
9. `negative_elbo_bound` in `fsvae.py` (bonus)

**Model Variants:**

- **VAE**: Basic variational autoencoder with IWAE objective
- **GMVAE**: Gaussian mixture model in latent space
- **SSVAE**: Semi-supervised learning with labeled/unlabeled data
- **FSVAE**: Factorized split of latent variables

### HW3: GANs - Architecture & Training

**Core Components:**

- **Generator**: Transforms random noise to data distribution
- **Discriminator**: Distinguishes real from generated samples
- **Training Loop**: Alternating optimization of G and D

**Loss Functions:**

- **Non-saturating**: Standard GAN loss with improved gradient flow
- **Wasserstein GP**: Wasserstein distance with gradient penalty for stability

**Data**: FashionMNIST (28×28 grayscale fashion images)

---

## Resources & Documentation

### Official Course Resources

- [Course Website](https://deepgenerativemodels.github.io/)
- [Course Notes](https://deepgenerativemodels.github.io/notes/index.html)
- [Stanford CS236 Syllabus](https://deepgenerativemodels.github.io/)

### Key Papers & Books

- **"Auto-Encoding Variational Bayes"** - Kingma & Welling (ICLR 2014)
- **"Generative Adversarial Nets"** - Goodfellow et al. (NIPS 2014)
- **"Deep Learning"** - Goodfellow, Bengio & Courville (Book)
- **"Improving Variational Inference with Inverse Autoregressive Flow"**
- **"Glow: Generative Flow with Invertible 1x1 Convolutions"**

### Additional Resources

- [PyTorch Introduction](notes/IntroductiontoPyTorch.pdf)
- [OpenAI Generative Models Blog](https://blog.openai.com/generative-models/)
- [Generative Models Collection](https://github.com/wiseodd/generative-models)

### LaTeX Templates

- Complete NIPS 2018 style template in `Template/nips_style_files.zip`
- Used for all assignment reports and final project
- Includes: `nips_2018.sty`, `nips_2018.tex`, `nips_2018.pdf`

---

## Final Project

The final project constitutes 40% of the course grade and involves implementing a novel generative model or applying existing techniques to a new domain.

### Project Components

1. **Proposal** (10%): Project idea, motivation, and planned methodology
2. **Implementation** (20%): Complete system with evaluation
3. **Report** (10%): Comprehensive write-up in NIPS format

### Available Resources

- [Project Guidelines](doc/CS236PosterGuidelines.pdf)
- [Proposal Guidelines](doc/CS236ProjectProposalGuidelines.pdf)
- [Final Report Guidelines](doc/CS236ProjectFinalReportGuide.pdf)
- [Project Examples](doc/CS236ProjectExamples.pdf)
- [Internal Project Assignment](doc/CS236ProjectAssignmentinternal-cs236.pdf)

### Timeline (Typical)

- **Week 6-7**: Project proposal submission
- **Week 8-9**: Proposal feedback and refinement
- **Week 10**: Midterm evaluation
- **Week 11-14**: Implementation and experimentation
- **Week 15**: Final report submission

---

## Troubleshooting

### Common Issues

**HW1 Issues:**

- **Import errors**: Ensure all requirements are installed
- **CUDA errors**: Use `--device cpu` if no GPU available
- **Memory errors**: Reduce batch size in config.yml

**HW2 Issues:**

- **Long training times**: Start with small number of epochs for testing
- **NaN losses**: Check gradient clipping and learning rates
- **Shape mismatches**: Verify tensor dimensions in forward passes

**HW3 Issues:**

- **Mode collapse**: Try different learning rates or architectures
- **Training instability**: Use gradient penalty (WGAN-GP)
- **Poor sample quality**: Increase training time or model capacity

### Performance Optimization

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU usage
nvidia-smi

# CPU optimization
export OMP_NUM_THREADS=4  # Limit CPU threads
```

### Debugging Tips

- Use `test_gan.py` to isolate GAN training issues
- Add gradient clipping to stabilize training
- Monitor losses: discriminator loss should hover around log(2)
- Visualize generated samples regularly during training

---

## Contributing

This repository contains course materials from Stanford CS236 Spring 2019. While the core assignments and materials are provided as-is, contributions for:

- Bug fixes in the provided code
- Additional documentation or examples
- Improved setup instructions
- Performance optimizations

are welcome. Please create an issue or pull request for any improvements.

### Academic Integrity

- Do not share solutions publicly
- Complete assignments independently
- Cite sources appropriately in reports
- Follow Stanford's honor code

---

**For questions or issues**, please refer to the course notes, check existing issues, or contact course staff through official channels.

_Last updated: December 2025_ | _Stanford CS236 - Deep Generative Models_

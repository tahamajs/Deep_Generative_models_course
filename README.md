# Deep Generative Models — Course Materials (DGM)

This repository collects lecture slides, assignments (CAs), code notebooks, reports, and reference papers used in the "Deep Generative Models" course (University of Tehran). The materials are organized to be reproducible and educational: each assignment contains an annotated Jupyter notebook, supporting code, and a report.

Course Overview

The "Deep Generative Models" (DGM) course covers advanced topics in machine learning focused on generative modeling techniques. Generative models learn the underlying distribution of data to generate new samples, enabling applications in image synthesis, anomaly detection, data augmentation, and more.

Key topics covered in the course include:

- **Variational Autoencoders (VAEs)**: Probabilistic latent variable models for learning compressed representations and generating new data.
- **Normalizing Flows**: Invertible transformations that allow exact density estimation and sampling.
- **Generative Adversarial Networks (GANs)**: Adversarial training frameworks for high-quality sample generation.
- **Diffusion Models**: Denoising diffusion probabilistic models for state-of-the-art image generation.
- **Score-based Generative Models**: Methods using score functions for sampling from complex distributions.

The course assignments (CA1-CA4) progressively build skills in implementing and evaluating these models on real datasets like CelebA, FashionMNIST, and custom image datasets.

## Key Concepts in Deep Generative Models

This section provides a high-level overview of the core mathematical and conceptual foundations that unify the different generative modeling approaches covered in the course.

### Probabilistic Generative Modeling

Generative models aim to learn the underlying data distribution $p(\mathbf{x})$ from samples $\mathbf{x} \sim p_{\text{data}}$. The goal is to:

1. **Density Estimation**: Approximate $p(\mathbf{x})$ or learn a tractable distribution $p_\theta(\mathbf{x})$ that matches the data.
2. **Sampling**: Generate new samples $\mathbf{x}' \sim p_\theta(\mathbf{x})$ from the learned distribution.
3. **Inference**: Compute posterior probabilities or latent representations for downstream tasks.

### Maximum Likelihood Estimation

Most generative models are trained by maximizing the log-likelihood:

$$
\theta^* = \arg\max_\theta \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} [\log p_\theta(\mathbf{x})]
$$

This is equivalent to minimizing the KL divergence between data and model distributions:

$$
\theta^* = \arg\min_\theta \text{KL}(p_{\text{data}} || p_\theta)
$$

### Latent Variable Models

Many generative models introduce latent variables $\mathbf{z}$ to simplify modeling:

- **Joint Distribution**: $p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z})$
- **Marginal Likelihood**: $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) d\mathbf{z}$
- **Posterior Inference**: $p_\theta(\mathbf{z}|\mathbf{x}) = \frac{p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z})}{p_\theta(\mathbf{x})}$

### Amortized Variational Inference

Exact inference in latent models is often intractable. Variational inference approximates posteriors using a recognition model:

- **Variational Distribution**: $q_\phi(\mathbf{z}|\mathbf{x}) \approx p_\theta(\mathbf{z}|\mathbf{x})$
- **Evidence Lower Bound (ELBO)**: $\log p_\theta(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [\log p_\theta(\mathbf{x}|\mathbf{z})] - \text{KL}(q_\phi(\mathbf{z}|\mathbf{x}) || p(\mathbf{z}))$

### Change of Variables and Normalizing Flows

Normalizing flows provide exact density estimation through invertible transformations:

- **Transformation**: $\mathbf{z} = f(\mathbf{x})$ where $f$ is invertible
- **Density**: $p_\mathbf{z}(\mathbf{z}) = p_\mathbf{x}(\mathbf{x}) |\det \frac{\partial f}{\partial \mathbf{x}}|$
- **Composition**: Complex transformations built from simple invertible layers

### Adversarial Training

GANs use adversarial objectives instead of explicit likelihoods:

- **Generator**: $G: \mathbf{z} \mapsto \mathbf{x}$, learns to fool discriminator
- **Discriminator**: $D: \mathbf{x} \mapsto [0,1]$, learns to distinguish real from fake
- **Minimax Objective**: $\min_G \max_D \mathbb{E}_{\mathbf{x}} [\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z}} [\log (1 - D(G(\mathbf{z})))]$

### Diffusion Processes

Diffusion models gradually add noise and learn to reverse the process:

- **Forward Process**: $q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})$
- **Reverse Process**: $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$
- **Training**: Predict noise added at each timestep

### Score-Based Generative Modeling

Score-based models learn the score function (gradient of log-density):

- **Score Function**: $\nabla_\mathbf{x} \log p_t(\mathbf{x})$ where $p_t$ is a noisy version of data
- **Score Matching**: Minimize $\mathbb{E}_{p_t(\mathbf{x})} [\frac{1}{2} || \mathbf{s}_\theta(\mathbf{x}, t) - \nabla_\mathbf{x} \log p_t(\mathbf{x}) ||^2 ]$
- **Sampling**: Use Langevin dynamics or SDEs to sample from learned scores

### Evaluation Metrics

Assessing generative model quality requires both quantitative and qualitative measures:

- **Likelihood-based**: Log-likelihood, bits-per-dimension (for flows)
- **Distribution-based**: Fréchet Inception Distance (FID), Kernel Inception Distance (KID)
- **Sample Quality**: Inception Score (IS), perceptual quality
- **Diversity**: Coverage, density of generated samples

### Connections Between Approaches

- **VAEs as Flow-like Models**: Reparameterization connects to normalizing flows
- **Diffusion as Hierarchical VAEs**: Diffusion steps can be viewed as latent layers
- **Score-Based and Diffusion**: Score functions are central to both
- **GANs and All**: Adversarial training can be applied to any generative model

Understanding these unifying principles helps in choosing appropriate models for different applications and in developing new generative techniques.

Prerequisites: Strong background in deep learning (PyTorch/TensorFlow), probability theory, and optimization. Specifically:

### Mathematical Prerequisites

- **Probability Theory**: Random variables, distributions (Gaussian, Bernoulli, Categorical), expectation, variance, Bayes' theorem, maximum likelihood estimation
- **Information Theory**: Entropy, cross-entropy, Kullback-Leibler divergence, mutual information
- **Linear Algebra**: Vector/matrix operations, eigenvalues/eigenvectors, singular value decomposition, tensor operations
- **Calculus**: Partial derivatives, chain rule, gradient descent, automatic differentiation
- **Statistics**: Hypothesis testing, confidence intervals, bias-variance tradeoff

### Machine Learning Fundamentals

- **Supervised Learning**: Classification, regression, loss functions, regularization
- **Neural Networks**: Feedforward networks, backpropagation, activation functions, initialization
- **Convolutional Networks**: Convolutional layers, pooling, receptive fields, modern architectures (ResNet, Transformer)
- **Optimization**: Stochastic gradient descent variants (Adam, RMSProp), learning rate scheduling, batch normalization
- **Regularization**: Dropout, weight decay, early stopping, data augmentation

### Programming and Tools

- **Python**: Advanced features (decorators, context managers, multiprocessing), NumPy/Pandas proficiency
- **Deep Learning Frameworks**: PyTorch (tensors, autograd, nn.Module, DataLoader) or TensorFlow/Keras
- **Version Control**: Git basics, collaborative workflows
- **Development Environment**: Jupyter notebooks, IDEs (VS Code, PyCharm), command-line tools

### Recommended Background Reading

- "Deep Learning" by Goodfellow, Bengio, Courville (Chapters 1-5, 13-20)
- "Pattern Recognition and Machine Learning" by Bishop (Chapters 1-4, 8-10)
- "Probabilistic Machine Learning" by Murphy (Chapters 1-3, 21-24)

Students without this background may find the course challenging and are encouraged to review these topics beforehand.

## Table of contents

- Course Overview
- Repository structure and purpose
- Quick start (setup & run)
- Notebooks and assignments (CA1..CA4) — summary and status
- Deep dive: CA1 (Variational Autoencoders)
- Deep dive: CA2 (GANs & Normalizing Flows)
- Deep dive: CA3 (Diffusion and Score-based Models)
- Deep dive: CA4 (Fine-Tuning Vision-Language Models)
- Data, storage and artifact management
- Reproducibility checklist and recommended configuration
- Testing and lightweight smoke checks
- Common issues and troubleshooting
- References and further reading
- Credits and license

---

## Repository structure (top-level)

- `CA1_Variational_Autoencoders/` — Course Assignment 1: Variational Autoencoders
  - `code/` — Jupyter notebooks and code used for experiments (e.g., `code.ipynb`).
  - `description/` — Assignment description PDF.
  - `report/` — PDF reports and figures.
  - `images/` — Generated images and visualizations.
  - `train/` — Training datasets (CelebA subset: smile/non-smile images).
  - `README.md` — Detailed documentation for CA1.
- `CA2_GANs_Normalizing_Flows/` — Course Assignment 2: GANs and Normalizing Flows
  - `code/` — Jupyter notebooks (e.g., `CA2_DGM.ipynb`, `Q2_final_res.ipynb`).
  - `description/` — Assignment description PDF.
  - `report/` — PDF reports and figures.
  - `images/` — Generated samples and visualizations.
  - `README.md` — Detailed documentation for CA2.
- `CA3_Diffusion_Models/` — Course Assignment 3: Diffusion and Score-based Models
  - `codes/` — Jupyter notebooks (e.g., `Diffusion_Models.ipynb`, `score_based_models.ipynb`).
  - `description/` — Assignment description PDF.
  - `report/` — PDF reports and figures.
  - `images/` — Generated samples and visualizations.
  - `README.md` — Detailed documentation for CA3.
- `CA4_Vision_Language_Model/` — Course Assignment 4: Vision-Language Models
  - `code/` — Jupyter notebooks (e.g., `final_CA4_training.ipynb`, evaluation notebooks).
  - `description/` — Assignment description PDF.
  - `report/` — PDF reports and figures.
  - `images/` — Generated images and visualizations.
  - `README.md` — Detailed documentation for CA4.
- `Slides/` — Lecture slides and course material used in class.
  - `DGM_Fall_2023_Slides/` — Course lecture slides.
  - `Stanford_slides/` — Supplementary slides from Stanford's CS236 course.
  - `MITSlides/` — Additional slides from MIT and other sources.
- `Exams/` — Past exams and solutions.
- `ExtraNotes/` — Additional notes, homework templates, supplementary PDFs, and exploratory materials (e.g., `homework_template/`, research papers on D-separation).
- `OtherTermAssignments/` — Assignments from other terms or related courses, including CA1, CA2, CA3 from previous semesters.
- `OtherUniversityLecturs/` — Lecture materials from other universities and courses.
  - `CS236_DGM/` — Complete materials from Stanford's CS236 Deep Generative Models course.
  - `SUT/` — Materials from Sharif University of Technology.
  - `notes/` — General notes and documentation.
  - `ssi2023/` — Materials from SSI 2023.
- `PaperSLecturs/` — Research papers, codes, and lecture materials on advanced topics.
  - `AI4Science_Codes/` — Code implementations for AI for Science applications.
  - `AI4Science_Papers/` — Research papers on AI for Science.
  - `Generative_Codes/` — Codes for generative models (e.g., conditional flow matching, Mamba).
  - `Generative_Models_Papers/` — Papers on generative models.
  - `LLM_Codes/` — Large Language Model implementations.
  - `LLM_Papers/` — Papers on LLMs.
  - `Vision_Codes/` — Computer vision codes.
  - `Vision_Papers/` — Papers on computer vision.

This repository is primarily an educational resource. Notebooks are annotated for readability and (where possible) reorganized to centralize imports and configuration.

---

## Quick start — environment and running notebooks

Recommended steps to set up a local, reproducible environment. We recommend using virtual environments to isolate dependencies.

### Option 1: Using venv (Python built-in)

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install core dependencies (adjust PyTorch install for your CUDA version):

```bash
pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Example for CUDA 11.8
pip install matplotlib numpy scipy scikit-learn jupyterlab pytorch-fid tqdm
```

### Option 2: Using conda

1. Create and activate a conda environment:

```bash
conda create -n dgm python=3.10
conda activate dgm
```

2. Install dependencies:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia  # Adjust CUDA version
conda install matplotlib numpy scipy scikit-learn jupyterlab tqdm
pip install pytorch-fid
```

### Additional Dependencies

- For advanced notebooks (e.g., diffusion models): `pip install torchdiffeq` (for ODE solvers in score-based models).
- For visualization: `pip install seaborn plotly`.
- For reproducibility: `pip install wandb` (optional, for experiment tracking).

Notes:

- If you have a CUDA-enabled GPU, install the matching `torch`/`torchvision` binaries using the official instructions at https://pytorch.org. For CPU-only, omit CUDA-specific installs.
- `pytorch-fid` is used in CA2 for FID computation. If installation fails, consider alternatives like `clean-fid`.
- For CA3 diffusion models, ensure you have sufficient GPU memory (at least 8GB VRAM recommended).

3. Launch JupyterLab from the project root and open the notebook you'd like to run:

```bash
jupyter lab
```

4. Important safety note: the notebooks in the assignment folders were edited as part of a documentation pass (imports consolidated, configuration cell added). The editorial pass did not execute the notebooks. Before running long training jobs, review the `Setup and Configuration` cell in each notebook and run smoke tests described below.

---

## Notebooks and assignments — short summary and status

This section summarizes the primary assignments and their current status in the repository.

- CA1 (folder: `CA1_Variational_Autoencoders/`)

  - Focus: Variational Autoencoders (VAE) and experiments exploring latent structure.
  - Key files: `code/code.ipynb`, `report/DGM_CA1_final_EN.pdf`, `README.md`.
  - Datasets: CelebA dataset (smiling/non-smiling classification task) stored in `train/smile/` and `train/non_smile/`.
  - Status: Fully improved with comprehensive README explaining all VAE concepts in depth, notebook reorganized (imports consolidated, configuration cell added, explanatory Markdown blocks added), and overview cell inserted for educational clarity. The accompanying PDF report summary was synthesized in the CA1 README due to extraction limitations.

- CA2 (folder: `CA2_GANs_Normalizing_Flows/`)

  - Focus: Normalizing flows (RealNVP) and GANs (DCGAN-style) applied to FashionMNIST. Includes OOD detection experiments (MNIST, KMNIST) and FID evaluation for GANs.
  - Key files: `code/CA2_DGM.ipynb`, `code/Q2_final_res.ipynb`, `README.md`.
  - Datasets: FashionMNIST, MNIST, KMNIST (downloaded via torchvision).
  - Status: Fully improved with comprehensive README explaining GANs and Normalizing Flows concepts in depth, notebook reorganized (imports consolidated, configuration cell added, explanatory Markdown blocks added), and overview cell inserted for educational clarity.

- CA3 (folder: `CA3_Diffusion_Models/`)

  - Focus: Diffusion Models and Score-based Generative Models.
  - Key files: `codes/Diffusion_Models.ipynb`, `codes/score_based_models.ipynb`, `report/DGM_CA3_EN_final.pdf`, `README.md`.
  - Datasets: Likely image datasets such as CIFAR-10 or custom datasets for diffusion processes.
  - Status: Fully improved with comprehensive README explaining Diffusion and Score-based Models concepts in depth, notebooks reorganized (imports consolidated, configuration cells added, explanatory Markdown blocks added), and overview cells inserted for educational clarity.

- CA4 (folder: `CA4_Vision_Language_Model/`)

  - Focus: Fine-tuning Vision-Language Models (Paligemma) on CLEVR dataset using PEFT/LoRA techniques.
  - Key files: `code/final_CA4_training.ipynb`, `code/eval_p1/final_CA4_results1.ipynb`, `code/eval_p2/final_CA4_results2.ipynb`, `README.md`.
  - Datasets: CLEVR (Compositional Language and Elementary Visual Reasoning).
  - Status: Fully improved with comprehensive README explaining Vision-Language Models, PEFT, and LoRA concepts in depth, notebook reorganized (imports consolidated, configuration cell added, explanatory Markdown blocks added), and overview cell inserted for educational clarity. Removed Google Drive dependencies, fixed evaluation metrics, and switched to Hugging Face model loading.

Other folders (`Slides/`, `Extra/`, `Exams/`) contain lecture materials, relevant readings, and supporting documents.

---

## Deep dive: CA1 (Variational Autoencoders)

CA1 introduces Variational Autoencoders (VAEs), a cornerstone of generative modeling that combines variational inference with autoencoder architectures.

### Key Concepts Explained

**Variational Autoencoders (VAEs)** are generative models that learn to encode input data into a low-dimensional latent space and decode it back to reconstruct the original data. Unlike traditional autoencoders, VAEs learn a probabilistic latent representation, allowing them to generate new samples by sampling from the learned distribution.

- **Encoder (Inference Network)**: Maps input data (e.g., images) to parameters of a latent distribution, typically a Gaussian with mean μ and variance σ².
- **Reparameterization Trick**: Enables gradient flow through stochastic sampling by expressing z = μ + σ \* ε, where ε ~ N(0,1).
- **Decoder (Generative Network)**: Takes latent samples and reconstructs the original data distribution.
- **Evidence Lower Bound (ELBO)**: The loss function combines reconstruction loss (how well the decoder reconstructs inputs) and KL divergence (regularization term that encourages the latent distribution to be close to a standard normal).
- **Latent Space Properties**: VAEs can learn disentangled representations where different dimensions correspond to interpretable factors of variation.

VAEs balance reconstruction fidelity with latent space regularization, making them useful for tasks like image generation, anomaly detection, and representation learning.

### Methodology

#### VAE Architecture

**Encoder Network**: The encoder progressively reduces spatial dimensions while increasing feature depth using convolutional layers, batch normalization, dropout, and LeakyReLU activations, culminating in linear layers that output mean (μ) and log variance (log σ²) for the latent distribution.

**Reparameterization Trick**: z = μ + σ ⊙ ε, where ε ~ N(0, I) and σ = exp(0.5 × logvar), enabling gradient flow through stochastic sampling.

**Decoder Network**: Mirrors the encoder with transposed convolutions, reconstructing images from latent vectors.

#### Loss Function

The Evidence Lower Bound (ELBO) combines reconstruction loss (MSE) and KL divergence regularization to balance fidelity and generalization.

#### Training Configuration

- Epochs: 1000
- Learning Rate: 0.0005
- Optimizer: Adam
- Batch Size: 32
- Latent Dimension: 32

### Results and Analysis

**Training Progress**: Loss reduced by ~57% over training, with stable convergence and good generalization (validation loss tracks training loss closely).

**Image Reconstruction**: Effective preservation of facial features with slight smoothing; quantitative metrics show significant error reduction.

**Image Generation**: Diverse, realistic facial images generated by sampling from the prior, demonstrating generative capabilities.

**Latent Space Analysis**: Embeddings show clustering by smile/non-smile classes, with successful interpolation and classification performance.

Key components in `CA1_Variational_Autoencoders/code/code.ipynb`:

1. **VAE Architecture**: Encoder (inference network) that maps images to latent distributions, decoder (generative network) that reconstructs images from latent samples.
2. **Reparameterization Trick**: Enables backpropagation through stochastic sampling.
3. **Loss Function**: Combination of reconstruction loss (e.g., MSE or BCE) and KL divergence regularization.
4. **Latent Space Analysis**: Visualization of latent representations, interpolation, and clustering.
5. **Experiments**: Training on CelebA dataset for smiling/non-smiling classification in latent space.

Why run CA1?

- Understand the trade-off between reconstruction quality and latent regularization.
- Explore disentangled representations and their applications in downstream tasks.
- Compare VAEs with other generative models introduced later in the course.

Files of interest in `CA1_Variational_Autoencoders/`:

- `code/code.ipynb` — the annotated notebook (imports consolidated and configuration cell added).
- `README.md` — detailed documentation with synthesized report summary.
- `train/` — contains CelebA subset (smile/non-smile) for training.
- `report/` — PDF reports including `DGM_CA1_final_EN.pdf`.
- `images/` — generated visualizations and outputs.

High-level suggested execution order:

1. Review the configuration cell for hyperparameters (latent_dim, learning_rate, etc.).
2. Load and preprocess the CelebA dataset.
3. Train the VAE and monitor reconstruction quality and KL divergence.
4. Analyze latent space: visualize embeddings, perform interpolations, evaluate classification performance.

---

## Deep dive: CA2 (GANs & Normalizing Flows)

CA2 is both pedagogical and experimental. It demonstrates two complementary approaches to deep generative modeling:

### Key Concepts Explained

**Normalizing Flows** are generative models that learn invertible transformations to map a simple base distribution (like a standard normal) to a complex data distribution. They provide exact likelihood computation and can be trained via maximum likelihood.

- **RealNVP (Real-valued Non-Volume Preserving)**: Uses coupling layers that split input dimensions and transform one half conditioned on the other using scale and shift functions.
- **Invertibility**: The transformation must be invertible to compute both forward (data to latent) and inverse (latent to data) mappings.
- **Log-Determinant of Jacobian**: Tracks volume changes during transformation for exact density estimation.
- **Advantages**: Exact density evaluation enables likelihood-based evaluation and out-of-distribution detection.

**Generative Adversarial Networks (GANs)** consist of two neural networks trained simultaneously: a generator that creates fake data and a discriminator that distinguishes real from fake. They learn through adversarial training without requiring explicit density estimation.

- **Generator**: Learns to map random noise to realistic data samples.
- **Discriminator**: Learns to classify real vs. generated samples.
- **Adversarial Loss**: Generator minimizes the probability of discriminator correctly identifying fakes, while discriminator maximizes classification accuracy.
- **DCGAN**: Uses convolutional architectures with batch normalization and specific activation functions for stable training.
- **Evaluation Challenges**: Lack of explicit likelihood makes evaluation tricky; metrics like FID (Fréchet Inception Distance) compare distributions in feature space.

These approaches complement each other: flows provide mathematical rigor and exact evaluation, while GANs excel at generating high-quality samples.

### Results Summary

**GAN Training on FashionMNIST**: Best FID score of 171.67 achieved at epoch 8, with overall ~23% improvement over 10 epochs. Samples show progressive quality improvement from blurry initial images to sharp, detailed fashion items.

**Normalizing Flows**: RealNVP implementation with coupling layers for invertible transformations, enabling exact density estimation and OOD detection via log-likelihoods on MNIST/KMNIST.

1. RealNVP (normalizing flows): an explicit density model trained by maximum likelihood. The notebook contains:

   - Implementation of coupling layers and RealNVP stacking.
   - Training using negative log-likelihood (NLL).
   - Computation of log-likelihoods for in-distribution and out-of-distribution (OOD) datasets (MNIST, KMNIST).
   - Visualization of generated samples via the inverse mapping.

2. GAN (DCGAN-style): an adversarial generator trained to produce realistic fashion images. The notebook contains:

   - DCGAN-style `Generator` and `Discriminator` classes implemented in PyTorch.
   - A training loop alternating generator and discriminator updates.
   - Fixed noise vectors to produce consistent image grids for visual progress.
   - FID evaluation using `pytorch-fid` computed per-epoch.

Why run CA2?

- RealNVP gives explicit densities and allows for direct OOD detection experiments based on log-likelihood.
- Training RealNVP in a learned latent space (via an encoder-decoder) reduces dimensionality and speeds up flow training.
- GAN training provides qualitative sample generation and a complementary evaluation via FID.

Files of interest in `CA2_GANs_Normalizing_Flows/`:

- `code/CA2_DGM.ipynb` — the annotated notebook (imports consolidated and a configuration cell added).
- `code/Q2_final_res.ipynb` — additional results and experiments.
- `README.md` — localized instructions, reproducibility notes and quick-start steps.
- `report/` — PDF reports including `DGM_CA2_final_EN.pdf`.
- `images/` — generated samples and training progress visualizations.

High-level suggested execution order (no code is run by the editor):

1. Edit the top `Setup and Configuration` cell to set `device`, `latent_dim`, `batch_size`, `epochs`, and `image_size`.
2. Run the data preparation cells to download datasets and build DataLoaders.
3. Train and evaluate RealNVP (or train RealNVP on learned latent representations after training the encoder-decoder).
4. Train the GAN and observe per-epoch outputs and FID metrics.

---

## Deep dive: CA3 (Diffusion and Score-based Models)

CA3 explores cutting-edge generative modeling techniques: Denoising Diffusion Probabilistic Models (DDPM) and Score-based Generative Models.

### Key Concepts Explained

**Denoising Diffusion Probabilistic Models (DDPM)** are generative models that learn to reverse a gradual noising process. They consist of two processes:

- **Forward Process (Diffusion)**: Gradually adds Gaussian noise to data over T timesteps, following a variance schedule β₁ to β_T.
- **Reverse Process (Denoising)**: Learns to remove noise step-by-step using a neural network (typically a U-Net) that predicts noise at each timestep.
- **Training Objective**: Simplified loss that predicts the added noise, enabling stable training.
- **Sampling**: Iterative denoising starting from pure noise to generate new samples.
- **DDIM**: Denoising Diffusion Implicit Models provide faster sampling by taking larger steps while maintaining quality.

**Score-based Generative Models** learn the score function (gradient of the log-density) of the data distribution. They can generate samples using stochastic processes:

- **Score Function**: ∇_x log p(x), the gradient pointing toward higher probability regions.
- **Score Matching**: Objective to learn the score function by matching it to the true score.
- **Langevin Dynamics**: MCMC sampling using ∇_x log p(x) to move toward data distribution.
- **Annealed Langevin Dynamics**: Multi-scale sampling with different noise levels for efficiency.
- **Connection to Diffusion**: Score-based models are related to diffusion through the concept of time-reversal.

These models represent the current state-of-the-art in generative modeling, offering superior sample quality compared to earlier approaches like VAEs and GANs.

### Results Summary

**Score-based Models**: Trained on 2D Gaussian mixture data with different noise levels (σ=1,3,7). Best performance at σ=3 with final loss 0.012, demonstrating effective score field learning and sampling via Langevin dynamics. Annealed sampling provides robust results across noise levels.

**Diffusion Models**: Implementation of DDPM with forward diffusion process and reverse denoising, enabling high-quality image generation through iterative noise removal.

Key components in `CA3_Diffusion_Models/codes/`:

1. **Diffusion Models**: Forward process (adding noise) and reverse process (denoising) for generating high-quality images.
2. **Score-based Models**: Learning the score function (gradient of log-density) for sampling via Langevin dynamics or ODE solvers.
3. **Training Objectives**: Simplified loss for diffusion, score matching for score-based models.
4. **Sampling**: Iterative denoising or stochastic differential equations (SDEs) for generation.

Why run CA3?

- Experience state-of-the-art image generation quality.
- Understand the connection between diffusion, score-based models, and energy-based models.
- Compare with earlier models (VAEs, GANs, Flows) in terms of sample quality and training stability.

Files of interest in `CA3_Diffusion_Models/`:

- `codes/Diffusion_Models.ipynb` — Implementation of DDPM.
- `codes/score_based_models.ipynb` — Score-based generative modeling.
- `report/DGM_CA3_EN_final.pdf` — Detailed report on experiments and results.
- `README.md` — Comprehensive documentation for CA3.
- `images/` — Generated samples and visualizations from diffusion and score-based models.

High-level suggested execution order:

1. Start with diffusion models: implement forward/reverse processes, train on a dataset like CIFAR-10.
2. Experiment with different noise schedules and sampling steps.
3. For score-based models: train the score network and sample using annealed Langevin dynamics.

---

## Deep dive: CA4 (Fine-Tuning Vision-Language Models)

CA4 explores advanced applications of deep generative models in vision-language tasks, specifically fine-tuning Google's Paligemma Vision-Language Model (VLM) on the CLEVR dataset using Parameter-Efficient Fine-Tuning (PEFT) techniques like Low-Rank Adaptation (LoRA).

### Key Concepts Explained

**Vision-Language Models (VLMs)** are multi-modal models that can process both visual and textual information simultaneously. They typically consist of:

- **Vision Encoder**: Processes images into visual features (e.g., using Vision Transformers or CNNs).
- **Text Encoder/Decoder**: Handles text input/output, often based on large language models.
- **Cross-Modal Fusion**: Mechanisms to combine visual and textual representations for joint understanding.

**Parameter-Efficient Fine-Tuning (PEFT)** addresses the challenge of adapting large pre-trained models without updating all parameters:

- **Low-Rank Adaptation (LoRA)**: Adds trainable low-rank matrices to frozen pre-trained weights, significantly reducing trainable parameters.
- **Benefits**: Faster training, lower memory usage, prevention of catastrophic forgetting, easier deployment.
- **How it Works**: For a weight matrix W, LoRA adds W + ΔW where ΔW = A×B, with A and B being low-rank matrices.

**Fine-Tuning VLMs** involves adapting general-purpose models to specific tasks:

- **Task-Specific Adaptation**: Training on domain-specific data to improve performance on targeted applications.
- **Instruction Tuning**: Teaching models to follow natural language instructions for vision-language tasks.
- **Evaluation**: Using metrics like ROUGE for text generation quality and task-specific accuracy.

**CLEVR Dataset** is designed for evaluating visual reasoning:

- **Synthetic Scenes**: Rendered images with multiple objects having various attributes (color, shape, size, position).
- **Complex Questions**: Require counting, comparison, spatial reasoning, and logical operations.
- **Ground Truth Answers**: Enables precise evaluation of reasoning capabilities.

This assignment bridges traditional generative modeling with modern multi-modal AI, showing how generative techniques extend beyond image synthesis to language and reasoning tasks.

Key components in `CA4_Vision_Language_Model/code/final_CA4_training.ipynb`:

1. **Vision-Language Model**: Paligemma-3B, a state-of-the-art VLM for understanding images and answering questions.
2. **Dataset**: CLEVR (Compositional Language and Elementary Visual Reasoning), featuring synthetic scenes with multiple objects and complex questions.
3. **PEFT with LoRA**: Efficient fine-tuning by adapting only low-rank matrices, reducing computational requirements.
4. **Quantization**: 8-bit quantization for memory efficiency during training.
5. **Evaluation**: ROUGE metrics to assess answer quality and model performance.

Why run CA4?

- Learn to adapt large pre-trained models for specific tasks without full fine-tuning.
- Understand vision-language integration and multi-modal generative modeling.
- Experience real-world application of generative techniques in AI assistants and chatbots.
- Compare PEFT approaches with traditional full fine-tuning in terms of efficiency and performance.

Files of interest in `CA4_Vision_Language_Model/`:

- `code/final_CA4_training.ipynb` — Complete implementation of Paligemma fine-tuning on CLEVR with LoRA.
- `code/eval_p1/final_CA4_results1.ipynb` — Evaluation notebook for part 1.
- `code/eval_p2/final_CA4_results2.ipynb` — Evaluation notebook for part 2.
- `description/DGM_HW4.pdf` — Assignment description and requirements.
- `report/` — Student analysis and experimental results.
- `README.md` — Comprehensive documentation for CA4.

High-level suggested execution order:

1. Set up environment and install dependencies (transformers, PEFT, etc.).
2. Configure model and LoRA parameters.
3. Load and preprocess CLEVR dataset subset.
4. Fine-tune Paligemma with LoRA on visual question answering.
5. Evaluate using ROUGE metrics and qualitative sample analysis.
6. Save the fine-tuned model for inference and deployment.

---

## Data, storage, and artifact management

Proper data and artifact management is crucial for reproducible experiments in generative modeling.

### Dataset Handling

- **Download and Caching**: Datasets are downloaded by `torchvision` into `./data/` by default. To avoid re-downloads and manage storage:
  - Set `torchvision` data directory: `export TORCH_HOME=./data` before running notebooks.
  - For large datasets like CelebA, consider using a shared cache directory if multiple users will run experiments.
- **Preprocessing**: Ensure consistent preprocessing pipelines across experiments (e.g., resize, normalization, data augmentation).
- **Custom Datasets**: For CA1's CelebA subset, the `CA1_Variational_Autoencoders/train/` folder contains pre-split smile/non-smile images. Verify integrity and consider backing up.

### Model Checkpoints and Artifacts

- **Saving Models**: Save PyTorch state_dicts (`.pth` files) for generators, discriminators, VAEs, flows, etc.
  - Example: `torch.save(generator.state_dict(), 'generator_epoch_50.pth')`
- **Run Metadata**: Save a `run_info.json` for each experiment including hyperparameters, random seed, Git commit, and timestamps.
- **Generated Samples**: Save image grids or sample batches as PNG/JPG for qualitative evaluation.
- **Logs and Metrics**: Use TensorBoard or Weights & Biases for tracking losses, FID scores, etc.

### Directory Structure for Experiments

Organize outputs like this:

```
experiments/
├── run_2023_10_01_vae_baseline/
│   ├── checkpoints/
│   │   ├── vae_epoch_10.pth
│   │   └── vae_final.pth
│   ├── samples/
│   │   ├── reconstructions.png
│   │   └── latent_interpolations.png
│   ├── logs/
│   │   └── tensorboard_logs/
│   └── run_info.json
└── run_2023_10_02_gan_fid/
    ├── ...
```

### Storage Tips

- Use Git LFS for large checkpoints or datasets if committing to repo.
- For FID evaluation, keep `real_images/` fixed: create a reproducible reference set (e.g., 2048 images sampled from training set with fixed RNG) and reuse across runs.
- Monitor disk usage: Generative models can produce many images; clean up intermediate results.

---

## Reproducibility checklist and recommended configuration

Reproducibility is essential in machine learning research. Follow these steps for reliable, comparable results.

### Environment Consistency

- **Virtual Environments**: Always use isolated environments (venv, conda) with pinned versions.
- **Package Versions**: Create a `requirements.txt` or `environment.yml` and commit it.
  - Example `requirements.txt`:
    ```
    torch==2.0.1
    torchvision==0.15.2
    numpy==1.24.3
    matplotlib==3.7.1
    pytorch-fid==0.10.1
    ```
- **Python Version**: Specify and use consistent Python versions (e.g., 3.10).

### Randomness Control

- **Seeds**: Set seeds for all sources of randomness at the start of each notebook's configuration cell.

  - Example:

    ```python
    import random
    import numpy as np
    import torch

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    ```

- **Deterministic Operations**: For PyTorch, set `torch.backends.cudnn.deterministic = True` to ensure reproducible convolutions.

### Version Control

- **Git Commits**: Record the exact commit hash for each experiment.
  ```bash
  git rev-parse --short HEAD
  ```
- **Code Snapshots**: Consider tagging releases or creating branches for major experiments.

### Experiment Tracking

- **Metadata Logging**: Save comprehensive `run_info.json` for each run.
  - Suggested schema:
    ```json
    {
      "experiment_name": "vae_baseline",
      "commit": "abc123d",
      "timestamp": "2023-10-01T12:00:00Z",
      "seed": 42,
      "hyperparameters": {
        "latent_dim": 128,
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 50
      },
      "model_config": {
        "encoder_layers": [784, 512, 256, 128],
        "decoder_layers": [128, 256, 512, 784]
      },
      "dataset": "CelebA_smile",
      "notes": "Baseline VAE with KL annealing"
    }
    ```
- **Metrics and Logs**: Log losses, evaluation metrics (FID, IS, etc.), and qualitative samples.

### Hardware Consistency

- **GPU/CPU**: Note the hardware used; results may vary between CPU/GPU or different GPU models.
- **Memory**: Ensure sufficient RAM/VRAM; document batch sizes that fit your hardware.

### Data Consistency

- **Fixed Splits**: Use fixed train/val/test splits with seeded random splits.
- **Preprocessing**: Apply identical preprocessing to all data (e.g., same normalization stats).

By following this checklist, experiments should be reproducible across different machines and time.

---

## Testing and lightweight smoke checks

Before committing to long training runs (which can take hours or days), perform these quick checks to catch issues early.

### Shape and Forward Pass Tests

- **VAE (CA1)**:

  - Assert encoder outputs mean/logvar with correct shapes: `assert mu.shape == (batch_size, latent_dim)`
  - Assert decoder reconstructs to original image shape.
  - Test reparameterization: sample z and verify gradients flow.

- **RealNVP (CA2)**:

  - Assert forward pass returns `(z, log_det_jacobian)` with correct shapes.
  - Assert inverse maps z back to x: `torch.allclose(x, inverse(z), atol=1e-5)`
  - Check log_det_jacobian is finite and reasonable.

- **GAN (CA2)**:

  - Assert generator output shape: `(batch_size, channels, height, width)`
  - Assert discriminator output: scalar per image.
  - Test with fixed noise: verify consistent outputs.

- **Diffusion/Score-based (CA3)**:

  - Assert noise addition/removal preserves shapes.
  - Verify score function gradients are finite.

### Mini-Training Runs

- Set small parameters: `batch_size=16`, `epochs=1`, `latent_dim=10`, `N=128` samples.
- Run training loop and check:
  - Losses decrease (not NaN/inf).
  - No runtime exceptions.
  - Checkpoints save/load correctly.
  - Generated samples look plausible (not all black/white).

### Evaluation Sanity Checks

- **FID (CA2)**:

  - Compute on small sets (200 real vs 200 generated).
  - Expect noisy values but end-to-end pipeline works.
  - Verify preprocessing: images resized to 299x299, normalized to [-1,1] then [0,1] for Inception.

- **Log-Likelihood (CA2)**:

  - Compute on small batch; check values are negative and finite.

- **Reconstruction/Generation Quality**:

  - Visual inspection: save and view sample grids.
  - Quantitative: PSNR/SSIM for reconstructions, diversity metrics for generations.

### Automated Testing

Consider adding unit tests using `pytest`:

```python
def test_vae_forward():
    vae = VAE(latent_dim=10)
    x = torch.randn(4, 3, 64, 64)
    recon, mu, logvar = vae(x)
    assert recon.shape == x.shape
    assert mu.shape == (4, 10)
```

Run tests: `pytest tests/` (create a `tests/` directory with test files).

---

## Common issues and troubleshooting

This section covers frequent problems encountered when running the notebooks and suggested fixes.

### Installation Issues

- **PyTorch/CUDA Mismatch**: Ensure PyTorch version matches your CUDA toolkit. Use `nvidia-smi` to check CUDA version, then install matching PyTorch from https://pytorch.org.
- **pytorch-fid Errors**: If `calculate_fid_given_paths` fails, try `pip install clean-fid` and use `clean_fid.compute_fid` instead. Ensure images are in [0,1] range and RGB.
- **Missing Dependencies**: For diffusion models, install `torchdiffeq` for ODE integration. If ODE solvers fail, fall back to simpler Euler discretization.

### Runtime Errors

- **CUDA Out of Memory**: Reduce `batch_size` (e.g., from 64 to 16), or use gradient accumulation. Enable mixed precision: `scaler = torch.cuda.amp.GradScaler()`.
- **NaN Losses**: Check for exploding gradients; add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`. Verify input normalization.
- **Shape Mismatches**: Double-check tensor shapes in forward passes. Use `print(x.shape)` liberally during debugging.
- **Slow Training**: Profile with `torch.profiler` or `cProfile`. Ensure data loading is not bottlenecked.

### Model-Specific Issues

- **VAE (CA1)**: If KL divergence explodes, anneal it: multiply by a coefficient that increases from 0 to 1 over epochs.
- **RealNVP (CA2)**: If log_det_jacobian is NaN, check for zero determinants in affine transformations; add small epsilon to denominators.
- **GAN (CA2)**: Mode collapse: monitor diversity in generated samples. If discriminator overpowers, adjust learning rates or use WGAN-GP.
- **Diffusion (CA3)**: If sampling fails, reduce noise schedule steps or use DDIM for faster sampling.

### Evaluation Problems

- **FID Too High/Low**: Ensure real and generated images are preprocessed identically. For FashionMNIST, resize to 64x64, normalize to [-1,1], then for FID convert to [0,1] and resize to 299x299.
- **Log-Likelihood Negative Infinity**: Clamp log probabilities to avoid -inf; add `log_prob = torch.clamp(log_prob, min=-1e10)`.

### Data Issues

- **Dataset Download Fails**: Check internet; for CelebA, may need manual download due to licensing. Use `wget` or browser to download and place in `./data/`.
- **Corrupted Images**: Verify dataset integrity; torchvision may redownload if files are missing.

### Jupyter/Environment Issues

- **Kernel Crashes**: Restart kernel; check for infinite loops or memory leaks.
- **Import Errors**: Ensure all packages are installed in the active environment. Use `conda list` or `pip list` to verify.
- **Notebook Not Saving**: Check disk space; try saving as .py and converting back.

### Performance Tips

- Use `torch.compile` (PyTorch 2.0+) for speedups: `model = torch.compile(model)`.
- For multi-GPU, use `torch.nn.DataParallel` or DDP.
- Profile memory: `torch.cuda.memory_summary()`.

If issues persist, check GitHub issues for similar problems or post with full error traceback and environment details.

---

## References and further reading

This section lists key papers, books, and resources related to the course topics.

### Core Papers

- **VAEs**:

  - Kingma, D.P. and Welling, M. "Auto-Encoding Variational Bayes." ICLR 2014.
  - Rezende, D.J., Mohamed, S. and Wierstra, D. "Stochastic Backpropagation and Approximate Inference in Deep Generative Models." ICML 2014.

- **Normalizing Flows**:

  - Dinh, L., Sohl-Dickstein, J. and Bengio, S. "Density estimation using Real NVP." ICLR 2017.
  - Kingma, D.P. and Dhariwal, P. "Glow: Generative Flow with Invertible 1x1 Convolutions." NeurIPS 2018.
  - Papamakarios, G., et al. "Normalizing Flows for Probabilistic Modeling and Inference." JMLR 2021.

- **GANs**:

  - Goodfellow, I., et al. "Generative Adversarial Nets." NeurIPS 2014.
  - Radford, A., Metz, L. and Chintala, S. "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks." ICLR 2016.
  - Gulrajani, I., et al. "Improved Training of Wasserstein GANs." NeurIPS 2017.

- **Diffusion Models**:

  - Sohl-Dickstein, J., et al. "Deep Unsupervised Learning using Nonequilibrium Thermodynamics." ICML 2015.
  - Ho, J., Jain, A. and Abbeel, P. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
  - Song, Y., et al. "Score-Based Generative Modeling through Stochastic Differential Equations." ICLR 2021.
  - Dhariwal, P. and Nichol, A. "Diffusion Models Beat GANs on Image Synthesis." NeurIPS 2021.

### Books and Tutorials

- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (Chapter 20 on Generative Models).
- "Probabilistic Machine Learning: An Introduction" by Kevin P. Murphy.
- Lilian Weng's blog: "What are Diffusion Models?" (lilianweng.github.io/posts/2021-07-11-diffusion-models/)

### Online Resources

- PyTorch tutorials on VAEs, GANs: https://pytorch.org/tutorials/
- Hugging Face Diffusers library: https://huggingface.co/docs/diffusers/index
- OpenAI's improved DDPM: https://github.com/openai/improved-diffusion

### Related Datasets

- CelebA: Liu, Z., et al. "Deep Learning Face Attributes in the Wild." ICCV 2015.
- FashionMNIST: Xiao, H., et al. "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms." arXiv 2017.
- CIFAR-10/100: Krizhevsky, A. "Learning Multiple Layers of Features from Tiny Images." 2009.

For the latest research, check arXiv, NeurIPS, ICML, ICLR proceedings.

### Additional Resources in Repository

- **Slides/**: Lecture slides from the course, including:
  - `DGM_Fall_2023_Slides/`: Course lecture slides with annotated versions and PDFs on topics like Mean-Field VI, Normalizing Flows, VAEs, Diffusion Models.
  - `Stanford_slides/`: Supplementary slides from Stanford's CS236 (Deep Generative Models) course.
  - `MITSlides/`: Additional slides from MIT and other sources.
- **Exams/**: Past midterm and final exams with solutions, useful for review and practice.
- **ExtraNotes/**: Miscellaneous resources including:
  - `homework_template/`: Homework templates and utility scripts.
  - Research papers on D-separation, bidirectional VAEs, etc.
  - Supplementary PDFs and notes.
- **OtherTermAssignments/**: Assignments from previous terms, including CA1, CA2, CA3 implementations.
- **OtherUniversityLecturs/**: Materials from other universities:
  - `CS236_DGM/`: Complete Stanford CS236 course materials.
  - `SUT/`: Sharif University of Technology resources.
  - `notes/`: General academic notes.
  - `ssi2023/`: SSI 2023 materials.
- **PaperSLecturs/**: Research papers and codes on advanced topics like AI for Science, generative models, LLMs, and computer vision.

---

## Getting Help and Support

If you encounter issues with the course materials:

1. **Check the Troubleshooting Section**: Common problems and solutions are documented above.
2. **Review Prerequisites**: Ensure you meet the mathematical and programming requirements.
3. **GitHub Issues**: Post detailed bug reports or questions in the repository issues, including:
   - Full error traceback
   - Your environment (Python version, PyTorch version, OS)
   - Steps to reproduce
   - Expected vs. actual behavior
4. **Course Discussion**: For conceptual questions, refer to the lecture slides or contact the course instructor.
5. **Community Resources**: Check PyTorch forums, Stack Overflow, or arXiv for related research questions.

When posting issues, provide minimal reproducible examples and avoid sharing sensitive data.

---

## Future Directions and Research Opportunities

The field of deep generative models is rapidly evolving. Based on this course, consider exploring:

### Advanced Architectures

- **Hierarchical VAEs**: Multi-scale latent representations
- **Flow-based VAEs**: Combining variational inference with normalizing flows
- **Energy-based Models**: Unnormalized probabilistic models with contrastive divergence

### Applications

- **Molecular Design**: Generating novel drug compounds
- **Art and Creativity**: AI-assisted content creation
- **Anomaly Detection**: Identifying outliers in high-dimensional data
- **Data Augmentation**: Synthetic data generation for limited datasets

### Theoretical Advances

- **Optimal Transport**: Using Wasserstein distances in generative training
- **Neural ODEs**: Continuous-time generative processes
- **Self-Supervised Learning**: Learning representations without explicit labels

### Ethical Considerations

- **Bias and Fairness**: Ensuring generated data doesn't perpetuate societal biases
- **Deepfakes Detection**: Developing methods to identify synthetic media
- **Privacy**: Balancing generative capabilities with data protection

This course provides a solid foundation for contributing to these exciting research directions.

---

## Credits and license

This repository contains course materials for the Deep Generative Models course. The code and notebooks are intended for educational and research use. If you reuse code or figures derived from these materials in publications or public projects, please credit the course author and repository.

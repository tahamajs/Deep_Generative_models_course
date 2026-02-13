# CA2: GANs and Normalizing Flows on FashionMNIST

This project compares two generative modeling families on FashionMNIST:

- **RealNVP Normalizing Flows** (explicit likelihood, invertible mapping)
- **DCGAN-style GAN** (implicit likelihood, adversarial training)

It also includes a latent-space flow variant and FID-based evaluation.

## Project Structure

- `code/CA2_GANs_and_NormalizingFlows_main.ipynb`: main implementation notebook.
- `code/CA2_question2_results.ipynb`: results-focused notebook.
- `requirements.txt`: pinned dependencies.
- `images/`: exported figures and generated samples used in report.
- `description/DGM_HW2.pdf`: assignment prompt.
- `report/`: report files.

## Setup

```bash
cd CA2_GANs_Normalizing_Flows
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run notebook:

```bash
jupyter lab code/CA2_GANs_and_NormalizingFlows_main.ipynb
```

## Frozen Results (No Re-Run)

This repository now supports a **frozen-results** workflow:

- No notebook execution
- No retraining
- Uses only saved `.ipynb` outputs and existing files in `images/`

Generate frozen metrics, figure manifest, and report include files:

```bash
python3 scripts/freeze_notebook_results.py
```

Verify report consistency and asset completeness:

```bash
python3 scripts/verify_frozen_results.py
```

Rebuild report PDFs (LaTeX only):

```bash
cd report
latexmk -pdf -interaction=nonstopmode CA2_Complete_Solutions.tex
cp -f CA2_Complete_Solutions.pdf DGM_CA2_final_EN.pdf
```

Frozen artifacts are written to:

- `artifacts/frozen_results/metrics.json`
- `artifacts/frozen_results/figure_manifest.json`
- `report/generated_metrics.tex`
- `report/generated_fid_table.tex`

## Methods Used (Explained)

### 1) RealNVP in Pixel Space

### Core classes/methods

- `class CouplingLayer`:
  - Implements affine coupling transform with alternating masks.
  - `forward(x)`: computes transformed output `y` and log-determinant Jacobian.
  - `inverse(y)`: exact inverse mapping.
- `class RealNVP`:
  - Stacks coupling layers.
  - `forward(x)`: maps data -> latent `z`, accumulates log-det Jacobian.
  - `inverse(z)`: latent -> data generation.

### Training method

- Negative log-likelihood optimization using change-of-variables:
  - `z, log_det = model(x)`
  - `log p(x) = log p(z) + log_det` with standard normal prior on `z`.
- Optimizer: Adam.

### Evaluation/analysis methods

- `compute_log_likelihood(model, data_loader)`: estimates sample likelihoods.
- `plot_kde(...)`: compares likelihood distributions across datasets.
- `visualize_real_vs_generated(...)`: side-by-side samples.
- `realnvp_latent_interpolation(...)`: interpolation in flow latent space.
- `visualize_reconstruction_quality(...)`: checks invertibility/reconstruction.

### 2) Latent-Space Flow (Encoder/Decoder + RealNVP)

This branch first compresses images, then fits flow in latent space.

### Core classes/methods

- `class Encoder`: MLP mapping flattened image -> latent vector.
- `class Decoder`: MLP mapping latent vector -> reconstructed image.
- `encode_dataset(encoder, data_loader)`: encodes full datasets.
- `compute_log_likelihood_latent(model, data)`: latent-space likelihood analysis.

### Training sequence

1. Train `Encoder+Decoder` with MSE reconstruction loss.
2. Encode FashionMNIST/MNIST/KMNIST sets.
3. Train RealNVP on encoded FashionMNIST latent vectors.
4. Compare latent log-likelihood distributions across datasets.

### 3) GAN (DCGAN-style)

### Core classes/methods

- `class Generator`:
  - Transposed-convolution network from latent noise to image.
  - BatchNorm + ReLU, final `tanh` output.
- `class Discriminator`:
  - Convolutional classifier producing real/fake probability.
  - LeakyReLU + BatchNorm + sigmoid output.
- `class GAN`:
  - Wraps generator/discriminator, BCE loss, Adam optimizers (`betas=(0.5, 0.999)`).

### Training method

- Alternating adversarial updates per batch:
  - Update `G` to fool `D` (`D(G(z)) -> real`).
  - Update `D` with real and detached fake batches.
- Tracks `G_losses` and `D_losses` during training.

### GAN analysis methods

- `plot_enhanced_loss_analysis(...)`: detailed loss behavior visualization.
- `visualize_discriminator_outputs(...)`: confidence distributions for real/fake.
- `latent_space_interpolation(...)`: semantic interpolation in GAN latent space.

### 4) Quality Metric Method

- FID is computed via `pytorch-fid` (`calculate_fid_given_paths`) using:
  - real image set
  - generated image set
- Used to track image quality across epochs.

## Hyperparameters Used in Notebook

### RealNVP block

- `input_dim = 28*28`
- `hidden_dim = 1024`
- `num_coupling_layers = 8`
- `num_epochs = 10`
- `learning_rate = 1e-3`
- `batch_size = 128`

### GAN block

- `latent_dim = 100`
- `lr = 2e-4`
- `batch_size = 64`
- `epochs = 10`

## Outputs

- Generated and real image folders (created during FID workflow):
  - `real_images/`
  - `generated_images/`
- Exported report figures: `images/`.

## Practical Notes

- The code is notebook-first; run cells in order.
- FID trends can fluctuate in GAN training; best epoch is not always final epoch.
- RealNVP offers exact likelihood and invertibility; GAN typically yields sharper samples.

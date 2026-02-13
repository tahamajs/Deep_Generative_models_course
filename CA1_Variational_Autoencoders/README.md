# CA1: Variational Autoencoder (VAE) for Face Images

This project implements a convolutional **beta-VAE** for 128x128 RGB face images (smile / non-smile classes), with training, reconstruction, random generation, and latent-space analysis.

## Project Structure

- `code/CA1_VAE_training_and_evaluation.ipynb`: main notebook workflow.
- `code/vae_training.py`: script version of the same pipeline.
- `description/HW1-DGM.pdf`: assignment statement.
- `images/`: exported figures used in the report.
- `report/`: final report and LaTeX sources.

## Methods Used (and What Each Does)

### 1) Data and Reproducibility Methods

- `set_seed(seed)`:
  - Sets Python, NumPy, and PyTorch seeds.
  - Enables deterministic CuDNN behavior on CUDA.
- `get_dataloaders(cfg)`:
  - Loads data with `torchvision.datasets.ImageFolder`.
  - Applies preprocessing: resize to `128x128`, tensor conversion, normalization to `[-1, 1]`.
  - Uses seeded `random_split` for deterministic train/val split.

### 2) Model Methods (`BetaVAE`)

- `BetaVAE.__init__(latent_dim=32, dropout=0.2)`:
  - Defines encoder and decoder.
  - Encoder: strided conv blocks downsample `128 -> 8` spatially.
  - Latent heads: `fc_mu`, `fc_logvar`.
  - Decoder: transposed conv blocks reconstruct image to `3x128x128` with `tanh` output.
- `encode(x)`:
  - Maps input image to latent Gaussian parameters `(mu, logvar)`.
- `reparameterize(mu, logvar)`:
  - Uses reparameterization trick `z = mu + sigma * eps` (with logvar clamping).
- `decode(z)`:
  - Maps latent `z` back to image space.
- `forward(x)`:
  - Full VAE path: encode -> sample -> decode.

### 3) Objective and Optimization Methods

- `elbo_loss(recon, x, mu, logvar, beta)`:
  - Reconstruction term: MSE (summed per batch, normalized by batch size).
  - KL divergence to standard Gaussian prior.
  - Total: `loss = recon_loss + beta * kl`.
- `train_one_epoch(...)`:
  - One training epoch with backprop and Adam update.
- `eval_epoch(...)`:
  - Validation pass without gradient updates.
- `run_training(cfg)` (notebook) / `train(cfg)` (script):
  - Runs full epoch loop, logs train/val metrics, periodically writes sample grids.

### 4) Visualization / Output Methods

- `save_samples(model, device, data, out_dir, step_label, num_samples)`:
  - Saves reconstruction grids (`original + recon`) and random generations.
- `preview_reconstructions(model, loader, max_images=16)`:
  - Quick post-training preview helper.

### 5) CLI Utility Method (script)

- `parse_args()`:
  - Parses script flags (`--data-root`, `--epochs`, `--batch-size`, `--lr`, `--beta`, ...).

## Architecture Summary

- **Encoder**: Conv2d blocks with BatchNorm + LeakyReLU + Dropout.
- **Latent dimension**: 32.
- **Decoder**: mirrored ConvTranspose2d stack to RGB output.
- **Output activation**: `tanh` (fits normalized image range `[-1,1]`).

## Training Configuration (Current Code Defaults)

Notebook defaults (smoke run):
- `epochs=5`, `batch_size=32`, `lr=5e-4`, `beta=1.0`.

Script defaults (full run style):
- `epochs=1000`, `batch_size=128`, `lr=5e-4`, `beta=1.0`.

## How to Run

### Notebook

```bash
cd CA1_Variational_Autoencoders/code
jupyter lab CA1_VAE_training_and_evaluation.ipynb
```

### Script

From repository root:

```bash
python CA1_Variational_Autoencoders/code/vae_training.py \
  --data-root CA1_Variational_Autoencoders/train \
  --epochs 5 \
  --out-dir CA1_Variational_Autoencoders/output
```

For full training, increase `--epochs` (e.g., 1000).

## Outputs

- Training log CSV: `output/train_log.csv` (script path).
- Sample images: `output/recon_*.png`, `output/gen_*.png`.
- Curated figures for report: `images/`.

## Notes

- Dataset directory must follow `ImageFolder` format (class subfolders).
- If you regenerate outputs, keep report figure references consistent with filenames in `images/`.

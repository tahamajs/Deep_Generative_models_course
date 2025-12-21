# Copilot instructions — CA2 GANs & Normalizing Flows repo

This file helps AI coding assistants get productive quickly in this folder (CA2: MAF + CycleGAN experiments).
Be concise and focused: prefer edits that keep behavior unchanged unless tests or the user ask for improvements.

Key points
- Big picture: this project implements two separate experiments:
  - MAF (Masked Autoregressive Flow) for unconditional image modeling / anomaly detection (see `maf.py`).
  - CycleGAN for image-to-image translation (see `cyclegan.py`).
- The entrypoint for running experiments is `code/run.py` (run from `code/`):
  - Train MAF: `python run.py maf --mode train --epochs 20 --dataset capsule/train/good`
  - Generate from MAF: `python run.py maf --mode generate --model maf_final.pth --num_samples 5`
  - Train CycleGAN: `python run.py cyclegan --mode train --dataset horse2zebra --epochs 20`
  - Test CycleGAN: `python run.py cyclegan --mode test --dataset horse2zebra --model <G_AB.pth>`

Project layout (what to inspect first)
- `maf.py` — MAD/E-based MAF implementation, `train_maf`, `generate_images_maf`.
- `cyclegan.py` — ResNet generators + patch discriminators, full train/test loops.
- `datasets.py` — `CapsuleDataset` (MVTec capsule subset) and `ImageDataset` helpers + standard transforms `capsule_transform` and `cyclegan_transform`.
- `utils.py` — plotting, ROC & AUROC helpers, visualization utilities.
- `run.py` — CLI wrapper that wires datasets + models into common experiments and includes `--quick` smoke-test modes.

Important conventions & gotchas (use these when making edits)
- Image normalization: transforms use mean/std = [0.5,0.5,0.5] so image pixel range is expected to be [-1, 1] for generators and flow training.
- Input dimensionality for MAF: images are flattened to vectors of length `H * W * 3`. Default image size is 128x128.
- Reproducibility: random seeds are set in some notebooks but NOT centrally in the modules — add explicit seed control where reproducibility is required.
- Training artifacts: `train_cyclegan` returns a `history` dict and optionally writes checkpoints to `checkpoints/`.

Testing and quick debug
- Use `--quick` flags in `run.py` to run smoke tests with small synthetic datasets for fast feedback.
- To validate imports/syntax: run `python -c "import maf, cyclegan, datasets, utils"` from `code/`.
- To visually inspect outputs: use `utils.visualize_samples()` which accepts tensors and handles moving them to CPU and denormalization.

Dependencies
- Primary: `torch`, `torchvision`, `numpy`, `matplotlib`, `tqdm`, `scikit-learn` (for AUROC). Prefer to match the environment used in the notebooks (Python 3.8+ recommended).

Editing guidance for AI agents
- Prefer small, focused changes with unit-testable behavior. For example:
  - If changing training hyperparameters, return an option through `run.py` rather than hardcoding it.
  - When changing data transforms, ensure both training and evaluation transforms match normalization.
- When adding new utilities, place them in `utils.py` and keep I/O side-effects in `run.py` or explicit helper scripts.
- Avoid altering notebook files — keep experiments reproducible by updating the .py modules and `run.py` instead.

When in doubt, check these files for examples of usage:
- `code/run.py` (how CLI wires models + datasets)
- `maf.py` (example: `samples, t = generate_images_maf(model, num_images=5, img_size=128)`)
- `cyclegan.py` (train loop and `test_cyclegan` return format)

If you add new features, update this file with short, concrete examples so future agents can find and reuse them quickly.

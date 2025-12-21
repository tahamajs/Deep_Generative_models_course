# CA2 Code (Refactored)

Files:
- `maf.py` — MADE / MAF implementation and training/generation helpers.
- `cyclegan.py` — Generator / Discriminator and CycleGAN training/testing helpers.
- `datasets.py` — `CapsuleDataset`, `ImageDataset`, and standard transforms.
- `utils.py` — Visualization and evaluation helpers.
- `run.py` — CLI entry point to run MAF and CycleGAN experiments.

Quick examples:
- Train MAF (quick smoke):
  `python run.py maf --mode train --epochs 2 --quick`

- Generate with MAF:
  `python run.py maf --mode generate --model maf_final.pth --num_samples 4`

- Evaluate MAF for anomaly detection (requires `capsule/test`):
  `python run.py maf --mode eval --model maf_final.pth`

- Train CycleGAN (quick smoke):
  `python run.py cyclegan --mode train --epochs 2 --quick`

- Test CycleGAN (requires dataset):
  `python run.py cyclegan --mode test --dataset horse2zebra --model G_AB_final.pth`

Notes:
- Use `--quick` to run small smoke tests on machines without datasets or GPUs.
- Long training should be run on a GPU-enabled machine with full datasets.

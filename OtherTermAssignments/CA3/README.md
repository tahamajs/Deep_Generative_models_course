# CA3 — Diffusion, EBM, and Score-Based Models (Next-Year Draft)

This folder hosts the future-course CA3 implementation for energy-based models (EBM) and noise-conditional score networks (NCSN) on MNIST. It follows the project conventions in `../CLAUDE.md` and the root repository rules.

## Structure

- `codes/`: Python package with training and inference utilities
  - `ebm_train.py`, `ebm_infer.py`: Conv-EBM training and sampling/denoising
  - `ncsn_train.py`, `ncsn_infer.py`: Score-based training, sampling, and denoising
  - `config.py`: Central dataclass configs (paths, data, EBM, NCSN)
  - `data.py`: MNIST loaders with optional [-1, 1] normalization
  - `utils.py`: Seeding, I/O helpers, experiment metadata writer
- `description/`: Assignment PDF
- `report/`: LaTeX report draft
- `requirements.txt`: Pinned dependencies (Python 3.10+ recommended)

## Setup

1. Create a virtual environment (`python -m venv .venv && source .venv/bin/activate`).
2. Install dependencies: `pip install -r requirements.txt`.
3. GPU is preferred; CPU also works for smoke tests.

## Running Experiments

All scripts assume execution from the repo root to keep paths stable.

- Train EBM: `python -m next_year.CA3.codes.ebm_train`
- Train NCSN (unconditional then conditional): `python -m next_year.CA3.codes.ncsn_train`
- EBM inference/denoising: `python -m next_year.CA3.codes.ebm_infer`
- NCSN sampling/denoising: `python -m next_year.CA3.codes.ncsn_infer`

Artifacts save under `next_year/CA3/images/` and checkpoints inside each run folder. MNIST downloads to `next_year/CA3/data/mnist` automatically.

## Reproducibility

- Seeds for `random`, `numpy`, `torch`, and CUDA are set before data loading and training.
- Deterministic cuDNN flags are enabled when CUDA is present.
- Each training run writes `run_info.json` (timestamp, git commit hash, device, and configs) into its output directory for traceability.
- Configs are centralized in `codes/config.py`; modify there or pass overrides when importing the modules.

## Quick Smoke Test

To validate setup without long runs:

- Reduce epochs (e.g., `EBMConfig(epochs=1)` or `NCSNConfig(epochs=1, langevin_steps=10)`).
- Run the corresponding train script; ensure loss is finite and sample grids render in `images/`.

## Notes

- Keep code DRY and reuse utilities in `codes/utils.py`.
- Do not commit large datasets or checkpoints; rely on the default MNIST download cache.
- Report drafts live in `report/`; keep IEEE format per directory rules.


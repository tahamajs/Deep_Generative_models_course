CA1 — Deep Generative Models (VAE + PGM)

This folder contains a lightweight Python package `ca1` extracted from the original notebook.

Quick start

1. Install dependencies (recommended to use a venv):

```bash
pip install -r ../requirements.txt
# or if in this folder:
# pip install torch torchvision numpy matplotlib networkx scikit-learn tqdm scipy
```

2. Run CLI from this folder:

```bash
# Draw PGM diagrams
python run.py pgm

# Quick smoke test (1-epoch training on a small subset)
python run.py smoke

# Full training (example)
python run.py train --epochs 30 --beta 1.0 --batch-size 128

# End-to-end report figure generation (PGM + beta-VAE plots)
python run.py report --output-dir ../report/DGM_Report_Template/figures --epochs 30 --subset 30000 --betas 1,2,5

# Generate extra report assets and numbered LaTeX aliases (1.png..13.png)
python generate_report_assets.py --figures-dir ../report/DGM_Report_Template/figures --report-dir ../report --subset 5000 --betas 1,2,5 --mig-max-samples 1000
```

Notes
- The script expects the dSprites `.npz` file at the path configured in `ca1/config.py` (`data_path`). If not present it will attempt to download automatically.
- No training is run by default; use the `train` or `smoke` subcommands explicitly.
- `report` saves all report figures plus checkpoints and run metadata in the specified output directory.

Development
- Package location: `ca1/`
- Entry point: `run.py`
- Tests: `tests/test_imports.py` (run with `pytest` or `python -m pytest`)

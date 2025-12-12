# CA1_Variational_Autoencoders — Claude Code Guidance

**Technology**: Python, Jupyter, PyTorch  
**Primary Notebook**: `code/code.ipynb`  
**Reports**: `report/DGM_CA1_Exercise_Solutions.tex` → `DGM_CA1_final_EN.pdf` (IEEE format, keep in sync)  
**Parent Context**: Extends root `../CLAUDE.md`

## Development & Commands
- From repo root: `jupyter lab` then open `CA1_Variational_Autoencoders/code/code.ipynb`.
- Environment: activate venv/conda, install PyTorch + dependencies (see root README).
- Syntax check (if `.py` helpers added): `python -m py_compile CA1_Variational_Autoencoders/**/*.py`.
- Render report: use LaTeX toolchain to build `report/DGM_CA1_Exercise_Solutions.tex` to PDF (IEEE style).
- Smoke run (recommended): set `batch_size=16`, `epochs=1`, small subset of data; confirm losses finite.

## Architecture & Patterns
- Structure notebook sections: overview → setup/config (imports, seeds, device, hyperparameters) → data → model → training → evaluation.
- Models: encoder/decoder modules must be shape-safe; assert output shapes for reconstructions and latent parameters.
- Reparameterization: keep KL + reconstruction loss clear; log both separately.
- Keep shared utilities (e.g., data transforms, plotting) in one cell; avoid duplicating across notebook.
- Hyperparameters: define once in config cell; avoid hardcoding values later.
- Data handling: images reside in `train/` (smile/non-smile). Do not move/delete without confirmation.
- Checkpoints and samples: store under `images/` or `experiments/` with run metadata; avoid committing large outputs.
- Reports: ensure notebook logic matches claims in `.tex` and final PDF; update figures if regenerated.

## Key Files
- `code/code.ipynb` — main VAE implementation and experiments.
- `train/` — CelebA subset (smile/non-smile); treat as read-mostly.
- `images/` — generated visualizations; regenerate via notebook when needed.
- `report/DGM_CA1_Exercise_Solutions.tex` — authoritative source for PDF; maintain IEEE formatting.
- `report/DGM_CA1_final_EN.pdf` — output artifact; regenerate after `.tex` updates.

## Quick Search (rg)
```bash
rg -n "class .*Encoder|Decoder" CA1_Variational_Autoencoders
rg -n "def reparameterize" CA1_Variational_Autoencoders/code
rg -n "KL|kl_div" CA1_Variational_Autoencoders
rg -n "seed|torch.backends.cudnn" CA1_Variational_Autoencoders/code
```

## Common Gotchas
- Forgetting to set seeds and deterministic flags → non-reproducible results.
- KL term collapse/explosion → consider annealing or clipping small variances.
- Shape mismatches in encoder/decoder → add assertions on latent and reconstruction shapes.
- Committing large image grids/checkpoints → keep out of git; store locally.
- Report drift: if code changes, update `.tex` and rebuild PDF.

## Testing & Smoke
- Mini-train: `batch_size=16`, `epochs=1`, confirm ELBO decreases and no NaN/inf.
- Reconstruction sanity: visualize a small batch, check pixel range scaling.
- Latent sampling: sample `z ~ N(0,1)` to ensure decoder is stable.
- Save/load: checkpoint once, reload, and run a forward pass to confirm parity.
- Document smoke outcomes in notebook Markdown or a short note in `report/`.

## Documentation & Reporting
- Keep Markdown cells explaining objectives, architecture, and findings.
- Update `.tex` whenever experiments change; ensure IEEE format (two-column, references, figures).
- Sync figures: regenerate images used in the report if data/model changed.

## Safety
- Do not delete or relocate `train/` contents.
- Avoid editing PDFs directly; regenerate from `.tex`.
- Avoid long training before a smoke pass.

## Pre-PR/Pre-Commit Checklist (CA1)
- Seeds set and logged; config centralized.
- Smoke run completed (1 epoch, small batch) with finite losses.
- Notebook saved/clean; no massive outputs embedded.
- `.tex` updated to reflect code changes; PDF rebuilt if `.tex` changed.
- No large binaries staged; only intentional artifacts tracked.

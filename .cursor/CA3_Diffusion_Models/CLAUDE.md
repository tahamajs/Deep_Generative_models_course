# CA3_Diffusion_Models — Claude Code Guidance

**Technology**: Python, Jupyter, PyTorch  
**Primary Notebooks**: `codes/CA3_Diffusion_Models_overview.ipynb`, `codes/CA3_Score_Based_Models.ipynb`  
**Reports**: `report/*.tex` → PDFs (IEEE format; keep LaTeX as source of truth)  
**Parent Context**: Extends root `../CLAUDE.md`

## Development & Commands

- Launch via `jupyter lab` → open notebooks under `codes/`.
- Environment: PyTorch + optional `torchdiffeq` for SDE/ODE sampling; ensure CUDA if available.
- Syntax check (if `.py` helpers added): `python -m py_compile CA3_Diffusion_Models/**/*.py`.
- Render report: build `.tex` files in `report/` to IEEE PDF; update figures from latest runs.
- Smoke runs:
  - Small image batch, reduced timesteps (e.g., 50), and single epoch/loop.
  - Use DDIM or few-step sampling for quick validation of the pipeline.

## Architecture & Patterns

- Notebook flow: overview → config (imports, seeds, betas/noise schedule) → data → model (UNet/score net) → training → sampling/evaluation.
- Noise schedules: document beta schedules; keep them configurable in the setup cell.
- Sampling: provide deterministic seeds for sample grids; log sampler settings (DDPM/DDIM, steps).
- Stability: clamp/log-sigma where needed; monitor for NaNs during training and sampling.
- Checkpointing: save model + optimizer state; track sampler parameters with the checkpoint.
- Reports: synchronize figures (sample grids, loss curves) with the `.tex` content.

## Key Files

- `codes/CA3_Diffusion_Models_overview.ipynb` — DDPM-focused content.
- `codes/CA3_Score_Based_Models.ipynb` — score-based SDE/ODE content.
- `images/` — generated samples; regenerate instead of manual edits.
- `report/` — LaTeX sources and PDFs; maintain IEEE format.

## Quick Search (rg)

```bash
rg -n "beta|schedule" CA3_Diffusion_Models/codes
rg -n "UNet|epsilon" CA3_Diffusion_Models/codes
rg -n "DDIM|DDPM" CA3_Diffusion_Models/codes
rg -n "seed|torch.backends.cudnn" CA3_Diffusion_Models/codes
```

## Common Gotchas

- NaNs during training → reduce lr, enable grad clipping, check beta schedule.
- Sampling instability → use fewer steps for smoke; ensure model eval mode and no_grad.
- Device mismatch errors → ensure tensors moved to `device` defined in config cell.
- Large images/pdfs creeping into git; keep only intended artifacts.
- Report drift: update `.tex` after rerunning experiments or changing schedules.

## Testing & Smoke

- Short training: few batches with small image size; confirm loss finite and decreasing.
- Sampling smoke: generate small grid (e.g., 4x4) with fixed seed; verify non-noisy outputs.
- Score-based: check gradient norms and ensure no NaNs in SDE/ODE solver steps.
- Save/load: checkpoint once, reload, and run a single sampling step to confirm integrity.
- Log sampler parameters (steps, schedule, guidance) in notes or `run_info`.

## Documentation & Reporting

- Add Markdown explaining noise schedules, sampler choices, and evaluation metrics.
- Keep `.tex` IEEE formatted; update figures and tables when experiments change.
- Note hardware (GPU/VRAM) used for runs that produce figures.

## Safety

- Avoid long diffusion runs without prior smoke; start small.
- Do not commit large sample archives; keep only representative figures.
- Do not edit rendered PDFs directly; regenerate from LaTeX.

## Pre-PR/Pre-Commit Checklist (CA3)

- Seeds/config centralized; schedule documented.
- Smoke training + sampling completed (few steps, fixed seed) with finite results.
- Checkpoints reload correctly for a sampling step.
- `.tex` updated and PDF rebuilt if `.tex` changed.
- No unintended large artifacts staged.

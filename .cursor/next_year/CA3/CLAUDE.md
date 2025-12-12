# next_year/CA3 — Claude Code Guidance

**Purpose**: Future CA3 drafts (Diffusion/Score-based) under `next_year/CA3`.  
**Parent Context**: Extends `../CLAUDE.md` (next_year rules) and root `../../CLAUDE.md`.

## Rules

- Draft workspace; keep distinct from current-year CA3 assets.
- Config/seed/device must be defined up front in notebooks/modules.
- Reports should stay in `.tex` (IEEE); regenerate PDFs from LaTeX.
- Keep SDE/ODE configs and noise schedules documented alongside experiments.

## Safety

- Avoid large artifact commits (checkpoints/images); keep local unless necessary.
- Document sampler settings with saved outputs for reproducibility.

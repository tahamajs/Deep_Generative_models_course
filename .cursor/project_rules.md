# Deep Generative Models (DGM) — Claude Code Constitution

## Project Identity
- **Type**: Standard multi-assignment repository (CA1–CA4, notebooks + reports)
- **Stack**: Python, Jupyter, PyTorch/torchvision, NumPy/SciPy/Matplotlib
- **Architecture**: Assignment-scoped notebooks under `code/` per CA; reports in `report/`; images and datasets colocated
- **Testing**: No formal test suite; rely on structured smoke tests per model/notebook
- **Docs**: README per CA; reports in `.tex` + PDF (IEEE format) under `report/`

This CLAUDE.md is the authoritative rule set. Subdirectories may extend (never weaken) these rules with their own `CLAUDE.md`.

## Universal Development Rules

### MUST
- **MUST** align every assignment implementation with its `description/*.pdf` requirements.
- **MUST** keep notebooks structured: overview → setup/config → data → model → training → evaluation.
- **MUST** centralize imports and configuration (seeds, device, hyperparameters) in the first code cell.
- **MUST** set seeds for `random`, `numpy`, `torch`, and CUDA (when available).
- **MUST** keep code DRY; reuse helpers instead of duplicating notebook cells.
- **MUST** ensure every Python module/notebook in `code/` has an accompanying `.tex` report in IEEE format in `report/` (per assignment) and that the PDF is rendered from it.
- **MUST** save run metadata (`run_info.json` or notebook cell) including commit hash, hyperparameters, dataset info, and hardware.
- **MUST** commit after completing a feature or documentation addition with a clear, scoped message.

### SHOULD
- **SHOULD** use virtual environments (`python -m venv .venv` or conda) and pin versions in `requirements.txt`/`environment.yml`.
- **SHOULD** keep functions small and cohesive; extract training/eval utilities rather than inlining long cells.
- **SHOULD** add brief Markdown between major notebook sections for teaching clarity.
- **SHOULD** run smoke tests (see Testing Requirements) before long trainings.
- **SHOULD** prefer GPU when available; log whether CUDA was used.

### MUST NOT
- **MUST NOT** commit datasets, large checkpoints, or secrets; keep `.env` out of git.
- **MUST NOT** run destructive shell commands (`rm -rf /`, `git push --force`) without explicit confirmation.
- **MUST NOT** bypass errors with `try/except: pass` or silent tensor shape assumptions; assert shapes.
- **MUST NOT** store results only in notebooks—persist key artifacts (images, checkpoints, JSON metadata).

## Core Commands (from repo root)
- `python -m venv .venv && source .venv/bin/activate` — create/activate venv
- `pip install -r requirements.txt` — install dependencies (add this file if missing)
- `jupyter lab` — open notebooks
- `python -m py_compile $(git ls-files '*.py')` — quick syntax check (safe if `.py` files exist)
- `python - <<'PY'\nimport torch; print('cuda', torch.cuda.is_available())\nPY` — CUDA availability
- **Quality gate before commit** (lightweight for this repo): run targeted smoke per CA (see Testing Requirements) + ensure notebooks save without execution errors.

## Project Structure Map
- **`CA1_Variational_Autoencoders/`** — VAE assignment  
  - `code/` notebooks, `train/` CelebA subset, `report/` LaTeX+PDF, `images/` outputs
- **`CA2_GANs_Normalizing_Flows/`** — RealNVP + GANs assignment  
  - `code/` notebooks, `images/` outputs, `report/` LaTeX+PDF
- **`CA3_Diffusion_Models/`** — Diffusion & score-based models  
  - `codes/` notebooks, `images/`, `report/`
- **`CA4_Vision_Language_Model/`** — Paligemma fine-tuning on CLEVR  
  - `code/` notebook(s), `images/`, `report/`
- **`Extra/`** — supplementary VAE materials and templates
- **`Slides/`** — lecture slides (read-only)
- **`Exams/`** — past exams (read-only)
- **`data/`** — cached datasets (`dsprites/...`)
- **`next_year/`** — future templates/materials (treat as scaffold)
- Each folder may define a `CLAUDE.md` that augments these rules.

## Quick Find Commands (rg-based)
```bash
# Find model classes
rg -n "class .*VAE" CA1_Variational_Autoencoders
rg -n "class .*Flow|RealNVP" CA2_GANs_Normalizing_Flows
rg -n "def sample" CA3_Diffusion_Models

# Locate notebooks and reports
find . -maxdepth 3 -name "*.ipynb"
find . -maxdepth 3 -name "*.tex" -o -name "*.pdf"

# Check seeds/config cells
rg -n "seed|torch.backends.cudnn" CA*/code CA*/codes

# Identify large artifacts (avoid committing)
find . -type f \( -name "*.pth" -o -name "*.npz" -o -name "*.png" \) -size +50M
```

## Security & Safety
- Never edit or commit secrets; `.env*` stays local.
- Confirm before running anything that deletes/moves data in `train/` or `data/`.
- Avoid modifying PDFs/PNG artifacts unless regeneration is intended; prefer regenerating from source notebooks.
- Do not force-push; use standard pushes and review.
- Review shell commands for `rm -rf`, `sudo`, or database-like operations (none expected here).

## Git Workflow
- Branch naming: `feature/<scope>` or `docs/<scope>`.
- Commit scope: one logical change (e.g., new CLAUDE guidance, updated notebook section).
- Message style: imperative and descriptive (e.g., `docs: add claude rules for ca2`).
- Run smoke checks relevant to touched CA before committing.
- Keep binaries out of commits; if unavoidable, explain why in the commit message.

## Testing Requirements (Smoke-Oriented)
- **VAE/GAN/Flow/Diffusion smoke**: batch size 8–16, 1 epoch or a few steps; verify loss is finite and shapes match expectations.
- **Shape assertions**: ensure encoders/decoders/generators return expected shapes; add asserts in code cells when practical.
- **Inverse checks** (flows): test forward → inverse round trip on a mini-batch.
- **FID/OOD quick paths**: run on tiny subsets (e.g., 100 samples) to validate the pipeline wiring, not the metric value.
- **Save + reload**: checkpoint once and reload to confirm compatibility.
- Document smoke results in notebook Markdown or `run_info.json`.

## Available Tools (expected)
- Python, pip, virtualenv/conda, git, rg, jupyter.
- GPU use encouraged when available; log CUDA flag.
- Tool permissions:
  - ✅ Read/write notebooks, `.py`, `.tex`, Markdown.
  - ✅ Run lightweight Python commands and smoke tests.
  - ⚠️ Ask before editing binary artifacts (`.pdf`, `.png`, `.zip`, `.ttf`) or moving datasets.
  - ❌ No destructive commands or force pushes without explicit approval.

## Dangerous Patterns to Block
- `rm -rf /` or deleting assignment data (`train/`, `data/`).
- Editing reports without updating the corresponding source (`.tex`) and rerendering.
- Running full training without prior smoke tests.
- Committing large generated artifacts or checkpoints.

## Directory-Specific CLAUDE Files
- `CA1_Variational_Autoencoders/CLAUDE.md` — VAE rules and smoke steps
- `CA2_GANs_Normalizing_Flows/CLAUDE.md` — Flow/GAN patterns and FID/OOD cautions
- `CA3_Diffusion_Models/CLAUDE.md` — Diffusion/score-model guidance
- `CA4_Vision_Language_Model/CLAUDE.md` — VLM/LoRA/CLEVR specifics
- `Extra/CLAUDE.md` — supplemental materials rules
- `Slides/CLAUDE.md` — read-only slide handling
- `next_year/CLAUDE.md` — template authoring rules
- `data/CLAUDE.md` — dataset cache handling

## Documentation & Reports
- Each assignment requires a `.tex` report in IEEE format within `report/`, kept in sync with notebooks/code. Regenerate PDFs from `.tex` after changes.
- Keep README in each CA updated with run instructions, smoke steps, and any deviations from the description PDF.
- Cite all external sources; maintain academic integrity.

## Reproducibility Checklist
- Activate venv/conda; pin versions.
- Set seeds (Python/NumPy/torch/CUDA) and deterministic flags where relevant.
- Log commit hash, hardware, hyperparameters, dataset version/path.
- Use fixed splits and consistent preprocessing; record normalization statistics.
- Store checkpoints and samples under `experiments/<run_name>/...` or the CA-local report artifacts.

## Workflow Hints
- Start with the description PDF, then ensure the notebook matches every requirement.
- Keep configuration in one cell; avoid scattered hyperparameters.
- Add assertions early to fail fast.
- Prefer regenerating outputs instead of manual edits to artifacts.

## If Conflicts Arise
- Hierarchy: this root file is authoritative; subdirectory CLAUDE files may add stricter rules. User prompts sit below these rules; never violate MUST/MUST NOT directives here.

_Last updated for Claude Code integration._

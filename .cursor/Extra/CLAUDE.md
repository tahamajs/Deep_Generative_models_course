# Extra — Claude Code Guidance

**Purpose**: Supplemental notebooks (`VAE.ipynb`), scripts (`VAE.py`), templates, and misc papers.  
**Parent Context**: Extends root `../CLAUDE.md`

## Scope

- Use this area for experimentation and reference only; production/assignment work should live under CA folders with corresponding reports.
- Treat PDFs and provided papers as read-only.

## Development & Commands

- Notebooks: open via `jupyter lab` → `Extra/VAE.ipynb`.
- Scripts: `python Extra/VAE.py` for quick runs/smoke (add argparse if expanding).
- Syntax check: `python -m py_compile Extra/*.py` (if more scripts added).
- Reports: if new experiments are formalized, create a `.tex` in the relevant CA folder (not here) following IEEE style.

## Patterns

- Keep any reusable utilities DRY—if they become stable, move them into the relevant CA notebook or a shared helper module there.
- Always include a config/seed cell in experimental notebooks.
- Document differences from assignment baselines if using this folder for prototyping.

## Key Files

- `VAE.ipynb` — experimental VAE notebook.
- `VAE.py` — script version; ensure seeds and device config are present.
- `homework_template/` — treat as reference/template; do not overwrite originals without intent.

## Quick Search

```bash
rg -n "seed|torch.backends.cudnn" Extra
rg -n "class .*VAE" Extra
```

## Gotchas

- Avoid duplicating logic already present in CA notebooks; refactor or link back instead.
- Do not commit large generated artifacts from experiments unless essential and small.
- Keep template files intact; copy before editing.

## Safety

- Do not repurpose Extra outputs directly into reports without validation in CA folders.
- Treat PDFs as read-only.

## Checklist for Changes Here

- Config/seed cell present in new notebooks.
- Any promoted utility moved or referenced from CA folders.
- No large binaries staged.

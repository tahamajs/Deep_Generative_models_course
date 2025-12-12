# CA2_GANs_Normalizing_Flows — Claude Code Guidance

**Technology**: Python, Jupyter, PyTorch  
**Primary Notebooks**: `code/CA2_GANs_and_NormalizingFlows_main.ipynb`, `code/CA2_question2_results.ipynb`  
**Reports**: `report/CA2_Complete_Solutions.tex` → `DGM_CA2_final_EN.pdf` (IEEE format)  
**Parent Context**: Extends root `../CLAUDE.md`

## Development & Commands

- Open via `jupyter lab` → notebooks under `code/`.
- Environment: venv/conda with PyTorch + `torchvision`, `pytorch-fid` (or `clean-fid` alternative).
- Syntax check (if `.py` helpers added): `python -m py_compile CA2_GANs_Normalizing_Flows/**/*.py`.
- Render report: build `report/CA2_Complete_Solutions.tex` to PDF; keep IEEE compliance.
- Smoke run recommendations:
  - **Flows**: small batch (16), few steps, verify log-det finite and inverse works.
  - **GAN**: 1–2 mini-epochs with fixed noise; ensure losses finite and images non-degenerate.

## Architecture & Patterns

- Separate sections: overview → config (imports, seeds, hyperparameters) → data → models (RealNVP, DCGAN) → training loops → evaluation (FID/OOD/samples).
- RealNVP: enforce invertibility; validate forward/inverse round-trip; clamp/log-det stability.
- GAN: keep fixed noise vector for progression; balance generator/discriminator lr; monitor mode collapse.
- Data: primarily FashionMNIST/MNIST/KMNIST via torchvision; avoid altering dataset download paths mid-run.
- Metrics: FID and log-likelihood/OOD—run on small subsets for smoke.
- Hyperparameters: declare once in config cell; avoid duplicate definitions later.
- Reports must mirror experiments; update figures and tables when code changes.

## Key Files

- `code/CA2_GANs_and_NormalizingFlows_main.ipynb` — primary notebook for flows + GANs.
- `code/CA2_question2_results.ipynb` — supplemental experiments.
- `images/` — generated samples and outputs.
- `report/CA2_Complete_Solutions.tex` — authoritative LaTeX source; IEEE format.
- `report/DGM_CA2_final_EN.pdf` — rendered PDF; regenerate after `.tex` updates.

## Quick Search (rg)

```bash
rg -n "RealNVP|coupling" CA2_GANs_Normalizing_Flows
rg -n "Generator|Discriminator" CA2_GANs_Normalizing_Flows
rg -n "FID|frechet|clean-fid" CA2_GANs_Normalizing_Flows
rg -n "seed|torch.backends.cudnn" CA2_GANs_Normalizing_Flows/code
```

## Common Gotchas

- Log-det NaN/inf in flows → add eps/clamp; check scale outputs.
- GAN mode collapse → adjust lr, add label smoothing, or reduce discriminator steps.
- FID pipeline failures due to preprocessing mismatch (normalize/resize); verify ranges.
- OOD detection: ensure consistent preprocessing across MNIST/KMNIST.
- Large image grids clog git; keep out of commits unless essential.
- Report drift: rerender `.tex` after updates to experiments.

## Testing & Smoke

- Flow inverse test: sample batch, run forward + inverse, assert `allclose` within tolerance.
- GAN quick check: generate fixed-noise grid after few steps; ensure diversity.
- FID smoke: tiny sample (≤200 images) to ensure pipeline wiring, ignore absolute value.
- Save/load: checkpoint generator/flow once; reload and recompute a batch to confirm.
- Document smoke results in notebook Markdown or `report/` notes.

## Documentation & Reporting

- Maintain Markdown explanations for coupling layers, adversarial objectives, and evaluation.
- Keep `.tex` aligned with notebook figures/tables; ensure IEEE format (two-column, refs).
- Cite datasets and external code if adapted.

## Safety

- Do not delete `images/` unless regenerating intentionally.
- Avoid committing datasets/checkpoints; keep large artifacts local.
- Confirm before altering dataset cache paths.

## Pre-PR/Pre-Commit Checklist (CA2)

- Seeds + config centralized and visible.
- Flow inverse/round-trip smoke passed; GAN mini-train smoke passed with finite losses.
- Fixed-noise samples saved for comparison.
- `.tex` updated and PDF regenerated if `.tex` changed.
- No oversized artifacts staged unintentionally.

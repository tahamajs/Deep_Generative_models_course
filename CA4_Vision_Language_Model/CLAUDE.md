# CA4_Vision_Language_Model — Claude Code Guidance

**Technology**: Python, Jupyter, PyTorch/Transformers, PEFT/LoRA  
**Primary Notebook**: `code/final_CA4_training.ipynb` (plus evaluation notebooks under `code/eval_*`)  
**Reports**: `.tex` in `report/` → PDF (IEEE format, keep synchronized)  
**Parent Context**: Extends root `../CLAUDE.md`

## Development & Commands
- Start via `jupyter lab` → open `code/final_CA4_training.ipynb`.
- Environment: `transformers`, `peft`, `accelerate`, `datasets`, `evaluate`, `bitsandbytes` (if quantization), PyTorch with CUDA if available.
- Syntax check (if `.py` helpers appear later): `python -m py_compile CA4_Vision_Language_Model/**/*.py`.
- Render report: build `.tex` in `report/` to IEEE PDF after updating experiments.
- Smoke run: tiny subset of CLEVR (few samples), very small batch, few steps; verify loss finite and VRAM within limits.

## Architecture & Patterns
- Notebook flow: overview → config (imports, seeds, model names, LoRA ranks, precision) → data → model setup (Paligemma + LoRA) → training → evaluation → logging.
- Keep all hyperparameters (LoRA rank/alpha, lr, batch size, max steps, gradient accumulation, precision) in the config cell.
- Use deterministic seeds; log CUDA availability and dtype (fp16/bf16/int8).
- Data: CLEVR; ensure splits and preprocessing (tokenization + image transforms) are consistent; document path or download behavior.
- LoRA/PEFT: isolate adapter save paths; avoid overwriting base model; log adapter config.
- Evaluation: ROUGE/accuracy; store small qualitative samples (Q/A pairs with images) with seed noted.
- Reports: align metrics/tables with notebook outputs; include hardware notes for reproducibility.

## Key Files
- `code/final_CA4_training.ipynb` — main fine-tuning workflow.
- `code/eval_p1/final_CA4_results1.ipynb`, `code/eval_p2/final_CA4_results2.ipynb` — evaluation notebooks.
- `images/` — generated visualizations; keep representative outputs only.
- `report/` — LaTeX sources and PDFs; maintain IEEE formatting.

## Quick Search (rg)
```bash
rg -n "LoRA|lora" CA4_Vision_Language_Model
rg -n "Paligemma|vision-language" CA4_Vision_Language_Model
rg -n "seed|torch.backends.cudnn" CA4_Vision_Language_Model/code
rg -n "ROUGE|evaluate" CA4_Vision_Language_Model
```

## Common Gotchas
- VRAM exhaustion with large batch/sequence/image sizes → reduce batch, enable gradient accumulation, or use 8-bit.
- Forgetting `torch.set_grad_enabled(False)` during eval → unnecessary memory use.
- Missing `device_map`/`dtype` leading to CPU fallback; log actual device in config.
- Dataset/tokenizer mismatch → ensure vocab/model names align; cache path stable.
- Report drift: update `.tex` when metrics or qualitative results change.

## Testing & Smoke
- Mini run: very small subset (e.g., 16 samples), batch size 1–2, few steps; confirm loss finite and checkpoint saves.
- Adapter load test: save adapters, reload, run one eval batch to confirm weights.
- Evaluation smoke: run evaluation notebooks on tiny split to validate pipeline.
- Log seeds, dataset slice size, and hardware used for the smoke.

## Documentation & Reporting
- Add Markdown for model setup, LoRA rationale, and evaluation protocol.
- Keep `.tex` IEEE compliant; include tables/figures sourced from latest runs.
- Note hardware (GPU model, memory) and precision used for reported metrics.

## Safety
- Avoid committing large checkpoints; keep adapters local unless explicitly needed.
- Do not modify rendered PDFs manually; regenerate from `.tex`.
- Confirm before downloading large models if disk space is limited.

## Pre-PR/Pre-Commit Checklist (CA4)
- Config/seed cell present; hyperparameters centralized.
- Smoke train + eval on tiny subset completed with finite losses.
- Adapter save/load verified.
- `.tex` updated and PDF rebuilt if `.tex` changed.
- No unintended large artifacts staged.

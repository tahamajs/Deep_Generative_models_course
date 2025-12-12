Run a lightweight smoke test for the targeted assignment ($ARGUMENTS = CA folder):

1. Activate the correct environment (venv/conda) with PyTorch and required libs.
2. Open the notebook in `$ARGUMENTS/code` (or `codes`) and set:
   - `batch_size=16` (or smaller for VLM), `epochs=1` or a few steps
   - Fixed `seed` for `random`, `numpy`, `torch`, and CUDA (if available)
3. Execute only the minimal data/load + model + one training/eval step to ensure finite losses.
4. For flows: check forward+inverse round-trip on a mini-batch; for GAN: generate fixed-noise grid; for diffusion: short sampling with few steps; for VLM: tiny subset run.
5. Save minimal artifacts (small image grid/checkpoint) if useful; do not commit large outputs.
6. Record outcome (loss finite? shapes correct?) in notebook Markdown or a brief note.

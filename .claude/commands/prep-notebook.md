Prepare a notebook to comply with Claude rules:

1. Add an Overview Markdown cell (objectives, dataset, expected outputs).
2. Add a Setup/Configuration cell at the top:
   - Imports
   - Seeds for `random`, `numpy`, `torch`, CUDA deterministic flags
   - `device` selection, hyperparameters (batch_size, lr, epochs, latent_dim, etc.)
3. Ensure section order: overview → config → data → model → training → evaluation/results.
4. Centralize hyperparameters in one place; remove duplicates further down.
5. Insert shape assertions for key tensors (latent, reconstructions, generator outputs, flow inverse).
6. Add Markdown between major sections to explain intent and observations.
7. If the notebook feeds a report, note where figures/tables are produced and keep them synced with the `.tex`.

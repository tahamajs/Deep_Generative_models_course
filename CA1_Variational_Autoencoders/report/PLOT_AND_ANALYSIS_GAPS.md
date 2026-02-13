# CA1_Variational_Autoencoders - PLOT_AND_ANALYSIS_GAPS

## Gap Register
| Gap ID | Type | Location | Current State | Needed Artifact | Priority |
|---|---|---|---|---|---|
| GAP-CA1-02 | Analysis linkage | `report/DGM_CA1_Exercise_Solutions.tex` | Figures are present but mostly auto-named (`output_cell_*`), reducing experiment-to-figure traceability. | Add a small figure index mapping each figure to experiment step/cell and metric context. | P2 |

## Notes
- Missing/weak plots are documented only (per static-audit scope).
- No figure regeneration or retraining was performed in this pass.
- Resolution note for GAP-CA1-02: use this quick figure map when reading the report:
  - `output_cell_11_img_0.png`: Exercise 1 reconstruction sample.
  - `output_cell_19_img_[0-2].png`: Exercise 2 reconstruction grid variations.
  - `output_cell_21_img_0.png`: Training/validation loss curves.
  - `output_cell_23_img_0.png`: Generation sample grid.
  - `output_cell_25_img_0.png`: Latent interpolation grid.
  - `output_cell_27_img_0.png`: Latent space visualization (2D).
  - `output_cell_29_img_0.png`: Latent dimension analysis plot.

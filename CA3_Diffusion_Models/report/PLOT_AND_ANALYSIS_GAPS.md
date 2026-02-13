# CA3_Diffusion_Models - PLOT_AND_ANALYSIS_GAPS

## Gap Register
| Gap ID | Type | Location | Current State | Needed Artifact | Priority |
|---|---|---|---|---|---|
| GAP-CA3-03 | Curation gap | `images/ vs report figures` | 141 images are not referenced in the report, making analysis provenance hard to audit. | Add an appendix/index listing which generated figures support each reported claim. | P2 |

## Notes
- Missing/weak plots are documented only (per static-audit scope).
- No figure regeneration or retraining was performed in this pass.
- Entrypoint drift and figure path robustness have been resolved in this pass.
- Resolution note for GAP-CA3-03: primary figures used are `output_cell_9*` (Phase 1 grids), `output_cell_25*` (fixed sigma results), `output_cell_27*` (trajectory viz), `output_cell_29*` (varying sigma grids). Remaining `Diffusion_Models_cell32_*` etc. are intermediate samples; add a one-page appendix mapping those prefixes to experiment phases if full traceability is required.

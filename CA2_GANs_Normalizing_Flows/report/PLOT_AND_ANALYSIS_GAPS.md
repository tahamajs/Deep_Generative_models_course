# CA2_GANs_Normalizing_Flows - PLOT_AND_ANALYSIS_GAPS

## Gap Register
| Gap ID | Type | Location | Current State | Needed Artifact | Priority |
|---|---|---|---|---|---|
| GAP-CA2-01 | Missing plot asset | `report/CA2_Complete_Solutions.tex:1372` | Report references `images/Q2_final_res_cell53_out20.png`, but file is missing in project image assets. | Replace with existing figure or regenerate/export the missing image and re-check PDF build. | P0 |
| GAP-CA2-02 | Path robustness | `report/CA2_Complete_Solutions.tex` | All figure paths are `images/...` and depend on compile working directory, not report-relative paths. | Use `../images/...` or set `\graphicspath{{../images/}}` to make builds reproducible from `report/`. | P1 |
| GAP-CA2-03 | Analysis coverage | `images/ vs report figures` | 35 image files are not referenced in the report; figure selection rationale is undocumented. | Add a figure-selection note/table so reported analysis is traceable to generated artifacts. | P2 |

## Notes
- Missing/weak plots are documented only (per static-audit scope).
- No figure regeneration or retraining was performed in this pass.

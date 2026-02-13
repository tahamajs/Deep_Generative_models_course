# CA2_GANs_Normalizing_Flows - PLOT_AND_ANALYSIS_GAPS

## Gap Register
| Gap ID | Type | Location | Current State | Needed Artifact | Priority |
|---|---|---|---|---|---|
| GAP-CA2-03 | Analysis coverage | `images/ vs report figures` | 35 image files are not referenced in the report; figure selection rationale is undocumented. | Add a figure-selection note/table so reported analysis is traceable to generated artifacts. | P2 |

## Notes
- Missing/weak plots are documented only (per static-audit scope).
- No figure regeneration or retraining was performed in this pass.
- Missing figure reference and path robustness have been resolved in this pass.
- Resolution note for GAP-CA2-03: report currently uses epochs 1,3,5,8,10 grids plus select RealNVP outputs; other `CA2_DGM_cell21_*` samples are exploratory and can stay unreferenced. If desired, add a short appendix table mapping each included figure to epoch/checkpoint to fully close this gap.

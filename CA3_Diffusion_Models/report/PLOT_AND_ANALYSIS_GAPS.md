# CA3_Diffusion_Models - PLOT_AND_ANALYSIS_GAPS

## Gap Register
| Gap ID | Type | Location | Current State | Needed Artifact | Priority |
|---|---|---|---|---|---|
| GAP-CA3-01 | Run instruction drift | `README.md notebook paths` | Primary notebook names in README are outdated and can break reproducibility. | Update README run and file-tree sections with current notebook filenames. | P1 |
| GAP-CA3-02 | Path robustness | `report/DGM_CA3_Complete_Report.tex` | Figure paths are `images/...` and compile only when working directory assumptions hold. | Switch to report-relative paths (`../images/...`) or define `\graphicspath`. | P1 |
| GAP-CA3-03 | Curation gap | `images/ vs report figures` | 141 images are not referenced in the report, making analysis provenance hard to audit. | Add an appendix/index listing which generated figures support each reported claim. | P2 |

## Notes
- Missing/weak plots are documented only (per static-audit scope).
- No figure regeneration or retraining was performed in this pass.

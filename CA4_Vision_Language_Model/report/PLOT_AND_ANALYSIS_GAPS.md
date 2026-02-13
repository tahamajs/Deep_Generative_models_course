# CA4_Vision_Language_Model - PLOT_AND_ANALYSIS_GAPS

## Gap Register
| Gap ID | Type | Location | Current State | Needed Artifact | Priority |
|---|---|---|---|---|---|
| GAP-CA4-01 | Entrypoint drift | `README.md + CLAUDE.md` | Documented notebook names do not exist, breaking reproducibility path for training/evaluation. | Update all references to existing notebook names under `code/`, `code/eval_p1/`, and `code/eval_p2/`. | P0 |
| GAP-CA4-02 | Asset curation | `images/ vs report` | 14 images are not referenced in the final report; evidence selection is not documented. | Add a short figure inventory describing selected vs omitted qualitative samples. | P2 |
| GAP-CA4-03 | Artifact naming quality | `report/ PDF artifacts` | Final PDF name uses typo (`fainal`) which can cause automation/path confusion. | Standardize filename spelling while keeping backward-compatible copy if needed. | P2 |

## Notes
- Missing/weak plots are documented only (per static-audit scope).
- No figure regeneration or retraining was performed in this pass.

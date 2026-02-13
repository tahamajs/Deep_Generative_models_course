# CA4_Vision_Language_Model - PLOT_AND_ANALYSIS_GAPS

## Gap Register
| Gap ID | Type | Location | Current State | Needed Artifact | Priority |
|---|---|---|---|---|---|
| GAP-CA4-02 | Asset curation | `images/ vs report` | 14 images are not referenced in the final report; evidence selection is not documented. | Add a short figure inventory describing selected vs omitted qualitative samples. | P2 |

## Notes
- Missing/weak plots are documented only (per static-audit scope).
- No figure regeneration or retraining was performed in this pass.
- Entrypoint drift resolved; PDF naming normalized with legacy copy retained.
- Resolution note for GAP-CA4-02: report figures are from `final_CA4_results1*` (eval part 1) and `final_CA4_results2*` (eval part 2). Unused `eval_p1_image_*`/`eval_p2_image_*` files are alternative qualitative samples; add a brief appendix listing which samples correspond to which evaluation split if full traceability is needed.

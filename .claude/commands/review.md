Perform a focused review of recent changes:

1. Verify code follows root and directory `CLAUDE.md` rules (seeds/config first cell, DRY, IEEE reports synced).
2. Check for correctness against assignment description PDFs in `description/`.
3. Ensure no large binaries (datasets, checkpoints, oversized images) are staged.
4. Confirm notebooks have clear sectioning and Markdown explanations; no giant embedded outputs.
5. For Python: look for shape assertions, no silent exception swallowing, proper device handling.
6. For reports: `.tex` must be IEEE formatted and consistent with notebook results; PDF regenerated if needed.
7. Security: no secrets/PII; no destructive shell commands or force pushes.
8. Reproducibility: seeds set, hardware/device logged, hyperparameters centralized.
   Provide actionable feedback with file references.

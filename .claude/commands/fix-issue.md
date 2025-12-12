Analyze and fix an issue (ID or description provided as $ARGUMENTS):

1. Read relevant `CLAUDE.md` (root + subdir) and the assignment description PDF if applicable.
2. Locate affected code/notebooks with `rg` and directory structure cues.
3. Plan a minimal, DRY fix; keep config/seeds centralized.
4. Implement the change; add/adjust smoke tests or mini-runs appropriate to the CA.
5. If Python files changed, ensure syntax OK (`python -m py_compile`) and run smoke steps.
6. Update documentation/Markdown and the corresponding `.tex` report if outputs/claims change; regenerate PDF.
7. Check git status for unintended binaries; stage only intended files.
8. Commit with a descriptive message scoped to this fix.

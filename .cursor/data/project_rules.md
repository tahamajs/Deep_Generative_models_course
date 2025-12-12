# data — Claude Code Guidance

**Purpose**: Dataset cache/storage (e.g., `dsprites/`).  
**Parent Context**: Extends root `../CLAUDE.md`

## Rules

- Treat datasets as read-mostly; do not commit new large datasets or caches.
- Keep download/cache paths consistent (`TORCH_HOME=./data` recommended).
- Verify integrity after downloads; avoid duplicating datasets across folders.
- Do not rename or relocate existing dataset folders without explicit intent.

## Allowed Actions

- Read datasets for experiments.
- Add small metadata/readme files describing dataset versions and sources.

## Disallowed Actions

- Committing large archives/checkpoints into git.
- Deleting cached datasets without confirmation.

## Safety

- If space is constrained, document which datasets can be safely removed locally; never delete shared canonical copies without approval.

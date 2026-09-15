# CLI + Gates Rule

## CLI constraints
- CLI subagents must be NON-INTERACTIVE.
- Avoid commands that prompt for input, auth, editors, pagers.
- If a command might page output, add flags (e.g., --no-pager) or redirect output.

## Gates (defaults)
If repo defines explicit gate commands, use them.
Otherwise use best-effort defaults:

Python DS repos (cookiecutter-ish):
- `python -m pytest -q` (or `pytest -q`)
- `python -m ruff check .` (if ruff present)
- `python -m ruff format --check .` (if formatting enforced)
- `python -m mypy .` (if mypy configured)

Node/TS repos:
- `npm test` or `pnpm test`
- `npm run lint`
- `npm run typecheck`
- `npm run build`

## Gate behavior
- If gates fail:
  - capture failures in `.cursor/state/ralf/GATES.md`
  - fix in smallest possible diff
  - rerun gates
- Never “handwave” a failing gate.

## Safety
- Don’t run destructive commands without explicit user intent.
- Don’t modify secrets, credentials, or production resources.

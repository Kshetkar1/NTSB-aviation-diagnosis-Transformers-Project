# Cursor Orchestrator: GSD -> RALF (CLI Subagents)

This .cursor setup turns your Cursor editor chat into an orchestrator that:
1) Runs a **GSD-style** spec workflow (onboard -> PRD -> plan -> task DAG).
2) Runs a **RALF-style** execution loop (implement -> verify -> review -> loop until gates pass -> ship).
3) Uses **Cursor CLI** as "subagents" (spawned from the editor via terminal), with logs and state captured under `.cursor/state/`.

## Why this exists
- Cursor supports project Rules + Commands + Hooks (.cursor/rules, .cursor/commands, .cursor/hooks.json).
- Cursor 2.4 adds Subagents + Skills, but **subagents cannot use skills automatically yet**.
  Workaround: instruct subagents to read the relevant SKILL.md. (We do that everywhere.)
- You want: "Editor orchestrator + CLI subagents + robust logging + gates + triad reasoning (analyze / challenge / explore)."

## Quick start (typical workflow)
1. Run: `/gsd-spec` -- Produces PRD, then get user approval.
2. Run: `/gsd-plan` -- Produces Plan + TASKS.yaml.
3. Run: `/ralf` -- Executes tasks via async loop with CLI subagents.
4. Run: `/verify` -- Runs gates (tests/lint/typecheck/etc).
5. Run: `/ship` -- Creates PR-ready summary + checklist.

## Command Reference

| Command | Purpose |
|---------|---------|
| `/gsd` | Entry point: routes to /gsd-spec then /gsd-plan |
| `/gsd-spec` | Phase 1: onboard + PRD generation |
| `/gsd-plan` | Phase 2: plan + task DAG generation |
| `/ralf` | Execute tasks via async RALF loop with CLI subagents |
| `/verify` | Run gate commands (tests, lint, typecheck) |
| `/ship` | PR description draft + changelog update |
| `/council` | Triad reasoning: analyze, challenge, explore, synthesize |
| `/plan` | Refine PRD into implementation plan |
| `/taskgraph` | Generate or refine the task DAG (TASKS.yaml) |
| `/research` | Spawn researcher subagent for codebase exploration |
| `/audit` | Deep codebase audit: logical correctness, design reasoning |
| `/quick-audit` | Lightweight audit of a single file/function |
| `/analyze` | Interpret code, data, results, or notebooks |
| `/check-plan` | Verify implementation matches a LOCKED PLAN |
| `/document` | Differential documentation: sync docs with code changes |
| `/onboard` | Re-enter project context or explain to a new contributor |
| `/smart-commit` | Synchronized commit: code + docs + changelog |
| `/smart-pr` | Verified PR creation with plan compliance check |
| `/planning-workflow` | Iterative planning directive (audit, critique, refine, lock) |
| `/status` | Quick project state overview: tasks, gates, memory |
| `/reset-state` | Archive current state and start fresh for a new feature |
| `/experiment` | Autonomous experimentation loop: hypothesize, execute, analyze, iterate |

## Requirements
- Cursor editor (obviously).
- Cursor CLI present as one of: `cursor-agent`, `agent`, or `cursor` (the script auto-detects).
- Python 3 installed (for the optional stop-hook script).
- Strongly recommended: add `.cursor/state/` to .gitignore (keep templates + rules/skills/commands committed).

## RALF Stop Hook

The file `.cursor/hooks/ralf_stop_hook.py` enables automatic RALF loop continuation. When active, it checks `TASKS.yaml` after each orchestrator turn and prompts continuation if tasks remain.

**Enabling the hook:**
```bash
touch .cursor/state/ralf/ENABLE_STOP_HOOK
```

**Disabling the hook:**
```bash
rm .cursor/state/ralf/ENABLE_STOP_HOOK
```

**Safety:** The hook is disabled by default. It includes a hard cap of 8 iterations (`MAX_ITERATIONS`) to prevent infinite loops. The flag file must exist for the hook to trigger.

## Notes on Skills
Skills are defined as `SKILL.md` files and loaded dynamically (not always-on like rules).
If you are not on a Cursor build that supports skills yet, the commands/rules still work.

## Notes on CLI subagents
We spawn CLI calls via `.cursor/scripts/spawn_cli_subagent.sh`.
The orchestrator (editor chat) will:
- generate a "subagent prompt"
- spawn the CLI
- capture output + exit status
- update TASKS.yaml statuses
- run gates and iterate

## State layout
- `.cursor/state/gsd/` : PRD, plan, task graph
- `.cursor/state/ralf/` : execution logs and checkpoints
- `.cursor/state/_archive/` : archived state from previous phases (created by `/reset-state`)

If you ever get lost: run `/status` for a quick overview, or open `.cursor/state/gsd/PRD.md` and `.cursor/state/gsd/TASKS.yaml` directly.

## Full Index

See `.cursor/index.mdc` for the complete index of all rules, skills, commands, agents, and templates.

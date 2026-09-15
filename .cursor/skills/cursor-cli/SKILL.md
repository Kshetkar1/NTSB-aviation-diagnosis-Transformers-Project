---
name: cursor-cli
description: Rules for subagent execution -- both native Task tool (preferred) and CLI fallback. Non-interactive, logged, reproducible.
---

# Cursor Subagent Skill

## Execution Methods

### Primary: Native Task Tool (in-editor)
When running inside the Cursor editor, subagents are spawned via the Task tool.
- **Context isolation**: intermediate output stays in the subagent; parent sees only the final summary.
- **Model selection**: specify `model="fast"` for mechanical work, default for reasoning-heavy tasks.
- **Readonly mode**: set `readonly=true` for subagents that should not modify files (reviewer, challenger, explorer, researcher).
- No process management needed -- the Task tool handles lifecycle.

### Fallback: CLI Scripts (terminal/headless)
When running outside the editor, use `.cursor/scripts/spawn_cli_subagent.sh`.
- Set model via `MODEL=<model>` environment variable.
- CLI subagents get full tool access (no readonly enforcement).
- Output is tee'd to a file; context isolation is weaker.

## Non-interactive Rule
All subagents (both Task tool and CLI) must not rely on:
- interactive auth prompts
- opening editors
- pagers

If a command might page, use flags like `--no-pager` or redirect output.

## Output Discipline
Write deterministic, structured output:
- Include the STATUS token from your agent definition's Output Contract
- Summarize what you changed
- List files modified
- Include commands run
- Keep output concise -- the orchestrator only needs the summary

## If You Get Stuck
Return:
- what you tried
- what failed
- exact error output excerpt (max 30 lines)
- suggested next step for the orchestrator
- the appropriate circuit breaker STATUS token from your agent definition

# RALF Execution Rule

RALF loop = implement -> verify -> review -> iterate until gates pass.

## Subagent Dispatch

All subagents are spawned via the **native Task tool** (preferred) or CLI scripts (fallback).

Model and access assignments:
| Role | Model | Readonly | Why |
|------|-------|----------|-----|
| implementer | default | false | Needs full reasoning + write access |
| verifier | fast | false | Mechanical gate-running; needs shell |
| reviewer | default | true | Deep reasoning; read-only is safer |
| challenger | fast | true | Structured critique; no write needed |
| explorer | fast | true | Brainstorming; no write needed |
| researcher | fast | true | Search and summarize; no write needed |

## When running /ralf:

1) Read `.cursor/state/gsd/TASKS.yaml`
2) Identify runnable tasks (deps satisfied)
3) For each runnable task:
   - **Context Pack:** Use `prompt-engineer` skill to create a high-fidelity prompt.
     - Include `.cursor/state/ralf/MEMORY.md`.

   - **Step A: Implement**
     - Spawn IMPLEMENTER via Task tool (model=default, readonly=false).
     - Check returned summary for "ARCHITECTURE_MISMATCH". If found, **PAUSE** and request manual intervention.

   - **Step B: Verify**
     - Spawn VERIFIER via Task tool (model=fast, readonly=false).
     - Or run /verify directly if preferred.

   - **Step C: Review**
     - Spawn REVIEWER via Task tool (model=default, readonly=true).
     - If reviewer fails, feed feedback back to Implementer (Step A).

   - **Completion:**
     - If all pass: Mark "done".
     - **Update Memory:** Append learnings to `MEMORY.md`.

4) After a "batch" completes:
   - Summarize what changed
   - Note any remaining risks
   - Update `.cursor/state/ralf/EXECUTION.md` (create if missing)

## Parallelism

- Spawn up to 4 independent tasks simultaneously (Task tool limit).
- Tasks must have no dependency edges between them.
- Tasks must touch disjoint file sets.
- Prefer "one subagent per task" for clean logs.

## Logging

For every task T###, write:
  - `.cursor/state/ralf/logs/T###/prompt.txt`
  - `.cursor/state/ralf/logs/T###/output.txt`
  - `.cursor/state/ralf/logs/T###/notes.md` (what was changed + why)

## Retry Policy

- Each task gets max 3 retries (default). Override via `max_retries` in TASKS.yaml per task.
- Retry 1: re-run with failure feedback + fresh MEMORY.md read.
- Retry 2: run /council first (challenger + explorer on fast model, in parallel), then retry with council synthesis.
- Retry 3: attempt smallest conservative fix.
- After 3 failures: mark task `blocked`, append failure summary to MEMORY.md, escalate to user.
- Do NOT block independent tasks on a single task's retries.

## Partial Failure

- If a task is `blocked`, its dependents stay `todo` (deps unsatisfied).
- Independent tasks continue executing regardless.
- User can: "retry T###" (reset count), "skip T###" (unblock dependents), or fix manually.

## Important

- Subagents can't auto-use skills; tell them to read the right SKILL.md in the prompt.
- Always include MEMORY.md contents in the prompt -- it carries forward learnings.

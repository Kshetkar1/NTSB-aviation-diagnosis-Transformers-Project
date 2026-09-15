You are now running the **RALF Execution Loop** using native Task tool subagents.

DO THIS EXACTLY:

1) Read `.cursor/state/gsd/TASKS.yaml`.
   - If missing, instruct user to run `/gsd` first.

2) Read `.cursor/state/ralf/MEMORY.md` (if exists) for context from prior runs.

3) **TASK EXECUTION LOOP:**

   A) **Find Runnable Tasks:**
      - status == "todo"
      - all depends_on are "done" (check TASKS.yaml)
      - touch disjoint file sets (parallelism rule from agent definitions)

   B) **Spawn Independent Tasks in Parallel (via native Task tool):**
      For each runnable task T###:
      - Create log dir: `.cursor/state/ralf/logs/T###/`
      - Build the implementer prompt using `prompt-engineer` skill:
        * Include MEMORY.md contents
        * Copy exact AC from TASKS.yaml
        * List specific files to read (from files_hint)
        * Save prompt to `.cursor/state/ralf/logs/T###/prompt.txt`
      - **Spawn via Task tool:**
        ```
        Task(
          subagent_type="implementer",
          description="Implement T###: <title>",
          prompt=<the built prompt>,
          model=<from agent definition: default for implementer>,
          readonly=false
        )
        ```
      - Launch up to 4 independent tasks simultaneously (Task tool limit).
      - The Task tool provides TRUE context isolation: intermediate output stays in the subagent, only the final summary returns.

   C) **Process Results:**
      For each completed Task subagent:
      - Save the returned summary to `.cursor/state/ralf/logs/T###/output.txt`
      - Parse the output for STATUS tokens:
        * `IMPLEMENT_DONE` -> proceed to verification
        * `ARCHITECTURE_MISMATCH` -> pause, escalate to user
        * `IMPLEMENT_PARTIAL` -> assess whether to retry or escalate

   D) **Verify (via Task tool):**
      For each implemented task T###:
      - **Spawn verifier subagent:**
        ```
        Task(
          subagent_type="verifier",
          description="Verify T###: run gates",
          prompt="Run these gate commands for task T###: <gates from TASKS.yaml>. Report pass/fail with error excerpts.",
          model="fast",
          readonly=false  # needs shell to run gates
        )
        ```
      - If `GATES_PASS` -> proceed to review
      - If `GATES_FAIL` -> mark for retry with failure feedback

   E) **Review (via Task tool):**
      For each verified task T###:
      - **Spawn reviewer subagent:**
        ```
        Task(
          subagent_type="reviewer",
          description="Review T###: semantic check",
          prompt="Review the implementation of T###. AC: <acceptance criteria>. Check changed files against PRD. Read .cursor/state/gsd/PRD.md first.",
          readonly=true  # reviewer should never modify files
        )
        ```
      - If `REVIEW_PASS` -> mark task done
      - If `REVIEW_FAIL` -> mark for retry with review feedback

   F) **Complete or Retry:**
      - If gates + review pass:
        * Update TASKS.yaml -> status "done"
        * Append learnings to MEMORY.md
        * Append to EXECUTION.md
      - If any step failed: apply retry policy (see below)
      - Check for newly runnable tasks (deps may now be satisfied)
      - Loop back to step A

4) **Final Verification:**
   - Run `/verify` as a final sweep
   - Ensure all tasks marked "done" in TASKS.yaml

**Stop condition:**
- All tasks are "done" in TASKS.yaml
- All gates pass (`/verify`)

**At the end:**
- Run `/ship`.

## Why Native Task Tool Over CLI Scripts

The native Task tool provides advantages the CLI scripts cannot:
- **True context isolation**: intermediate output stays in the subagent; only the final summary returns to the orchestrator. CLI scripts tee all output into a file that the orchestrator reads entirely.
- **Model flexibility**: specify `model="fast"` for verifier/challenger/explorer (cheap, fast) or default for implementer/reviewer (needs full reasoning). CLI scripts default everything to `auto`.
- **Readonly enforcement**: `readonly=true` prevents the reviewer from accidentally modifying files. CLI scripts give full access to all subagents.
- **No process management**: no PIDs to track, no `ps -p` polling, no background processes. The Task tool handles this natively.
- **Cost efficiency**: fast model subagents cost ~1/10th and run significantly faster than default model subagents.

**CLI fallback**: If the Task tool is unavailable (e.g., running outside the editor), use `.cursor/scripts/spawn_cli_subagent.sh` with `MODEL=<model>` env var. The scripts still work but lack context isolation and readonly enforcement.

## Retry and Backoff Policy

- Each task has a max of **3 retries** (configurable via `max_retries` in TASKS.yaml, default 3).
- Track retry count per task: `retry_count["T###"] = 0`
- On verify or review failure:
  1. **Retry 1**: Re-read MEMORY.md, re-spawn implementer with failure feedback appended to prompt.
  2. **Retry 2**: Spawn `/council` (challenger + explorer in parallel via Task tool, both on fast model) on the failing task to get alternative approaches, then retry with council synthesis.
  3. **Retry 3 (final)**: Attempt with the most conservative fix possible (smallest diff).
  4. **After 3 failures**: Mark task as `status: blocked` in TASKS.yaml, append failure summary to MEMORY.md, and **escalate to user** with:
     - What was tried (3 attempts)
     - Last error output
     - Suggested manual intervention
- **Do NOT block other tasks** on a single task's retries. Continue spawning independent tasks.

## Partial Failure Recovery

- If task T003 fails but T004/T005 (which don't depend on T003) succeed:
  - Mark T003 as `blocked`, T004/T005 as `done`
  - Tasks that depend on T003 remain `todo` (their deps aren't satisfied)
  - Report the partial completion state to the user
- If a task is `blocked`, the user can:
  - Fix manually and run `/verify` to re-validate
  - Say "retry T003" to reset its retry count and re-attempt
  - Say "skip T003" to mark it as `skipped` and unblock dependents (if safe)

## Subagent Model Assignments

| Role | Model | Readonly | Rationale |
|------|-------|----------|-----------|
| implementer | default | false | Needs full reasoning to write correct code |
| verifier | fast | false | Mechanical: run commands, parse output |
| reviewer | default | true | Deep semantic reasoning for correctness |
| challenger | fast | true | Structured critique from template |
| explorer | fast | true | Breadth-oriented brainstorming |
| researcher | fast | true | Bulk search and summarization |

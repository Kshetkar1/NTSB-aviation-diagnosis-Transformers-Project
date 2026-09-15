# Project Status Dashboard

Show a quick overview of the current project state. Useful for re-entering context after a break.

## Instructions

1) **Task Summary** -- Read `.cursor/state/gsd/TASKS.yaml` and report:
   - Total tasks
   - Breakdown by status: `todo` | `doing` | `done` | `blocked` | `skipped`
   - Next runnable tasks (deps satisfied, status=todo)
   - Blocked tasks with reason (if any)

2) **Gate History** -- Read `.cursor/state/ralf/GATES.md` (if it exists) and report:
   - Last gate run date
   - Pass/fail status per gate command
   - Any outstanding failures

3) **Memory Highlights** -- Read `.cursor/state/ralf/MEMORY.md` (if it exists) and report:
   - Number of learning entries
   - Last 3 entries (summarized to 1 line each)
   - Any "Common Learnings" section highlights

4) **Execution Progress** -- Read `.cursor/state/ralf/EXECUTION.md` (if it exists) and report:
   - Last completed task and date
   - Overall progress percentage (done / total)

5) **PRD Status** -- Check if `.cursor/state/gsd/PRD.md` exists:
   - If yes: report title and number of acceptance criteria
   - If no: report "No PRD found -- run /gsd-spec to create one"

## Output Format

```
=== Project Status ===

PRD:        [exists/missing] - [title if exists]
Tasks:      X total | Y done | Z todo | W blocked
Progress:   XX% complete
Next up:    T### - [title]
Gates:      [last run status]
Memory:     N entries, last: "[summary]"

[Details if any tasks are blocked or gates failing]
```

## Tips

- Run `/status` at the start of a session to orient yourself.
- If everything looks stale, consider running `/reset-state` to start fresh.
- If tasks are blocked, check MEMORY.md for lessons from previous attempts.

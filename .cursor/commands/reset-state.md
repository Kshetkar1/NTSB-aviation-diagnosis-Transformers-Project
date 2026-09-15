# Reset State

Archive the current `.cursor/state/` directory and start fresh for a new feature or project phase.

## Instructions

1) **Confirm with user** -- Before resetting, show the current state summary (run `/status` logic) and ask:
   "This will archive all current GSD and RALF state. Proceed? (yes/no)"

2) **Create archive** -- If user confirms:
   - Create archive directory: `.cursor/state/_archive/<YYYY-MM-DD>_<short-label>/`
   - The `<short-label>` should be derived from the PRD title or user-provided label.
   - Move (not copy) the following into the archive:
     - `.cursor/state/gsd/PRD.md`
     - `.cursor/state/gsd/PLAN.md`
     - `.cursor/state/gsd/TASKS.yaml`
     - `.cursor/state/gsd/ONBOARD.md`
     - `.cursor/state/gsd/RESEARCH_QUERY.md`
     - `.cursor/state/gsd/RESEARCH_RESULTS.md`
     - `.cursor/state/ralf/EXECUTION.md`
     - `.cursor/state/ralf/GATES.md`
     - `.cursor/state/ralf/SHIP.md`
     - `.cursor/state/ralf/COUNCIL.md`
     - `.cursor/state/ralf/logs/` (entire directory)
   - Skip files that don't exist (not an error).

3) **Preserve persistent files** -- Do NOT archive these. They carry forward across phases:
   - `MEMORY.md` -- append a separator:
     ```
     ---
     ## [Archive: <label> archived on <date>]
     Previous learnings above this line are from the archived phase.
     ---
     ```
   - `.cursor/state/experiments/` -- NEVER touch. The experiment journal is a permanent research record.

4) **Reset state files** -- Ensure these exist but are empty/templated:
   - `.cursor/state/gsd/` -- only `.gitkeep` remains
   - `.cursor/state/ralf/` -- only `MEMORY.md` and `.gitkeep` remain

5) **Confirm** -- Report:
   - What was archived and where
   - That MEMORY.md was preserved
   - "Run /gsd-spec to start a new feature."

## Safety

- Never delete state without archiving first.
- Never touch MEMORY.md learnings -- only append the separator.
- If the archive directory already exists for today's date + label, append a counter (e.g., `_2`).

## Tips

- Use when switching between major features or project phases.
- The archive is kept in `.cursor/state/_archive/` for reference. Add it to `.gitignore` if you don't want it in version control.
- MEMORY.md accumulates across all phases intentionally -- it's the persistent learning store.

---
name: ralf
description: Implementation loop (implement ➜ verify ➜ review ➜ iterate) with logs and gates; designed to run via Cursor CLI subagents.
---

# RALF Skill: Loop Until Truth

## Ground rule
“Done” is not a feeling. Done = acceptance criteria met + gates pass + review passes.

## Standard subagent prompt header (MANDATORY)
Every CLI subagent prompt MUST start with:
1) "You are a CLI subagent. Do not ask interactive questions."
2) "First read .cursor/skills/cursor-cli/SKILL.md and follow it."
3) "Read .cursor/state/ralf/MEMORY.md (if it exists) to learn from past mistakes."
4) "Then execute the task below exactly."

## Loop structure
1) **Prep:** Orchestrator packs context.
2) **Implement:** Subagent writes code. (Check for "ARCHITECTURE_MISMATCH").
3) **Verify:** Run gates (tests/lint).
4) **Review:** Spawn REVIEWER subagent to check logic vs PRD.
5) **Learn:** If failure occurred, append lesson to `MEMORY.md`.
6) **Iterate:** If any step fails, loop back.
7) **Done:** Mark task done only if Verify + Review pass.

## Architecture Drift
If the implementer finds the plan is impossible:
- Output "STATUS: ARCHITECTURE_MISMATCH"
- Stop execution.
- Orchestrator will pause and ask for replanning.

## Parallelism rule
Parallelize only if:
- tasks have no dependency edge
- they touch disjoint file sets
Otherwise keep serial execution.

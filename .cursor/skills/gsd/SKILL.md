---
name: gsd
description: Spec-first workflow (onboard ➜ PRD ➜ plan ➜ task DAG) inspired by Claude Code “Get Shit Done” paradigms.
---

# GSD Skill: Spec-First Execution

## Purpose
Convert a vague goal into:
- a crisp PRD (testable acceptance criteria)
- a concrete plan (files + milestones)
- a task DAG (dependency-aware tasks)

## Output locations (required)
- `.cursor/state/gsd/ONBOARD.md`
- `.cursor/state/gsd/PRD.md`
- `.cursor/state/gsd/PLAN.md`
- `.cursor/state/gsd/TASKS.yaml`

## Onboard checklist
- What is the repo? What are the core modules?
- What does “done” mean in measurable terms?
- What constraints exist (time, tooling, infra, policy)?
- What could break? (top 5 failure modes)

## PRD quality bar
Acceptance criteria must be:
- observable
- testable
- phrased as “Given/When/Then” or equivalent.

## Task DAG quality bar
Each task must include:
- deps
- success checks
- expected files
- rollback note

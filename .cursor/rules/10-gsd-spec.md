# GSD Spec Rule

When running the GSD path (via /gsd, /plan, /taskgraph):
- Optimize for a *correct spec* over speed.
- Ask targeted questions only when truly blocking. Otherwise proceed with best-effort + explicit assumptions.

Deliverables you must produce/update:
1) `.cursor/state/gsd/PRD.md`
2) `.cursor/state/gsd/PLAN.md`
3) `.cursor/state/gsd/TASKS.yaml`

Minimum PRD sections:
- Problem statement (1 paragraph)
- Users / stakeholders
- Goals + non-goals
- Success metrics (measurable)
- Constraints (tech, time, policy)
- Acceptance criteria (testable)
- Risks + mitigations

Minimum plan sections:
- Architecture / approach
- Milestones (mapped to tasks)
- Dependencies
- Verification plan (gates)
- Rollback strategy

TASKS.yaml must:
- include dependencies (DAG)
- define gates for each task (commands/checks)
- define acceptance criteria

You are now running **Step 2 of GSD: Planning**.

Prerequisites:
- `.cursor/state/gsd/PRD.md` exists and is approved.

Steps:
1) Create PLAN:
   - Read `.cursor/state/gsd/PRD.md`.
   - Use `.cursor/templates/PLAN_TEMPLATE.md`.
   - Output to `.cursor/state/gsd/PLAN.md`.
   - Include verification strategy + gate commands.

2) Create TASK DAG:
   - Read Plan and PRD.
   - Use `.cursor/templates/TASKS_TEMPLATE.yaml`.
   - Output to `.cursor/state/gsd/TASKS.yaml`.

3) Review:
   - Show the Task DAG to the user.
   - Ask for confirmation before running `/ralf`.

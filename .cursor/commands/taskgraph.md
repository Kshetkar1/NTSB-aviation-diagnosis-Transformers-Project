Generate or refine the task DAG.

Instructions:
1) Read:
   - `.cursor/state/gsd/PRD.md`
   - `.cursor/state/gsd/PLAN.md`
2) Create/update:
   - `.cursor/state/gsd/TASKS.yaml`

Task design constraints:
- Tasks are small enough to complete in 1–2 focused agent runs.
- Dependencies are explicit (DAG).
- Avoid tasks that touch the same files concurrently (to preserve parallelism).
- Every task has:
  - acceptance_criteria
  - gates (commands/checks)
  - definition_of_done

After writing TASKS.yaml:
- Add a short “Execution Notes” section to PLAN explaining how to run /ralf safely.

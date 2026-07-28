# Cursor / agent documentation (this folder)

Agent-oriented docs for the NTSB diagnosis project live **here** so they are easy to `@` from the repo root.

| Start here | File |
|------------|------|
| Full implementation contract | [`Project_Outline_For_Cursor.md`](Project_Outline_For_Cursor.md) |
| Plain-language “what we’re doing” | [`Project_Primer_For_Agents.md`](Project_Primer_For_Agents.md) |
| Parallel chats playbook | [`Multi_Agent_Workflow.md`](Multi_Agent_Workflow.md) |
| Per-track task briefs (`01`–`05`) | [`agent_tasks/README.md`](agent_tasks/README.md) |
| Plan QA before coding | [`Plan_Review_Checklist.md`](Plan_Review_Checklist.md) |
| Professor feedback table | [`Faculty_Feedback_And_Fixes.md`](Faculty_Feedback_And_Fixes.md) |
| Decisions & rationale | [`What_Worked_And_Why.md`](What_Worked_And_Why.md) |

**Still in `docs/`:** other project markdown/PDFs, including **[`docs/Train_Test_Split_Plan.md`](../docs/Train_Test_Split_Plan.md)** (copy or symlink into a new repo if you want the split plan next to code).

**New / second repo:** Copy the entire **`DOCS_NEW_FILE/`** tree (or rename to `DOCS_NEW_FILE` in the new project) so `@` paths and relative links keep working. Update only if your new repo moves `docs/Train_Test_Split_Plan.md`—fix links that point to `../docs/`.

**Strict eval:** Retrieval must use **`merged_dataset_train.json`** + **`embeddings_train.*`**, not the full merge — set **`NTSB_USE_TRAIN_INDEX=1`** (see [`evaluation/README.md`](../evaluation/README.md)).

**Cursor:** `@DOCS_NEW_FILE/README.md` or `@DOCS_NEW_FILE/Project_Outline_For_Cursor.md`.

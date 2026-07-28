# Agent task briefs (parallel Cursor chats)

Start here from [`Multi_Agent_Workflow.md`](../Multi_Agent_Workflow.md).

**Before coding:** [`Project_Primer_For_Agents.md`](../Project_Primer_For_Agents.md) (understand) · [`Plan_Review_Checklist.md`](../Plan_Review_Checklist.md) (review each brief) · [`Faculty_Feedback_And_Fixes.md`](../Faculty_Feedback_And_Fixes.md) (professor issues) · [`What_Worked_And_Why.md`](../What_Worked_And_Why.md) (rationale).

| Order | File | Role |
|-------|------|------|
| Parallel | [01_Track_Splits_and_Refined.md](01_Track_Splits_and_Refined.md) | Train/test IDs + `merged_dataset_train.json` |
| Parallel | [02_Track_Preprocessing_Hooks.md](02_Track_Preprocessing_Hooks.md) | Train-mode embedding / 2b / 3 outputs |
| After A (for full runs) | [03_Track_Core_Eval.md](03_Track_Core_Eval.md) | `exclude_ev_ids` + `evaluate_diagnosis.py` |
| Parallel | [04_Track_UI_Meeting_Fixes.md](04_Track_UI_Meeting_Fixes.md) | Streamlit / LOTP / ties / captions |
| Last | [05_Integrator_Merge.md](05_Integrator_Merge.md) | Merge config + verify |

**Always attach** [`../Project_Outline_For_Cursor.md`](../Project_Outline_For_Cursor.md) in every chat (or `@DOCS_NEW_FILE/Project_Outline_For_Cursor.md`).

**Non‑negotiable for strict eval:** the live retrieval/diagnosis corpus is **`merged_dataset_train.json`** + **`embeddings_train.*`**, **not** the full `merged_dataset.json`. Full merge is for storage, eligibility, and building the train file only. See **`NTSB_USE_TRAIN_INDEX`** in [`evaluation/README.md`](../../evaluation/README.md) and the golden-rule callouts in **01–03**.

**New project:** Copy this whole **`DOCS_NEW_FILE/`** folder so outline, primer, checklist, faculty table, and these briefs stay in sync. Track **A → B → C** is the usual dependency chain for strict eval; **D** can run in parallel; **E** last.

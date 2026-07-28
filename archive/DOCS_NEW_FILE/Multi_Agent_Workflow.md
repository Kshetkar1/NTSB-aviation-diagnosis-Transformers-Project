# Multi-agent workflow (parallel Cursor chats)

Use this when you want **several Cursor agent chats** working at the same time, then **one integrator chat** to merge and verify.

## Chat types (recommended)

| Chat | Attach | Purpose |
|------|--------|---------|
| **0 — Understand** | [`Project_Primer_For_Agents.md`](Project_Primer_For_Agents.md) + [`Project_Outline_For_Cursor.md`](Project_Outline_For_Cursor.md) §1–5 | No code: restate goals, leakage rules, and ask clarifying questions. |
| **0b — Review a plan** | Task brief or Cursor plan + [`Plan_Review_Checklist.md`](Plan_Review_Checklist.md) | Find gaps before anyone implements (`Partial` rows). |
| **A–D — Build** | Outline + **one** [`agent_tasks/`](agent_tasks/) brief | Parallel implementation with clear file ownership. |
| **D — Faculty** | [`Faculty_Feedback_And_Fixes.md`](Faculty_Feedback_And_Fixes.md) + outline | Close open rows; update Status/Notes in the table. |
| **E — Integrate** | [`agent_tasks/05_Integrator_Merge.md`](agent_tasks/05_Integrator_Merge.md) + outline | Merge, verify, run tests; optionally refresh [`What_Worked_And_Why.md`](What_Worked_And_Why.md). |

## How many files do you need?

| Approach | Files | When to use |
|----------|-------|-------------|
| **Recommended** | **6** = this playbook + **5** task briefs under [`agent_tasks/`](agent_tasks/) | Clean `@` references per chat (one brief per chat). |
| **Minimal** | **2** = this playbook + one long `Agent_Task_Briefs_All.md` with all tracks as `##` sections | Fewer files; more scrolling and merge risk from copy-paste errors. |

You do **not** need more than one “integration” document: the integrator follows [`agent_tasks/05_Integrator_Merge.md`](agent_tasks/05_Integrator_Merge.md).

Non-negotiable context for **every implementation** chat:

- [`Project_Outline_For_Cursor.md`](Project_Outline_For_Cursor.md) (contract).

**Strict eval data:** The searchable index must be **`merged_dataset_train.json` + `embeddings_train.*`**, not the full `merged_dataset.json`. Full merge is allowed on disk for audit / building the train subset; runtime strict mode uses **`NTSB_USE_TRAIN_INDEX=1`** (see [`evaluation/README.md`](../evaluation/README.md)).

For **understanding-only** chats, prefer [`Project_Primer_For_Agents.md`](Project_Primer_For_Agents.md) first, then outline.

Optional deep copy:

- [`docs/Train_Test_Split_Plan.md`](../docs/Train_Test_Split_Plan.md)

---

## Tracks and file ownership

Avoid two agents editing the same file without ordering. **Own** means “only this track touches it unless integrator fixes conflicts.”

| Track | Brief | Owns (typical) | Can read |
|-------|--------|----------------|----------|
| **A** — Splits & train merged | [`agent_tasks/01_Track_Splits_and_Refined.md`](agent_tasks/01_Track_Splits_and_Refined.md) | `evaluation/splits/*`, script that writes `merged_dataset_train.json`, small `evaluation/README.md` if needed | `01_create_refined_dataset.py`, `config.py` (paths only if unavoidable—prefer integrator) |
| **B** — Preprocessing hooks | [`agent_tasks/02_Track_Preprocessing_Hooks.md`](agent_tasks/02_Track_Preprocessing_Hooks.md) | `data/preprocessing/2_*.py`, `3_*.py`, train path constants in `config.py` (`--train` on scripts) | `config.py` |
| **C** — Core retrieval & eval | [`agent_tasks/03_Track_Core_Eval.md`](agent_tasks/03_Track_Core_Eval.md) | `main_app.py` (only `find_top_matches` + helpers), `evaluation/evaluate_diagnosis.py` (new), tests under `tests/` for eval | `config.py` |
| **D** — UI / meeting fixes | [`agent_tasks/04_Track_UI_Meeting_Fixes.md`](agent_tasks/04_Track_UI_Meeting_Fixes.md) | `streamlit_app.py`, copy/strings only in UI layer | `main_app.py` read-only unless small call-site tweaks—coordinate with C |
| **E** — Integrator | [`agent_tasks/05_Integrator_Merge.md`](agent_tasks/05_Integrator_Merge.md) | Merge conflicts, `config.py` consolidation, wiring imports, CI/pytest runner | whole repo |

**Hot files** (`config.py`, `main_app.py`): at most one **parallel** owner; integrator resolves overlaps.

---

## Suggested order

```text
Parallel wave 1 (after Phase 0 bootstrap):
  A ─┐
  B ─┼─► wait until A has train_ev_ids + merged_dataset_train.json OR B uses CLI paths that don’t depend on A’s script name
  D ─┘   (D can start anytime; may need rebase after C if diagnosis API changes)

Sequence-sensitive:
  A before full B pipeline runs (train refined file must exist).
  C should land after A has split files (eval script needs test IDs).

Wave 2:
  C — implement exclude-self + evaluate_diagnosis once train index story is clear.

Finally:
  E — merge, single config story, run tests + one smoke eval.
```

If **B** only adds CLI flags and does not run the pipeline, **B** can start in parallel with **A**. If **B** must *run* embeddings on train data, **A** must finish first.

---

## How to start each chat (template)

Paste at the top of every new agent chat:

```text
Context (read first):
@DOCS_NEW_FILE/Project_Outline_For_Cursor.md

Your assignment ONLY:
@DOCS_NEW_FILE/agent_tasks/NN_Track_....md

Rules:
- Small diffs; match existing style.
- Do not edit files owned by another track (see Multi_Agent_Workflow.md table).
- End with: files changed, how to run smoke test, and any API/path the integrator must know.
```

---

## Integration checklist (high level)

Fully detailed steps live in [`agent_tasks/05_Integrator_Merge.md`](agent_tasks/05_Integrator_Merge.md). Summary:

1. One unified `config.py` story for train vs full paths (`TRAIN_MERGED_DATA_PATH`, `EMBEDDINGS_TRAIN_PATH`, …).
2. `main_app.py`: `find_top_matches` + any helpers; no duplicate definitions.
3. Preprocessing scripts agree on output names (`*_train.*`) per [Project outline §7](Project_Outline_For_Cursor.md).
4. Run §6 checks from outline + pytest for new tests.
5. Optional: `streamlit run streamlit_app.py` smoke test.

---

## When to avoid parallel agents

- One huge refactor of `main_app.py`.
- You have not frozen **normalization** for eval hits and **τ usage** (decide in outline or integrator first).

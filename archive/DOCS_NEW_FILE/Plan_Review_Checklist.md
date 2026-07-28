# Plan review checklist — make every plan “complete enough”

Use this when **you** (or an agent) draft or update a **Cursor plan**, a **task brief** under [`agent_tasks/`](agent_tasks/), or a **design note**. Goal: catch gaps **before** parallel implementation.

How to use:

1. Open the plan or task file side-by-side with this checklist.
2. Every section below: mark **Yes / Partial / N/A** and fix **Partial** before treating the plan as “ready.”
3. For parallel work, **do not start Track C** until **eligibility, τ usage, and hit normalization** are at least **Partial → Yes** (or explicitly deferred in writing).

---

## A. Problem and outcome

| # | Question | Notes |
|---|----------|--------|
| A1 | What user-visible or paper-visible **outcome** does this plan produce? | e.g. “Recall@5 on strict test set exported to CSV” |
| A2 | **Non-goals** stated? | What we explicitly will not change |
| A3 | **Success criteria** measurable? | Not “make it better”—define pass/fail |

---

## B. Data and leakage (evaluation tracks)

| # | Question | Notes |
|---|----------|--------|
| B1 | Split unit is **`ev_id`**, not rows? | |
| B2 | **Train ∩ test = ∅** enforced in artifacts and in code paths? | |
| B3 | Query text is **narrative-only** for eval (no `finding_description` in embedding input)? | |
| B4 | **Labels** loaded **after** prediction, for scoring only? | |
| B5 | **Exclude self** from neighbors (or test ∉ train index) specified? | |
| B5b | **Strict index** = **`merged_dataset_train.json`** + train embeddings — **not** full `merged_dataset.json` / full `embeddings.npy` for retrieval? (`NTSB_USE_TRAIN_INDEX` or explicit paths) | |
| B6 | **τ**: where chosen (dev/train), and forbidden (tune on test)? | |
| B7 | **Cause_Factor** filter for labels matches outline §10 unless user overrides? | |
| B8 | **Multi-label** hits defined (match **any** reference finding)? | |
| B9 | **Normalization** for predicted `cause` vs `finding_description` defined **once** and exported in results? | |

---

## C. Artifacts and paths

| # | Question | Notes |
|---|----------|--------|
| C1 | Full vs **train** merged JSON path named? (`MERGED_DATA_PATH` / `TRAIN_MERGED_DATA_PATH`) | |
| C2 | Full vs **train** `embeddings.npy` + `embeddings_map.json` named? | |
| C3 | Split files and **eval results** directory named? | See outline §7 |
| C4 | If new scripts: **CLI flags / env** documented with examples? | |
| C5 | Backward compatibility: default remains **full corpus** where applicable? | |

---

## D. Algorithms and metrics

| # | Question | Notes |
|---|----------|--------|
| D1 | **Similarity** = cosine / dot consistent with `find_top_matches`? | |
| D2 | **Which**: similarity-weighted vs LOTP path—or both with separate runs? | |
| D3 | **K** for Recall@K and **MRR** definition unambiguous (rank of first hit)? | |
| D4 | **τ** applied **where** (neighbor filter vs post-filter)? Matches export JSON? | |
| D5 | Skipped incidents (missing narrative, no labels) counted and reported? | |

---

## E. Code touch list and ownership

| # | Question | Notes |
|---|----------|--------|
| E1 | **Files owned** by this plan listed? | Avoid two agents on `main_app.py` |
| E2 | **Public API** changes (function signatures)? If yes, who updates call sites? | |
| E3 | **Dependencies** between tasks (A before B, etc.) explicit? | See `Multi_Agent_Workflow.md` |

---

## F. Verification

| # | Question | Notes |
|---|----------|--------|
| F1 | **Smoke test** command(s) in the plan? | |
| F2 | **Automated tests** (outline §6) updated or added? | |
| F3 | **Faculty / UI** items in [`Faculty_Feedback_And_Fixes.md`](Faculty_Feedback_And_Fixes.md) cross-checked if UI changes? | |

---

## G. Professor-facing narrative

| # | Question | Notes |
|---|----------|--------|
| G1 | User-visible math labeled **law of total probability** where appropriate? | |
| G2 | **Ties** and **Mermaid** semantics addressed if this plan touches UI? | |
| G3 | Any new claim on slides or UI tied to **implementable** behavior? | |

---

## H. Handoff (for integrator / next chat)

| # | Question | Notes |
|---|----------|--------|
| H1 | **End state**: list of new/changed files? | |
| H2 | **What could break** other tracks? | |
| H3 | **Open questions** escalated to user, not hidden? | |

---

## Agent prompt: “review my plan”

```text
Review this plan against every section of @DOCS_NEW_FILE/Plan_Review_Checklist.md.
List gaps as a numbered table: checklist ID, Yes/Partial/N/A, what is missing, suggested fix.
Do not write implementation code—only the review.
```

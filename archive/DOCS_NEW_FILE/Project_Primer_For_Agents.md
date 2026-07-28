# Project primer — what we are doing (read this first)

Use this file when you want a **new chat or collaborator to understand the project in plain language** before touching code. It complements the implementation contract in [`Project_Outline_For_Cursor.md`](Project_Outline_For_Cursor.md), which is optimized for *building*; this file is optimized for *understanding*.

---

## The problem

The **National Transportation Safety Board (NTSB)** publishes aviation accident data: narratives, structured findings, event sequences, and more. A natural question is: **given a new incident description, can we suggest plausible causes** in a way that is grounded in history and can be **evaluated fairly**—not just “looks good in a demo”?

This repository builds a **prototype pipeline** that:

1. **Merges** raw tables into one **merged incident dataset** (one JSON record per `ev_id`; on disk **`merged_dataset.json`**, `config.MERGED_DATA_PATH`). Older repos may still have `refined_dataset.json`—rename or re-run the merge script.
2. **Embeds** text into vectors for the **retrieval index**. The index may include multiple rows per incident (e.g. narratives and finding text as **separate embedding rows**). For **strict evaluation**, the **query** must be **narrative-only**—never embed `finding_description` as the string you score against; labels come from findings **after** prediction (see outline).
3. **Retrieves** similar past incidents by **cosine similarity** (dot product on normalized embeddings).
4. **Diagnoses** probable causes in two main ways:
   - **Similarity-weighted** pooling of causes seen in neighbors.
   - **Mixture over clusters** (user-facing: **law of total probability**): weight clusters by how well the query matches them, then combine within-cluster cause probabilities.
5. Optionally shows **prognosis** (sequence / time-oriented views) and a **Streamlit** UI.

---

## What “success” means here

Success is **not** “the LLM sounds smart.” Success is a **defensible evaluation**:

- **Ground truth** comes from official **findings** (`finding_description`, `Cause_Factor`), not from the model paraphrasing the narrative. Incidents often have **several** findings; a prediction **hits** if it matches **any** reference finding under the scoring rule (details and `Cause_Factor` filter in the outline §10).
- **Metrics** (e.g. **Recall@K**, **MRR**) compare **ranked predicted causes** to those findings under a clear protocol.
- **Strict evaluation** avoids **leakage**: test incidents are queried with **narrative-only** text; the retrieval index is built from **train-only** data; the test incident does not retrieve **itself**; **findings are not fed into the query embedding**; **τ** (a **similarity threshold** on the same cosine score as retrieval—see outline) is chosen without **tuning on the test split** to inflate metrics.

That strict setting is what makes results credible for **research, sponsors, and professors**.

### Which merged file for retrieval?

| File | Role |
|------|------|
| **`merged_dataset.json`** | Full corpus (e.g. all 2,200+ incidents): **master** record; use to **build** train subset and to **look up test labels** when scoring. |
| **`merged_dataset_train.json`** | **Train-only keys:** this (with **`embeddings_train.npy`** / **`embeddings_map_train.json`**) is the **only** merged JSON that should back the **similarity index** for strict eval. |

Do **not** load the full merge into `find_top_matches` for strict metrics. In this repo, set **`NTSB_USE_TRAIN_INDEX=1`** before starting Python so `main_app` loads the train paths (see [`evaluation/README.md`](../evaluation/README.md)).

---

## What exists vs what is still being built

**Already in the repo (typical):** preprocessing merge, embedding generation, diagnostic augmentation (`2b`), optional clustering (`3`), core logic in `main_app.py`, Streamlit UI.

**Still coming (see outline Phase 1–3):** reproducible **train/test splits** and **`merged_dataset_train.json`** (Track A), then running **`2` / `2b` / `3` with `--train`** for train-only embeddings (Track B — implemented). **Not done until wired:** **`exclude_ev_ids`** in `find_top_matches` (or equivalent), and an **`evaluate_diagnosis.py`** script that exports metrics and metadata (Track C).

---

## How faculty / presentation feedback fits in

Professors may challenge **wording** (e.g. “chain rule” vs **law of total probability**), **honesty** when many causes tie, **diagram semantics** (Mermaid edges), or **prognosis** explanations. Those items are tracked in [`Faculty_Feedback_And_Fixes.md`](Faculty_Feedback_And_Fixes.md). Implementation tracks (especially UI) should close those items **without** weakening the eval protocol.

---

## Where to go next

| Need | Document |
|------|----------|
| Defaults, paths, protocol, code entry points | [`Project_Outline_For_Cursor.md`](Project_Outline_For_Cursor.md) |
| Split/eligibility detail | [`docs/Train_Test_Split_Plan.md`](../docs/Train_Test_Split_Plan.md) |
| Professor issues & status | [`Faculty_Feedback_And_Fixes.md`](Faculty_Feedback_And_Fixes.md) |
| Why past choices worked | [`What_Worked_And_Why.md`](What_Worked_And_Why.md) |
| Review a plan before coding | [`Plan_Review_Checklist.md`](Plan_Review_Checklist.md) |
| Parallel agents | [`Multi_Agent_Workflow.md`](Multi_Agent_Workflow.md) |

---

## Suggested prompt for an “understanding-only” chat

```text
Read and summarize back in your own words (no code yet):
@DOCS_NEW_FILE/Project_Primer_For_Agents.md
@DOCS_NEW_FILE/Project_Outline_For_Cursor.md §1–§5

Call out any contradictions or missing assumptions. Do not implement until I confirm.
```

# Faculty feedback — issues, fixes, status

**Purpose:** Central place to record **what professors challenged**, **where to fix it**, and **whether it is done**. When you start a UI or “presentation correctness” chat, attach this file and say: *“Work only on rows with Status ≠ Done”* or *“Add a new row for today’s feedback.”*

**Related:** narrative context in [`Project_Primer_For_Agents.md`](Project_Primer_For_Agents.md); user-facing math naming in [`Project_Outline_For_Cursor.md`](Project_Outline_For_Cursor.md) §2. Implementation track for UI rows: [`agent_tasks/04_Track_UI_Meeting_Fixes.md`](agent_tasks/04_Track_UI_Meeting_Fixes.md).

---

## How to add new feedback

Copy a row into the table (or add at bottom):

- **ID:** `F-00x` incrementing  
- **Source:** meeting date, slide #, or quote (short)  
- **Issue:** what is wrong from faculty’s perspective  
- **Where to fix:** `streamlit_app.py`, slides, `main_app.py` strings, docs—be specific  
- **Correct behavior:** one sentence  
- **Status:** `Open` | `In progress` | `Done` | `Won’t fix` (with reason)  
- **Notes:** PR link, commit, or caveat  

---

## Tracking table

| ID | Source | Issue | Where to fix | Correct behavior | Status | Notes |
|----|--------|--------|--------------|------------------|--------|-------|
| F-001 | Presentation / theory | Internal name “chain rule” confuses; mixture is **law of total probability** | UI copy, any user-visible formula in Streamlit or docs | Show **Σ P(cause\|cluster)·P(cluster\|query)** and call it LOTP in user-facing text | Open | Code may keep internal function names; users see LOTP |
| F-002 | UI review | Many causes share same probability; looks arbitrary or misleading | `streamlit_app.py` (ranking display), any cause list from diagnosis | **Deterministic** tie-break (e.g. sort by cause string); explain ties in UI | Open | |
| F-003 | Diagrams | Mermaid **100%** edges confuse readers | Caption / help text near generated Mermaid | State edges are **empirical P(next\|node)**; often one dominant successor | Open | |
| F-004 | Prognosis | Anchor event unclear vs narrative; **Defining_ev** underused | Prognosis section UI + any slide text | Prefer **defining event** as anchor when data supports it | Open | |
| F-005 | Slides / narrative | Wrong or oversimplified **multi-step probability** story | Slides, not necessarily repo | Align slide math with code or label as illustrative | Open | Repo may only need a comment pointing to slide fix |
| F-006 | Evaluation rigor | Self-retrieval inflates metrics | `main_app.py` `find_top_matches`, eval script | **Exclude** query `ev_id` from neighbors; or exclude test from train index | Open | Tied to strict eval track |

---

## Prompt for an agent: “fix faculty items”

```text
Context:
@DOCS_NEW_FILE/Project_Outline_For_Cursor.md
@DOCS_NEW_FILE/Faculty_Feedback_And_Fixes.md

Task:
1. For every row with Status = Open or In progress, implement the fix in the listed files.
2. Update the Status and Notes columns (in Faculty_Feedback_And_Fixes.md) when done.
3. Small diffs; do not weaken eval protocol (narrative-only query, **train-only merged JSON + train embeddings** as the index — not full `merged_dataset.json`, `NTSB_USE_TRAIN_INDEX=1` for strict mode, no test leakage).
4. End with what you changed and how to verify in the UI or tests.
```

---

## Prompt: “incorporate new professor comments”

```text
Add new rows to @DOCS_NEW_FILE/Faculty_Feedback_And_Fixes.md from the bullet list below.
Do not implement yet—only catalog with clear Where to fix and Status = Open.

[Paste professor notes here]
```

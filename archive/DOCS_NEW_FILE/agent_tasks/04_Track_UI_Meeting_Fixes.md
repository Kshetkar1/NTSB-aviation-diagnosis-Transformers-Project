# Track D — UI & meeting / faculty fixes

## Goal

Apply **user-facing** and presentation fixes tracked in [`Faculty_Feedback_And_Fixes.md`](../Faculty_Feedback_And_Fixes.md) (update **Status / Notes** when done—rows **F-001–F-006**). Use [`Project_Outline_For_Cursor.md`](../Project_Outline_For_Cursor.md) for protocol constraints. **Do not** weaken eval rules (narrative-only query, train-only index for strict eval, no test leakage).

> **Strict eval UI:** To manually verify the same index as Track C, run Streamlit with **`NTSB_USE_TRAIN_INDEX=1`** so `main_app` loads **`merged_dataset_train.json`** and **`embeddings_train.*`**, not the full merged corpus. Default (unset) = full-corpus demo index.

## Scope (you own)

- `streamlit_app.py` — primary.
- **Copy only** in UI: “law of total probability” vs internal “chain rule” naming where user-visible.
- **Tied causes:** deterministic tie-break in displayed rankings; honest messaging when many causes share the same probability.
- **Mermaid** captions or helper text: explain 100% edges, empirical P(next|node).
- **Prognosis:** prefer `Defining_ev` as anchor where data supports; fix misleading multi-step probability narrative **in UI text** if present.

## Do not touch (unless unavoidable)

- Heavy changes to `main_app.py` math—prefer **display-layer** sorting and labels. If a one-line call-site change is required, keep it minimal and tell integrator.

## Dependency

- If Track C changes return shapes for diagnosis results, **rebase** after C or let integrator merge.

## Definition of done

- [ ] LOTP wording is consistent where the mixture formula is shown.
- [ ] Ties have stable ordering (e.g. secondary sort by cause name).
- [ ] Mermaid section has a short explanation if the app generates those charts.
- [ ] Streamlit still runs: `streamlit run streamlit_app.py`.

## Smoke test

Manual: open app, run one diagnosis path, confirm no exceptions and copy reads correctly.

## End-of-chat report

- Sections of UI changed:
- Any requested change to `main_app.py` (yes/no + lines):
- Screenshots optional:

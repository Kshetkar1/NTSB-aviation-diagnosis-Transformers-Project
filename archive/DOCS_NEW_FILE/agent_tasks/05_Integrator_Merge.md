# Track E — Integrator (merge & verify)

**File:** `agent_tasks/05_Integrator_Merge.md` (same as “Track E” in [`Multi_Agent_Workflow.md`](../Multi_Agent_Workflow.md).)

## Goal

Turn parallel branches into **one consistent repo**: single config story, no duplicate logic, tests pass, smoke eval runs.

> **Verify:** Strict eval and smoke eval use **`merged_dataset_train.json`** + train embeddings via **`NTSB_USE_TRAIN_INDEX=1`** (or equivalent explicit paths). **Do not** claim strict numbers if `main_app` still loads only the full `merged_dataset.json` / full `embeddings.npy` without the env var.

## When to start

After Tracks **A–D** have landed **or** when each track has pushed commits to a branch you can merge. Prefer **not** to run integrator in parallel with heavy edits to the same files.

## Scope (you own everything others touched)

1. **`config.py`**
   - One place for: `MERGED_DATA_PATH`, `TRAIN_MERGED_DATA_PATH`, full vs train embeddings + map + cause stats + checkpoint paths, split paths, results dir. (`REFINED_DATA_PATH` may remain as alias to `MERGED_DATA_PATH` for legacy imports.)
   - Remove duplicate or conflicting env var names.

2. **`main_app.py`**
   - Ensure `find_top_matches` + exclude-self behavior is coherent with `evaluate_diagnosis.py`.
   - No dead imports; **`NTSB_USE_TRAIN_INDEX=1`** switches loaded data to **train merged + train embeddings** (see `ACTIVE_INCIDENT_DATA_PATH`, `ACTIVE_EMBEDDINGS_PATH`, `ACTIVE_EMBEDDINGS_MAP_PATH` in `config.py`).

3. **Preprocessing**
   - Verify Track B flags work with Track A paths.

4. **Tests**
   - Run outline §6 checks + any new pytest.
   - Fix import/order issues only—avoid feature changes unless blocking.

5. **Streamlit**
   - Quick run; fix breakage from signature changes.

6. **Faculty table**
   - Confirm [`Faculty_Feedback_And_Fixes.md`](../Faculty_Feedback_And_Fixes.md) rows touched by D match merged code; no `Open` items that were claimed done.

## Merge checklist

- [ ] Train ∩ test = ∅ (script or test).
- [ ] Train-only merged JSON has no test keys.
- [ ] `embeddings_train.shape[0] == len(embeddings_map_train)` when train artifacts present.
- [ ] Eval script finds splits, merged JSON, train embeddings/map; exclude-self verified on one known `ev_id`.
- [ ] `streamlit run streamlit_app.py` loads.

## Commands (adapt to repo)

```bash
# pytest if configured
pytest -q

# smoke eval (example)
python evaluation/evaluate_diagnosis.py --limit 10   # if implemented

# Strict-eval UI / eval (train merged + train embeddings)
export NTSB_USE_TRAIN_INDEX=1
streamlit run streamlit_app.py
```

## Definition of done

- [ ] Main branch (or target branch) builds with no merge markers.
- [ ] Short **INTEGRATION_NOTES.md** optional: one page for team—what landed, how to run full eval, known gaps.

## End-of-chat report

- Conflicts resolved (files):
- Final path table (full vs train):
- Test results summary:
- Follow-ups for user (manual steps, API keys, data files):

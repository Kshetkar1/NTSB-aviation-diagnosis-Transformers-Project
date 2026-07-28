# Track C — Core retrieval & evaluation script

## Goal

Implement **exclude-self** retrieval and an **offline evaluation** script that reports Recall@K / MRR under the strict protocol (train-only index, narrative-only query, findings labels after prediction).

> **Golden rule:** The eval loop **must not** load **`MERGED_DATA_PATH` / full `merged_dataset.json`** or **`embeddings.npy`** for the retrieval index. Use **`TRAIN_MERGED_DATA_PATH`**, **`EMBEDDINGS_TRAIN_PATH`**, **`EMBEDDINGS_MAP_TRAIN_PATH`** only — or set **`NTSB_USE_TRAIN_INDEX=1`** before `import main_app`. **Labels** for test incidents still come from the **full** merge on disk (or from any file that contains test `ev_id` payloads) **only for scoring**, not as retrieval corpus.

## Prerequisites

- **Track A:** `evaluation/splits/train_ev_ids.txt`, `test_ev_ids.txt`, `data/processed/merged_dataset_train.json`.
- **Track B:** `embeddings_train.npy`, `embeddings_map_train.json` (and usually `2b` on `--train` so the map has `diagnostic_data`). Paths: `config.EMBEDDINGS_TRAIN_PATH`, `EMBEDDINGS_MAP_TRAIN_PATH`, `TRAIN_MERGED_DATA_PATH`.

## Scope (you own)

- `main_app.py`: extend `find_top_matches` (or add a thin wrapper) with **`exclude_ev_ids`** (e.g. `set[str]`) or equivalent filtering **immediately after** similarity sort—see [`Project_Outline_For_Cursor.md`](../Project_Outline_For_Cursor.md) §2.
- **New** `evaluation/evaluate_diagnosis.py`: loops test IDs, builds query from narrative (outline §10), loads **train-only** embeddings + map + **train merged** JSON, calls diagnosis aggregation consistent with chosen method (similarity-weighted vs LOTP path—**document which**).
- Tests: outline §6 (exclude-self, small synthetic Recall@K/MRR case if feasible). Place under `tests/` next to existing split/preprocessing tests.

## Config / loading

- For strict eval, use **train-only** artifacts. In this repo, set **`NTSB_USE_TRAIN_INDEX=1`** before importing `main_app`; it loads `TRAIN_MERGED_DATA_PATH`, `EMBEDDINGS_TRAIN_PATH`, `EMBEDDINGS_MAP_TRAIN_PATH` (see `config.py` and [`evaluation/README.md`](../../evaluation/README.md)). A dedicated `evaluate_diagnosis.py` should also set this (or pass paths explicitly).

## Do not touch (other tracks)

- Split generation (Track A).
- Preprocessing scripts (Track B), except **reading** the same paths.
- Streamlit copy (Track D).

## Coordinate with D

If you **change** public function signatures beyond adding optional kwargs, leave a one-line comment at call sites or notify integrator.

## Requirements (protocol)

- Query: narrative only; **no** `finding_description` in the embedding text.
- Labels: load findings **after** prediction; filter `Cause_Factor` per outline §10.
- Similarity: cosine / dot as in outline §2; **τ** handling and neighbor gating **must match** what you export in results JSON.
- **Hit** definition: normalized string match between predicted `cause` and reference `finding_description`—**same** normalization function for preds and labels; record rule in export.
- Exclude test `ev_id` from neighbors when scoring that incident (or ensure test ∉ train index—outline §5). Prefer **both** train-only index **and** exclude-self so faculty item **F-006** is satisfied.

## Definition of done

- [ ] `find_top_matches` supports exclusion (or documented filter helper used everywhere for eval).
- [ ] Eval script runs on a small test subset without crashing; writes JSON/CSV summary (paths outline §7).
- [ ] Exports document: model id, τ, K for Recall, split seed pointer, N scored/skipped.
- [ ] One pytest or smoke run documented for `evaluate_diagnosis` (even `--limit 5`).

## Smoke test

```text
# After train artifacts exist
python evaluation/evaluate_diagnosis.py --help
# Run with --limit 5 or similar if you add it
```

## End-of-chat report

- Diagnosis code path used (function names):
- Normalization rule for hits:
- How τ is applied (neighbor filter vs other):
- CLI flags and example command:
- Which config paths the eval script loads (train vs full):

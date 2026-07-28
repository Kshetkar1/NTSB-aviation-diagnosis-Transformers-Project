# What each file in `data/Testing_Data_Metrics` is for

Short map so you can find the right script without opening everything.

## Documentation (this folder)

| File | Purpose |
|------|---------|
| **`README.md`** | Main guide: diagnosis + prognosis **embedding-only** eval, splits, order of operations, outputs. |
| **`STRUCTURAL_MAPPING_EVAL.md`** | **A0 / A1 / A2** structural runs: modes, caches, commands, pointers to `Testing_Structural_Mapping`. |
| **`FILES_AND_SCRIPTS.md`** | This file — inventory of scripts and what they do. |

## Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| **`create_splits.py`** | Build `splits/train_ev_ids.txt` and `test_ev_ids.txt` from eligibility rules; mirrors train data for train-only indexing. |
| **`eval_common.py`** | Shared helpers: load test ids, CSV/JSON append, recall@k, MRR, cosine helpers, summary recompute for structural CSVs. |
| **`evaluate_diagnosis.py`** | **Baseline diagnosis eval (no structural term):** M1 (C findings) + M2 (`narr_cause` chunks), embedding match threshold. |
| **`evaluate_diagnosis_transformed.py`** | Diagnosis with **LLM query transform** + plain-language expansion; separate thresholds for M1/M2. |
| **`evaluate_prognosis.py`** | **Baseline prognosis:** next `sequence_of_events` step; exact / recall@5 / optional soft cosine. |
| **`run_structural_diagnosis_eval.py`** | **Launcher** for `Testing_Structural_Mapping/.../eval_diagnosis_structural.py` with presets **a0 / a1 / a2** (clear flags). |
| **`run_structural_prognosis_eval.py`** | Same for **prognosis** structural eval (`eval_prognosis_structural.py`). |

## Data paths (not scripts)

| Path | Purpose |
|------|---------|
| **`splits/train_ev_ids.txt`** | Train `ev_id` list for train-only embeddings and structural train caches. |
| **`splits/test_ev_ids.txt`** | Test `ev_id` list for evaluation. |
| **`outputs/`** | Default output location for `evaluate_*.py` runs (CSV / JSON / summary); filename set by `--output-stem`. |

## Related code outside this folder

| Location | Purpose |
|----------|---------|
| `Testing_Structural_Mapping/scripts/eval_diagnosis_structural.py` | Diagnosis with `--structural`, `--struct-version v1\|v2`, caches, `--alpha`. |
| `Testing_Structural_Mapping/scripts/eval_prognosis_structural.py` | Prognosis with the same structural options. |
| `Testing_Structural_Mapping/scripts/build_struct_cache_train.py` | Fill **v1** train struct JSONL. |
| `Testing_Structural_Mapping/scripts/build_struct_cache_train_v2.py` | Fill **v2** train struct JSONL. |

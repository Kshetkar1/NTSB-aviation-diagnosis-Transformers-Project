# Testing Data Metrics — Diagnosis + prognosis evaluation

Strict train/test evaluation uses **`NTSB_USE_TRAIN_INDEX=1`** where noted: retrieval uses **train-only** embeddings so test incidents are not in the index.

## Quick index (what lives here)

| Doc / script | Role |
|--------------|------|
| **`FILES_AND_SCRIPTS.md`** | One-line description of **every** file in this folder’s `scripts/` and related paths. |
| **`STRUCTURAL_MAPPING_EVAL.md`** | **A0 / A1 / A2** (baseline vs flat struct vs causal-chain struct): caches, commands, launchers. |
| **`scripts/run_structural_diagnosis_eval.py`** | Preset launcher (`a0` / `a1` / `a2`) → `Testing_Structural_Mapping/.../eval_diagnosis_structural.py`. |
| **`scripts/run_structural_prognosis_eval.py`** | Same for prognosis. |
| **`scripts/evaluate_diagnosis.py`** | Standard **embedding-only** diagnosis metrics (no structural reweighting). |
| **`scripts/evaluate_prognosis.py`** | Standard **embedding-only** next-event prognosis. |
| **`scripts/create_splits.py`** | Train/test `ev_id` lists under `splits/`. |
| **`scripts/eval_common.py`** | Shared CSV/JSON helpers for eval scripts. |

For structural similarity code (extraction, `struct_score`, fusion), see `Testing_Structural_Mapping/README.md` and `docs/CURSOR_STRUCTURAL_MAPPING_PLAN_v2.md`.

## Prognosis (next sequence event)

| Script | What it measures |
|--------|------------------|
| **`scripts/evaluate_prognosis.py`** | For each **test** `ev_id`, for up to **`--max-positions`** steps in `sequence_of_events`: **query** = text at step *i*, **ground truth** = text at step *i+1*. Runs `predict_future_events` (train-only neighbors). Reports **exact** match (normalized text) and **recall@5**. Optional **`--soft-threshold 0.85`**: cosine similarity between predicted top-1 and truth. |

```bash
python data/Testing_Data_Metrics/scripts/evaluate_prognosis.py --n 20 --output-stem eval_prognosis
python data/Testing_Data_Metrics/scripts/evaluate_prognosis.py --n all --soft-threshold 0.85 --output-stem eval_prognosis_soft
```

Requires the same **train-only** artifacts as diagnosis eval (`create_splits.py`, `2_generate_embeddings.py --train`, `merged_dataset_train.json`). Outputs: `outputs/eval_prognosis.csv` and `_summary.json`.

**Invariant tests** (optional API run): `RUN_PROGNOSIS_INTEGRATION=1 pytest tests/test_prognosis_invariants.py -q`

---

## Diagnosis evaluation

Strict train/test evaluation for **diagnosis** (cause prediction). The pipeline runs with **`NTSB_USE_TRAIN_INDEX=1`**: it searches **train-only** embeddings and never indexes test incidents.

## Two evaluation tracks

| Script | What it measures |
|--------|------------------|
| **`evaluate_diagnosis.py`** | **Structural / coded** causes: ranked taxonomy-style predictions vs **M1** (C findings) and **M2** (full `narr_cause` chunks). Single threshold **`MATCH_THRESHOLD = 0.75`**. **No M3.** |
| **`evaluate_diagnosis_transformed.py`** | **Query transformation** (LLM rewrites narrative for search) → diagnosis on transformed query; **plain-language** expansion of predictions and M1 via LLM; metrics compare **text vs text** with **separate** thresholds (defaults **M1: 0.65**, **M2: 0.55**; override with env `EVAL_TRANSFORM_THRESHOLD_M1` / `EVAL_TRANSFORM_THRESHOLD_M2`). |

Shared helpers live in **`scripts/eval_common.py`**.

## Narrative fields used as the query

| Field       | Meaning |
|------------|---------|
| **`narr_accp`** | Pilot/crew account; preferred — closest to what a user would type. |
| **`narr_accf`** | Investigator factual narrative; used if `narr_accp` is empty. |

The column **`query_source`** in results records which field was used.

## Structural eval metrics (ground truth)

| Metric | Source |
|--------|--------|
| **M1** | Structured findings: `finding_description` where `Cause_Factor == 'C'`. |
| **M2** | Full **`narr_cause`**. Long texts are split into chunks (~10k characters); **Match%** = max cosine over chunks vs the **structural** prediction. |

Scoring: same embedding model; **cosine** = dot product on normalized vectors. **Match%** = max similarity × 100. **Hit** = ≥ **`MATCH_THRESHOLD`** (0.75). **Recall@5** / **MRR** as in the standard script.

*Note:* **`2b_precompute_diagnostic_data.py`** still uses **keyword snippets** for `all_causes`; **eval M2** uses the **full** narrative for labels.

## Transformed eval (plain text)

- **`query_transformed`**: LLM rewrite for terminology (no new facts).
- **`top_prediction_detailed`** / **`top5_detailed_full`**: LLM expansion of structural cause lines.
- **`m1_truth_detailed`**: expanded official C findings.
- **M2** still uses **`narr_cause`** chunks as reference; **prediction side** is **plain** text, so the comparison is **prose vs narrative** (not slash-path vs narrative).

Uses **`LLM_MODEL`** from `config.py` (default `gpt-4o-mini`). Requires **`OPENAI_API_KEY`**.

## Order of operations

1. **Create splits and train JSON** (mirrors train JSON to `data/processed/merged_dataset_train.json`):

   ```bash
   python data/Testing_Data_Metrics/scripts/create_splits.py
   ```

2. **Train-only embeddings** (requires `OPENAI_API_KEY`):

   ```bash
   python data/preprocessing/2_generate_embeddings.py --train
   python data/preprocessing/2b_precompute_diagnostic_data.py --train
   ```

3. **Structural evaluation**

   ```bash
   python data/Testing_Data_Metrics/scripts/evaluate_diagnosis.py --n all --output-stem eval_results2
   python data/Testing_Data_Metrics/scripts/evaluate_diagnosis.py --n all --output-stem eval_results2 --resume
   ```

4. **Plain-text / transformed evaluation** (many LLM calls per incident)

   ```bash
   python data/Testing_Data_Metrics/scripts/evaluate_diagnosis_transformed.py --n 5
   python data/Testing_Data_Metrics/scripts/evaluate_diagnosis_transformed.py --n all --resume
   ```

Eligibility for the split: at least one **Cause_Factor `C`** finding **and** non-empty **`narr_accp` or `narr_accf`**.

Ground-truth incidents: **`REFINED_DATA_PATH`** from `config.py`.

## Outputs (`outputs/`)

| Pattern | Contents |
|---------|----------|
| `<stem>.csv` / `.json` / `_summary.json` | Per-row metrics and cumulative summary (`--output-stem` on each script). |

Structural default stem: **`eval_results`**. Transformed default stem: **`eval_results_transformed`**.

## Clearing results

Delete the files under `outputs/` if you want a clean run without mixing batches.

## Note on `config.py`

`REFINED_DATA_PATH` resolves to `refined_dataset.json` when present; otherwise `merged_dataset.json`.

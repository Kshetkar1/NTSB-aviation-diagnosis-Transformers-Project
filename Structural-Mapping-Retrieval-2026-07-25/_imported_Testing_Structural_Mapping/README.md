# Testing_Structural_Mapping

Pilot pipeline: **LLM-extracted incident structure** (train cache) + **rule similarity** to **reweight** embedding neighbors for diagnosis (chain-rule path) and prognosis (query-weighted transitions). Same train/test splits as `data/Testing_Data_Metrics`.

## Prerequisites

- `OPENAI_API_KEY` set (extraction + query structs + embeddings).
- Train-only artifacts from the main README / `data/Testing_Data_Metrics/README.md`: `merged_dataset_train.json`, `embeddings_train.npy`, `embeddings_map_train.json`, diagnostics precompute.
- Full corpus JSON `data/processed/refined_dataset.json` or `merged_dataset.json` for **test** narratives (eval scripts load test `ev_id`s from there).

## 1) Build train struct cache (one-time, resumable)

```bash
cd /path/to/NTSB_Shivy
export OPENAI_API_KEY=...

# Pilot: first 100 train ids
python Testing_Structural_Mapping/scripts/build_struct_cache_train.py --limit 100 --resume

# Full train list from splits
python Testing_Structural_Mapping/scripts/build_struct_cache_train.py --resume
```

Output: `Testing_Structural_Mapping/cache/struct_train_v1.jsonl` (gitignored).

## 2) Diagnosis eval

**A0 — baseline** (same as main app, no structural term):

```bash
python Testing_Structural_Mapping/scripts/eval_diagnosis_structural.py --n 20 \
  --output-stem eval_diagnosis_A0
```

**A1 — structural reweight**:

```bash
python Testing_Structural_Mapping/scripts/eval_diagnosis_structural.py --n 20 --structural \
  --alpha 0.5 --output-stem eval_diagnosis_A1
```

Query-side structs are cached in `Testing_Structural_Mapping/cache/query_struct_v1.jsonl`.

Resume a partial run (skip `ev_id`s already in the CSV):

```bash
python Testing_Structural_Mapping/scripts/eval_diagnosis_structural.py --n all --structural \
  --output-stem full_diagnosis_A1 --resume
```

## 3) Prognosis eval

```bash
python Testing_Structural_Mapping/scripts/eval_prognosis_structural.py --n 20 \
  --output-stem eval_prog_A0

python Testing_Structural_Mapping/scripts/eval_prognosis_structural.py --n 20 --structural \
  --alpha 0.5 --output-stem eval_prog_A1
```

## A2 — causal chain (v2)

Build the **v2** train cache (separate JSONL; uses `schema/extraction_prompt_v2.txt` and Chat Completions like v1):

```bash
python Testing_Structural_Mapping/scripts/build_struct_cache_train_v2.py --resume
```

Diagnosis / prognosis with **`--struct-version v2`** (defaults to `struct_train_v2.jsonl` / `query_struct_v2.jsonl`):

```bash
python Testing_Structural_Mapping/scripts/eval_diagnosis_structural.py --n 20 --structural \
  --struct-version v2 --alpha 2.0 --output-stem eval_diagnosis_A2

python Testing_Structural_Mapping/scripts/eval_prognosis_structural.py --n 20 --structural \
  --struct-version v2 --alpha 2.0 --output-stem eval_prog_A2
```

Shortcut launchers from `data/Testing_Data_Metrics/scripts/`: `run_structural_diagnosis_eval.py a2 ...` (see `data/Testing_Data_Metrics/STRUCTURAL_MAPPING_EVAL.md`).

## Weighting

For each retrieved neighbor, weight becomes:

`max(0, cosine_similarity) * exp(alpha * struct_similarity)`

- **v1** (`--struct-version v1`, default): `struct_similarity` from `scripts/struct_score.py` (enums + Jaccard on `contributing_factors`).
- **v2** (`--struct-version v2`): `struct_similarity` from `scripts/struct_score_v2.py` (Needleman–Wunsch–style alignment on `causal_chain`).

Missing train struct for a neighbor → no change vs baseline for that row.

## Code hooks

`main_app.diagnose_with_conditional_probabilities(..., score_adjust_fn=None)`  
`main_app.predict_future_events(..., score_adjust_fn=None)`

`score_adjust_fn(cosine_score, match_dict) -> float`.

## Files

| Path | Role |
|------|------|
| `schema/extraction_prompt_v1.txt` | LLM extraction instructions (flat enums, A1) |
| `schema/extraction_prompt_v2.txt` | LLM extraction instructions (causal chain, A2) |
| `schema/incident_struct_v1.json` | JSON Schema (documentation) |
| `scripts/struct_score.py` | Rule similarity (v1) |
| `scripts/struct_score_v2.py` | Chain alignment similarity (v2) |
| `scripts/extract_struct_v2.py` | v2 extraction + `extract_struct_from_incident_v2` |
| `scripts/build_struct_cache_train_v2.py` | Build `cache/struct_train_v2.jsonl` |
| `scripts/reweight.py` | `exp(alpha * sim)` blend |
| `scripts/struct_hooks.py` | Builds `score_adjust_fn` for eval (v1 or v2 similarity) |

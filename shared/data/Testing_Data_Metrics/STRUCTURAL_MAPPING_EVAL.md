# Structural mapping evaluation (A0 / A1 / A2)

This folder holds **train/test splits**, **shared eval helpers**, and **entrypoint scripts** for baseline vs structural runs. The heavy implementation lives under `Testing_Structural_Mapping/scripts/` (extraction, caches, `struct_score`, fusion).

## Modes (what each letter means)

| Mode | Meaning | Retrieval | Structural term |
|------|---------|-----------|-----------------|
| **A0** | Baseline | Embedding cosine only | Off |
| **A1** | Flat structure | Same | LLM → enum fields (`extraction_prompt_v1`) → `struct_score.py` → fuse with `reweight.reweighted_incident_score` |
| **A2** | Relational causal chain | Same | LLM → `causal_chain` (`extraction_prompt_v2`) → `struct_score_v2.py` (alignment) → same fusion |

Fusion: `weight = max(0, cosine) * exp(α * struct_sim)` (see `Testing_Structural_Mapping/scripts/reweight.py`). Tune **`--alpha`** per mode; v2 often uses higher α in the plan (e.g. 2.0) than early A1 runs (e.g. 0.5).

## Caches (do not mix v1 / v2)

| File | Role |
|------|------|
| `Testing_Structural_Mapping/cache/struct_train_v1.jsonl` | Train incidents → flat struct (A1) |
| `Testing_Structural_Mapping/cache/query_struct_v1.jsonl` | Test queries → flat struct (A1) |
| `Testing_Structural_Mapping/cache/struct_train_v2.jsonl` | Train incidents → causal chain (A2) |
| `Testing_Structural_Mapping/cache/query_struct_v2.jsonl` | Test queries → causal chain (A2) |

Build train caches:

```bash
# A1 train cache
python Testing_Structural_Mapping/scripts/build_struct_cache_train.py --resume

# A2 train cache (separate API calls)
python Testing_Structural_Mapping/scripts/build_struct_cache_train_v2.py --resume
```

## Commands (from repo root)

**Diagnosis** (M1/M2 metrics; `NTSB_USE_TRAIN_INDEX` is set inside the script):

```bash
# A0
python data/Testing_Data_Metrics/scripts/run_structural_diagnosis_eval.py a0 --n all --resume

# A1 — needs struct_train_v1 + query cache populated on demand
python data/Testing_Data_Metrics/scripts/run_structural_diagnosis_eval.py a1 --n all --resume

# A2 — needs struct_train_v2 first
python data/Testing_Data_Metrics/scripts/run_structural_diagnosis_eval.py a2 --n 10 --alpha 2.0
```

**Prognosis** (next sequence event):

```bash
python data/Testing_Data_Metrics/scripts/run_structural_prognosis_eval.py a0 --n 20
python data/Testing_Data_Metrics/scripts/run_structural_prognosis_eval.py a1 --n 20 --structural
python data/Testing_Data_Metrics/scripts/run_structural_prognosis_eval.py a2 --n 20 --structural --alpha 2.0
```

Direct equivalents (same behavior) are `Testing_Structural_Mapping/scripts/eval_diagnosis_structural.py` and `eval_prognosis_structural.py` with `--structural` / `--struct-version v2`.

## Outputs

Structural eval CSV/JSON summaries are written under `Testing_Structural_Mapping/outputs/` unless you pass `--output-stem` (see each script’s help).

## Design reference

See `docs/CURSOR_STRUCTURAL_MAPPING_PLAN_v2.md` for schema details, NW alignment sketch, and tuning notes.

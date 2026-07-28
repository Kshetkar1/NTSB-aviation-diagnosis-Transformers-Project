# Methodology Audit — June 22, 2026

Goal: make the **diagnosis methodology bulletproof** against the hardest questions
a reviewer (Maha) can ask — leakage, baselines, circular metrics, calibration —
using **real, computed evidence**, not assertions. No Bayesian network, no agents.

## The one-line result

> On the held-out test set, run **leak-free** (train-only index **+ exclude-self**),
> scored with a **non-circular** metric, the method reaches **0.74 top-1 accuracy**,
> beating the base-rate baseline (0.34) by **+0.40**, and is **well-calibrated**
> (Brier 0.097; high-confidence predictions are 100% accurate).

## The four holes this closes

| # | Hole (what Maha attacks) | How it's closed | Evidence |
|---|--------------------------|-----------------|----------|
| 1 | **Data leakage** — does a test case retrieve itself? | train-only index + new `exclude_ev_ids` guard in `find_top_matches` | Probe: **100%** of test cases self-leak in the naive full-corpus setup; **0%** here |
| 2 | **Circular metric** — grading retrieval with the retrieval model | score by **NTSB finding-category string match**, not embeddings | metric is taxonomy equality, model-independent |
| 3 | **No baseline** — does it beat guessing? | base-rate + random baselines, same metric | method 0.74 vs base-rate 0.34 vs random 0.20 |
| 4 | **Uncalibrated probabilities** | Brier + reliability table | Brier 0.097 < base-rate 0.148; reliability monotonic |

## Run it (fully offline — no OpenAI call)

```bash
# from repo root; uses the project's .venv which has numpy/sklearn/openai
.venv/bin/python methodology_audit_june_22_2026/run_leakfree_eval.py
# → methodology_audit_june_22_2026/outputs/leakfree_eval.{json,md}
```

Why it runs offline: query vectors are read from the **cached** full embedding
index (each test incident's `narrative` row), and retrieval is pure NumPy dot
product against the **cached** train index. All 177 train incidents already carry
a precomputed `cluster_label`, so no LLM classification fires.

## What it actually does (per test incident)

1. Take the test incident's cached narrative embedding as the query.
2. `main_app.find_top_matches(query, exclude_ev_ids={ev})` against the **train** index.
3. `cluster_incidents_by_type` → `calculate_cause_probabilities_per_cluster`
   → `calculate_chain_rule_diagnosis` (the real production LTP path).
4. Reduce predicted causes to **NTSB finding categories**; compare top-1 to the
   incident's true `Cause_Factor='C'` categories (string equality).
5. Compare against base-rate / random; accumulate Brier + reliability.

## Engine change made (backward compatible)

`main_app.find_top_matches(query_embedding, exclude_ev_ids=None)` and
`diagnose_with_conditional_probabilities(..., exclude_ev_ids=None)` now accept an
exclude set. Default `None` = original behavior, so the demo/UI is unaffected; eval
and any leak-sensitive use pass the query's own `ev_id`.

## Honest limits (state these too)

- Metric is **category-level** (level-1/level-2 of the finding taxonomy), not the
  exact finding leaf. Level-2 accuracy is 0.53.
- Test set is small (77 incidents); report this with the numbers.
- This audits **diagnosis**. Prognosis needs the same treatment next.

# Reproduce — Frozen BN + Narrative Evidence

**Status:** ACCEPTED (main paper path)  
**Modes:** Diagnosis **and** prognosis (injury + damage on held-out eval)

## Prerequisites

- Python 3.11 (validated environment)
- Repo root `.env` with `OPENAI_API_KEY` (only for LLM-tier / held-out with tier-2)
- Processed data under `shared/data/processed/` (see `shared/data/README.md`)

From repository root:

```bash
export PYTHONPATH="shared/code:Frozen-BN-Narrative-Evidence-2026-07-20/code:${PYTHONPATH:-}"
```

## 1. Foundation (Zhang reproduction)

```bash
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/reproduce_all_examples.py
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/zhang_conditional_method.py
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/reproduce_fig3_from_tables.py
```

Expected: prior ~5.53e-7, Table 7 85/85, Section 4.3 brake/wiring in ~1–11% band.

## 2. Main held-out eval (296 accidents, 2007–2019)

```bash
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/frozenbn_heldout_narrative_bn_eval.py
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/lr_baseline_heldout.py
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/embedding_lr_baseline.py
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/heldout_significance.py
```

Results: `Frozen-BN-Narrative-Evidence-2026-07-20/outputs/heldout_significance.md`

Headline (leak-safe **bn-sev** = k-NN severity as virtual evidence through the
frozen BN, Jul 2026):
- Injury top-1 **90.9%** (CI 87.5–93.9%) vs parsed-LR **87.8%** (McNemar p=0.02)
- Damage top-1 **77.4%** (CI 72.6–82.1%) vs parsed-LR **64.2%** (p<0.001)
- vs embedding-LR (strongest supervised baseline, 91.6% / 74.0%): not
  significantly different (p=0.50 injury, p=0.11 damage)
- BN prior alone: injury 58.4%, damage 42.6%
- Binary severe screening: severe injury 93.5% sens / 96.8% spec;
  severe damage 75.3% / 89.8%

LLM-tier ablation (needs API key; writes `*_llm.json` so the canonical
outputs above are untouched):

```bash
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/frozenbn_heldout_narrative_bn_eval.py --llm gpt-4.1 --tag _llm
```

Expected: llm-tier 68.9% / 51.4%, llm-first 69.9% / 57.8% (both below
bn-sev -- the LLM reads text, it is not the predictive signal); tier usage
176 deterministic / 120 LLM / 0 failures. Query embeddings are disk-cached
(`shared/data/processed/query_emb_cache.npz`), so re-runs are
deterministic and cost no embedding API calls.

The eval self-tests that severity virtual evidence propagates (hard failure
otherwise). `bn-fused` (event evidence + severity evidence together) is a
NEGATIVE ablation: both signals derive from the same narrative, so
product-of-experts fusion double-counts and collapses to 38.5% / 41.9%.
Outcome phrases and NTSB report boilerplate are stripped before any
embedding; stated-severity evidence is off by default.

## 2a. Diagnosis eval (cause-category level, era-fair)

```bash
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/diagnosis_heldout_eval.py
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/diagnosis_emb_lr_baseline.py
```

Outputs: `outputs/diagnosis_heldout_eval.md`, `outputs/diagnosis_emb_lr.md`.
253 held-out accidents with C/F cause findings; both coding eras rolled up
to CICTT top-level categories (legacy subjects mapped by keyword rules,
98.3% coverage). Headline: narrative retrieval **83.8%** top-1 / 0.912 MRR
vs frequency baseline 45.8% (McNemar p<0.0001); supervised emb-LR 88.1%
(beats retrieval p=0.027 -- disclosed; retrieval needs zero training).
BN event path 57.7% (beats baseline, p=0.0004); lift ranking 49.8%
(disclosed negative result).

## 2b. Leakage + robustness audits

```bash
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/heldout_leak_audit.py
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/redaction_leak_probe.py
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/retrieval_hyperparam_sensitivity.py
```

Outputs: `heldout_leak_audit.md`, `redaction_leak_probe.md`,
`hyperparam_sensitivity.md`. Free-parameter inventory (the "what is
trained?" answer): `docs_FrozenBN/FREE_PARAMETERS.md`.

## 3. Streamlit demo

```bash
./run_app.sh
```

Opens the diagnosis + prognosis tree demo (`apps/frozenbn_streamlit_diagnosis_prognosis_demo.py`).

## 4. Key entry scripts

| Script | Purpose |
|--------|---------|
| `code/query_to_bn.py` | Narrative → evidence → inference |
| `code/llm_evidence.py` | Tier-2 LLM fallback parser |
| `tests/frozenbn_maha_narrative_evidence_ladder_demo.py` | Maha ladder demo |
| `tests/frozenbn_tiered_parser_validation_11_scenarios.py` | Parser scoreboard |

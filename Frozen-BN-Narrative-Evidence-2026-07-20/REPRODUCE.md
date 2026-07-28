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
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/heldout_significance.py
```

Results: `Frozen-BN-Narrative-Evidence-2026-07-20/outputs/heldout_significance.md`

Headline (full BN + narrative evidence):
- Injury top-1 ~88.5% vs prior ~58.4%
- Damage top-1 ~64.2% vs prior ~42.6%
- LR baseline (trained on build set only): injury ~87.5%, damage ~64.5%

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

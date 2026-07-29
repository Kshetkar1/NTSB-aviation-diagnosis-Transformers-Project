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

Convenience wrapper: `./scripts/reproduce_foundation.sh` (runs the three commands below).

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
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/tfidf_lr_baseline_heldout.py
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/heldout_significance.py
```

Results: `Frozen-BN-Narrative-Evidence-2026-07-20/outputs/heldout_significance.md`

Headline (leak-safe **bn-sev** = k-NN severity as virtual evidence through
the frozen BN, Jul 2026; all baselines on the identical 296-accident cohort,
`outputs/cohort_manifest.json`; p-values Holm-corrected over the 5 primary
comparisons per target):
- Injury top-1 **90.9%** (CI 87.5–93.9%) vs parsed-LR **85.5%** (Holm p=0.006)
- Damage top-1 **77.4%** (CI 72.6–82.1%) vs parsed-LR **60.1%** (Holm p<0.0001)
- vs embedding-LR (91.6% / 74.0%): not significantly different
  (Holm p=1.0 injury, p=0.29 damage)
- vs TF-IDF-LR (92.2% / 73.3%; the leak probe promoted to a baseline,
  `tests/tfidf_lr_baseline_heldout.py`): not significantly different
  (Holm p=0.87 injury, p=0.29 damage); bn-sev keeps best damage Macro-F1
  (0.697) and damage severe-screen sensitivity (75.3% vs 58.0%)
- vs retrieval-sev: identical by construction (0/296 discordant) — the BN
  mediates the signal losslessly, it does not add accuracy
- BN prior alone: injury 58.4%, damage 42.6%
- Binary severe screening: severe injury 93.5% sens / 96.8% spec;
  severe damage 75.3% / 89.8%

LLM-tier ablation (needs API key; writes `*_llm.json` so the canonical
outputs above are untouched):

```bash
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/frozenbn_heldout_narrative_bn_eval.py --llm gpt-4.1 --tag _llm
```

Expected: llm-tier **68.2% / 51.7%**, llm-first **66.9% / 57.8%** (both below
bn-sev — the LLM reads text, it is not the predictive signal); tier usage
176 deterministic / 120 LLM / 0 failures. Rerun date 2026-07-29 under full
redaction (parsers receive redacted text only). Query embeddings are disk-cached
(`shared/data/processed/query_emb_cache.npz`), so re-runs are
deterministic and cost no embedding API calls.

The eval self-tests that severity virtual evidence propagates (hard failure
otherwise). `bn-fused` (event evidence + severity evidence together) is a
NEGATIVE ablation: both signals derive from the same narrative, so
product-of-experts fusion double-counts and collapses to 38.5% / 41.9%.
Outcome phrases and NTSB report boilerplate are stripped before ANY use of
the text (embedding, retrieval, deterministic parse, LLM parse); the
stated-severity readout is off by default and exists only as an ablation.

## 2a. Diagnosis eval (cause-category level, era-fair)

```bash
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/diagnosis_heldout_eval.py
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/diagnosis_emb_lr_baseline.py
```

Outputs: `outputs/diagnosis_heldout_eval.md`, `outputs/diagnosis_emb_lr.md`.
253 held-out accidents with C/F cause findings; both coding eras rolled up
to CICTT top-level categories (legacy subjects mapped by keyword rules,
97.6% coverage). Headline: narrative retrieval **84.2%** top-1 / 0.915 MRR
vs frequency baseline 45.8% (McNemar, Holm-corrected p<0.0001); supervised
emb-LR 88.1% (beats retrieval p=0.041 -- disclosed; retrieval needs zero
training). BN event path 57.7% (beats baseline, Holm p=0.0007); lift
ranking 50.2% (disclosed negative result). The legacy-to-category rollup is a
published keyword rule set; a 75-row stratified sample was reviewed for
internal consistency (2 rule defects corrected, 5 boundary rows kept) and the
contested mass (316/5,062 findings, 6.2%) bounds the mapping's effect at
<= 2 pp in the same direction for every predictor, so the ordering is
insensitive to mapping choices; observed effect was retrieval +0.4 pp only.
Not independent dual coding, and not claimed as such -- see
`outputs/mapping_audit_sample.csv` + `outputs/mapping_audit_summary.md`.

## 2b. Leakage + robustness audits

```bash
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/heldout_leak_audit.py
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/redaction_leak_probe.py
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/retrieval_hyperparam_sensitivity.py
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/build_cohort_manifest.py
```

Outputs: `heldout_leak_audit.md`, `redaction_leak_probe.md` (window-trained
probe + in-sample upper bound), `hyperparam_sensitivity.md`,
`cohort_manifest.json` (exact accident IDs per cohort + cross-checks).
Free-parameter inventory (the "what is trained?" answer):
`docs_FrozenBN/FREE_PARAMETERS.md`.

## 2c. Fire-node cross-inference (negative result)

```bash
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/fire_node_cross_inference.py
```

Outputs: `outputs/fire_node_cross_inference.md`, `.json`. Headline: BN ROC AUC
~0.38–0.46 on coded fire (below chance); retrieval neighbour fire rate on
identical text ~0.96–0.98; keyword-fire regex ~0.94. Does **not** support
held-out predictive lift from BN cross-node inference over retrieval.

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

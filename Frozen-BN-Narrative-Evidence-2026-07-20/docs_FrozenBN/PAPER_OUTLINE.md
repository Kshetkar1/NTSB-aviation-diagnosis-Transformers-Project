# Paper outline — every section backed by an existing result

Working title: *Narrative-driven probabilistic diagnosis and prognosis of
airline accidents: validating, upgrading, and language-enabling a published
Bayesian network*

Target: Reliability Engineering & System Safety (same venue as Zhang &
Mahadevan 2021) or Safety Science.

The rule this outline follows: **match Zhang wherever we claim to do what he
did; measure and explain the difference wherever we claim to do something he
couldn't.**

---

## 1. Introduction
- Zhang & Mahadevan (RESS 2021) built a BN for airline accident diagnosis /
  prognosis from the CODED NTSB database (1982–2006). Two open questions:
  (a) is the model reproducible from the published methodology, and
  (b) can the free-text narratives — untouched in the original — be used?
- Contributions (one paragraph each):
  1. Independent reproduction + a quantified reproducibility finding
     (build variance of the randomized structure step).
  2. Two methodological upgrades that recover published behavior a faithful
     first-pass build cannot run (person nodes; multi-state severity).
  3. A narrative-to-evidence bridge (deterministic + retrieval + Jeffrey
     conditioning) with out-of-sample validation.
  4. A measured decomposition of what LLMs can and cannot do in this
     pipeline (four experiments, two positive, two negative).

## 2. Background and data
- Zhang's Section 4 recipe summary (priors Eq. 6, edge ratios Eq. 8,
  Beta-CDF CPTs Eqs. 10–14, 12-parent cap).
- Our refined dataset: 1,742 accidents 1982–2006 (window matches his),
  1,288 with factual narratives; held-out 2007–2019 (296 with narratives).
- Artifacts: `data/processed/refined_dataset_1982_2006.json`.

## 3. Faithful reproduction and the 93-item validation
- Counting layer: Table 7 **85/85 exact** (`docs/TABLE7_FULL_REPRODUCTION.md`),
  priors and Beta-CDF anchors pass (`outputs/reproduce_all_examples_results.json`).
- Network: `tests/bn_build_ours.py`; scoreboard 43 exact / 16 close /
  26 differ / 8 qualitative BEFORE upgrades
  (`outputs/BN_COMPARISON_REPORT.md`).
- **Reproducibility finding**: his randomized parent selection makes the
  published numbers one draw from a distribution; variance envelope in
  `outputs/bn_variance_envelope.json`. Frame as a finding about learned-BN
  reproducibility, not a defect of either build.
- Fig 3 internal-consistency check (`tests/reproduce_fig3_from_tables.py`)
  resolves the Table 3/4/5 question: tutorial values, not data-derived.

## 4. Two upgrades
- Person-finding nodes (unlocks Fig 12 pilot queries — previously not
  runnable) and multi-state severity nodes (fixes the impossible
  P(no injury) cells). `tests/bn_upgraded.py`.
- After upgrades: **48 exact / 29 close / 12 differ / 4 qualitative**
  (`tests/bn_upgraded_full.py`); each remaining gap traced.
- Full-precision side-by-sides for every recreatable table:
  `docs/ALL_TABLES_EXACT_COMPARISON.md`.

## 5. The narrative-to-evidence bridge
- Architecture: narrative → structured evidence {node: confidence} → BN.
  Hard evidence (named facts), soft evidence (retrieval f_q + Jeffrey
  conditioning via likelihood ratio against the node prior). `query_to_bn.py`.
- Scenario validation: Zhang's queries typed as English reproduce his evidence
  and posteriors exactly (`tests/all_tables_exact.py`).
- **Held-out validation (the key table)**: 296 accidents 2007–2019, leak-safe.
  Injury top-1 58.4% (prior) → 90.9% (bn-sev); damage 42.6% → 77.4%
  (`tests/frozenbn_heldout_narrative_bn_eval.py`,
  `outputs/heldout_significance.md`). State alongside it, in the same
  paragraph, the two facts that keep this honest: (a) `bn-sev` and
  `retrieval-sev` are identical (0/296 discordant), so the network adds **zero
  predictive lift** — the accuracy is the k-NN narrative signal, and the
  network's contribution is joint reasoning at zero accuracy cost; (b) the
  chain only *ties* the supervised text baselines (TF-IDF LR 92.2% injury,
  Holm p = 0.87, 8 discordant; emb-LR 91.6%/74.0%), and is numerically best
  only on damage (77.4%), damage Macro-F1 (0.697), and severe-damage
  sensitivity (75.3%). Do not present 58% → 90% as a standalone headline.
- Calibration of soft-evidence confidences: reliability diagram, retrieval
  f_q ECE ≈ 0.12 (`docs/figures/confidence_calibration.png`,
  `tests/confidence_calibration.py`).
- Baseline: supervised logistic regression on the same parsed features
  (`tests/lr_baseline_heldout.py`) — report side by side; note the BN is
  unsupervised for this task and additionally provides posteriors over all
  nodes + what-if reasoning.

## 6. What can the LLM do? Four measured answers
1. **Query interface — works.** gpt-4o-mini extracts Zhang's exact evidence
   11/11; posteriors identical to hand-set evidence
   (`docs/LLM_VS_ZHANG_TABLES.md`). Robustness on unseen sentences:
   `tests/llm_paraphrase_robustness.py` (report strict score + failure
   modes). Guardrails: verbatim vocabulary, hallucination guard, generic-vs-
   specific coding convention (justified by the measured zero co-occurrence
   of generic and specific codes).
2. **Evidence extractor on real narratives — usable, not better.** Held-out
   comparison of prior / det / LLM / hybrid / supplement predictors
   (`tests/llm_heldout_eval.py`, full-296 run). LLM self-confidences are
   uncalibrated (ECE ≈ 0.74 vs 0.12 for measured f_q) — the reliability
   diagram is the money figure.
3. **Replacement for the coded database — fails, with the reason.**
   Narrative-only rebuild of the whole dataset (1,288 accidents re-coded by
   the LLM): outcomes survive (injury 81%, damage 61% with honest
   abstentions), causal attribution inverts (Table 7 Spearman −0.43) —
   narratives describe events, not investigative findings
   (`docs/LLM_NARRATIVE_ONLY_REBUILD.md`).
4. **Replacement for the network — fails.** Asked directly for the
   probabilities, median error ~10x, no evidence propagation (flat answers
   on the Sec 5.2 ladder) (`docs/LLM_DIRECT_VS_BN.md`).
- Contamination note: NTSB reports are public; extraction (not prediction)
  framing + CPTs from pre-2007 coded data insulate reported probabilities.

## 7. Demonstration system
- Streamlit app: diagnosis / prognosis trees (Table 7 exact at level 1),
  multi-evidence BN section, narrative pre-parsing. One figure + pointer to
  repository. Keep short — the app is evidence of usability, not a claim.

## 8. Discussion, limitations, conclusion
- **Canonical limitations list: `docs_FrozenBN/RESULTS_SECTION.md` §5.5.**
  Write the paper's limitations section from that file, not from this outline —
  it is maintained against the current leak-safe results.
- Limitations specific to the LLM experiments below (§6), which §5.5 does not
  cover: single domain/dataset; gpt-4o-mini only (justify: failures are
  calibration/over-extraction, not comprehension); truth signal for
  extraction is the coded record (conservative); prompt rule 1c added after
  in-sample diagnosis (paraphrase suite is the out-of-sample check).
- Conclusion mirrors the architecture sentence: the LLM is the interface,
  the data-built network is the calculator; we measured every link and both
  swap directions fail.

---

## Figures / tables shortlist
1. Pipeline diagram (narrative → evidence → BN → posterior).
2. 93-item scoreboard before/after upgrades (canvas exists).
3. Table 9 side-by-side (Zhang / ours) — from `docs/ALL_TABLES_EXACT_COMPARISON.md`.
4. Build-variance envelope plot.
5. Held-out accuracy/Brier table (296 accidents, all predictors).
6. Calibration reliability diagram (`docs/figures/confidence_calibration.png`).
7. Narrative-only rebuild: fire-cause scatter (coded vs LLM) showing the
   inversion.
8. LLM-direct vs BN order-of-magnitude error summary.

# Viability report — three paper goals

Generated from repo audit (Aug 2026). Canonical numbers; use while writing §5–§8.

---

## Goals at a glance

| Goal | Viable? | Headline evidence |
|------|---------|-------------------|
| **1. Narratives → downstream events/severity + probabilities** | **Yes, partial** | 90.9% injury / 77.4% damage (n=296, bn-sev); Table 9 demos when parser hits |
| **2. Narratives → upstream causes + probabilities** | **Yes, partial** | Retrieval 84.2% (n=253); BN event path 57.7% |
| **3. Compare to Zhang published numbers** | **Yes coded; demo-only for narrative** | Table 7 85/85; 77/93 posteriors; held-out ≠ Zhang comparison |

**Ship as:** narrative evidence bridge + frozen BN reasoning — **not** "we beat Zhang with text" or "BN adds predictive accuracy."

---

## What to claim vs not claim

### Accept (Jesse/Maha)

- Reproduce Zhang counting layer: 85/85 Table 7, 102 fires, P(fire) formula
- Narrative → evidence → frozen BN is auditable (`query_to_bn.py`)
- Held-out 90.9%/77.4% severity, 84.2% diagnosis (leak-safe)
- BN event path 57.7% as partial success + honest negative vs retrieval
- Fire experiment: BN AUC ~0.38–0.46; retrieval ~0.96–0.98
- Zhang scenario posteriors when parsing lands same evidence as hand-set queries

### Reject

- "BN improves prediction accuracy" (0/296 severity discordant vs retrieval)
- "Cross-node inference beats retrieval" (fire disproves)
- Finding-level diagnosis validated on held-out (only 4-category rollup)
- Calling retrieval output a "posterior"
- 93%/81% pre-leakage numbers
- LLM-primary (68.2%/51.7% — rejected path)

---

## Zhang comparison — two estimands (do not merge)

1. **Coded reproduction (1982–2006):** Table 7, Table 8/9 scoreboard, demo scenarios — compare to Zhang directly.
2. **Held-out narratives (2007–2019):** 296/253 eval — **new contribution**; not a Zhang baseline comparison.

---

## Forward vs backward (BN terms)

| Direction | Primary readout | Accuracy | Limitation |
|-----------|-----------------|----------|------------|
| **Forward / prognosis** | bn-sev (k-NN → virtual evidence → BN) | 90.9% / 77.4% | Signal from retrieval; BN lossless pass-through |
| **Forward / events only** | soft-priority | 89.9% / 55.4% | Damage weak; fire fails |
| **Backward / diagnosis** | retrieval (neighbor voting) | 84.2% | Not a BN posterior — weighted neighbor distribution |
| **Backward / diagnosis** | BN event path | 57.7% | Parser enters sparse/generic nodes |

---

## Not viable without (future work)

1. 2020–2024 one-shot confirmatory held-out (submission prerequisite per RESULTS §5.6)
2. Finding-level held-out diagnosis scoring (pipeline exists, not scored)
3. Rare-class recovery (3 fatal, 25 organizational — n too small)

---

## Paste-ready framing paragraph (§8 / abstract)

We reproduce Zhang and Mahadevan's data-built airline-accident Bayesian network from 1982–2006 coded records—matching all 85 Table 7 conditional cause distributions, 102 fire occurrences, and 77 of 93 published posterior quantities—and freeze it without retraining. At query time, NTSB investigation narratives are converted into evidence on the network's own variables through deterministic parsing, embedding-based neighbor voting, and k-nearest-neighbor severity likelihoods, then propagated through the frozen network for joint diagnosis and prognosis. On 296 held-out accidents (2007–2019) under a leak-safe protocol, this zero-parameter chain reaches 90.9% injury and 77.4% damage top-1 accuracy—statistically tying strong supervised text baselines while leading on damage Macro-F1—and 84.2% cause-category diagnosis via retrieval, with the BN event path reaching 57.7% as evidence that parsed facts propagate to cause nodes but not competitively with neighbor voting. The network adds no severity accuracy by construction (it losslessly mediates the retrieval signal) and fails held-out fire cross-inference (ROC AUC ~0.38–0.46 vs retrieval ~0.97), so our contribution is not predictive dominance over simpler methods but a validated, auditable probabilistic layer where narrative evidence, coded structure, and what-if queries compose coherently—demonstrated on Zhang's benchmark scenarios and measured honestly where the BN does and does not add value.

---

## Source files

- `CRIB_SHEET.md`, `RESULTS_SECTION.md`, `SECTION_DRAFTS_TIER1_TIER2.md`
- `outputs/heldout_significance.md`, `outputs/diagnosis_heldout_eval.md`
- `outputs/BN_COMPARISON_REPORT.md`, `outputs/fire_node_cross_inference.md`
- `apps-parity/README.md`, `tests/trace_narrative_vs_direct.py`

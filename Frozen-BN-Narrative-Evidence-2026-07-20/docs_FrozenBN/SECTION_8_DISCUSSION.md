# Section 8 — Discussion, limitations, and conclusion

**Time:** ~30–45 min (paste into Word; no figures required).

**Source of truth:** your status doc + `RESULTS_SECTION.md` §5.6.

**Banned in this section:** 93%/81%; “BN improves predictive accuracy”; “audit” for category mapping; bn-sev / retrieval-sev as jargon (say “severity predictor” / “neighbor voting” instead).

---

## 8. Discussion

We set out to reproduce Zhang and Mahadevan’s airline-accident Bayesian network from coded NTSB records and to test whether investigation narratives can drive that same network at query time without retraining. The results support a modular architecture: narratives supply evidence; the frozen, data-built network supplies auditable probabilistic reasoning.

On reproduction, the counting layer that anchors Zhang’s model matches exactly: 102 fire occurrences in 1982–2006, the prior formula P(fire) = 102/184,517,128, and all 85 rows of Table 7 within tolerance. Full-network posteriors show residual disagreement with 12 of 93 published quantities even after person-node and multi-state severity upgrades; tie-breaking randomness in parent selection explains most of this, and Zhang’s own released model file does not reproduce every published cell either. We therefore claim faithfulness to method and data-derived anchors, not bitwise identity of every posterior.

On held-out evaluation (296 accidents, 2007–2019, leak-safe protocol), the narrative pipeline reaches 90.9% top-1 injury accuracy and 77.4% on damage, large gains over the network prior (58.4% / 42.6%) and statistically indistinguishable from strong supervised text baselines on injury (TF-IDF logistic regression 92.2%, Holm-adjusted p = 0.87). The chain is numerically strongest on damage Macro-F1 (0.697). Diagnosis at four-category granularity is led by narrative retrieval (84.2% top-1); structured event evidence alone reaches 57.7%, above the frequency baseline but well below retrieval.

Several negative results are informative rather than failures. Severity accuracy from the full chain equals direct neighbor voting on all 296 accidents—the network mediates that signal rather than adding information to it, while still enabling joint conditioning and what-if queries. Fusing event evidence with neighbor severity in one inference collapses accuracy (38.5% / 41.9%), exposing double-counting when the same narrative is used twice. A held-out fire-node experiment shows parsed event evidence does not yield usable cross-node prediction of fire (network ROC AUC below 0.46; retrieval and keyword baselines near 0.94–0.98). Large language models work as a query interface on Zhang’s benchmark scenarios and as a usable—but uncalibrated—evidence extractor on real narratives; they fail as replacements for the coded database (causal attribution inverts on narrative-only rebuild) or for the network itself (order-of-magnitude probability errors when asked directly).

---

## 8.1 Limitations

We state the following constraints explicitly; each bounds what the evidence supports.

**Retrospective narratives.** Every narrative is written after the investigation closed. Defensible use cases are triage of an existing report, coding assistance, and what-if analysis—not real-time forecasting of an unfolding accident.

**Severity accuracy is carried by retrieval, not added by the network.** On held-out severity prediction, the full Bayesian-network readout and direct k-nearest-neighbor voting on injury and damage agree on all 296 accidents. The network faithfully propagates neighbor-derived severity evidence; its contribution is joint probabilistic reasoning at zero severity-accuracy cost. We do not claim predictive superiority over neighbor voting on severity.

**Cross-node inference does not beat retrieval on held-out fire.** In a designed experiment, event-only evidence produces network fire posteriors with ROC AUC roughly 0.38–0.46 on the coded fire label, while retrieval neighbor rates and a simple fire-keyword match reach roughly 0.94–0.98 on the same narratives. The surviving value of the network is auditable evidence composition and what-if semantics—not held-out predictive lift on unobserved nodes such as fire.

**Rare classes are not reliably recovered.** The held-out set contains 3 fatal-injury accidents, 16 minor-injury accidents (none ranked top-1 by any predictor), and 25 organizational-cause accidents (at most one recovered top-1). We report per-class recall and full confusion matrices rather than hiding these misses in headline accuracy.

**Embedding pretraining may include test-era text.** The embedding model’s training corpus is undisclosed and may contain 2007–2019 NTSB reports; redaction cannot reach inside a pretrained encoder. Comparisons remain internally fair because all embedding-based methods share the same channel; the TF-IDF text baseline (92.2% injury / 73.3% damage) provides a pretraining-free reference.

**Incomplete posterior parity with Zhang.** After upgrades, 48 of 93 published quantities match exactly and 29 closely; 12 still differ, attributable to stochastic parent selection and encoding differences. We claim faithfulness to Zhang’s method and counting-layer quantities (85/85 Table 7, 102 fires, prior formula), not cell-for-cell identity of every posterior.

**Diagnosis only at four-category granularity.** NTSB’s 2008 taxonomy change (legacy subject codes to CICTT) prevents exact code matching across our temporal split. Legacy and modern codes are rolled to CICTT’s four top-level categories (Personnel, Aircraft, Environment, Organizational) via a documented keyword rule set reviewed for internal consistency. Contested mappings touch at most roughly 2 accuracy points in the same direction for every predictor—a sensitivity bound, not independent validation by a second coder.

**Held-out reuse during development.** The 296-accident window was scored repeatedly while leakage fixes and ablations were designed. No model parameter was fitted to held-out data, but design decisions saw held-out results; reported accuracies may carry selection optimism. A one-shot confirmatory evaluation on a never-scored window (2020–2024) is planned before journal submission.

**LLM experiments.** Language-model tests use a single model family (gpt-4o-mini), a single aviation domain, and coded NTSB records as the reference for extraction quality—a conservative choice that favors the pipeline when wording aligns with official vocabulary. Reported extraction failures reflect miscalibration and over-extraction rather than lack of comprehension; out-of-sample paraphrase tests are the appropriate robustness check.

---

## 8.2 Conclusion

We reproduced and froze Zhang and Mahadevan’s airline-accident Bayesian network, extended it minimally so published benchmark queries run, and built a leak-safe pipeline that converts free-text NTSB narratives into evidence on the network’s own variables without retraining. On 296 held-out accidents, the chain achieves 90.9% injury and 77.4% damage top-1 accuracy under explicit outcome redaction, ties strong supervised text baselines on injury, and leads on damage Macro-F1; narrative retrieval remains the primary diagnosis readout at 84.2% four-category top-1 accuracy. Controlled experiments show that language models serve effectively as query interfaces and usable evidence extractors but cannot replace either the coded investigative record or the network as a probability engine. The architecture we advocate separates roles: narratives and language models supply evidence; the data-built network supplies auditable probabilistic reasoning. We measured every link in that chain—and both swap directions fail when tested honestly.

---

## Optional closing paragraph (reproducibility)

All tables and figures regenerate from the public repository (`REPRODUCE.md`). Narrative embeddings are cached by content hash; bootstrap and significance machinery use fixed seeds.

---

## Paste order (if you want to go step by step)

| Step | Paste | Time |
|------|-------|------|
| 1 | **8. Discussion** (opening synthesis — 4 paragraphs) | 10 min |
| 2 | **8.1 Limitations** (9 bullets → short paragraphs) | 15 min |
| 3 | **8.2 Conclusion** (1 paragraph) | 5 min |
| 4 | Optional reproducibility sentence | 2 min |
| 5 | Read aloud; trim if over ~3 pages | 10 min |

Send a screenshot when done for a grade.

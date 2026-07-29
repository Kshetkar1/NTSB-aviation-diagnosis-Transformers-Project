# Fire-node cross-inference (held-out 2007-2019)

## Verdict

* **`coded-field`**: best BN cross-inference arm reaches ROC AUC 0.392; the retrieval baseline on the same text reaches 0.975; a single fire-word regex reaches 0.940. **The BN arm does NOT beat the retrieval baseline, and does not beat chance (0.500) either.** Its 95% CI upper bound is 0.521, so the data are inconsistent with the BN carrying a useful fire signal, in either direction of effect.
* **`occurrence`**: best BN cross-inference arm reaches ROC AUC 0.457; the retrieval baseline on the same text reaches 0.981; a single fire-word regex reaches 0.933. **The BN arm does NOT beat the retrieval baseline, and does not beat chance (0.500) either.** Its 95% CI upper bound is 0.588, so the data are inconsistent with the BN carrying a useful fire signal, in either direction of effect.

Reading this honestly: on this cohort the frozen BN's narrative-event evidence carries NO usable information about the unobserved fire node. The claim "the frozen BN performs cross-node inference that retrieval alone cannot" is NOT supported by this experiment. The mechanism is visible in the diagnostics: `fire` has only 13 ancestors out of 783 nodes, 6 of them barred by the leak guard, and the only enterable one the parser ever hits is the generic `person: flightcrew`. Narrative event evidence is almost never d-connected to `fire` in a fire-specific way, so the posterior barely moves and what movement there is is not informative. Meanwhile the coded fire label is nearly fully recoverable from the words of the narrative alone, which is why neighbour voting and a one-line regex both do well. Any surviving version of the joint-reasoning claim has to be narrowed to what was actually demonstrated (coherent what-if/composition semantics on a frozen model) and must stop implying held-out predictive lift on unobserved nodes.

**Question.** Feed the frozen 1982-2006 network ONLY parsed narrative event evidence -- no severity evidence, no fire evidence -- and read the posterior on the `fire` occurrence node. Score it against the coded fire field. This is a CROSS-NODE query: the pipeline was never pointed at fire, and no arm except the last one in each table ever sees a fire label.

**Cohort.** The same n = 296 held-out accidents (2007-2019, `narr_accf` >= 100 chars) as the severity eval, verified identical to `outputs/cohort_manifest.json`. Text treatment is identical: `redact_severity_phrases(narr[:4000])` before any parsing or embedding.

**Leak guards.** (1) every node whose name matches the fire lexicon (`fire|smoke|explos|burn|ignit|combust|extinguish|overheat|...`), 23 nodes in all, is barred from the entered evidence set, including the `fire` target itself; (2) the two severity nodes are removed from the parser's name space, so the collider `fire -> aircraft damage` is never opened; (3) the `masked` text variant additionally deletes the fire lexicon from the narrative before embedding and parsing, so retrieval cannot select neighbours on fire words either. Both variants are reported; `masked` is the stricter number.

**Frozen prior.** P(fire = Yes) with no evidence = 7.7229e-08. This is Zhang's PER-FLIGHT scale (102 fire accidents / 184,517,128 departures), not a per-accident probability, so every BN posterior here is ~1e-7 in absolute terms. Brier scores and absolute calibration for BN arms are therefore dominated by that scale mismatch and are reported only for completeness -- the meaningful BN metrics are the RANK-based ones (ROC AUC, average precision).

## Metric definitions

* **base rate** -- fraction of the scored accidents whose coded label is fire = yes. With a rare positive class, accuracy is meaningless (always-no already scores 1 - base rate), which is why no accuracy column appears below.
* **ROC AUC** -- probability that a randomly chosen fire accident gets a strictly higher score than a randomly chosen non-fire accident, with tied scores counted as half. 0.5 = no ranking information. Ties matter here: BN arms whose evidence cannot reach the fire node return exactly the prior, and every such accident is tied with every other.
* **Average precision (PR AUC)** -- area under the precision-recall curve, sum over thresholds of (R_k - R_{k-1}) * P_k, with tied scores grouped. Better suited to rare positives than ROC AUC; the no-skill reference equals the base rate.
* **Brier** -- mean squared error of the probability against the 0/1 label. Read the caveat above for BN arms.
* **95% CI** -- percentile bootstrap over accidents (4,000 resamples, seed 42); resamples containing a single class are discarded.

**`keyword-fire` is the reference that matters most.** It is one regex over the redacted narrative -- 1.0 if any fire word appears, 0.0 otherwise -- with no network, no retrieval and no training. Being binary, its ROC AUC is just (sensitivity + specificity) / 2, so it is not directly comparable to a ranked score; it is here to show how much of the fire signal is sitting in plain sight in the text. Any claim that the BN contributes cross-node reasoning has to clear this bar, not just the prior.

## Truth definition: `coded-field`

Fire = yes when the NTSB coded aircraft-fire field `acft_fire` is GRD (on ground), IFLT (in flight) or BOTH; no when it is NONE; UNKNOWN (excluded) otherwise. This field is populated for almost the whole cohort.

**Base rate: 29/295 = 9.8% fire = yes** (1 of 296 excluded as unknown).

| arm | ROC AUC | 95% CI | avg precision | Brier | needs fire labels? |
|---|---|---|---|---|---|
| prior | 0.500 | [0.500, 0.500] | 0.098 | 0.0983 | no |
| keyword-fire | 0.940 | [0.897, 0.970] | 0.533 | 0.0814 | no |
| bn-soft-priority[plain] | 0.381 | [0.256, 0.511] | 0.092 | 0.0978 | no |
| bn-all-events[plain] | 0.385 | [0.261, 0.513] | 0.093 | 0.0978 | no |
| bn-fire-family[plain] | 0.377 | [0.252, 0.509] | 0.095 | 0.0978 | no |
| retrieval[plain] | 0.975 | [0.948, 0.992] | 0.847 | 0.0450 | no |
| retrieval-w[plain] | 0.975 | [0.949, 0.993] | 0.848 | 0.0445 | no |
| bn-soft-priority[masked] | 0.374 | [0.249, 0.507] | 0.095 | 0.0978 | no |
| bn-all-events[masked] | 0.392 | [0.268, 0.521] | 0.094 | 0.0977 | no |
| bn-fire-family[masked] | 0.376 | [0.250, 0.508] | 0.095 | 0.0978 | no |
| retrieval[masked] | 0.956 | [0.919, 0.983] | 0.748 | 0.0561 | no |
| retrieval-w[masked] | 0.958 | [0.924, 0.983] | 0.752 | 0.0557 | no |
| tfidf-lr[plain] | 0.986 | [0.972, 0.996] | 0.905 | 0.0554 | YES |
| tfidf-lr[masked] | 0.961 | [0.927, 0.986] | 0.789 | 0.0644 | YES |

No-skill references: ROC AUC 0.500, average precision 0.098 (= base rate).

Mean score by true class (direction check -- a useful predictor scores fire accidents HIGHER):

| arm | mean score, fire = yes | mean score, fire = no |
|---|---|---|
| prior | 7.7229e-08 | 7.7229e-08 |
| keyword-fire | 9.6552e-01 | 8.6466e-02 |
| bn-soft-priority[plain] | 2.9169e-03 | 3.2732e-03 |
| bn-all-events[plain] | 2.9169e-03 | 3.3390e-03 |
| bn-fire-family[plain] | 2.9169e-03 | 3.2744e-03 |
| retrieval[plain] | 4.4196e-01 | 7.2603e-02 |
| retrieval-w[plain] | 4.4443e-01 | 7.1744e-02 |
| bn-soft-priority[masked] | 2.9169e-03 | 3.2757e-03 |
| bn-all-events[masked] | 2.9376e-03 | 3.3415e-03 |
| bn-fire-family[masked] | 2.9169e-03 | 3.2769e-03 |
| retrieval[masked] | 3.2929e-01 | 6.8471e-02 |
| retrieval-w[masked] | 3.3060e-01 | 6.7591e-02 |
| tfidf-lr[plain] | 3.2365e-01 | 9.2363e-02 |
| tfidf-lr[masked] | 2.6921e-01 | 1.0105e-01 |

### Head-to-head: BN cross-inference vs the retrieval baseline

Paired bootstrap over accidents (4,000 resamples, seed 43) on the AUC difference. A positive interval that excludes 0 would mean the BN ranks fire better than neighbour voting on the same retrieval pool.

| A | B | AUC(A) - AUC(B) | 95% CI | bootstrap p |
|---|---|---|---|---|
| bn-soft-priority[plain] | retrieval[plain] | -0.594 | [-0.723, -0.461] | 0.0000 |
| bn-all-events[plain] | retrieval[plain] | -0.590 | [-0.719, -0.456] | 0.0000 |
| bn-fire-family[plain] | retrieval[plain] | -0.598 | [-0.730, -0.465] | 0.0000 |
| bn-soft-priority[masked] | retrieval[masked] | -0.582 | [-0.717, -0.442] | 0.0000 |
| bn-all-events[masked] | retrieval[masked] | -0.564 | [-0.696, -0.425] | 0.0000 |
| bn-fire-family[masked] | retrieval[masked] | -0.580 | [-0.714, -0.440] | 0.0000 |

### Supplementary: subset analyses (both directions of the leak-guard artifact)

The guard strips fire evidence, and it strips it DISPROPORTIONATELY from fire accidents. Two consequences, isolated here so the headline number above is read correctly:

* **moved-only** -- accidents whose BN posterior actually left the prior. Accidents tied at the prior all share the lowest score, and fire accidents are over-represented among them (the guard removed the very evidence that would have moved them), which pushes the pooled AUC DOWN for a reason unrelated to BN reasoning.
* **guard-silent** -- accidents where the guard excluded nothing at all, so it leaves no footprint. The cleanest subset.

The retrieval competitor is rescored on the identical rows, so these are apples-to-apples.

| subset | arm | n | n fire | ROC AUC | avg precision |
|---|---|---|---|---|---|
| moved-only[plain] | bn-soft-priority[plain] | 246 | 15 | 0.626 | 0.087 |
| moved-only[plain] | bn-all-events[plain] | 246 | 15 | 0.644 | 0.088 |
| moved-only[plain] | bn-fire-family[plain] | 246 | 15 | 0.625 | 0.087 |
| moved-only[plain] | retrieval[plain] | 246 | 15 | 0.981 | 0.891 |
| guard-silent[plain] | -- | 253 | 0 | n/a (single class) | n/a |
| moved-only[masked] | bn-soft-priority[masked] | 245 | 13 | 0.720 | 0.091 |
| moved-only[masked] | bn-all-events[masked] | 245 | 13 | 0.720 | 0.091 |
| moved-only[masked] | bn-fire-family[masked] | 245 | 13 | 0.720 | 0.091 |
| moved-only[masked] | retrieval[masked] | 245 | 13 | 0.957 | 0.744 |
| guard-silent[masked] | bn-soft-priority[masked] | 276 | 14 | 0.459 | 0.055 |
| guard-silent[masked] | bn-all-events[masked] | 276 | 14 | 0.503 | 0.056 |
| guard-silent[masked] | bn-fire-family[masked] | 276 | 14 | 0.460 | 0.055 |
| guard-silent[masked] | retrieval[masked] | 276 | 14 | 0.930 | 0.477 |

### Calibration (quantile bins over the score)

Equal-width probability bins are useless for the BN arms (every posterior sits near the per-flight prior), so bins are score QUANTILES: within each bin, mean predicted probability vs observed fire frequency. A useful ranking shows `obs_freq` rising across bins even when `mean_pred` is off by orders of magnitude.

**bn-soft-priority[plain]**

| bin | n | score range | mean pred | observed fire freq | n fire |
|---|---|---|---|---|---|
| 1 | 59 | 7.723e-08 - 8.227e-08 | 7.740e-08 | 27.1% | 16 |
| 2 | 59 | 8.265e-08 - 1.966e-07 | 9.724e-08 | 0.0% | 0 |
| 3 | 59 | 1.966e-07 - 6.507e-03 | 3.177e-03 | 3.4% | 2 |
| 4 | 59 | 6.507e-03 - 6.507e-03 | 6.507e-03 | 6.8% | 4 |
| 5 | 59 | 6.507e-03 - 6.507e-03 | 6.507e-03 | 11.9% | 7 |

**bn-soft-priority[masked]**

| bin | n | score range | mean pred | observed fire freq | n fire |
|---|---|---|---|---|---|
| 1 | 59 | 7.723e-08 - 8.265e-08 | 7.745e-08 | 27.1% | 16 |
| 2 | 59 | 8.388e-08 - 1.966e-07 | 9.806e-08 | 0.0% | 0 |
| 3 | 59 | 1.966e-07 - 6.507e-03 | 3.188e-03 | 3.4% | 2 |
| 4 | 59 | 6.507e-03 - 6.507e-03 | 6.507e-03 | 6.8% | 4 |
| 5 | 59 | 6.507e-03 - 6.507e-03 | 6.507e-03 | 11.9% | 7 |

**retrieval[plain]**

| bin | n | score range | mean pred | observed fire freq | n fire |
|---|---|---|---|---|---|
| 1 | 59 | 0.000e+00 - 0.000e+00 | 0.000e+00 | 0.0% | 0 |
| 2 | 59 | 0.000e+00 - 3.000e-02 | 1.079e-02 | 0.0% | 0 |
| 3 | 59 | 3.000e-02 - 8.081e-02 | 5.543e-02 | 0.0% | 0 |
| 4 | 59 | 8.602e-02 - 1.684e-01 | 1.158e-01 | 1.7% | 1 |
| 5 | 59 | 1.700e-01 - 6.289e-01 | 3.625e-01 | 47.5% | 28 |

**retrieval[masked]**

| bin | n | score range | mean pred | observed fire freq | n fire |
|---|---|---|---|---|---|
| 1 | 59 | 0.000e+00 - 0.000e+00 | 0.000e+00 | 0.0% | 0 |
| 2 | 59 | 0.000e+00 - 3.000e-02 | 1.062e-02 | 0.0% | 0 |
| 3 | 59 | 3.000e-02 - 8.000e-02 | 5.353e-02 | 1.7% | 1 |
| 4 | 59 | 8.000e-02 - 1.600e-01 | 1.119e-01 | 1.7% | 1 |
| 5 | 59 | 1.616e-01 - 5.052e-01 | 2.945e-01 | 45.8% | 27 |

## Truth definition: `occurrence`

Fire = yes when any occurrence in `sequence_of_events` has a description containing the token "fire". This is Zhang's own field logic: his 102 fire accidents are the 1982-2006 rows whose `Occurrence_Description` is exactly "Fire" (the family including "Fire/explosion" is 113). The 2008+ CICTT taxonomy has no bare "Fire" label, only "<phase> - Fire/smoke (non-impact)/(post-impact)", so the token test is the era-parallel form. Accidents with an EMPTY `sequence_of_events` are scored UNKNOWN and excluded rather than silently labelled fire = no.

**Base rate: 25/257 = 9.7% fire = yes** (39 of 296 excluded as unknown).

| arm | ROC AUC | 95% CI | avg precision | Brier | needs fire labels? |
|---|---|---|---|---|---|
| prior | 0.500 | [0.500, 0.500] | 0.097 | 0.0973 | no |
| keyword-fire | 0.933 | [0.882, 0.967] | 0.505 | 0.0895 | no |
| bn-soft-priority[plain] | 0.431 | [0.296, 0.564] | 0.097 | 0.0966 | no |
| bn-all-events[plain] | 0.436 | [0.301, 0.570] | 0.098 | 0.0966 | no |
| bn-fire-family[plain] | 0.424 | [0.287, 0.560] | 0.100 | 0.0966 | no |
| retrieval[plain] | 0.981 | [0.961, 0.996] | 0.890 | 0.0451 | no |
| retrieval-w[plain] | 0.981 | [0.961, 0.996] | 0.890 | 0.0449 | no |
| bn-soft-priority[masked] | 0.423 | [0.285, 0.560] | 0.100 | 0.0966 | no |
| bn-all-events[masked] | 0.457 | [0.327, 0.588] | 0.099 | 0.0966 | no |
| bn-fire-family[masked] | 0.423 | [0.286, 0.559] | 0.100 | 0.0966 | no |
| retrieval[masked] | 0.960 | [0.921, 0.987] | 0.786 | 0.0596 | no |
| retrieval-w[masked] | 0.961 | [0.926, 0.987] | 0.789 | 0.0595 | no |
| tfidf-lr[plain] | 0.989 | [0.977, 0.997] | 0.918 | 0.0658 | YES |
| tfidf-lr[masked] | 0.955 | [0.911, 0.985] | 0.790 | 0.0724 | YES |

No-skill references: ROC AUC 0.500, average precision 0.097 (= base rate).

Mean score by true class (direction check -- a useful predictor scores fire accidents HIGHER):

| arm | mean score, fire = yes | mean score, fire = no |
|---|---|---|
| prior | 7.7229e-08 | 7.7229e-08 |
| keyword-fire | 9.6000e-01 | 9.4828e-02 |
| bn-soft-priority[plain] | 3.3836e-03 | 3.3236e-03 |
| bn-all-events[plain] | 3.3836e-03 | 3.3721e-03 |
| bn-fire-family[plain] | 3.3836e-03 | 3.3249e-03 |
| retrieval[plain] | 3.6779e-01 | 3.3617e-02 |
| retrieval-w[plain] | 3.6944e-01 | 3.3172e-02 |
| bn-soft-priority[masked] | 3.3836e-03 | 3.3264e-03 |
| bn-all-events[masked] | 3.4077e-03 | 3.3749e-03 |
| bn-fire-family[masked] | 3.3836e-03 | 3.3278e-03 |
| retrieval[masked] | 2.4763e-01 | 2.9949e-02 |
| retrieval-w[masked] | 2.4817e-01 | 2.9506e-02 |
| tfidf-lr[plain] | 2.0048e-01 | 4.9756e-02 |
| tfidf-lr[masked] | 1.6022e-01 | 5.6156e-02 |

### Head-to-head: BN cross-inference vs the retrieval baseline

Paired bootstrap over accidents (4,000 resamples, seed 43) on the AUC difference. A positive interval that excludes 0 would mean the BN ranks fire better than neighbour voting on the same retrieval pool.

| A | B | AUC(A) - AUC(B) | 95% CI | bootstrap p |
|---|---|---|---|---|
| bn-soft-priority[plain] | retrieval[plain] | -0.550 | [-0.692, -0.407] | 0.0000 |
| bn-all-events[plain] | retrieval[plain] | -0.545 | [-0.685, -0.401] | 0.0000 |
| bn-fire-family[plain] | retrieval[plain] | -0.557 | [-0.701, -0.410] | 0.0000 |
| bn-soft-priority[masked] | retrieval[masked] | -0.537 | [-0.688, -0.381] | 0.0000 |
| bn-all-events[masked] | retrieval[masked] | -0.503 | [-0.647, -0.355] | 0.0000 |
| bn-fire-family[masked] | retrieval[masked] | -0.537 | [-0.688, -0.381] | 0.0000 |

### Supplementary: subset analyses (both directions of the leak-guard artifact)

The guard strips fire evidence, and it strips it DISPROPORTIONATELY from fire accidents. Two consequences, isolated here so the headline number above is read correctly:

* **moved-only** -- accidents whose BN posterior actually left the prior. Accidents tied at the prior all share the lowest score, and fire accidents are over-represented among them (the guard removed the very evidence that would have moved them), which pushes the pooled AUC DOWN for a reason unrelated to BN reasoning.
* **guard-silent** -- accidents where the guard excluded nothing at all, so it leaves no footprint. The cleanest subset.

The retrieval competitor is rescored on the identical rows, so these are apples-to-apples.

| subset | arm | n | n fire | ROC AUC | avg precision |
|---|---|---|---|---|---|
| moved-only[plain] | bn-soft-priority[plain] | 215 | 15 | 0.620 | 0.097 |
| moved-only[plain] | bn-all-events[plain] | 215 | 15 | 0.637 | 0.099 |
| moved-only[plain] | bn-fire-family[plain] | 215 | 15 | 0.619 | 0.097 |
| moved-only[plain] | retrieval[plain] | 215 | 15 | 0.983 | 0.862 |
| guard-silent[plain] | bn-soft-priority[plain] | 218 | 1 | 0.058 | 0.005 |
| guard-silent[plain] | bn-all-events[plain] | 218 | 1 | 0.051 | 0.005 |
| guard-silent[plain] | bn-fire-family[plain] | 218 | 1 | 0.058 | 0.005 |
| guard-silent[plain] | retrieval[plain] | 218 | 1 | 0.991 | 0.333 |
| moved-only[masked] | bn-soft-priority[masked] | 214 | 13 | 0.714 | 0.102 |
| moved-only[masked] | bn-all-events[masked] | 214 | 13 | 0.714 | 0.102 |
| moved-only[masked] | bn-fire-family[masked] | 214 | 13 | 0.714 | 0.102 |
| moved-only[masked] | retrieval[masked] | 214 | 13 | 0.957 | 0.729 |
| guard-silent[masked] | bn-soft-priority[masked] | 240 | 12 | 0.521 | 0.060 |
| guard-silent[masked] | bn-all-events[masked] | 240 | 12 | 0.573 | 0.063 |
| guard-silent[masked] | bn-fire-family[masked] | 240 | 12 | 0.521 | 0.060 |
| guard-silent[masked] | retrieval[masked] | 240 | 12 | 0.933 | 0.445 |

### Calibration (quantile bins over the score)

Equal-width probability bins are useless for the BN arms (every posterior sits near the per-flight prior), so bins are score QUANTILES: within each bin, mean predicted probability vs observed fire frequency. A useful ranking shows `obs_freq` rising across bins even when `mean_pred` is off by orders of magnitude.

**bn-soft-priority[plain]**

| bin | n | score range | mean pred | observed fire freq | n fire |
|---|---|---|---|---|---|
| 1 | 52 | 7.723e-08 - 8.388e-08 | 7.764e-08 | 23.1% | 12 |
| 2 | 52 | 8.511e-08 - 1.966e-07 | 1.064e-07 | 0.0% | 0 |
| 3 | 51 | 1.966e-07 - 6.507e-03 | 3.764e-03 | 5.9% | 3 |
| 4 | 51 | 6.507e-03 - 6.507e-03 | 6.507e-03 | 3.9% | 2 |
| 5 | 51 | 6.507e-03 - 6.507e-03 | 6.507e-03 | 15.7% | 8 |

**bn-soft-priority[masked]**

| bin | n | score range | mean pred | observed fire freq | n fire |
|---|---|---|---|---|---|
| 1 | 52 | 7.723e-08 - 8.511e-08 | 7.775e-08 | 23.1% | 12 |
| 2 | 52 | 8.511e-08 - 1.966e-07 | 1.073e-07 | 0.0% | 0 |
| 3 | 51 | 1.966e-07 - 6.507e-03 | 3.776e-03 | 5.9% | 3 |
| 4 | 51 | 6.507e-03 - 6.507e-03 | 6.507e-03 | 3.9% | 2 |
| 5 | 51 | 6.507e-03 - 6.507e-03 | 6.507e-03 | 15.7% | 8 |

**retrieval[plain]**

| bin | n | score range | mean pred | observed fire freq | n fire |
|---|---|---|---|---|---|
| 1 | 52 | 0.000e+00 - 0.000e+00 | 0.000e+00 | 0.0% | 0 |
| 2 | 52 | 0.000e+00 - 0.000e+00 | 0.000e+00 | 0.0% | 0 |
| 3 | 51 | 0.000e+00 - 2.439e-02 | 1.294e-02 | 0.0% | 0 |
| 4 | 51 | 2.500e-02 - 8.434e-02 | 4.762e-02 | 2.0% | 1 |
| 5 | 51 | 8.451e-02 - 5.417e-01 | 2.727e-01 | 47.1% | 24 |

**retrieval[masked]**

| bin | n | score range | mean pred | observed fire freq | n fire |
|---|---|---|---|---|---|
| 1 | 52 | 0.000e+00 - 0.000e+00 | 0.000e+00 | 0.0% | 0 |
| 2 | 52 | 0.000e+00 - 0.000e+00 | 0.000e+00 | 0.0% | 0 |
| 3 | 51 | 0.000e+00 - 2.174e-02 | 1.242e-02 | 2.0% | 1 |
| 4 | 51 | 2.174e-02 - 7.000e-02 | 4.283e-02 | 2.0% | 1 |
| 5 | 51 | 7.000e-02 - 4.271e-01 | 2.024e-01 | 45.1% | 23 |

## Diagnostics

* Fire-lexicon nodes barred from evidence: 23 of 783 candidate nodes.
* `fire` has 13 ancestors in the frozen DAG (6 of them barred by the fire lexicon, 7 still enterable: ['airport facilities, refueling truck/equipment', 'brakes, anti-skid', 'communication/navigation equipment, antenna', 'electrical system, auxiliary power unit (apu)', 'exhaust system, stack', 'fuel system, primer system', 'person: flightcrew']) and only the two severity nodes as children. Severity is never observed, so that collider stays blocked; the posterior can still move through any OTHER open path (typically a shared ancestor of the evidence node and one of `fire`'s parents), which is why the empirical 'posterior moved' counts below are the honest measure of how often the evidence reaches `fire` at all.
* Leak guard [plain]: 43/296 accidents had >= 1 fire-lexicon evidence node excluded (44 node-instances).
* Leak guard [masked]: 19/296 accidents had >= 1 fire-lexicon evidence node excluded (19 node-instances).
* Which enterable ancestors of `fire` actually received evidence: {'person: flightcrew': 143}. If that list is a single generic node, the posterior movement is not fire-specific reasoning.
* Evidence landed on one of `fire`'s 7 enterable ancestors in only 143/296 accidents [plain]. Everywhere else the posterior can only move through remote correlations (a shared ancestor of the evidence node and one of `fire`'s parents), which is weak and not fire-specific.
* Evidence landed on one of `fire`'s 7 enterable ancestors in only 143/296 accidents [masked]. Everywhere else the posterior can only move through remote correlations (a shared ancestor of the evidence node and one of `fire`'s parents), which is weak and not fire-specific.
* `bn-all-events[masked]`: posterior differed from the prior for 253/296 accidents; the rest are exactly tied at the prior.
* `bn-all-events[plain]`: posterior differed from the prior for 251/296 accidents; the rest are exactly tied at the prior.
* `bn-fire-family[masked]`: posterior differed from the prior for 244/296 accidents; the rest are exactly tied at the prior.
* `bn-fire-family[plain]`: posterior differed from the prior for 243/296 accidents; the rest are exactly tied at the prior.
* `bn-soft-priority[masked]`: posterior differed from the prior for 246/296 accidents; the rest are exactly tied at the prior.
* `bn-soft-priority[plain]`: posterior differed from the prior for 247/296 accidents; the rest are exactly tied at the prior.


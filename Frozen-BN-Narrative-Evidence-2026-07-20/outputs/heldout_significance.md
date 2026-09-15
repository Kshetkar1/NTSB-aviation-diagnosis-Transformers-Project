# Held-out significance report

n = 296 held-out accidents (2007-2019). Leak-safe eval (outcome phrases stripped before retrieval). 95% CIs: bootstrap, 10,000 resamples, seed 42. Pairwise: McNemar exact test on top-1 correctness.

**Metric definitions.** Accuracy = fraction of accidents whose top-1 predicted class (argmax of the 4-state distribution) equals the NTSB-coded class. Macro-F1 = unweighted mean of per-class F1 (robust to class imbalance). Brier = squared error of the full distribution (lower better).

## Injury accuracy with 95% CI

Majority-class baseline: always predict NONE = 58.4% accuracy (any useful model must beat this AND have higher Macro-F1).

| Predictor | Accuracy | 95% CI | Macro-F1 | n |
|---|---|---|---|---|
| prior | 58.4% | [52.7%, 63.9%] | 0.184 | 296 |
| soft-only | 89.2% | [85.5%, 92.6%] | 0.461 | 296 |
| hard | 63.9% | [58.4%, 69.3%] | 0.263 | 296 |
| hard+soft | 82.1% | [77.7%, 86.5%] | 0.420 | 296 |
| soft-priority | 89.9% | [86.5%, 93.2%] | 0.465 | 296 |
| retrieval-sev | 90.9% | [87.5%, 93.9%] | 0.470 | 296 |
| bn-sev | 90.9% | [87.5%, 93.9%] | 0.470 | 296 |
| bn-fused | 90.9% | [87.5%, 93.9%] | 0.470 | 296 |
| bn-fused-t | 90.9% | [87.5%, 93.9%] | 0.470 | 296 |
| narrative-evidence | 90.9% | [87.5%, 93.9%] | 0.470 | 296 |
| lr | 85.5% | [81.4%, 89.2%] | 0.440 | 296 |
| emb-lr | 91.6% | [88.2%, 94.6%] | 0.474 | 296 |
| tfidf-lr | 92.2% | [89.2%, 95.3%] | 0.603 | 296 |

## Damage accuracy with 95% CI

Majority-class baseline: always predict NONE = 42.6% accuracy (any useful model must beat this AND have higher Macro-F1).

| Predictor | Accuracy | 95% CI | Macro-F1 | n |
|---|---|---|---|---|
| prior | 42.6% | [36.8%, 48.3%] | 0.149 | 296 |
| soft-only | 55.1% | [49.3%, 60.5%] | 0.324 | 296 |
| hard | 44.6% | [38.9%, 50.3%] | 0.214 | 296 |
| hard+soft | 50.7% | [44.9%, 56.4%] | 0.309 | 296 |
| soft-priority | 55.4% | [49.7%, 60.8%] | 0.342 | 296 |
| retrieval-sev | 77.4% | [72.3%, 82.1%] | 0.697 | 296 |
| bn-sev | 77.4% | [72.6%, 82.1%] | 0.697 | 296 |
| bn-fused | 77.4% | [72.6%, 82.1%] | 0.697 | 296 |
| bn-fused-t | 77.4% | [72.6%, 82.1%] | 0.697 | 296 |
| narrative-evidence | 77.4% | [72.6%, 81.8%] | 0.697 | 296 |
| lr | 60.1% | [54.4%, 65.5%] | 0.408 | 296 |
| emb-lr | 74.0% | [68.9%, 79.1%] | 0.543 | 296 |
| tfidf-lr | 73.3% | [68.2%, 78.0%] | 0.621 | 296 |

## Injury: per-class recall

| Predictor | FATL recall | SERS recall | MINR recall | NONE recall |
|---|---|---|---|---|
| bn-sev | 0/3 | 99/104 | 0/16 | 170/173 |
| retrieval-sev | 0/3 | 99/104 | 0/16 | 170/173 |
| emb-lr | 0/3 | 100/104 | 0/16 | 171/173 |
| tfidf-lr | 1/3 | 100/104 | 0/16 | 172/173 |
| lr | 0/3 | 89/104 | 0/16 | 164/173 |
| soft-priority | 0/3 | 96/104 | 0/16 | 170/173 |
| prior | 0/3 | 0/104 | 0/16 | 173/173 |

## Damage: per-class recall

| Predictor | DEST recall | SUBS recall | MINR recall | NONE recall |
|---|---|---|---|---|
| bn-sev | 1/3 | 59/78 | 61/89 | 108/126 |
| retrieval-sev | 1/3 | 59/78 | 61/89 | 108/126 |
| emb-lr | 0/3 | 53/78 | 56/89 | 110/126 |
| tfidf-lr | 1/3 | 44/78 | 62/89 | 110/126 |
| lr | 0/3 | 19/78 | 52/89 | 107/126 |
| soft-priority | 0/3 | 35/78 | 8/89 | 121/126 |
| prior | 0/3 | 0/78 | 0/89 | 126/126 |

### Injury confusion matrix: bn-sev (rows = truth, cols = predicted)

| truth \ pred | FATL | SERS | MINR | NONE |
|---|---|---|---|---|
| **FATL** | 0 | 1 | 0 | 2 |
| **SERS** | 0 | 99 | 0 | 5 |
| **MINR** | 0 | 3 | 0 | 13 |
| **NONE** | 0 | 3 | 0 | 170 |

### Injury confusion matrix: retrieval-sev (rows = truth, cols = predicted)

| truth \ pred | FATL | SERS | MINR | NONE |
|---|---|---|---|---|
| **FATL** | 0 | 1 | 0 | 2 |
| **SERS** | 0 | 99 | 0 | 5 |
| **MINR** | 0 | 3 | 0 | 13 |
| **NONE** | 0 | 3 | 0 | 170 |

### Injury confusion matrix: emb-lr (rows = truth, cols = predicted)

| truth \ pred | FATL | SERS | MINR | NONE |
|---|---|---|---|---|
| **FATL** | 0 | 1 | 0 | 2 |
| **SERS** | 0 | 100 | 0 | 4 |
| **MINR** | 0 | 2 | 0 | 14 |
| **NONE** | 0 | 2 | 0 | 171 |

### Damage confusion matrix: bn-sev (rows = truth, cols = predicted)

| truth \ pred | DEST | SUBS | MINR | NONE |
|---|---|---|---|---|
| **DEST** | 1 | 1 | 1 | 0 |
| **SUBS** | 0 | 59 | 14 | 5 |
| **MINR** | 0 | 19 | 61 | 9 |
| **NONE** | 0 | 3 | 15 | 108 |

### Damage confusion matrix: retrieval-sev (rows = truth, cols = predicted)

| truth \ pred | DEST | SUBS | MINR | NONE |
|---|---|---|---|---|
| **DEST** | 1 | 1 | 1 | 0 |
| **SUBS** | 0 | 59 | 14 | 5 |
| **MINR** | 0 | 19 | 61 | 9 |
| **NONE** | 0 | 3 | 15 | 108 |

### Damage confusion matrix: emb-lr (rows = truth, cols = predicted)

| truth \ pred | DEST | SUBS | MINR | NONE |
|---|---|---|---|---|
| **DEST** | 0 | 2 | 0 | 1 |
| **SUBS** | 0 | 53 | 16 | 9 |
| **MINR** | 0 | 23 | 56 | 10 |
| **NONE** | 0 | 2 | 14 | 110 |

## Binary severe-outcome screen

Severe injury = FATL or SERS; severe damage = DEST or SUBS. Sensitivity = severe accidents flagged severe; specificity = non-severe accidents not flagged. This is the operating view for triage use.

| Predictor | Target | Sensitivity | Specificity | n severe / n |
|---|---|---|---|---|
| bn-sev | injury | 93.5% (100/107) | 96.8% | 107 / 296 |
| retrieval-sev | injury | 93.5% (100/107) | 96.8% | 107 / 296 |
| emb-lr | injury | 94.4% (101/107) | 97.9% | 107 / 296 |
| tfidf-lr | injury | 94.4% (101/107) | 98.9% | 107 / 296 |
| lr | injury | 84.1% (90/107) | 94.2% | 107 / 296 |
| soft-priority | injury | 90.7% (97/107) | 97.4% | 107 / 296 |
| prior | injury | 0.0% (0/107) | 100.0% | 107 / 296 |
| bn-sev | damage | 75.3% (61/81) | 89.8% | 81 / 296 |
| retrieval-sev | damage | 75.3% (61/81) | 89.8% | 81 / 296 |
| emb-lr | damage | 67.9% (55/81) | 88.4% | 81 / 296 |
| tfidf-lr | damage | 58.0% (47/81) | 93.0% | 81 / 296 |
| lr | damage | 23.5% (19/81) | 94.4% | 81 / 296 |
| soft-priority | damage | 43.2% (35/81) | 91.2% | 81 / 296 |
| prior | damage | 0.0% (0/81) | 100.0% | 81 / 296 |

## Multiplicity policy

Five comparisons per target are designated PRIMARY (confirmatory): bn-sev vs prior, bn-sev vs lr, bn-sev vs emb-lr, bn-sev vs tfidf-lr, and bn-sev vs retrieval-sev. Holm-Bonferroni correction is applied within each target's primary family (m = 5). All other rows are EXPLORATORY ablations; their raw p-values are shown without correction and should not be read as confirmatory tests.

## Injury: paired comparisons (McNemar exact + Brier delta)

| A vs B | Role | A only right | B only right | McNemar p | p (Holm) | Brier delta (A-B) 95% CI | n |
|---|---|---|---|---|---|---|---|
| bn-sev vs prior | primary | 99 | 3 | 0.0000 | 0.0000 * | [-0.767, -0.546] | 296 |
| bn-sev vs lr | primary | 20 | 4 | 0.0015 | 0.0062 * | [-0.079, -0.020] | 296 |
| bn-sev vs emb-lr | primary | 0 | 2 | 0.5000 | 1.0000 | [+0.000, +0.021] | 296 |
| bn-sev vs tfidf-lr | primary | 2 | 6 | 0.2891 | 0.8672 | [-0.036, -0.003] | 296 |
| bn-sev vs retrieval-sev | primary | 0 | 0 | 1.0000 | 1.0000 | [-0.000, +0.000] | 296 |
| tfidf-lr vs emb-lr | exploratory | 5 | 3 | 0.7266 | -- | [+0.016, +0.044] | 296 |
| tfidf-lr vs lr | exploratory | 21 | 1 | 0.0000 | -- | [-0.059, -0.001] | 296 |
| bn-sev vs bn-fused | exploratory | 0 | 0 | 1.0000 | -- | [-0.000, +0.000] | 296 |
| bn-sev vs hard+soft | exploratory | 27 | 1 | 0.0000 | -- | [-0.224, -0.134] | 296 |
| bn-sev vs soft-priority | exploratory | 4 | 1 | 0.3750 | -- | [-0.104, -0.057] | 296 |
| retrieval-sev vs lr | exploratory | 20 | 4 | 0.0015 | -- | [-0.079, -0.020] | 296 |
| retrieval-sev vs emb-lr | exploratory | 0 | 2 | 0.5000 | -- | [+0.000, +0.021] | 296 |
| emb-lr vs lr | exploratory | 21 | 3 | 0.0003 | -- | [-0.088, -0.032] | 296 |
| narrative-evidence vs prior | exploratory | 99 | 3 | 0.0000 | -- | [-0.768, -0.546] | 296 |
| narrative-evidence vs hard+soft | exploratory | 27 | 1 | 0.0000 | -- | [-0.225, -0.134] | 296 |
| narrative-evidence vs soft-only | exploratory | 6 | 1 | 0.1250 | -- | [-0.070, -0.009] | 296 |
| narrative-evidence vs lr | exploratory | 20 | 4 | 0.0015 | -- | [-0.078, -0.019] | 296 |
| soft-only vs lr | exploratory | 19 | 8 | 0.0522 | -- | [-0.048, +0.030] | 296 |
| hard+soft vs prior | exploratory | 75 | 5 | 0.0000 | -- | [-0.578, -0.378] | 296 |
| hard+soft vs soft-only | exploratory | 3 | 24 | 0.0000 | -- | [+0.092, +0.187] | 296 |
| hard+soft vs lr | exploratory | 18 | 28 | 0.1839 | -- | [+0.079, +0.183] | 296 |

## Damage: paired comparisons (McNemar exact + Brier delta)

| A vs B | Role | A only right | B only right | McNemar p | p (Holm) | Brier delta (A-B) 95% CI | n |
|---|---|---|---|---|---|---|---|
| bn-sev vs prior | primary | 121 | 18 | 0.0000 | 0.0000 * | [-0.872, -0.674] | 296 |
| bn-sev vs lr | primary | 68 | 17 | 0.0000 | 0.0000 * | [-0.136, -0.073] | 296 |
| bn-sev vs emb-lr | primary | 21 | 11 | 0.1102 | 0.2884 | [-0.013, +0.020] | 296 |
| bn-sev vs tfidf-lr | primary | 28 | 16 | 0.0961 | 0.2884 | [-0.030, +0.001] | 296 |
| bn-sev vs retrieval-sev | primary | 0 | 0 | 1.0000 | 1.0000 | [-0.000, +0.000] | 296 |
| tfidf-lr vs emb-lr | exploratory | 19 | 21 | 0.8746 | -- | [+0.001, +0.037] | 296 |
| tfidf-lr vs lr | exploratory | 56 | 17 | 0.0000 | -- | [-0.121, -0.059] | 296 |
| bn-sev vs bn-fused | exploratory | 0 | 0 | 1.0000 | -- | [-0.000, +0.000] | 296 |
| bn-sev vs hard+soft | exploratory | 96 | 17 | 0.0000 | -- | [-0.337, -0.216] | 296 |
| bn-sev vs soft-priority | exploratory | 83 | 18 | 0.0000 | -- | [-0.236, -0.144] | 296 |
| retrieval-sev vs lr | exploratory | 68 | 17 | 0.0000 | -- | [-0.136, -0.074] | 296 |
| retrieval-sev vs emb-lr | exploratory | 21 | 11 | 0.1102 | -- | [-0.013, +0.021] | 296 |
| emb-lr vs lr | exploratory | 59 | 18 | 0.0000 | -- | [-0.145, -0.072] | 296 |
| narrative-evidence vs prior | exploratory | 121 | 18 | 0.0000 | -- | [-0.875, -0.672] | 296 |
| narrative-evidence vs hard+soft | exploratory | 96 | 17 | 0.0000 | -- | [-0.335, -0.215] | 296 |
| narrative-evidence vs soft-only | exploratory | 88 | 22 | 0.0000 | -- | [-0.336, -0.205] | 296 |
| narrative-evidence vs lr | exploratory | 68 | 17 | 0.0000 | -- | [-0.136, -0.074] | 296 |
| soft-only vs lr | exploratory | 44 | 59 | 0.1674 | -- | [+0.099, +0.231] | 296 |
| hard+soft vs prior | exploratory | 33 | 9 | 0.0003 | -- | [-0.579, -0.415] | 296 |
| hard+soft vs soft-only | exploratory | 15 | 28 | 0.0660 | -- | [-0.054, +0.064] | 296 |
| hard+soft vs lr | exploratory | 33 | 61 | 0.0051 | -- | [+0.111, +0.231] | 296 |

`*` = significant at Holm-adjusted p < 0.05 (primary comparisons only). Negative Brier delta favors A (lower Brier is better). Exploratory rows: raw p shown for transparency, no correction, no significance claims.

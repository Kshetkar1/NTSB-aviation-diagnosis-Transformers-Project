# Held-out significance report

n = 296 held-out accidents (2007-2019). Leak-safe eval (outcome phrases stripped before retrieval). 95% CIs: bootstrap, 10,000 resamples, seed 42. Pairwise: McNemar exact test on top-1 correctness.

**Metric definitions.** Accuracy = fraction of accidents whose top-1 predicted class (argmax of the 4-state distribution) equals the NTSB-coded class. Macro-F1 = unweighted mean of per-class F1 (robust to class imbalance). Brier = squared error of the full distribution (lower better).

## Injury accuracy with 95% CI

Majority-class baseline: always predict NONE = 58.4% accuracy (any useful model must beat this AND have higher Macro-F1).

| Predictor | Accuracy | 95% CI | Macro-F1 | n |
|---|---|---|---|---|
| prior | 58.4% | [52.7%, 63.9%] | 0.184 | 296 |
| soft-only | 89.2% | [85.5%, 92.6%] | 0.461 | 296 |
| hard | 64.2% | [58.4%, 69.6%] | 0.267 | 296 |
| hard+soft | 82.4% | [78.0%, 86.8%] | 0.422 | 296 |
| soft-priority | 89.9% | [86.5%, 93.2%] | 0.465 | 296 |
| retrieval-sev | 90.9% | [87.5%, 93.9%] | 0.470 | 296 |
| bn-sev | 90.9% | [87.5%, 93.9%] | 0.470 | 296 |
| bn-fused | 38.5% | [33.1%, 43.9%] | 0.230 | 296 |
| bn-fused-t | 38.5% | [33.1%, 44.3%] | 0.229 | 296 |
| narrative-evidence | 90.9% | [87.5%, 93.9%] | 0.470 | 296 |
| lr | 85.5% | [81.4%, 89.2%] | 0.440 | 296 |
| emb-lr | 91.6% | [88.2%, 94.6%] | 0.474 | 296 |

## Damage accuracy with 95% CI

Majority-class baseline: always predict NONE = 42.6% accuracy (any useful model must beat this AND have higher Macro-F1).

| Predictor | Accuracy | 95% CI | Macro-F1 | n |
|---|---|---|---|---|
| prior | 42.6% | [36.8%, 48.3%] | 0.149 | 296 |
| soft-only | 55.1% | [49.3%, 60.8%] | 0.324 | 296 |
| hard | 44.6% | [38.9%, 50.3%] | 0.214 | 296 |
| hard+soft | 50.7% | [44.9%, 56.4%] | 0.309 | 296 |
| soft-priority | 55.4% | [49.7%, 60.8%] | 0.342 | 296 |
| retrieval-sev | 77.4% | [72.6%, 82.1%] | 0.697 | 296 |
| bn-sev | 77.4% | [72.3%, 82.1%] | 0.697 | 296 |
| bn-fused | 41.9% | [36.5%, 47.3%] | 0.367 | 296 |
| bn-fused-t | 41.9% | [36.1%, 47.6%] | 0.367 | 296 |
| narrative-evidence | 77.4% | [72.6%, 82.1%] | 0.697 | 296 |
| lr | 60.1% | [54.4%, 65.9%] | 0.408 | 296 |
| emb-lr | 74.0% | [68.6%, 79.1%] | 0.543 | 296 |

## Injury: per-class recall

| Predictor | FATL recall | SERS recall | MINR recall | NONE recall |
|---|---|---|---|---|
| bn-sev | 0/3 | 99/104 | 0/16 | 170/173 |
| retrieval-sev | 0/3 | 99/104 | 0/16 | 170/173 |
| emb-lr | 0/3 | 100/104 | 0/16 | 171/173 |
| lr | 0/3 | 89/104 | 0/16 | 164/173 |
| soft-priority | 0/3 | 96/104 | 0/16 | 170/173 |
| prior | 0/3 | 0/104 | 0/16 | 173/173 |

## Damage: per-class recall

| Predictor | DEST recall | SUBS recall | MINR recall | NONE recall |
|---|---|---|---|---|
| bn-sev | 1/3 | 59/78 | 61/89 | 108/126 |
| retrieval-sev | 1/3 | 59/78 | 61/89 | 108/126 |
| emb-lr | 0/3 | 53/78 | 56/89 | 110/126 |
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
| lr | injury | 84.1% (90/107) | 94.2% | 107 / 296 |
| soft-priority | injury | 90.7% (97/107) | 97.4% | 107 / 296 |
| prior | injury | 0.0% (0/107) | 100.0% | 107 / 296 |
| bn-sev | damage | 75.3% (61/81) | 89.8% | 81 / 296 |
| retrieval-sev | damage | 75.3% (61/81) | 89.8% | 81 / 296 |
| emb-lr | damage | 67.9% (55/81) | 88.4% | 81 / 296 |
| lr | damage | 23.5% (19/81) | 94.4% | 81 / 296 |
| soft-priority | damage | 43.2% (35/81) | 91.2% | 81 / 296 |
| prior | damage | 0.0% (0/81) | 100.0% | 81 / 296 |

## Multiplicity policy

Four comparisons per target are designated PRIMARY (confirmatory): bn-sev vs prior, bn-sev vs lr, bn-sev vs emb-lr, and bn-sev vs retrieval-sev. Holm-Bonferroni correction is applied within each target's primary family (m = 4). All other rows are EXPLORATORY ablations; their raw p-values are shown without correction and should not be read as confirmatory tests.

## Injury: paired comparisons (McNemar exact + Brier delta)

| A vs B | Role | A only right | B only right | McNemar p | p (Holm) | Brier delta (A-B) 95% CI | n |
|---|---|---|---|---|---|---|---|
| bn-sev vs prior | primary | 99 | 3 | 0.0000 | 0.0000 * | [-0.768, -0.543] | 296 |
| bn-sev vs lr | primary | 20 | 4 | 0.0015 | 0.0046 * | [-0.078, -0.020] | 296 |
| bn-sev vs emb-lr | primary | 0 | 2 | 0.5000 | 1.0000 | [+0.001, +0.021] | 296 |
| bn-sev vs retrieval-sev | primary | 0 | 0 | 1.0000 | 1.0000 | [-0.000, +0.000] | 296 |
| bn-sev vs bn-fused | exploratory | 169 | 14 | 0.0000 | -- | [-0.888, -0.698] | 296 |
| bn-sev vs hard+soft | exploratory | 26 | 1 | 0.0000 | -- | [-0.222, -0.133] | 296 |
| bn-sev vs soft-priority | exploratory | 4 | 1 | 0.3750 | -- | [-0.105, -0.058] | 296 |
| retrieval-sev vs lr | exploratory | 20 | 4 | 0.0015 | -- | [-0.078, -0.020] | 296 |
| retrieval-sev vs emb-lr | exploratory | 0 | 2 | 0.5000 | -- | [+0.000, +0.021] | 296 |
| emb-lr vs lr | exploratory | 21 | 3 | 0.0003 | -- | [-0.088, -0.032] | 296 |
| narrative-evidence vs prior | exploratory | 99 | 3 | 0.0000 | -- | [-0.767, -0.544] | 296 |
| narrative-evidence vs hard+soft | exploratory | 26 | 1 | 0.0000 | -- | [-0.223, -0.133] | 296 |
| narrative-evidence vs soft-only | exploratory | 6 | 1 | 0.1250 | -- | [-0.071, -0.009] | 296 |
| narrative-evidence vs lr | exploratory | 20 | 4 | 0.0015 | -- | [-0.079, -0.021] | 296 |
| soft-only vs lr | exploratory | 19 | 8 | 0.0522 | -- | [-0.049, +0.030] | 296 |
| hard+soft vs prior | exploratory | 76 | 5 | 0.0000 | -- | [-0.579, -0.382] | 296 |
| hard+soft vs soft-only | exploratory | 3 | 23 | 0.0001 | -- | [+0.092, +0.185] | 296 |
| hard+soft vs lr | exploratory | 18 | 27 | 0.2327 | -- | [+0.076, +0.181] | 296 |

## Damage: paired comparisons (McNemar exact + Brier delta)

| A vs B | Role | A only right | B only right | McNemar p | p (Holm) | Brier delta (A-B) 95% CI | n |
|---|---|---|---|---|---|---|---|
| bn-sev vs prior | primary | 121 | 18 | 0.0000 | 0.0000 * | [-0.872, -0.674] | 296 |
| bn-sev vs lr | primary | 68 | 17 | 0.0000 | 0.0000 * | [-0.137, -0.073] | 296 |
| bn-sev vs emb-lr | primary | 21 | 11 | 0.1102 | 0.2204 | [-0.012, +0.020] | 296 |
| bn-sev vs retrieval-sev | primary | 0 | 0 | 1.0000 | 1.0000 | [-0.000, +0.000] | 296 |
| bn-sev vs bn-fused | exploratory | 121 | 16 | 0.0000 | -- | [-0.695, -0.484] | 296 |
| bn-sev vs hard+soft | exploratory | 96 | 17 | 0.0000 | -- | [-0.334, -0.214] | 296 |
| bn-sev vs soft-priority | exploratory | 83 | 18 | 0.0000 | -- | [-0.237, -0.143] | 296 |
| retrieval-sev vs lr | exploratory | 68 | 17 | 0.0000 | -- | [-0.137, -0.073] | 296 |
| retrieval-sev vs emb-lr | exploratory | 21 | 11 | 0.1102 | -- | [-0.013, +0.021] | 296 |
| emb-lr vs lr | exploratory | 59 | 18 | 0.0000 | -- | [-0.145, -0.072] | 296 |
| narrative-evidence vs prior | exploratory | 121 | 18 | 0.0000 | -- | [-0.873, -0.672] | 296 |
| narrative-evidence vs hard+soft | exploratory | 96 | 17 | 0.0000 | -- | [-0.334, -0.214] | 296 |
| narrative-evidence vs soft-only | exploratory | 88 | 22 | 0.0000 | -- | [-0.332, -0.206] | 296 |
| narrative-evidence vs lr | exploratory | 68 | 17 | 0.0000 | -- | [-0.137, -0.073] | 296 |
| soft-only vs lr | exploratory | 44 | 59 | 0.1674 | -- | [+0.100, +0.232] | 296 |
| hard+soft vs prior | exploratory | 33 | 9 | 0.0003 | -- | [-0.582, -0.418] | 296 |
| hard+soft vs soft-only | exploratory | 15 | 28 | 0.0660 | -- | [-0.056, +0.063] | 296 |
| hard+soft vs lr | exploratory | 33 | 61 | 0.0051 | -- | [+0.111, +0.229] | 296 |

`*` = significant at Holm-adjusted p < 0.05 (primary comparisons only). Negative Brier delta favors A (lower Brier is better). Exploratory rows: raw p shown for transparency, no correction, no significance claims.

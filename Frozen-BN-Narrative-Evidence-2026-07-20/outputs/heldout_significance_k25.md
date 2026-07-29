# Held-out significance report

n = 296 held-out accidents (2007-2019). Leak-safe eval (outcome phrases stripped before retrieval). 95% CIs: bootstrap, 10,000 resamples, seed 42. Pairwise: McNemar exact test on top-1 correctness.

## Injury accuracy with 95% CI

| Predictor | Accuracy | Macro-F1 | 95% CI (acc) | n |
|---|---|---|---|---|
| majority class (NONE) | 58.4% | 0.184 | [52.7%, 63.9%] | 296 |
| prior | 58.4% | 0.184 | [52.7%, 64.2%] | 296 |
| soft-only | 89.2% | 0.461 | [85.5%, 92.6%] | 296 |
| hard | 63.9% | 0.263 | [58.4%, 69.3%] | 296 |
| hard+soft | 82.1% | 0.420 | [77.7%, 86.5%] | 296 |
| soft-priority | 89.9% | 0.465 | [86.5%, 93.2%] | 296 |
| retrieval-sev | 89.9% | 0.466 | [86.1%, 93.2%] | 296 |
| narrative-evidence | 89.9% | 0.466 | [86.1%, 93.2%] | 296 |
| bn-sev | 89.9% | 0.466 | [86.1%, 93.2%] | 296 |
| bn-fused | 38.9% | 0.237 | [33.1%, 44.3%] | 296 |
| bn-fused-t | 38.2% | 0.221 | [32.8%, 43.9%] | 296 |
| lr | 87.8% | 0.454 | [84.1%, 91.6%] | 296 |
| emb-lr | 91.6% | 0.474 | [88.2%, 94.6%] | 296 |

## Damage accuracy with 95% CI

| Predictor | Accuracy | Macro-F1 | 95% CI (acc) | n |
|---|---|---|---|---|
| majority class (NONE) | 42.6% | 0.149 | [36.8%, 48.3%] | 296 |
| prior | 42.6% | 0.149 | [37.2%, 48.3%] | 296 |
| soft-only | 55.1% | 0.324 | [49.3%, 60.8%] | 296 |
| hard | 44.6% | 0.214 | [38.9%, 50.0%] | 296 |
| hard+soft | 50.7% | 0.309 | [44.9%, 56.4%] | 296 |
| soft-priority | 55.4% | 0.342 | [49.7%, 60.8%] | 296 |
| retrieval-sev | 77.7% | 0.740 | [73.0%, 82.4%] | 296 |
| narrative-evidence | 77.7% | 0.740 | [73.0%, 82.1%] | 296 |
| bn-sev | 77.7% | 0.740 | [73.0%, 82.4%] | 296 |
| bn-fused | 43.9% | 0.378 | [38.5%, 49.7%] | 296 |
| bn-fused-t | 42.9% | 0.294 | [37.5%, 48.6%] | 296 |
| lr | 64.2% | 0.454 | [58.8%, 69.6%] | 296 |
| emb-lr | 74.0% | 0.543 | [68.9%, 78.7%] | 296 |

## Injury: per-class recall (correct / true count)

| Predictor | FATL | SERS | MINR | NONE |
|---|---|---|---|---|
| bn-fused | 0/3 | 103/104 | 11/16 | 1/173 |
| bn-fused-t | 0/3 | 103/104 | 8/16 | 2/173 |
| bn-sev | 0/3 | 101/104 | 0/16 | 165/173 |
| retrieval-sev | 0/3 | 101/104 | 0/16 | 165/173 |
| soft-priority | 0/3 | 96/104 | 0/16 | 170/173 |
| lr | 0/3 | 91/104 | 0/16 | 169/173 |
| emb-lr | 0/3 | 100/104 | 0/16 | 171/173 |

## Damage: per-class recall (correct / true count)

| Predictor | DEST | SUBS | MINR | NONE |
|---|---|---|---|---|
| bn-fused | 2/3 | 64/78 | 64/89 | 0/126 |
| bn-fused-t | 0/3 | 64/78 | 62/89 | 1/126 |
| bn-sev | 2/3 | 57/78 | 64/89 | 107/126 |
| retrieval-sev | 2/3 | 57/78 | 64/89 | 107/126 |
| soft-priority | 0/3 | 35/78 | 8/89 | 121/126 |
| lr | 0/3 | 32/78 | 49/89 | 109/126 |
| emb-lr | 0/3 | 53/78 | 56/89 | 110/126 |

### Injury confusion matrix -- bn-fused (rows=true, cols=pred)

| true \ pred | FATL | SERS | MINR | NONE |
|---|---|---|---|---|
| FATL | 0 | 2 | 1 | 0 |
| SERS | 0 | 103 | 1 | 0 |
| MINR | 1 | 4 | 11 | 0 |
| NONE | 4 | 48 | 120 | 1 |

### Injury confusion matrix -- bn-fused-t (rows=true, cols=pred)

| true \ pred | FATL | SERS | MINR | NONE |
|---|---|---|---|---|
| FATL | 0 | 3 | 0 | 0 |
| SERS | 0 | 103 | 1 | 0 |
| MINR | 1 | 7 | 8 | 0 |
| NONE | 5 | 62 | 104 | 2 |

### Damage confusion matrix -- bn-fused (rows=true, cols=pred)

| true \ pred | DEST | SUBS | MINR | NONE |
|---|---|---|---|---|
| DEST | 2 | 0 | 1 | 0 |
| SUBS | 3 | 64 | 11 | 0 |
| MINR | 0 | 25 | 64 | 0 |
| NONE | 4 | 13 | 109 | 0 |

### Damage confusion matrix -- bn-fused-t (rows=true, cols=pred)

| true \ pred | DEST | SUBS | MINR | NONE |
|---|---|---|---|---|
| DEST | 0 | 2 | 1 | 0 |
| SUBS | 2 | 64 | 12 | 0 |
| MINR | 0 | 26 | 62 | 1 |
| NONE | 0 | 8 | 117 | 1 |

## Binary severe-outcome screening (severe = worst two classes)

### severe injury (FATL+SERS)

| Predictor | Sensitivity | Specificity | Balanced acc | n severe / n |
|---|---|---|---|---|
| bn-fused | 98.1% | 69.8% | 84.0% | 107 / 296 |
| bn-fused-t | 99.1% | 60.3% | 79.7% | 107 / 296 |
| bn-sev | 95.3% | 94.7% | 95.0% | 107 / 296 |
| retrieval-sev | 95.3% | 94.7% | 95.0% | 107 / 296 |
| soft-priority | 90.7% | 97.4% | 94.0% | 107 / 296 |
| lr | 86.0% | 97.4% | 91.7% | 107 / 296 |
| emb-lr | 94.4% | 97.9% | 96.1% | 107 / 296 |

### severe damage (DEST+SUBS)

| Predictor | Sensitivity | Specificity | Balanced acc | n severe / n |
|---|---|---|---|---|
| bn-fused | 85.2% | 80.5% | 82.8% | 81 / 296 |
| bn-fused-t | 84.0% | 84.2% | 84.1% | 81 / 296 |
| bn-sev | 74.1% | 89.3% | 81.7% | 81 / 296 |
| retrieval-sev | 74.1% | 89.3% | 81.7% | 81 / 296 |
| soft-priority | 43.2% | 91.2% | 67.2% | 81 / 296 |
| lr | 39.5% | 89.8% | 64.6% | 81 / 296 |
| emb-lr | 67.9% | 88.4% | 78.1% | 81 / 296 |

## Injury: paired comparisons (McNemar exact + Brier delta)

| A vs B | A only right | B only right | McNemar p | Brier delta (A-B) 95% CI | n |
|---|---|---|---|---|---|
| bn-fused-t vs bn-fused | 1 | 3 | 0.6250 | [-0.098, -0.057] | 296 |
| bn-fused-t vs lr | 20 | 167 | 0.0000 * | [+0.569, +0.756] | 296 |
| bn-fused-t vs retrieval-sev | 10 | 163 | 0.0000 * | [+0.615, +0.782] | 296 |
| bn-fused vs prior | 114 | 172 | 0.0007 * | [-0.072, +0.317] | 296 |
| bn-fused vs lr | 23 | 168 | 0.0000 * | [+0.636, +0.842] | 296 |
| bn-fused vs emb-lr | 14 | 170 | 0.0000 * | [+0.690, +0.886] | 296 |
| bn-fused vs retrieval-sev | 13 | 164 | 0.0000 * | [+0.681, +0.867] | 296 |
| bn-fused vs soft-priority | 18 | 169 | 0.0000 * | [+0.593, +0.804] | 296 |
| bn-fused vs bn-sev | 13 | 164 | 0.0000 * | [+0.683, +0.867] | 296 |
| bn-sev vs retrieval-sev | 0 | 0 | 1.0000 | [-0.000, -0.000] | 296 |
| retrieval-sev vs lr | 13 | 7 | 0.2632 | [-0.066, -0.009] | 296 |
| retrieval-sev vs emb-lr | 1 | 6 | 0.1250 | [-0.002, +0.025] | 296 |
| emb-lr vs lr | 12 | 1 | 0.0034 * | [-0.073, -0.024] | 296 |
| soft-priority vs lr | 10 | 4 | 0.1796 | [+0.009, +0.071] | 296 |
| soft-only vs lr | 10 | 6 | 0.4545 | [-0.035, +0.033] | 296 |
| hard+soft vs prior | 75 | 5 | 0.0000 * | [-0.582, -0.383] | 296 |
| hard+soft vs soft-only | 3 | 24 | 0.0000 * | [+0.093, +0.186] | 296 |
| hard+soft vs lr | 6 | 23 | 0.0023 * | [+0.094, +0.183] | 296 |

## Damage: paired comparisons (McNemar exact + Brier delta)

| A vs B | A only right | B only right | McNemar p | Brier delta (A-B) 95% CI | n |
|---|---|---|---|---|---|
| bn-fused-t vs bn-fused | 4 | 7 | 0.5488 | [-0.025, +0.006] | 296 |
| bn-fused-t vs lr | 56 | 119 | 0.0000 * | [+0.362, +0.570] | 296 |
| bn-fused-t vs retrieval-sev | 20 | 123 | 0.0000 * | [+0.467, +0.654] | 296 |
| bn-fused vs prior | 130 | 126 | 0.8513 | [-0.411, -0.036] | 296 |
| bn-fused vs lr | 61 | 121 | 0.0000 * | [+0.367, +0.583] | 296 |
| bn-fused vs emb-lr | 31 | 120 | 0.0000 * | [+0.453, +0.654] | 296 |
| bn-fused vs retrieval-sev | 16 | 116 | 0.0000 * | [+0.472, +0.664] | 296 |
| bn-fused vs soft-priority | 89 | 123 | 0.0232 * | [+0.239, +0.483] | 296 |
| bn-fused vs bn-sev | 16 | 116 | 0.0000 * | [+0.472, +0.669] | 296 |
| bn-sev vs retrieval-sev | 0 | 0 | 1.0000 | [-0.000, +0.000] | 296 |
| retrieval-sev vs lr | 59 | 19 | 0.0000 * | [-0.132, -0.057] | 296 |
| retrieval-sev vs emb-lr | 26 | 15 | 0.1173 | [-0.035, +0.005] | 296 |
| emb-lr vs lr | 50 | 21 | 0.0008 * | [-0.115, -0.041] | 296 |
| soft-priority vs lr | 29 | 55 | 0.0060 * | [+0.065, +0.166] | 296 |
| soft-only vs lr | 32 | 59 | 0.0061 * | [+0.133, +0.262] | 296 |
| hard+soft vs prior | 33 | 9 | 0.0003 * | [-0.579, -0.416] | 296 |
| hard+soft vs soft-only | 15 | 28 | 0.0660 | [-0.054, +0.063] | 296 |
| hard+soft vs lr | 15 | 55 | 0.0000 * | [+0.145, +0.258] | 296 |

`*` = significant at p < 0.05. Negative Brier delta favors A (lower Brier is better).

# Held-out diagnosis evaluation (cause-category level)

n = 253 held-out accidents (2007-2019) with a narrative and >=1 C/F cause finding. Era-fair rollup to CICTT top-level categories (AIRCRAFT / PERSONNEL / ENVIRONMENT / ORGANIZATIONAL). Truth = category set of the accident's C/F findings; top-1 correct if the predicted #1 category is in that set. Leak-safe (redacted embeddings).

Window mapping coverage: 4939/5062 C/F findings mapped (97.6%).

## Metric definitions (read before comparing tables)

Truth for an accident is a SET of categories (an accident can have C/F findings in several). Two different metrics follow, and they must not be conflated:

* **Top-1 accuracy** (first table): the predictor's #1-ranked category is IN the truth set. Multi-category accidents give every predictor several chances to be 'right', so absolute values are higher than a single-label accuracy would be; the freq baseline row shows how much of that is set-membership generosity.
* **Per-category recall** (last table): among accidents whose truth set CONTAINS category c, the fraction where the predictor's #1 category EQUALS c exactly. This is stricter and column-wise; a predictor can have high top-1 accuracy while never ranking a rare category first (see ORGANIZATIONAL).
* **MRR**: reciprocal rank of the first truth-set category in the predicted ranking, averaged over accidents.

## Top-1 accuracy and MRR (95% CI bootstrap, 10k)

| Predictor | Top-1 | 95% CI | MRR | n |
|---|---|---|---|---|
| freq | 45.8% | [39.5%, 52.2%] | 0.685 | 253 |
| retrieval | 84.2% | [79.4%, 88.5%] | 0.915 | 253 |
| bn-post | 57.7% | [51.4%, 63.6%] | 0.759 | 253 |
| bn-lift | 50.2% | [43.9%, 56.1%] | 0.723 | 253 |

## Paired comparisons (McNemar exact, top-1)

PRIMARY (confirmatory) comparisons are each predictor vs the freq baseline; Holm-Bonferroni is applied within that family (m = 3). The predictor-vs-predictor rows are exploratory (raw p only, no significance claims).

| A vs B | Role | A only right | B only right | p | p (Holm) |
|---|---|---|---|---|---|
| retrieval vs freq | primary | 113 | 16 | 0.0000 | 0.0000 * |
| bn-lift vs freq | primary | 66 | 55 | 0.3634 | 0.3634 |
| bn-post vs freq | primary | 49 | 19 | 0.0004 | 0.0007 * |
| bn-lift vs bn-post | exploratory | 22 | 41 | 0.0226 | -- |
| retrieval vs bn-lift | exploratory | 102 | 16 | 0.0000 | -- |
| retrieval vs bn-post | exploratory | 81 | 14 | 0.0000 | -- |

## Per-category recall (top-1 predictions, truth contains category)

| Predictor | PERSONNEL | AIRCRAFT | ENVIRONMENT | ORGANIZATIONAL |
|---|---|---|---|---|
| freq | 116/116 | 0/100 | 0/99 | 0/25 |
| retrieval | 79/116 | 62/100 | 72/99 | 0/25 |
| bn-post | 83/116 | 39/100 | 23/99 | 1/25 |
| bn-lift | 35/116 | 69/100 | 23/99 | 0/25 |

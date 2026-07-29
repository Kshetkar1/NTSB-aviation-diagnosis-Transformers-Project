# Retrieval hyperparameter sensitivity (internal split)

Queries: 236 window accidents 2002-2006; pool: 1982-2001 only; 2007-2019 test never touched.

## k-NN severity readout vs top_k

| top_k | injury top-1 | damage top-1 |
|---|---|---|
| 25 | 86.0% | 66.5% |
| 50 | 83.5% | 66.5% |
| 100 (production) | 83.5% | 61.9% |
| 200 | 83.1% | 59.3% |

## Soft-fact gates (top-100 pool, max 3 facts)

| min_fq | min_lift | facts/query | fact precision |
|---|---|---|---|
| 0.1 | 2.0 | 2.79 | 47.6% |
| 0.1 | 3.0 | 2.60 | 46.5% |
| 0.1 | 5.0 | 2.06 | 44.8% |
| 0.15 | 2.0 | 2.44 | 52.1% |
| 0.15 (production) | 3.0 | 2.14 | 53.6% |
| 0.15 | 5.0 | 1.63 | 49.4% |
| 0.25 | 2.0 | 1.54 | 66.6% |
| 0.25 | 3.0 | 1.27 | 71.3% |
| 0.25 | 5.0 | 0.81 | 74.0% |

Fact precision = fraction of suggested soft facts that are actually coded (occurrence/finding) in the query accident.

# Diagnosis: supervised embedding-LR baseline

OvR logistic regression on redacted narrative embeddings, trained on 1250 window accidents (mapped C/F categories), tested on the same 253 held-out accidents as `diagnosis_heldout_eval.md`.

| Predictor | Top-1 | 95% CI | MRR | n |
|---|---|---|---|---|
| emb-lr (supervised) | 88.1% | [84.2%, 91.7%] | 0.936 | 253 |
| retrieval | 84.2% | [79.4%, 88.5%] | 0.915 | 253 |
| bn-post | 57.7% | [51.4%, 63.6%] | 0.759 | 253 |
| freq | 45.8% | [39.5%, 52.2%] | 0.685 | 253 |

## McNemar exact (top-1) vs emb-lr

| A vs B | A only right | B only right | p |
|---|---|---|---|
| retrieval vs emb-lr | 5 | 15 | 0.0414 * |
| bn-post vs emb-lr | 8 | 85 | 0.0000 * |
| freq vs emb-lr | 6 | 113 | 0.0000 * |

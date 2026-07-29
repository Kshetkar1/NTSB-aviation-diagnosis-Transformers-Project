# Free-parameters inventory — "what are you training?"

Jesse's question, answered precisely. Every quantity in the pipeline is one of
FROZEN (fixed by Zhang's recipe / external data), CALIBRATED (computed once
from the 1982–2006 training window, no gradient/iterative fitting), SELECTED
(hyperparameter chosen on an internal validation split of the training
window), or FITTED (supervised training with labels). **Nothing is ever fitted
or selected on the 2007–2019 held-out set — it is a TEST set, used once.**

## The one-sentence answer

> "Nothing in the primary pipeline is trained. The BN is frozen from Zhang's
> Section-4 recipe on 1982–2006; the retrieval index is precomputed embeddings
> of the same window; the k-NN severity readout has zero fitted parameters.
> The 296 accidents from 2007–2019 are a held-out **test** set — we evaluate
> on them, we never train on them. The only *fitted* models are the two
> supervised baselines we compare against."

## Inventory

| Quantity | Where | Status | Data used |
|---|---|---|---|
| BN structure (nodes = coded occurrences/findings/persons) | `bn_upgraded.build_upgraded` | FROZEN | 1982–2006 coded NTSB |
| BN CPTs (conditional counts + Beta-CDF smoothing, Zhang §4.2–4.3) | same | FROZEN | 1982–2006 coded NTSB + BTS flight totals |
| Prior P(occurrence) = counts / 184,517,128 flights | Zhang Eq. 6 | FROZEN | NTSB counts + BTS table 1-37 |
| Narrative embeddings (index) | `shared/data/processed/embeddings_1982_2006.npy` | FROZEN (precomputed) | 1982–2006 narratives |
| Deterministic parser (vocabulary match, assertion guard) | `query_to_bn.parse_query_to_bn_evidence` | FROZEN (rule-based, no parameters) | BN's own node names |
| Redaction lexicon (leak-safe severity stripping) | `query_to_bn._REDACT_ONLY` | FROZEN (rule-based); audited by `tests/redaction_leak_probe.py` | none |
| k-NN severity readout (similarity-weighted neighbor outcomes) | `query_to_bn.severity_retrieval_distributions` | ZERO parameters | neighbors from 1982–2006 index |
| Severity virtual-evidence likelihood L(j) ∝ f_q(j)/p0(j) | `retrieval_severity_virtual_evidence` | ZERO parameters (Jeffrey conditioning) | — |
| top_k neighbors for severity readout | eval `--sev-topk` | **k=100 (primary)** — the pre-set default, same pool size as evidence retrieval, fixed before any held-out run. The internal split (queries 2002–2006, pool 1982–2001) mildly preferred k=25 (injury 86.0% vs 83.5%), but the plateau is flat, and a held-out *ablation* confirms the choice is immaterial: k=25 gives 89.9%/77.7% vs k=100's 90.9%/77.4% (`outputs/heldout_significance_k25.md`). We report k=100 as primary and k=25 as sensitivity, never the better of the two post hoc. | training window only |
| Soft-fact gates (min_fq=0.15, min_lift=3.0, top_m=3, max_conf=0.95) | `retrieval_facts` | assumed defaults; sensitivity in `outputs/hyperparam_sensitivity.md` | training window only |
| Stated-severity confusion matrices P(stated \| coded) | `severity_likelihoods` | CALIBRATED (counted once, Laplace 0.5); **OFF by default** (leak-safe) | 1982–2006 narratives |
| Logistic regression baseline (parsed features) | `tests/lr_baseline_heldout.py` | FITTED (supervised) | 1982–2006 labels |
| Logistic regression baseline (narrative embedding) | `tests/embedding_lr_baseline.py` | FITTED (supervised) | 1982–2006 labels |

## Terminology discipline (Maha's rule)

- The 296 accidents (2007–2019) are the **held-out test set** — never say
  "we trained on 296 accidents".
- BN outputs after evidence propagation are **posteriors**; the raw k-NN
  neighbor frequencies are **empirical severity distributions**, not
  posteriors (they only become posteriors after entering the BN as virtual
  evidence).
- "Soft evidence" enters via **Pearl virtual evidence** implementing
  **Jeffrey conditioning**: likelihood ratio anchored on the node's own prior,
  so the updated belief lands at the parsed confidence c.

## Validation-set question, pre-answered

There IS an internal validation split: queries = 2002–2006 window accidents,
retrieval pool = 1982–2001 (see `tests/retrieval_hyperparam_sensitivity.py`).
It is used only to sanity-check retrieval hyperparameters — NOT to pick the
production values. The production values (top_k=100, min_fq=0.15,
min_lift=3.0, top_m=3) are the defaults set when the pipeline was built,
before the sensitivity study; the study then confirmed the results sit on a
flat plateau around them (and a held-out k=25 ablation confirms the same
out-of-sample). If the internal split HAD driven the choice, the primary k
would be 25; we deliberately did not switch, to avoid even the appearance of
tuning toward the metric. The test window (2007–2019) is disjoint from both
splits by accident ID and by year, verified by `tests/heldout_leak_audit.py`
(0 / 296 overlaps).

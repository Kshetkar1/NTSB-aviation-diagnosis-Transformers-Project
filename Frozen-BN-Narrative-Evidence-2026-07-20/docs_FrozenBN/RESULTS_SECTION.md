# Paper-ready Results section (leak-safe, 2026-07-29)

Drop-in text for the paper. Every number regenerates from `REPRODUCE.md`;
do not mix with any figure produced before the leakage fix (93%/81% era).

---

## 5. Results

### 5.1 Fidelity to the reference network

Before evaluating the narrative pipeline we verify that the frozen network
is a faithful reconstruction of Zhang & Mahadevan's model. On every
quantity the original paper reports, our rebuild matches exactly: 102 fire
occurrences in the 1982-2006 window, the prior formula P(fire) =
102/184,517,128 = 5.53e-7, and all 85 rows of the published Table 7
conditional cause distribution (85/85, contributory-factor counting,
tolerance +/-0.0005). This matters for what follows: all narrative evidence
is injected into a network whose parameters are pinned to a published,
independently constructed reference, not fitted to our evaluation data.

### 5.2 Evaluation protocol

The network and the retrieval index are built exclusively on the
1982-2006 window (1,742 accidents). The held-out set is every accident in
the corpus outside that window (2007-2019) with a factual narrative:
n = 296 for severity, of which 253 carry cause findings for diagnosis.
To prevent outcome leakage, severity-stating phrases ("was destroyed",
"received fatal injuries") are removed from every narrative by a
deterministic redaction pass before any embedding or parsing (all
predictors, including the deterministic and LLM parsers, receive only
redacted text), and narrative-stated severity is excluded as evidence.
Redaction's scope is stated precisely: it removes explicit outcome
statements, not outcome predictability -- mechanism wording (stall,
turbulence, gear collapse) legitimately predicts severity, and a
window-trained TF-IDF probe confirms the redacted text retains that
mechanism signal (full vs redacted differ by 0.0 pp injury / 2.4 pp
damage) while a token-attribution check confirms the predictive tokens
are mechanism words, not outcome words. A held-out ID audit shows 0/296
overlaps with the window. Retrieval hyperparameters are the pipeline's
pre-set defaults (k = 100 neighbors, matching the evidence-retrieval
pool); an internal 2002-2006 validation slice and a held-out k=25
ablation both show a flat plateau around them. Limitation, disclosed:
the held-out window was scored repeatedly during development (protocol
fixes, ablation design), so the reported accuracies may carry some
selection optimism; no model parameter was ever fitted to held-out data,
but a one-shot confirmation on a never-evaluated window (e.g. 2020-2024)
is the appropriate confirmatory follow-up. No component of the primary
pipeline is trained: the free-parameter inventory (FREE_PARAMETERS.md)
classifies every quantity as frozen (network CPTs), measured (evidence
strengths, counted as fractions of retrieved neighbors), or a fixed
default confirmed by sensitivity analysis.

### 5.3 Prognosis: held-out severity prediction

Table X reports 4-class top-1 accuracy, Macro-F1, and Brier score for
injury (fatal / serious / minor / none) and damage (destroyed /
substantial / minor / none), with bootstrap 95% CIs (10,000 resamples)
and exact McNemar tests for paired comparisons.

All predictors and baselines are scored on the identical 296-accident
cohort (same narrative filter, truncation, and redaction; the accident
IDs are published in cohort_manifest.json). Five comparisons per target
were designated primary and Holm-Bonferroni-corrected; all other
contrasts are reported as exploratory.

| Predictor | Injury acc. | Injury M-F1 | Damage acc. | Damage M-F1 |
|---|---|---|---|---|
| Majority class (prior) | 58.4% | 0.184 | 42.6% | 0.149 |
| Parsed events only (hard+soft) | 82.4% | 0.422 | 50.7% | 0.309 |
| Supervised LR (parsed features) | 85.5% | 0.440 | 60.1% | 0.408 |
| Supervised LR (TF-IDF text) | 92.2% | 0.603 | 73.3% | 0.621 |
| Supervised LR (embedding) | 91.6% | 0.474 | 74.0% | 0.543 |
| **Narrative -> BN (bn-sev)** | **90.9%** | **0.470** | **77.4%** | **0.697** |

The full chain improves on the network prior by +32.5 points on injury
and +34.8 on damage (both Holm-adjusted p < 1e-4, McNemar exact), and
significantly outperforms the supervised logistic regression on parsed
features (Holm p = 0.006 injury, p < 1e-4 damage). Against the two
strongest supervised baselines -- logistic regression on the raw
narrative embedding and on TF-IDF bag-of-words text, both trained on the
1,286 window accidents with usable narratives -- the zero-parameter chain
is statistically indistinguishable (embedding: Holm p = 1.0 injury /
0.29 damage; TF-IDF: Holm p = 0.87 injury / 0.29 damage). The TF-IDF
model is numerically the best injury predictor (92.2%, 8 discordant
accidents vs bn-sev) and has the best injury Macro-F1 (0.603); the chain
is numerically best on damage and clearly best on damage Macro-F1 (0.697
vs 0.621/0.543). We note the TF-IDF baseline originated as our redaction
leak probe and was promoted to the baseline table, configuration
unchanged, once its strength was apparent. As a severe-outcome screen
(fatal-or-serious vs rest), the chain reaches 93.5% sensitivity / 96.8%
specificity for injury and 75.3% / 89.8% for damage (TF-IDF: 94.4%/98.9%
injury but only 58.0%/93.0% damage). Remaining failure modes are the rare
classes: fatal injuries (3 cases) and minor injuries (16) are never
top-1; we report the full confusion matrices in the appendix.

Two ablations locate the contribution. Removing the network and reading
severity directly off the 100 nearest neighbors (retrieval-sev) gives the
same accuracy -- by construction, since a self-test asserts the network
posterior reproduces the virtual-evidence input when no other evidence
competes. Fusing event evidence and neighbor severity in a single
inference (bn-fused) collapses accuracy to 38.5%/41.9%: the event
evidence re-derives severity from the same narrative, so fusing the two
double-counts it -- a negative result we report deliberately. The
accuracy therefore comes from the narrative-retrieval signal; the network
contributes the reasoning layer (joint conditioning, what-if queries,
per-node explanations) at zero accuracy cost.

### 5.4 Diagnosis: held-out cause-category prediction

NTSB replaced its coding taxonomy in 2008 (legacy subject codes ->
CICTT), so exact cause-code matching across our temporal split is
impossible by design. We therefore evaluate at the level CICTT itself
defines: the four top-level cause categories (Personnel, Aircraft,
Environment, Organizational). Held-out truth is the category set of an
accident's coded cause findings; legacy-era findings are mapped by
auditable keyword rules covering 97.6% of window cause findings. A
dual-coded audit of a 75-row stratified mapping sample, adjudicated row by
row by the first author, found 68 rows correct, 2 wrong (both corrected in
the rules; the rerun moved retrieval by +0.4 points and no other predictor),
and 5 genuinely ambiguous boundary rows confirmed as mapped, bounding the
residual mapping uncertainty at roughly 2 accuracy points without affecting
any ordering (mapping_audit_summary.md). A prediction is correct if its top-ranked
category is in the truth set (n = 253); per-category recall, which is
stricter (exact top-1 match per category), is reported separately.

| Predictor | Top-1 | 95% CI | MRR |
|---|---|---|---|
| Category frequency baseline | 45.8% | [39.5, 52.2] | 0.685 |
| Frozen BN, event evidence | 57.7% | [51.4, 63.6] | 0.759 |
| **Narrative retrieval (zero-parameter)** | **84.2%** | **[79.4, 88.5]** | **0.915** |
| Supervised LR (embedding) | 88.1% | [84.2, 91.7] | 0.936 |

Retrieval nearly doubles the frequency baseline (Holm-adjusted p < 1e-4;
the three baseline comparisons are the designated primary family) and
is balanced across the three common categories (68/62/72% recall); no
predictor recovers the rare Organizational class (25 cases). The
supervised embedding model is 3.9 points better than retrieval
(p = 0.041, exploratory) -- the expected price of zero training -- while
the BN event path alone reaches 57.7%, significantly above baseline
(Holm p = 0.0007) but well below the narrative readouts. We also report a negative result:
ranking categories by posterior lift instead of posterior probability
degrades top-1 to 50.2%, because lift amplifies low-prior nodes.

### 5.5 Reproducibility

Every table regenerates from the public repository with one command per
result (REPRODUCE.md). Narrative embeddings are cached on disk keyed by
content hash, making reruns deterministic and API-free; all significance
machinery (bootstrap seeds, McNemar counts) is seeded and versioned.

---

### Numbers you must NOT put in the paper

- 93% / 81% severity accuracy (pre-leakage-fix; superseded by 90.9 / 77.4).
- The old "diagnosis similarity 36.6%" metric (replaced by the
  era-fair category evaluation).
- Any Brier/logloss from runs before 2026-07-28.

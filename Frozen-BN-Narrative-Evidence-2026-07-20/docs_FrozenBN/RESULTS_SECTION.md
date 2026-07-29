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
accident's coded cause findings; legacy-era findings are mapped by an
explicit, published keyword rule set covering 97.6% of window cause
findings. We do not claim the rollup is validated by independent coding;
we bound its influence instead. A 75-row stratified sample of the rule
set's decisions was reviewed for internal consistency against CICTT's own
top-level conventions: 68 rows consistent, 2 rule defects (both corrected;
the rerun moved retrieval by +0.4 points and no other predictor), and 5
genuine taxonomy-boundary rows where CICTT admits both readings and the
mapping was kept. Those seven contested rows together touch 316 of 5,062
window cause findings (6.2%), which bounds the mapping's effect at at most
2 accuracy points and -- because a reassignment perturbs every predictor's
inputs in the same direction -- leaves every claimed ordering intact
(mapping_audit_summary.md). A prediction is correct if its top-ranked
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

### 5.5 Limitations

This is the canonical limitations list for the paper; write the Discussion's
limitations subsection from here. Each item is a genuine constraint we cannot
engineer away, stated without hedging.

1. **Retrospective narratives.** Every narrative we consume is written after
   the investigation closed, so this is not real-time prediction. The
   defensible use cases are triage of an existing report, coding assistance,
   and what-if analysis -- not forecasting an accident's outcome as it unfolds.

2. **The network adds no severity accuracy, by construction.** `bn-sev` and
   `retrieval-sev` agree on all 296 held-out accidents (0 discordant), because
   the network faithfully mediates the k-NN severity evidence rather than
   adding information to it. Every severity accuracy figure in this paper is
   the narrative-retrieval signal's; the network's contribution is joint
   probabilistic reasoning at zero accuracy cost, and no claim of predictive
   superiority over retrieval is made anywhere.

3. **Rare classes are not learnable from this window.** The held-out set
   contains 3 fatal-injury accidents (1 flagged severe on injury), 16
   minor-injury accidents (0 ranked top-1), and 25 Organizational-cause
   accidents (0 recovered by any predictor). These n are too small to learn
   or retrieve reliably, and no amount of modeling fixes that; we publish
   per-class recall and full confusion matrices rather than absorbing the
   misses into an accuracy average.

4. **Embedding pretraining may have seen the test years.** The OpenAI
   embedding model's training corpus is undisclosed and may include
   2007-2019 NTSB reports; redaction cannot reach inside a pretrained
   encoder. Comparisons stay internally fair because emb-LR uses the same
   channel, and the TF-IDF text baseline (92.2% injury / 73.3% damage) is the
   pretraining-free reference point -- it matches the embedding pipeline, so
   the results do not depend on the embedding's provenance.

5. **12 of Zhang's 93 published BN numbers still differ.** After the person-node
   and multi-state-severity upgrades the scoreboard is 48 exact / 29 close /
   12 differ. The differences are attributed rather than hidden: Zhang's
   construction randomly breaks ties, so his published values are one draw
   from a distribution (build-variance envelope,
   `outputs/bn_variance_envelope.json`), and his own released `NTSB.xdsl` does
   not reproduce his published Table 8 either
   (`outputs/bn_posterior_parity.json`). We therefore claim faithfulness to
   his published *method and data-derived quantities* (85/85 Table 7 rows,
   102 fire occurrences, the prior formula), not cell-for-cell identity of
   every posterior.

6. **Diagnosis is only evaluated at four-category granularity.** NTSB's 2008
   taxonomy change means legacy and CICTT codes share no vocabulary below
   CICTT's top level, so the four-category rollup is forced by the data, not
   chosen for convenience. The rollup itself is a keyword rule set that no
   independent coder has checked; its influence is bounded (contested rows are
   316/5,062 window findings = 6.2%, worst case <= 2 accuracy points in the
   same direction for every predictor, so all claimed orderings survive), but
   a bound is not a validation. Finding-level and occurrence-level diagnosis
   accuracy are computed by the pipeline and not scored here.

7. **Held-out reuse during development.** The 296-accident window was scored
   repeatedly while protocol fixes and ablations were designed, so reported
   accuracies may carry selection optimism. No model parameter was ever fitted
   to held-out data, but decisions saw held-out results. **A one-shot
   confirmatory run on a never-evaluated window (2020-2024) is a prerequisite
   for submission,** not an optional extension; until it exists, every number
   here should be read as developed-on-test.

### 5.6 Reproducibility

Every table regenerates from the public repository with one command per
result (REPRODUCE.md). Narrative embeddings are cached on disk keyed by
content hash, making reruns deterministic and API-free; all significance
machinery (bootstrap seeds, McNemar counts) is seeded and versioned.

---

### Numbers you must NOT put in the paper

- 93% / 81% severity accuracy (pre-leakage-fix; superseded by 90.9 / 77.4).
  If it appears at all, it appears explicitly labelled as superseded, in the
  leakage narrative, never as a result.
- The old "diagnosis similarity 36.6%" metric (replaced by the
  era-fair category evaluation).
- Any Brier/logloss from runs before 2026-07-28.
- Pre-correction diagnosis numbers: retrieval 83.8%, coverage 98.3%,
  emb-LR vs retrieval +4.3 pp / p = 0.027. Current: 84.2%, 97.6%,
  +3.9 pp / p = 0.041. The old values are valid only inside the
  before/after sensitivity table in `outputs/mapping_audit_summary.md`.

### Framings you must NOT use

- **"Audit" or "dual-coded" for the category mapping.** The review pass was an
  AI assistant with rows checked by the first author -- self-review, not
  independent coding. Say "documented rule set" and give the sensitivity bound
  (<= 2 pp, same direction for every predictor). The leakage and train/test
  checks *are* audits and keep that word.
- **Any phrasing in which the Bayesian network improves predictive accuracy.**
  It does not: `bn-sev` = `retrieval-sev` exactly. The BN's contribution is the
  reasoning layer at zero accuracy cost.
- **The BN event path (57.7% diagnosis) as a headline.** It is a
  structured-inference result and simultaneously a partial negative result:
  significantly above the frequency baseline (Holm p = 0.0007), 26.5 points
  below retrieval. Retrieval (84.2%) is the primary diagnosis readout.

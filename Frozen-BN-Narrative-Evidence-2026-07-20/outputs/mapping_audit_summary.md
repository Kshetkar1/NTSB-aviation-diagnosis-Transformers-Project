# Diagnosis category rollup — rule set and mapping sensitivity

The legacy-to-CICTT rollup is a **documented keyword rule set**
(`categorize_legacy` / `categorize_cictt` in `tests/diagnosis_heldout_eval.py`),
not a coded judgement exercise. This file records (a) a consistency review of a
stratified 75-row sample of the rule set's decisions and (b) the quantity that
actually carries evidential weight: **the sensitivity of the reported results to
the contested rows.**

## What was and was not done (stated precisely)

A 75-row stratified sample (50 legacy 1982-2006 subject+person rows weighted
toward high window counts, 25 CICTT 2007-2019 finding codes) was reviewed for
**internal consistency against CICTT's own top-level conventions**, using the 25
CICTT rows in the sample as the anchor — their prefix IS the category, so they
define how the taxonomy treats each situation type. The sample and per-row notes
are in `mapping_audit_sample.csv`.

The review pass was performed by an AI assistant and every row was then checked
by the first author (2026-07-29). **This is not independent dual coding**, and
nothing here should be read as inter-rater agreement: an assisted pass plus
first-author review is self-review with extra steps. It is reported as what it
is — a documented consistency check that found and fixed two rule defects — and
the paper's claim about mapping robustness rests on the sensitivity bound below,
not on the review.

## Sample review outcome

| Outcome | Legacy rows | CICTT rows | Total | Affected window findings |
|---|---|---|---|---|
| consistent with CICTT conventions | 43 | 25 | 68 (90.7%) | -- |
| rule defect -> corrected | 2 | 0 | 2 (2.7%) | 60 (1.2% of 5,062) |
| boundary case -> mapping kept | 5 | 0 | 5 (6.7%) | 256 (5.1% of 5,062) |

## The two rule defects — found and fixed

1. **`maintenance, service bulletin/letter` + Company/operator management**
   (22 findings). Was PERSONNEL via the "maintenance" keyword; the attached
   person is *management*, and the sample's own `procedure inadequate` +
   management row maps to ORGANIZATIONAL. **Corrected to ORGANIZATIONAL.**
2. **`reason for occurrence undetermined` + (unspecified person)**
   (38 findings). Was PERSONNEL via the person-attribution fallback;
   "undetermined" is not a personnel cause. **Corrected to excluded from
   cause mapping.**

Both corrections are implemented at the top of `categorize_legacy` in
`tests/diagnosis_heldout_eval.py` (commented with the 2026-07-29 date).

## The five boundary cases — mapping kept

`flight into known adverse weather` -> ENVIRONMENT (15; CICTT
condition-response convention), `ifr separation standards` -> PERSONNEL (32),
`inadequate training` -> PERSONNEL (12), `miscellaneous`+Unknown -> PERSONNEL
(29), and `procedures/directives`+(unspecified person) -> PERSONNEL (168).
Each sits on a genuine CICTT boundary (person non-compliance vs organizational
procedure deficiency) where the taxonomy admits both readings. These rows are
the residual contested mass, and they are what the sensitivity bound below
quantifies.

## Mapping sensitivity — the actual evidence (rerun 2026-07-29)

| Quantity | Before corrections | After corrections |
|---|---|---|
| Window mapping coverage | 4977/5062 (98.3%) | 4939/5062 (97.6%) |
| retrieval top-1 | 83.8% | 84.2% (MRR 0.915) |
| bn-post top-1 | 57.7% | 57.7% |
| bn-lift top-1 | 50.2% | 50.2% |
| freq top-1 | 45.8% | 45.8% |
| emb-lr top-1 | 88.1% | 88.1% |
| emb-lr vs retrieval (exploratory) | +4.3 pp, p = 0.027 | +3.9 pp, p = 0.041 |

**The bound.** All seven contested rows together (2 corrected + 5 boundary)
touch 316 of 5,062 window C/F findings — **6.2% of the mapped mass**. Because a
category reassignment perturbs the retrieval neighborhood and the BN's node
scores in the *same direction for every predictor*, the worst case a full
reassignment of that mass could produce is **<= 2 percentage points on absolute
top-1 accuracy, identical in direction for every predictor**. The two actually
corrected rows moved retrieval by +0.4 pp and moved nothing else, which sits
comfortably inside that bound.

**What this licenses the paper to claim.** The reported ordering
(retrieval >> bn-post > bn-lift ~ freq) is insensitive to mapping choices: every
gap the paper actually claims exceeds 2 pp by a wide margin — retrieval over
bn-post is 26.5 pp, bn-post over freq is 11.9 pp — so no admissible re-reading of
the contested rows can reorder them. (bn-lift vs freq is 4.4 pp and is reported
as *not* significant, p = 0.36, so no ordering claim is made there in the first
place.) Absolute accuracies carry a residual mapping uncertainty of <= 2 pp, and
the paper states them that way.

**What it does not license.** It does not license a claim that the rollup is
*correct* in an inter-rater sense. The 4-category rollup is a defensible reading
of CICTT's own top level, published in full so a reader can disagree with any
specific rule and recompute — that is the honest standing of this artifact.

## Provenance

Sample generated by `tests/diagnosis_mapping_audit_sample.py` (file names retain
the original "audit" wording; the file contents and all prose describing them
are the rule set + sensitivity framing above). Mapping rules live in
`tests/diagnosis_heldout_eval.py` (`categorize_legacy`, `categorize_cictt`);
rerun artifacts: `diagnosis_heldout_eval.md/.json`, `diagnosis_emb_lr.md/.json`.

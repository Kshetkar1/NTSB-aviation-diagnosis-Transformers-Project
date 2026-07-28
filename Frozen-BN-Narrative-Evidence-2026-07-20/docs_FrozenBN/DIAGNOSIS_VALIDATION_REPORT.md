# Diagnosis-Quality Validation Report

Workstream: diagnosis quality for the NTSB Bayesian-Network reproduction (Zhang &
Mahadevan, *RESS* 209, 2021). Covers three deliverables: (1) Table 7 reproduction,
(2) the exact-filter diagnosis mode, (3) sparse-cell semantic smoothing vs Zhang's
Beta-CDF. All numbers below are reproducible with framework Python 3.11.

---

## Bottom lines (tell the advisor)

- **Table 7:** Now **85/85 exact** (was 67/85). All 18 residuals were label-mapping
  artefacts, fixed honestly in the label layer with **no dataset edits**.
- **Exact-filter diagnosis:** **Works** for arbitrary outcome+fact combinations and
  degrades safely; recommend presenting it as **"evidence-filtered cohort
  diagnosis"** (drop "conditional"). Usable today.
- **Sparse cell:** **HYBRID / lean Beta-CDF.** Semantic smoothing beats Beta-CDF in
  the sparse regime but does *not* cleanly win the leakage-free test (it has the
  worst log-loss). Default to Zhang's Beta-CDF (parity) or plain counting; keep
  semantic only as an experimental ultra-sparse-cell smoother.

---

## Task 1 — Table 7 reproduction: 67/85 → 85/85 (honest)

**Artifacts:** `docs/TABLE7_FULL_REPRODUCTION.md`, `docs/table7_full_reproduction.csv`,
`tests/reproduce_table7_full.py` (compares verbatim against the PDF, page 12).

The fix lives entirely in the label/edge-mapping layer of `zhang_diagnosis`
(new opt-in `cause_factor_only=True` path); **no dataset rows were mutated**, and the
default behaviour of every existing function is unchanged (backward-compatible for
the concurrent `trees.py` consumer, `sparse_cpt`, and `prognosis`).

Two root causes explained all 18 mismatches:

1. **Contributory-factor filter — closes 17 over-attributions.** Every NTSB legacy
   finding carries a `Cause_Factor` flag: `C` (cause), `F` (factor), or blank
   (a non-causal descriptive finding). Zhang's Table 7 counts *contributory
   factors* = `C`/`F` only. The prior reproduction counted **all** findings on the
   fire occurrence, so blank-flag descriptive findings inflated counts — most
   visibly `Emergency procedure – Performed` (11→1) and `Evacuation – Performed`
   (6→1), plus +1/+2 on APU, electric wiring, fuel, engine compartment, etc.
   Restricting to `Cause_Factor ∈ {C,F}` reproduces every published count exactly.
2. **Unresolved-code label — closes the 1 absent label.** Zhang's
   `deriveNamebyCode()` (his `main.py`, lines 520-527) returns the literal string
   **"Unknown quantity"** for any `Subj_Code` missing from his code→meaning lookup.
   The refined dataset stores those same unresolved findings with a `nan`
   description. They are the *same* records: `Subj_Code 92000`, present as a Factor
   on exactly **2 fire findings**. Normalizing nan→"Unknown quantity" reproduces
   Zhang's convention, simultaneously deleting the spurious `nan` cause and
   restoring `Unknown quantity` (n=2).
   - *Footnote (honesty):* the NTSB master code table `ct_seqevt` actually maps
     `92000 → Inadequate certification/approval`; **both** Zhang's lookup and the
     student's lookup lack this code, so both fall back to the unknown placeholder.
     We deliberately match Zhang's (imperfect) convention so the table is
     apples-to-apples.

**Denominator preserved at 102** in faithful mode: `count(fire)` counts every fire
accident regardless of the C/F filter, so the two incidents whose only fire-occurrence
findings were blank-flag remain in the denominator exactly as in Zhang.

**Residual differences: none.** All 85 of Zhang's Table 7 causes reproduce within
±0.0005 (Zhang's own 5-dp rounding). In faithful mode the reproduction surfaces
**exactly** Zhang's 85 factors — no spurious extras.

---

## Task 2 — Exact-filter diagnosis (the mode mislabeled "conditional diagnosis")

**What it does:** given an outcome + one or more known facts, hard-filter the corpus
to accidents whose node-set contains the outcome **and every fact**, then rank causes
by Zhang's `count(cause & outcome)/count(outcome)` over that cohort. It is a
deterministic, no-network set-intersection query (`diagnose_conditional(..., mode="global")`).

**Validation:** `tests/exact_filter_validation.py` → `docs/exact_filter_validation.json`.

| Outcome | Known fact(s) | Cohort n | Top cause (P) |
|---|---|--:|---|
| fire | electric wiring | 14 | Electrical system, electric wiring (**78.6%**, 11/14) |
| fire | fuel | 8 | Fluid, fuel (87.5%) |
| fire | auxiliary power unit | 8 | Auxiliary power unit (APU) (75.0%) |
| fire | maintenance | 3 | Maintenance (100%) |
| loss of engine power | fuel | 9 | Fluid, fuel (100%) |
| gear collapsed | landing gear | 3 | Landing gear (66.7%) |
| loss of engine power | carburetor | 1 | (tiny n — all causes 100%) |
| fire | electric wiring + maintenance | 0 | empty cohort (safe) |
| fire | wiring + fuel + maintenance | 0 | empty cohort (safe) |

**Findings:**
- **Reproduces the canonical example exactly:** fire + electric wiring → 14
  incidents, wiring 78.6%.
- **Generalizes** to arbitrary outcome+fact combinations (multiple outcomes and
  facts shown above), all returning sensible cohorts and rankings.
- **Degrades sensibly:** as the filter narrows, n shrinks monotonically; at n=1
  every cause trivially reads 100% (correctly a tiny-n signal, not a bug); when the
  intersection is empty the call returns a safe empty cohort, no crash.
- **One caveat (matcher, not logic):** the free-text condition matcher
  (`match_cause`) can map an ambiguous single word to an unintended label — e.g.
  bare `"wiring"` matched `thrust reverser, wiring` (cohort 0) instead of
  `electrical system, electric wiring`. Passing the specific label fixes it. The
  *filter/count* engine is sound; only the convenience phrase-matcher is loose on
  ambiguous one-word inputs.

**Recommended presentation name (do NOT rename the live function):**
1. **Evidence-filtered cohort diagnosis** ← recommended
2. Exact-filter diagnosis
3. Targeted (evidence-constrained) cohort diagnosis

"Evidence-filtered cohort diagnosis" is precise (it filters to a *cohort* by *evidence*
and reports empirical frequencies) and, unlike "conditional diagnosis", makes no claim
to a Bayesian-conditioning / probabilistic-inference semantics — addressing Maha's
objection.

**Verdict:** ✅ The user can confidently present and use this mode. Recommend (a)
showing the cohort size n alongside the ranking, (b) flagging tiny-n cohorts (e.g.
n<5) as low-confidence, and (c) using precise condition labels (or surfacing the
matched label, which the function already returns in `matched_conditions`).

---

## Task 3 — Sparse cell: semantic smoothing vs Zhang's Beta-CDF

**Artifacts:** `tests/sparse_robustness_validation.py`,
`docs/sparse_robustness_results.json`, `docs/figures/sparse_robustness.png`.
Cells = incident-level conditionals P(outcome | cause) on causes with ≥30 incidents
(reliable gold); 158 cells / 47 causes; k=50 semantic neighbours; 400 subsample draws.

Two complementary tests:

### (a) Subsample-to-recover (simulated sparsity; optimistic for semantic)
Mean MAE vs full-data gold (lower better):

| n | raw | cap | Beta-CDF | semantic |
|--:|--:|--:|--:|--:|
| 1 | 0.284 | 0.272 | 0.284 | **0.111** |
| 2 | 0.216 | 0.212 | 0.272 | **0.111** |
| 3 | 0.176 | 0.173 | 0.250 | **0.111** |
| 5 | 0.131 | 0.130 | 0.212 | **0.111** |
| 10 | **0.085** | **0.085** | 0.166 | 0.111 |

- Semantic beats **Beta-CDF at every n** (Wilcoxon p<1e-14) and is perfectly stable
  (across-draw STD = 0, since it is n-independent).
- Semantic beats raw counting for n≤5 but **raw wins at n=10** (p≈0.04).
- **Beta-CDF is dominated by raw counting** at every n here — applying `beta.cdf` to
  the per-cell ratio biases the estimate upward.
- *Honesty:* this is an **optimistic** bound — semantic's neighbour pool includes the
  cell's own incidents (leakage).

### (b) Leave-one-incident-out (leakage-controlled; the honest verdict)

| method | Brier (↓) | log-loss (↓) |
|---|--:|--:|
| raw | **0.148** | **0.471** |
| cap | **0.148** | **0.471** |
| Beta-CDF | 0.171 | 0.530 |
| semantic | 0.167 | 0.600 |

- By **Brier**: semantic edges Beta-CDF (Δ=−0.0042, Wilcoxon p≈3e-11) — *but* plain
  counting (raw/cap) beats both.
- By **log-loss**: semantic is the **worst** (0.600 > Beta-CDF 0.530 > raw 0.471) —
  the fixed corpus-level fraction is poorly calibrated for individual held-out cells.
- So once cells are not ultra-sparse, **plain counting is the best honest predictor**,
  and semantic does **not** reliably beat Beta-CDF (wins Brier, loses log-loss).

### Verdict: **HYBRID / lean Beta-CDF (NO-GO for semantic as the default)**
- Semantic's only honest edge is the **ultra-sparse regime (n≤3)**: lowest MAE and
  perfect stability. Outside that, it loses to plain counting on both Brier and
  log-loss and only ties/edges Beta-CDF.
- Per the advisor's rule (fall back to Beta-CDF unless semantic *clearly* wins) —
  it does not clearly win — so:
  - **Default:** Zhang's **Beta-CDF** for parity with the paper (or **plain
    counting**, which is empirically strongest on non-sparse cells).
  - **Optional:** use **semantic only for ultra-sparse cells (support ≤3)** where
    counting is degenerate (0/1), and label it experimental — with the explicit
    caveat that its calibration (log-loss) is the weakest of the four.
- **One-line bottom line:** *Semantic smoothing helps only in the ultra-sparse
  corner and isn't reliable enough to replace Beta-CDF; default to Beta-CDF (Zhang
  parity) and treat semantic as an experimental sparse-cell add-on.*

---

## Reproduce everything

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11
# Task 1 (no network):
$PY tests/reproduce_table7_full.py                 # -> 85/85
# Task 2 (no network):
$PY tests/exact_filter_validation.py
# Task 3 (network for the semantic lane; counts-only with --no-semantic):
$PY tests/sparse_robustness_validation.py
```

# Head-to-Head Predictive Scores — Narrative method vs Zhang's counting baseline

**Question (the publishability crux):** *Does the narrative method produce **better
predictive scores** than Zhang's counting method?*

**Crucial framing.** You **cannot** beat Zhang on his own descriptive quantities
(Table 7 `P(cause|fire)`, Table 9 edges) — reproducing them exactly is **parity by
definition** (and we already reproduce them: 85/85 Table-7 cells, prior
5.53×10⁻⁷; see `docs/ZHANG_REPRODUCTION_REPORT.md`). "Better" can only be claimed on
a **held-out predictive task** where neither method saw the test item. This report
measures that, honestly, on two tasks.

- **Script:** `tests/headtohead_scores.py` (framework Python 3.11, runs **offline**
  from cached embeddings; engine files unmodified).
- **Data / index:** Zhang window `refined_dataset_1982_2006.json` (1,742 incidents;
  1,703 indexed).
- **Machine-readable results:** `docs/head_to_head_scores_results.json`.

```bash
PY=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11
$PY tests/headtohead_scores.py            # both tasks, offline (no network)
```

---

## TL;DR verdict

| Task | Narrative vs Zhang baseline | Significance |
|---|---|---|
| **Diagnosis** (predict a held-out incident's cause from its narrative) | **BETTER** on every metric | top-1 +5.6 pp (McNemar p≈9e-9), MRR +0.062 (p≈5e-20), log-loss −0.23 (p≈5e-67) |
| **Prognosis** (predict the next event on held-out transitions) | **MARGINAL / TIE** — a small ranking edge, no calibration gain | top-1 +3.8 pp (McNemar p≈0.03) but log-loss **worse** (+0.26, p≈2e-5) |

**One line:** *The narrative method delivers a clear, statistically decisive
predictive win on **diagnosis**; on **prognosis** it only ties (a small,
leakage-suspect ranking edge that is cancelled by worse probability calibration).*

---

## Methods compared

**Diagnosis** — given a held-out incident's NARRATIVE, predict its true Zhang-edge
cause/finding label(s):

- **NARRATIVE (ours):** embed the held-out factual narrative → cosine-retrieve
  similar **training** incidents (**self excluded**) → rank causes by
  `P(cause | outcome, narrative-neighbours)` (the `diagnose_retrieval` mechanism,
  isolated).
- **ZHANG baseline:** leave-self-out `P(cause | outcome)` over **all** outcome
  accidents — the population prior, **no narrative conditioning**. This *is* Zhang's
  counting answer; it is wording-invariant and its top-1 is the
  majority-cause-given-outcome.
- **MAJORITY baseline:** global, outcome-agnostic cause frequency — the dumbest floor.

**Prognosis** — given the current event (and the incident's narrative), predict the
NEXT event on each held-out consecutive transition:

- **NARRATIVE (ours):** forward Markov transitions counted **only over the
  narrative-retrieved neighbour incidents** (self excluded).
- **GLOBAL MARKOV baseline:** leave-self-out global transition counts
  (`build_transition_counts` / `transition_step`).

**Scoring (identical for every method).** `p_hit` = probability mass the method puts
on the true label(s); `log-loss = −log(clip(p_hit, 1e-3, 1))`; `Brier = (1−p_hit)²`;
plus top-1, top-3 recall, MRR over the ranked labels. Generic catch-all causes
("Airframe/component/system failure/malfunction", "Miscellaneous/other") are reported
**both included AND excluded** (the "fairer" metric strips them from predictions and
ground truth). **Significance** (narrative vs baseline): McNemar exact on top-1;
Wilcoxon signed-rank + paired bootstrap 95% CI on MRR and log-loss.

**Leakage control.** Diagnosis queries are the **factual** narrative
(`narr_accf`/`narr_accp`) — the sequence-of-events report, not NTSB's probable-cause
prose — i.e. the validated leakage-free **stratum A** (report §14). We also report
the **A-clean** subset (factual text does not echo the cause; containment < 0.7).

---

## TASK 1 — DIAGNOSIS

Mean conditioned neighbourhood pool ≈ **35** outcome accidents. ↑ = higher better,
↓ = lower better. **Bold** = winner.

### Generic-INCLUDED (the number a user sees)

| Variant | n | Method | top-1 ↑ | top-3 ↑ | MRR ↑ | log-loss ↓ | Brier ↓ |
|---|--:|---|--:|--:|--:|--:|--:|
| all factual | 1283 | **Narrative** | **0.481** | **0.652** | **0.587** | **1.655** | **0.431** |
| | | Zhang | 0.425 | 0.565 | 0.525 | 1.884 | 0.523 |
| | | Majority | 0.085 | 0.281 | 0.235 | 4.133 | 0.942 |
| A-clean | 1102 | **Narrative** | **0.489** | **0.663** | **0.595** | **1.623** | **0.418** |
| | | Zhang | 0.436 | 0.577 | 0.536 | 1.819 | 0.507 |

### Generic-EXCLUDED (the "fairer" specific-mechanism metric)

| Variant | n | Method | top-1 ↑ | top-3 ↑ | MRR ↑ | log-loss ↓ | Brier ↓ |
|---|--:|---|--:|--:|--:|--:|--:|
| all factual | 1254 | **Narrative** | **0.469** | **0.625** | **0.570** | **2.611** | **0.696** |
| | | Zhang | 0.404 | 0.552 | 0.505 | 2.930 | 0.760 |
| | | Majority | 0.195 | 0.321 | 0.279 | 4.243 | 0.946 |
| A-clean | 1085 | **Narrative** | **0.477** | **0.636** | **0.578** | **2.577** | **0.691** |
| | | Zhang | 0.418 | 0.566 | 0.518 | 2.861 | 0.753 |

### Significance — Narrative vs Zhang baseline (paired)

| Variant | top-1 (McNemar) | MRR Δ [95% CI], p | log-loss Δ [95% CI], p | Verdict |
|---|---|---|---|---|
| incl, all | b=115 c=43, **p=9.0e-9** | **+0.062** [+0.047,+0.076], p=5e-20 | **−0.229** [−0.289,−0.167], p=5e-67 | **BETTER** |
| incl, A-clean | b=95 c=36, **p=2.6e-7** | **+0.058** [+0.043,+0.074], p=1e-16 | **−0.196** [−0.261,−0.128], p=3e-53 | **BETTER** |
| excl, all | b=120 c=39, **p=8.7e-11** | **+0.065** [+0.050,+0.080], p=2e-19 | **−0.319** [−0.371,−0.265], p=3e-82 | **BETTER** |
| excl, A-clean | b=102 c=37, **p=3.2e-8** | **+0.060** [+0.045,+0.076], p=1e-15 | **−0.285** [−0.342,−0.227], p=1e-66 | **BETTER** |

**Diagnosis verdict: BETTER.** On the held-out per-incident cause-prediction task,
conditioning on the factual narrative beats Zhang's counting prior on **every metric,
in every variant** (generic included/excluded, full and leakage-free A-clean), with
overwhelming significance (all paired CIs exclude 0; McNemar p ≤ 1e-7). The win
**survives leakage control** (A-clean ≈ full A) and is **largest on the fairer
specific-mechanism metric** (top-1 +6.5 pp, log-loss −0.32). Both methods crush the
majority floor. This is a legitimate "better scores than Zhang" result because Zhang's
own descriptive tables make no per-incident prediction — this task is *outside* what
counting can do, and conditioning adds real, calibrated signal (lower log-loss **and**
Brier, i.e. better calibration *and* sharpness).

---

## TASK 2 — PROGNOSIS

Next-event prediction on **496 held-out transitions** (incidents with ≥1 occurrence
transition, a usable narrative, and an index vector). **Coverage 95.0%** — the
narrative neighbourhood had transition data for the current event `a` on 471/496
transitions; the 25 uncovered ones penalise the narrative method (it has nothing to
say). "Covered subset" is the fair "where the narrative can speak" comparison.

| Subset | n | Method | top-1 ↑ | top-3 ↑ | MRR ↑ | log-loss ↓ | Brier ↓ |
|---|--:|---|--:|--:|--:|--:|--:|
| all transitions | 496 | Narrative | **0.464** | **0.631** | **0.552** | 2.731 | **0.536** |
| | | Global Markov | 0.425 | 0.601 | 0.535 | **2.474** | 0.597 |
| covered subset | 471 | Narrative | **0.488** | **0.665** | **0.581** | 2.510 | **0.511** |
| | | Global Markov | 0.435 | 0.616 | 0.548 | **2.354** | 0.589 |

### Significance — Narrative vs Global Markov (paired)

| Subset | top-1 (McNemar) | log-loss Δ [95% CI], p |
|---|---|---|
| all transitions | b=45 c=26, **p=0.032** (narr better) | **+0.258** [+0.091,+0.423], p=1.8e-5 (narr **worse**) |
| covered subset | b=45 c=20, **p=0.0026** (narr better) | +0.156 [−0.005,+0.316], p=1.9e-7 (narr **worse**, median) |

**Prognosis verdict: MARGINAL / TIE.** Narrative-conditioning gives a **small but
significant ranking edge** (top-1 +3.8 pp all / +5.3 pp covered, p ≤ 0.03; top-3 and
MRR also higher; Brier better), but it is **worse-calibrated**: its log-loss is
**significantly higher** than global Markov's because the neighbourhood transition
distributions are sparse and over-peaked, so its confident misses are punished. Brier
favours narrative (sharper, more mass on truth when right) while log-loss favours
global (fewer catastrophic confident misses) — the classic sharpness-vs-calibration
trade-off, with no method dominating. **Net: narrative does not *clearly* beat global
Markov on prognosis — it roughly ties, buying a little top-k accuracy at the cost of
calibration.**

**Honest caveat (why even the small prognosis ranking edge is suspect).** The factual
narrative often *describes the whole event sequence*, so neighbour selection can be
informed by events that occur *after* `a`. That is an indirect leakage channel the
global Markov baseline does not have, which would *inflate* the narrative's prognosis
ranking. Given that, the modest ranking edge should be read conservatively — the
defensible reading is **TIE**.

---

## Synthesis — do we produce better scores than Zhang, and if not, what IS our contribution?

Putting this head-to-head together with the prior findings
(`docs/ZHANG_REPRODUCTION_REPORT.md`, `docs/DIAGNOSIS_VALIDATION_REPORT.md`):

1. **Descriptive parity (not a win, by definition).** We reproduce Zhang's Table 7
   (85/85) and the prior (5.53×10⁻⁷) exactly. Matching a summary statistic is parity;
   it earns the right to extend the method but is not "better."

2. **Diagnosis prediction — a real, decisive win (this report).** On held-out
   per-incident cause prediction, the narrative method beats Zhang's counting prior on
   **every** metric with p ≤ 1e-7, leakage-controlled, and largest on the fairer
   specific-cause metric. This is the cleanest "better scores than Zhang" claim, and
   it is genuinely *outside* what counting offers (Zhang has no per-incident answer).
   It corroborates the §14 query-conditioning validation (top-1 +5.2 pp, MRR +0.052
   vs the unconditioned prior; +8.5 pp / +0.103 vs a concentration-matched random
   pool) using an independent, self-contained scoring harness.

3. **Prognosis prediction — a tie.** Narrative-conditioned transitions add only a
   small (and leakage-suspect) ranking edge and are *worse* calibrated than global
   Markov. Consistent with the **sparse-cell NO-GO** finding: conditioning on a small
   neighbourhood helps where there is dense outcome-cause signal (diagnosis) but not
   where the data is intrinsically sparse (only ~500 incidents have a usable
   transition; legacy sequences truncate at the terminal occurrence).

4. **Capability wins that are not "scores."** Query-conditioning is a *capability*
   Zhang's static tables lack (free-text questions, evidence-filtered cohort
   diagnosis, confidence-aware selective diagnosis, calibrated probabilities); and our
   semantic smoother is a **strict improvement over Zhang's actual Beta-CDF** in the
   ultra-sparse regime (n ≤ 5) where his headline forward "0.95" cells live — though
   it does **not** beat plain counting once n ≳ 10.

### Bottom line for the advisor (4–5 sentences)

We reproduce Zhang's descriptive tables exactly, which is parity, so the only place
"better" can be claimed is a held-out predictive task — and there the answer splits by
direction. On **diagnosis** (predict a held-out incident's cause from its narrative),
the narrative method beats Zhang's counting prior on every metric (top-1 +5.6 pp,
MRR +0.062, log-loss −0.23; all p ≤ 1e-8, leakage-controlled), a clean and defensible
win because counting offers no per-incident answer at all. On **prognosis** (predict
the next event), it only ties: a small, leakage-suspect ranking edge is cancelled by
worse probability calibration, consistent with our sparse-cell findings. So the honest,
publishable claim is **narrative-conditioned diagnosis produces better predictive
scores than Zhang's counting; prognosis does not** — and the broader contribution is a
*capability* layer (free-text/conditional/calibrated diagnosis and a Beta-CDF
replacement for ultra-sparse cells), not a blanket "we beat Zhang everywhere."

---

## Reproducibility

| File | Purpose |
|---|---|
| `tests/headtohead_scores.py` | This head-to-head harness (diagnosis + prognosis; offline) |
| `docs/head_to_head_scores_results.json` | Machine-readable metric tables + significance |
| `docs/qc_embed_vecs.npy`, `docs/qc_embed_keys.json` | Cached factual-narrative query embeddings (diagnosis) |
| `data/processed/embeddings_1982_2006.*` | Retrieval index reused for prognosis narrative vectors |

Run offline: `python3.11 tests/headtohead_scores.py`. Flags: `--task
{diagnosis,prognosis,both}`, `--top-n-incidents` (diagnosis retrieval depth, default
100), `--top-k-neighbors` (prognosis neighbours, default 100), `--embed-missing`
(fill any missing query embeddings via the OpenAI API — needs `OPENAI_API_KEY` +
network), `--limit` (smoke cap). Engine files (`zhang_diagnosis.py`, `prognosis.py`,
`trees.py`, `main_app.py`) are not modified.

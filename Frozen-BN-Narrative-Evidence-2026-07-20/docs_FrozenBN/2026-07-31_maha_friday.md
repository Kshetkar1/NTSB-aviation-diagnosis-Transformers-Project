# Friday in-person meeting with Maha - 2026-07-31

Source of truth for `presentations_FrozenBN/build_maha_jul31_pptx.py`.
Everything in plain words (Maha's instruction: explain every slide yourself,
no AI wording). Numbers are the leak-safe 2026-07-29 set.

## Slide 1 - Title

NTSB Narrative Project - Results & Paper Status.
Kanu Shetkar - Dr. Mahadevan - July 31, 2026 (in person).

## Slide 2 - What you asked for on Tuesday, and what I did

- "Clean up the confusing predictor table" -> every method now has a plain
  English name; one table, one metric, defined on the slide.
- "Explain everything yourself" -> one-page crib sheet written; every
  number on these slides traces to one script in the repo.
- "Everything on the slide you should be able to explain" -> each slide
  footer names the file that generates its numbers.

## Slide 3 - The system in one sentence

A narrative comes in; a reader turns it into facts the network knows
(NTSB's own vocabulary); each fact gets a strength counted from the 100
most similar past accidents; the frozen network (Zhang's, rebuilt and
verified) combines them and returns probabilities.

- Reader = vocabulary match first; LLM only when the text describes
  without naming (and the LLM never supplies a number).
- Strengths = counted fractions from data. Nothing in the chain is trained.
- Network = frozen after being built on 1982-2006 coded data.

## Slide 4 - The bar you set: match Zhang exactly

| Quantity Zhang published | Zhang | Ours | Status |
|---|---|---|---|
| Fire occurrences 1982-2006 | 102 | 102 | exact |
| Prior P(fire) formula | 5.53e-7 | 5.527942e-7 | exact |
| Table 7 cause distribution (published) | 85 rows | 85/85 | exact |

Footer: docs_FrozenBN/ZHANG_REPRODUCTION_REPORT.md

## Slide 5 - Severity on 296 unseen accidents (2007-2019)

Metric on this slide: 4-way exact-match accuracy - the model's top choice
among fatal/serious/minor/none (injury) or destroyed/substantial/minor/none
(damage) equals what NTSB coded.

| Method (plain name) | Injury | Damage |
|---|---|---|
| Always guess the most common outcome | 58.4% | 42.6% |
| Facts parsed from the text -> network | 82.4% | 50.7% |
| Similar-accidents readout -> network (OURS) | 90.9% | 77.4% |
| Trained embedding model (comparison) | 91.6% | 74.0% |
| Trained word-frequency model (comparison) | 92.2% | 73.3% |

- Outcome sentences are stripped from every narrative before ANY use --
  embedding AND parsing -- so the model can't read the answer in the text
  (that fix moved us from 93/81 to the honest 90.9/77.4).
- OURS vs both trained models: statistical ties (all Holm p >= 0.29). They
  edge us on injury, we edge them on damage, and we're clearly stronger on
  severe damage caught (75% vs 68% / 58%) -- with zero training.
- Honest framing: accuracy is a commodity any text model reaches; what the
  network adds is reasoning (joint queries, what-ifs, explanations), not
  accuracy.

Footer: outputs/heldout_significance.md

## Slide 6 - Because you'll ask what accuracy hides

- Class balance: accuracy alone flatters "always none"; we also report
  Macro-F1 (ours 0.470 injury / 0.697 damage vs 0.184 / 0.149 baseline).
- Triage view - "is this severe?": injury sensitivity 93.5%, specificity
  96.8%; damage sensitivity 75.3%, specificity 89.8%.
- Honest misses: fatal (3 cases) and minor injuries (16) never ranked
  top-1 - too rare in this window; stated in the paper.
- Every comparison has an exact McNemar test + bootstrap CI.

## Slide 7 - NEW: diagnosis on unseen accidents (cause categories)

NTSB changed its coding system in 2008, so old and new codes can't be
matched directly. Both eras roll up to the four official top-level cause
categories (Personnel / Aircraft / Environment / Organizational); a
prediction is right if its top category is among the coded causes (n=253).

| Method (plain name) | Top-1 | Rank quality (MRR) |
|---|---|---|
| Always guess the most common category | 45.8% | 0.685 |
| Facts parsed from the text -> network | 57.7% | 0.759 |
| Similar-accidents vote (OURS) | 84.2% | 0.915 |
| Trained model on the same text (comparison) | 88.1% | 0.936 |

- Balanced: 68/62/72% recall on Personnel/Aircraft/Environment.
- Disclosed: trained model is 4 points better (p=0.04) - the price of
  needing no labels; and no method catches the rare Organizational class.

Footer: outputs/diagnosis_heldout_eval.md, outputs/diagnosis_emb_lr.md

## Slide 8 - What the network itself contributes

- The prediction accuracy comes from the narratives (similar-accident
  signal). The network deliberately preserves it (verified by a built-in
  self-test) - it does not add accuracy, and we say so.
- What it adds instead: a verified causal structure (Slide 4), the ability
  to combine several pieces of evidence in one coherent calculation, and
  what-if questions ("same accident but IMC at night?") that a counting
  table cannot answer.
- One-line version: narratives give the signal; the network makes it
  explainable and interrogable.

## Slide 9 - Paper status

- Results section: written, every number regenerates from one command
  (docs_FrozenBN/RESULTS_SECTION.md, REPRODUCE.md).
- Free-parameter inventory (Jesse's "what is trained?"): written -
  frozen / measured / selected, nothing fitted in the primary chain.
- Leakage protocol + two audits: written.
- Remaining for full draft: intro/related-work polish, figures, and your
  pass on the framing of Slide 8.

## Slide 10 - Plan to submission

- This week: full draft assembled from the verified sections.
- You get: draft + reproduction commands + this deck's crib sheet.
- Open decision for you: is the Slide 8 framing (prediction from
  narratives, reasoning from the network) the story you want the paper
  to lead with?

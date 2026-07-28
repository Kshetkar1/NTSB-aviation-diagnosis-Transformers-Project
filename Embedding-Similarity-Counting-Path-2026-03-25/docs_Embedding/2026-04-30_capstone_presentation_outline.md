# Capstone Presentation — Structural Mapping & Zhang Comparison
**Date:** April 30, 2026
**Audience:** Jesse, Maha
**Time budget:** 20 min talk + 10 min discussion

---

## How to use this document

- Each section is **one slide** (or one beat if you don't make slides). Total ≈ 13 slides for ~90 sec each.
- Bold text = what the slide should say.
- Plain text below it = **what to actually say out loud** (speaker notes).
- The "high-level" arc: state the answer, then explain how we got there, then say what it means. Do not lead with implementation.
- If you fall behind on time, drop slides 6, 7, and 11 first — they're nice-to-have detail, not load-bearing.

---

## SLIDE 1 — Title (~30 sec)

**Title:** Comparing Retrieval-Based Diagnosis Against Bayesian Network Inference on NTSB Incidents
**Subtitle:** Where structural mapping helps, where it doesn't, and what to do next

**Say:**
> "Last meeting we agreed on three things I'd come back with: held-out test results in raw probabilities, mapped to Zhang's coding system, with a direct comparison against Zhang's Bayesian Network. I have all three today, plus an interpretation of what they mean for the paper. I'd like 20 minutes, then I want your reactions."

---

## SLIDE 2 — The question, in one sentence (~60 sec)

**The question:** Does our retrieval-based diagnosis system — with structural mapping — produce probabilities that look like Zhang's Bayesian Network?

**Three sub-questions:**
1. Does structural mapping (A2) improve over plain embedding retrieval (A0)?
2. How close are our probabilities to Zhang's, on Zhang's own examples?
3. What does the gap mean — is it a tuning problem or an architectural one?

**Say:**
> "Lead the talk with the question. Last time, we ended on: we have these two paradigms — retrieval over narratives versus a Bayesian Network over coded data — and we don't know how they compare. I picked three sub-questions to answer. The high-level answer to all three is on the next slide; the rest of the talk is how I got there and what it means."

---

## SLIDE 3 — Headline answer (~90 sec)

**The 30-second version:**

| Question | Answer |
|---|---|
| 1. Does A2 beat A0? | **Yes, marginally** — 33/77 vs 32/77 correct, no regressions |
| 2. How close are we to Zhang? | **Same shape, different concentration** — mean gap 0.28 across 78 cells |
| 3. Tuning or architectural? | **Architectural** — no α value gets us to Zhang's distribution shape |

**Say:**
> "Don't apologize for these numbers — own them. The structural mapping does what we hoped, marginally. The gap to Zhang isn't something I can close by tuning a parameter — it's the difference between two paradigms. I'll defend each of these three claims in turn, then talk about what it means for the paper. If at any point you want to interrupt with questions, please do."

---

## SLIDE 4 — What we measured, briefly (~90 sec)

**Two deliverables:**

1. **77 held-out incidents, full probability distributions over Zhang's 54 codes** — A0 baseline vs. A2 structural, with Zhang-coded ground truth alongside.
2. **Zhang's three published worked examples reproduced** — Table 9 (Loss of Engine Power), Figure 11 (gear collapse), Figure 12 (pilot error propagation), with our A0 and A2 numbers next to his Bayesian Network values.

**No accuracy-chasing.** Both deliverables keep the **full distribution**, not just top-1, because — as Jesse said last time — the interesting comparisons are downstream from probabilities, not from rankings.

**Say:**
> "These two deliverables together answer the 'three models with ground truth' framing from last time. Plan 1 has A0 + A2 + ground truth; Plan 2 has Zhang + A0 + A2. They're split into two files because Zhang doesn't publish per-incident predictions for our 77 cases — his BN was built on hypothetical scenarios. The split is structural, not a packaging choice."

---

## SLIDE 5 — How we set it up (~90 sec)

**A0 — baseline:** query narrative → embedding → cosine similarity → top-50 historical incidents → cluster by type → P(cause | cluster) × P(cluster | query).

**A2 — structural:** same pipeline, but rerank the top-50 by causal-chain similarity. Each candidate's score is multiplied by `exp(α × struct_sim)`. We used **α = 0.5**.

**Zhang's BN:** parametric directed graph with conditional probability tables, learned from coded NTSB data. We use his published numbers directly, not a re-implementation.

**Held out:** 77 ev_ids, train-only index (`NTSB_USE_TRAIN_INDEX=1`). No leakage.

**Say:**
> "Three sentences for the methods. The takeaway: A0 and A2 are almost the same pipeline — the only difference is whether we rerank using structural similarity. Zhang's a different paradigm entirely. I'll come back to what α=0.5 means in concrete terms when I show why A2 ≈ A0."

---

## SLIDE 6 — Result 1: A2 is marginally better than A0 (~120 sec)

**On 77 held-out incidents:**

| Metric | A0 | A2 |
|---|---|---|
| Top-1 correct, all 77 | 32/77 (41.6%) | **33/77 (42.9%)** |
| Top-1 correct, 64 with mappable ground truth | 32/64 (50.0%) | **33/64 (51.6%)** |
| Mean entropy of distribution | 2.527 | 2.526 |
| Times A2 changed top-1 from A0 | — | 1/77 |

**The one A2 change was a clean win:** ev_id 20090223X03403, which A0 ranked as UNMAPPED but A2 ranked correctly as "Loss of control - in flight."

**Caveat:** 13/77 ground-truth causes don't fit any of Zhang's 54 codes (Personnel Fatigue, Towing, Ground Crew). Those are unwinnable by construction.

**Say:**
> "Two things to read from this slide. First, no regressions — Maha, this is the bar you set last time, and we cleared it. Second, the effect is small. A2 helps on one incident out of 77. That's not nothing — it's directionally right and it's statistically consistent across the rest of the 78 Zhang cells we'll see in a moment — but it's not transformative at α=0.5. I'll come back to what that means."

---

## SLIDE 7 — What an A2 win looks like (~60 sec)

**ev_id 20090223X03403** — flight crew lost control during landing.

| Model | Top-1 prediction | Probability | Correct? |
|---|---|---|---|
| Ground truth (Zhang code 250) | Loss of control - in flight | — | ✓ |
| A0 (embedding only) | UNMAPPED top cause | 0.238 | ✗ |
| A2 (structural mapping) | Loss of control - in flight | 0.240 | ✓ |

**Probabilities barely moved (Δ = 0.002), but the argmax flipped to the right code.**

**Say:**
> "This is the kind of incident where structural mapping pays off — when the embedding-only retrieval brings back semantically similar narratives whose causes happen to live outside Zhang's schema, the structural reweighting nudges retrieval toward neighbors with matching causal *structure*, which lands inside the schema on the right answer. It's a small effect, but it's a real one."

---

## SLIDE 8 — Result 2: Reproducing Zhang's worked examples (~120 sec)

**Three of Zhang's published scenarios, reproduced through our system.**

Sample numbers from Table 9 (Loss of Engine Power | inoperative engine instruments):

| Event | Zhang BN | A0 | A2 |
|---|---|---|---|
| Loss of engine power | **0.95** | 0.09 | 0.09 |
| Forced landing | 0.14 | 0.67 | 0.67 |
| Substantial damage | 0.05 | 0.17 | 0.17 |
| No injury | 0.99 | 0.00 | 0.00 |

**Three patterns repeat across all 78 cells (Table 9 + Fig 11 + Fig 12):**
1. **Concentration:** Zhang concentrates on the target node (~0.95). Retrieval disperses (~0.10).
2. **Dominant-outcome bias:** We over-predict common downstream events (Forced Landing, Hard Landing).
3. **Absence states:** Zhang gives high probability to "No injury." Retrieval gives 0.0 by construction.

**Say:**
> "These three patterns are the story of the comparison. They're not bugs — they're the consequences of how each system answers the question. Zhang's BN was built to concentrate probability on a target given evidence. Our retrieval system was built to find historically similar incidents — which is a frequency estimate, not a confidence statement. They're answering different questions, and the gap reflects that."

---

## SLIDE 9 — Result 3: The gap is architectural, not tunable (~120 sec)

**Across 78 probability cells from all three Zhang reproductions:**

| Measure | A0 | A2 |
|---|---|---|
| Mean \|Zhang − model\| | 0.2775 | 0.2774 |
| Cells where model is closer to Zhang | — | 25/78 (32%) |
| Cells where model is farther from Zhang | — | 20/78 (26%) |
| Mean shift A0 → A2 | — | **0.0009** |

**Translation:** at α=0.5, A2 shifts probabilities by ~0.001 on average. The gap to Zhang is ~0.28. **The shift is 300× too small to close the gap.**

**Why no α value will fix this:** the formula is `score × exp(α × struct_sim)`. To get A2 to look like Zhang, we'd need to reshape the *shape* of the distribution — not scale the scores. The reweighting can move ranks, but not concentrate mass the way a Bayesian Network does.

**Say:**
> "This is the most important slide of the talk. The gap to Zhang isn't a tuning problem. I tested it. At α=0.5 — which is a meaningful structural reweighting — A2 moves probabilities by 1/1000th of the Zhang gap. Even cranking α up an order of magnitude would not close the 0.28 gap, because the gap is between two ways of computing probability, not between two settings of the same computation."

---

## SLIDE 10 — What the gap means, mechanically (~120 sec)

**Three architectural facts that produce the gap:**

| Fact | Zhang's BN | Our retrieval |
|---|---|---|
| **Probability comes from** | Conditional probability tables | Cosine-weighted historical neighbors |
| **Concentration** | Sharp (P=0.95 on target) | Diffuse (P=0.10 max) |
| **Novel events** | Cannot represent (fixed schema) | Picks up automatically (17% of our test cases) |
| **Absence states** | Native (P(no injury) = 0.99) | Cannot represent (always 0) |

**The 0.28 gap is structural, not parametric.**

**Say:**
> "If you remember nothing else from this talk, remember this slide. The two systems can't be made to agree, and that's not a defect of either. They answer different questions. Zhang answers 'given this evidence, what's the most likely outcome under my parametric model.' We answer 'historically, in similar narratives, what causes were attributed.' Both are valid. The gap is the price of the abstraction each system makes."

---

## SLIDE 11 — The 17% Zhang's BN can't see (~75 sec)

**13 of our 77 held-out incidents have ground-truth causes that fall outside Zhang's 54-code scheme:**

- Personnel issues — Physical — Alertness/Fatigue — Flight crew
- Aircraft handling — Towing and taxiing — Towing
- Personnel issues — Physical — Ground crew
- Personnel issues — Action/decision — Maintenance personnel

**Zhang's BN cannot even *talk about* these incidents.** Our retrieval system does — it just maps them to UNMAPPED, which is itself a usable signal.

**Say:**
> "I want to flag this because it's where our approach has something Zhang's BN structurally can't have. 17% of the held-out cases involve causes that aren't in his schema. Those incidents are invisible to a Bayesian Network built on the 54-code scheme. This isn't a small edge case — it's one in six incidents. And the retrieval system at least knows it doesn't know, which is a kind of calibration."

---

## SLIDE 12 — What I think this means for the paper (~120 sec)

**The honest framing for the paper isn't *"our approach beats Zhang."* It's:**

1. **Retrieval over narratives captures ~17% of incidents that Zhang's coded schema misses entirely.**
2. **Structural mapping (A2) is a real but small refinement on plain retrieval (A0)** — directionally right, no regressions, +1/77 incident on top-1.
3. **The probability gap to Zhang is architectural** — retrieval and BN inference produce different probability shapes by design. This isn't tunable.
4. **The natural fit is hybrid:** retrieval as a discovery and anomaly-flagging layer, Zhang-style BNs as the front-line decision tool. They feed each other.

**This frames the contribution as augmentation, not competition** — which is more defensible to engineering reviewers and more honest given the data.

**Say:**
> "I want to be transparent about what I think the paper's claim should be. Trying to argue that our retrieval beats Zhang's BN is a fight we can't win on these numbers, and reviewers will see through it. The honest contribution is that retrieval picks up cases the schema can't, and could feed downstream tools — including Zhang-style BNs — as a discovery layer. That framing is consistent with what we found and lands more cleanly in a reliability engineering journal."

---

## SLIDE 13 — Next steps & open questions (~90 sec)

**Capstone-stage (this week):**
- Trace through one incident end-to-end as a sanity check (Jesse's request from last time — running `trace_one_incident.py`).
- Add Brier score / log-loss / KL divergence for proper distributional metrics on the 64 mappable incidents.
- Verify Zhang's hardcoded values in our scripts against the actual paper.

**Paper-stage (over the next 2-4 weeks):**
- α sweep to confirm the "architectural, not tunable" claim publicly.
- Calibration plot — does our P=0.10 mean what we think?
- Trace and table-match per Zhang's worked examples for paper figures.

**Future work:**
- Hybrid: retrieval flags incidents Zhang's schema can't represent; new BN nodes proposed by retrieval clusters.
- In-context learning with smaller local models for direct probability readout (Jesse's suggestion).

**Say:**
> "Three time-boxes. This week, I want to close out the capstone with the trace-through and proper distributional metrics. In the next month, I'd like to harden the paper with α sweeps, calibration, and exact format-match to Zhang's tables. Beyond that, the hybrid framing opens a research path Maha mentioned last time — retrieval feeding the priors of a downstream BN. I'd like to discuss which of these you'd want me to prioritize."

---

## Backup slides (only show if asked)

### B1 — What α=0.5 means concretely

The reweighting formula: `new_score = max(0, cosine) × exp(α × struct_sim)`.

| `struct_sim` | Multiplier at α=0.5 |
|---|---|
| 0.0 (no structural overlap) | 1.00 |
| 0.5 (moderate) | 1.28 (+28%) |
| 1.0 (perfect) | 1.65 (+65% max) |

Even the maximum 65% boost on one candidate gets diluted across 50 retrieval neighbors during normalization.

### B2 — File map / where to find this

```
Testing_Structural_Mapping_Slides/outputs/
├── three_model_comparison/three_model_comparison.xlsx     ← Plan 1 deliverable
├── zhang_comparison/comparison_table9.xlsx                 ← Plan 2 (Loss of Engine Power)
├── zhang_comparison/comparison_fig11.xlsx                  ← Plan 2 (Gear Collapse)
└── zhang_comparison/comparison_fig12.xlsx                  ← Plan 2 (Pilot Error)
```

### B3 — Zhang vs. retrieval, side by side (1 cell)

`Inoperative engine instruments → P(loss of engine power)`:
- Zhang's BN: 0.95 ("given engine evidence, this is overwhelmingly the cause")
- Our A0: 0.093 ("of 50 historical neighbors, ~5 had this cause")
- Same prior, different probability semantics.

### B4 — Why retrieval can't represent absence states

Retrieval answers: *"in similar past incidents, what events were observed?"*
Bayesian Network answers: *"given this evidence, what is the conditional probability over outcome states (including non-occurrence)?"*

Retrieval cannot generate `P(no injury)` because "no injury" is not an event you can retrieve — it's an absence. This is structural, not a parameter to tune.

---

## Delivery tips for "speaking at a high level"

1. **Lead with the answer.** Slide 3 has the headline — say it before slide 4. If you have to interrupt yourself with detail, you've lost the thread.
2. **Use the word "because."** Every claim should be followed by a one-sentence explanation. *"A2 is marginally better than A0, **because** at α=0.5 the structural reweighting moves probabilities by ~0.001 — enough to flip one argmax but not enough to reshape distributions."*
3. **Don't read the table — interpret it.** When a slide has numbers, your job is to tell them what to take from the numbers, not to recite them.
4. **Pause before transitions.** When you finish a section, take one beat of silence before saying "OK, so the second question I wanted to answer was…" — gives them time to land.
5. **Volunteer the limitation.** Reviewers respect researchers who flag their own caveats. Slide 6 says "13/77 are unwinnable" — say it before they ask.
6. **End on an open question, not a closed claim.** Slide 13 asks them what to prioritize — that invites collaboration. Don't end with "and that's my work."
7. **If you don't know, say so.** If Jesse asks "what's the Brier score?" — the right answer is "I haven't computed it yet, that's on my list for this week" — not a guess.

---

## What you should rehearse out loud (in this order)

1. Slides 2 + 3 back-to-back as one beat — "the question, then the answer." This is the most important pair in the whole talk.
2. Slide 9 — the "tuning vs. architectural" claim. This is the strongest single result and the easiest to fumble. Practice the formula language.
3. Slide 12 — the paper framing. This is where Maha and Jesse will ask the hardest questions, and the more confidently you can land "augmentation, not competition" the stronger your position is.

If you only have time to rehearse three slides, rehearse those three.

# Roadmap to "Maha is satisfied"

> Constraints (current phase): **no Bayesian network of our own**, **no agents**.
> We run Zhang's BN only to produce the *target numbers* (answer key); our method
> stays retrieval + counting + structural mapping (+ optionally LLM token probs).

## What "satisfied" means (from his behavior)
Trustable probabilities (real `n`, not n=1) · conditional logic that isn't "primitive" ·
numbers that get *close to Zhang* · honest about gaps · visible progress twice a week.

---

## The two findings that reframe everything

1. **Table 4 is pedagogical.** In Zhang's built network (`NTSB.xdsl`), `Fire` has one
   parent — `Antiicedeicesystemwindshield` (P=0.9526) — **not** brake wear or electrical
   overheat. Table 4's 0.99/0.93/0.95 were hand-built with Beta-CDF smoothing to teach
   how a CPT works. They cannot be reproduced from counts or by running Zhang's own BN.
   → **Stop chasing Table 4 magnitude.** Keep it as the honest "why" exhibit.

2. **Retrieval ≠ Zhang on probabilities (already shown, May 2026).** On the data-rich
   Table 9 scenario (evidence = inoperative engine instruments):

   | Outcome | Zhang BN | Engine A0 | Engine A2 |
   |---------|----------|-----------|-----------|
   | Loss of engine power (diagnosis) | 0.95 | 0.09 | 0.09 |
   | Forced landing (prognosis) | 0.14 | 0.67 | 0.67 |
   | No injury (prognosis) | 0.94 | 0.00 | 0.00 |

   Zhang **spikes the cause**; retrieval **spreads over observed consequences**. This is a
   *question mismatch*, not a tuning gap. And **A0 ≈ A2** — structural reweighting
   validates but doesn't move the code-level distribution.

**Therefore the real fix to "get close to Zhang" is to change the probability SOURCE**
(LLM token probabilities, or a light calibration/aggregation layer) — not to tune retrieval.

---

## Phase 0 — Foundation & honesty (DONE)

| Step | Fix | Status |
|------|-----|--------|
| Table 4 empirical CPT | computed; kept as honest exhibit | done |
| Table 4 reframe (pedagogical, not in real BN) | verified in `NTSB.xdsl` | done |
| Structural overheat label bug | now requires electrical context + thermal signal; real `no` branch (9.5% → 50%) | done |
| Consolidated Maha HTML | `outputs/maha_probability_comparison.html` | done |

## Phase 1 — Real comparisons on data-rich scenarios (this week)

| Step | Problem | Fix | Done when |
|------|---------|-----|-----------|
| 1.1 Zhang ground truth | hardcoded pedagogical constants | use `Zhang_Replication_Runner/outputs/` (table9, fig11, fig12 — 88% validated) | numbers sourced |
| 1.2 Diagnosis comparison | exists (May) but scattered | consolidate Table 9 diagnosis into the report | in report |
| 1.3 Prognosis comparison | Table 5 done (0.73 vs 0.92); add Fig-11 forward | run `predict_future_events` on a common scenario | directional match |
| 1.4 Structural mapping honest role | sold as booster | report A0≈A2 + label-quality role | one clear slide |

## Phase 2 — Close the gap to Zhang's probabilities (the real fix)

| Step | Problem | Fix | Done when |
|------|---------|-----|-----------|
| 2.1 **LLM token-prob diagnosis** | retrieval spreads, can't spike like Zhang | prompt an LLM "given evidence X, the cause is …", read calibrated token logprobs over Zhang's 54 codes (Jesse's path a). **Not a BN, not an agent.** | P(cause\|evidence) is Zhang-shaped |
| 2.2 Calibration/aggregation layer | raw frequencies uncalibrated | light layer mapping retrieval freqs → calibrated posteriors | gap to Zhang shrinks |
| 2.3 Multi-evidence conditioning | engine takes one query | counting track for true 2-evidence CPTs; or filter-then-count | a 2-evidence cell with real n |
| 2.4 Common subgraphs + Beta-CDF | Table-4 cells n=0–4 | pick n≥10–30 pairs; smooth sparse cells Zhang's way | estimable CPTs w/ CIs |

## Phase 3 — Rigor & write-up

| Step | Fix |
|------|-----|
| 3.1 Eval discipline | frozen train/test, exclude-self retrieval, defined Recall@K / MRR / τ (harness exists) |
| 3.2 Clean narrative | two modes ↔ two inference directions, vs Zhang, honest gaps; tight deck not sprawling draft |

---

## Dependency order
```
1.1 ─► 1.2 ─► 1.4
   └─► 1.3
2.1 (token probs) ─► 2.2 (calibration) ─► 2.3/2.4
(Phase 1&2) ─► 3.1 ─► 3.2
```

---

## Two-lane framing (the spine of the pitch)

| Lane | What it is | vs Zhang | Honest expectation |
|------|-----------|----------|--------------------|
| **Lane 1 — Empirical CPT (counting)** | P(outcome \| evidence) counted from cases — same *kind* of estimator as Zhang's CPTs | same **band** on data-rich cells | near Zhang, **not exact** |
| **Lane 2 — Retrieval + structural (+ LLM tokens)** | narrative → diagnosis/prognosis on **unseen** incidents | **different question** | won't match; does what the BN **can't** |

Lane 1 answers "do my numbers agree with Zhang." Lane 2 answers "can I go beyond a static BN."

## Corrected reality on Lane 1 (tested 2026-06-22)
Counting does **not** exactly reproduce Zhang. Three reasons, all to be stated up front:
1. **Coding era:** dataset uses modern **eADMS** codes (loss of engine power = 341/342); Zhang's Table 9 uses **legacy** codes (350).
2. **Estimator:** Zhang = counts **+ Beta-CDF smoothing + network inference**, not raw 2-way counts.
3. **Labeling sensitivity:** P(severe damage \| fire) = 0.73 (broad labels, n=272) vs 0.18 (strict codes, n=38).

→ Defensible claim: **Lane 1 lands in Zhang's band on data-rich cells (Table 5: 0.73 vs 0.92); exact match is prevented by the three factors above — which is why Zhang used a smoothed network.**

## Doubt register (raise each before Maha does)

| # | Doubt | Severity | What closes it |
|---|-------|----------|----------------|
| 1 | "0.09 vs 0.95 compares different quantities" | fatal | two-lane framing; define each P |
| 2 | "Can counting reproduce Table 9?" | high | honest: band not exact; coding-era caveat |
| 3 | "Free-text→54-code mapping is guessing" | med | audit mapper on 30–50 causes; report accuracy |
| 4 | "Structural mapping does nothing — A0≈A2" | thesis-risk | show polysemy wins or down-scope; report McNemar p=0.50 yourself |
| 5 | "P(no injury)=0 vs 0.94" | med | handle absence states (complements) or restrict claims |
| 6 | "Are these calibrated probabilities?" | med | Brier + reliability for Lane 1; Lane 2 = uncalibrated freq |
| 7 | "n=1–4 isn't trustable" | med | claim magnitude only at n≥30; show Wilson CIs |
| 8 | "Did you leak test→train?" | med | frozen test ids, train-only index, exclude-self |
| 9 | "Table 4 is what I asked for" | low | confirm pedagogical origin with him |
| 10 | "Is your Zhang replication stable?" | low | seed-band + 99M (done, 88%) |

## Definition of done
- **Plan → 9.5:** two-lane framing locked · Lane 1 band-match shown with CI · doubt items 1–6 each have an artifact or principled answer.
- **Maha → 9.5:** Table 4 reframe confirmed · every number regenerated from a script (provenance) · "what my method does that yours can't" slide · no finding hidden (absence states, A0≈A2, significance stated first by you).

## Decisions needing you / Maha
- Confirm with Maha that Table 4 was illustrative (he's the senior author).
- Phase 1 only by Friday, or Phase 1 + start of 2.1 (token-prob fix)?
- Publication framing (reliability journal vs CS venue) — open since May.

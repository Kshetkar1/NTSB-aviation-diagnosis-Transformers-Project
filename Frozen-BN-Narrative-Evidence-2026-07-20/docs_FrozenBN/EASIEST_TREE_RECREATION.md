# Easiest Tree Recreation — Zhang (2021) vs Our Tree Code

**Question:** Does our current tree code (`trees.py`, `zhang_diagnosis.py`, `prognosis.py`)
recreate the single easiest diagnosis and prognosis examples from Zhang & Mahadevan,
*Bayesian network modeling of accident investigation reports for aviation safety
assessment*, **Reliability Engineering and System Safety 209 (2021) 107371**?

- **Dataset / index:** Zhang comparison window 1982–2006
  (`data/processed/refined_dataset_1982_2006.json`), the apples-to-apples corpus.
- **How reproduced:** ran the existing code unmodified via
  `tests/recreate_easiest_examples.py` (framework Python 3.11). No engine files edited.

---

## 1. Chosen examples (quotes + page numbers)

### Diagnosis — easiest = the fire cause distribution, Zhang **Table 7 (p. 12)**
> Caption: *"The contributory factors to fire occurrence and the corresponding
> conditional probabilities."* Probabilities are **P(cause | fire) = count / 102**
> (102 = number of fire accidents).

The anchor is also stated explicitly in the text, **Eq. (9), p. 8**:
> *P(ω = fire | e = airframe component malfunction) = 32/102 = 31.37×10⁻².*

Comparison targets (Table 7, p. 12):

| Cause | Zhang P(cause\|fire) | Zhang n |
|---|--:|--:|
| Airframe/component/system failure/malfunction | 0.31372 | 32 |
| Loss of engine power (total) – mechanical failure/malfunction | 0.08823 | 9 |
| Electrical system, electric wiring | 0.08823 | 9 |
| Fluid, fuel | 0.05882 | 6 |
| Auxiliary power unit (APU) | 0.04901 | 5 |

### Prognosis — easiest single-hop = loss of engine power → forced landing
From the **Loss of engine power** scenario, **§5.4 (p. 16)**:
> *"the loss of engine power leads to forced landing, and the forced landing results
> in the collapse of main gear …"*

Quantified in **Table 9 (p. 17)**, column **Loss of engine power**:
> Forced landing = **14.29×10⁻² = 0.1429**.

This is the simplest single forward hop in the paper (one event → the next event).

---

## 2. Side-by-side: Zhang vs ours

### 2A. Diagnosis — clean cross-check (population-level, Zhang's exact method)
`zhang_diagnosis.empirical_cause_distribution("fire", cause_factor_only=True)` —
denominator = **102** fire accidents, 85 causes.

| Cause | Zhang P | Zhang n | Ours P | Ours n | Verdict |
|---|--:|--:|--:|--:|:--:|
| Airframe/component/system failure/malfunction | 0.31372 | 32 | 0.31373 | 32 | **exact** |
| Loss of engine power (total) – mechanical | 0.08823 | 9 | 0.08824 | 9 | **exact** |
| Electrical system, electric wiring | 0.08823 | 9 | 0.08824 | 9 | **exact** |
| Fluid, fuel | 0.05882 | 6 | 0.05882 | 6 | **exact** |
| Auxiliary power unit (APU) | 0.04901 | 5 | 0.04902 | 5 | **exact** |

(Differences are pure 5th-decimal rounding; the full 85/85 match is documented in
`docs/TABLE7_FULL_REPRODUCTION.md`.)

### 2B. Diagnosis — the actual TREE (`trees.build_diagnosis_tree`)
Query `"engine caught fire during takeoff"`, level-1 edges = P(cause | fire) over the
**query-retrieved pool**. Pool had **69** fire accidents (not the full 102), and the
tree uses the *default* counting (every finding on the fire occurrence, **no**
Cause/Factor contributory filter).

| Cause | Zhang P (n=102) | Tree P (n=69 pool) | Tree n | Verdict |
|---|--:|--:|--:|:--:|
| Airframe/component/system failure/malfunction | 0.31372 (32) | 0.28986 | 20/69 | **close**, stays #1 |
| Loss of engine power (total) – mechanical | 0.08823 (9) | 0.11594 | 8/69 | close |
| Fluid, fuel | 0.05882 (6) | 0.08696 | 6/69 | close |
| Auxiliary power unit (APU) | 0.04901 (5) | 0.07246 | 5/69 | close |
| Electrical system, electric wiring | 0.08823 (9) | — | — | dropped below top-8 |

Tree top-6 (pool-restricted): Airframe 0.290, **Emergency procedure 0.145**, LOEP-total
0.116, **Evacuation 0.087**, Fluid-fuel 0.087, APU 0.072. The injected non-cause
descriptive findings (*Emergency procedure*, *Evacuation*) are exactly the labels the
Cause/Factor filter removes in the faithful reproduction.

### 2C. Prognosis — the TREE (`trees.build_prognosis_tree`)
Query `"loss of engine power"`. Seed resolved to the most-connected family member
`loss of engine power (total) – mechanical failure/malfunction`. Edge =
P(next | current) as a first-order Markov hop over consecutive occurrence pairs.

| Hop | Zhang Table 9 | Our tree (query pool) | Our tree (global) | Verdict |
|---|--:|--:|--:|:--:|
| Loss of engine power → forced landing | 0.1429 | 0.1905 (n=4/21) | 0.1818 (n=4/22) | **close, differs in kind** |

Our top next-events from the seed (global): fire 0.409, airframe 0.182, **forced
landing 0.182**, miscellaneous 0.091, …

---

## 3. Verdict (per example)

- **Diagnosis (Table 7) — EXACT** via the clean cross-check. Our population method
  `empirical_cause_distribution(..., cause_factor_only=True)` reproduces every Table 7
  anchor (and all 85 causes) to the 5th decimal: airframe 0.31373 (n=32) vs Zhang
  0.31372 (n=32), etc.

- **Diagnosis TREE — CLOSE (ranking), DIFFERS in magnitude (and why).** The tree keeps
  Airframe as the #1 fire cause and surfaces the same major mechanisms, but the
  probabilities shift for two honest, expected reasons: (1) the tree is **query-first**,
  so it conditions on the retrieved pool (denominator 69 fire accidents, not 102), and
  (2) the tree path uses **default counting without Zhang's Cause/Factor contributory
  filter**, so descriptive findings (*Emergency procedure*, *Evacuation*) enter and
  push specific causes around. Apply the filter / full population and it collapses to
  the exact Table 7 numbers (§2A).

- **Prognosis (Table 9) — CLOSE in value, DIFFERS in kind (and why).** Our Markov hop
  P(forced landing | loss of engine power) ≈ **0.18–0.19** lands in the same
  neighborhood as Zhang's **0.1429**, and forced landing correctly appears as a leading
  next-event. But they are **not the same quantity**: ours is an empirical first-order
  transition over consecutive occurrence pairs (n=4 supporting incidents), whereas
  Zhang's 0.1429 is a **full Bayesian-network posterior** (his Beta-CDF conditional
  function calibrated on single-event probabilities, propagated through the forced-landing
  node with all its parents). Same direction, similar size, different machinery — so
  "close" rather than "exact" is the honest call.

> Note: Zhang's other single-hop prognosis cell, P(loss of engine power | inoperative
> engine instruments) = 0.95 (a raw ratio of 1.0 capped at 0.95), is **not directly
> recreatable by label** because "inoperative engine instruments" is not a verbatim
> node label in our occurrence/finding vocabulary (the lookup returns no node). That is
> why the loss-of-engine-power → forced-landing hop was chosen as the easiest prognosis
> example instead.

---

## 4. Plain-English summary for the advisor

> Our code reproduces Zhang's easiest diagnosis example **exactly**: the population-level
> P(cause | fire) from Table 7 matches the paper to the fifth decimal (e.g. the airframe
> anchor 32/102 = 0.314). Our branching *diagnosis tree* keeps the same top cause and the
> same major fire mechanisms, but its magnitudes shift slightly because it answers a
> **query-specific** version (it conditions on the retrieved fire accidents and doesn't
> apply Zhang's "contributory-factor only" filter). For prognosis, our forward escalation
> "loss of engine power → forced landing" comes out at ~0.18–0.19, close to Zhang's 0.143,
> but it's a simple data-frequency transition rather than Zhang's full Bayesian-network
> posterior — same story, slightly different number, by design.

---

*Generated by `tests/recreate_easiest_examples.py` (read-only run; no engine files modified).*

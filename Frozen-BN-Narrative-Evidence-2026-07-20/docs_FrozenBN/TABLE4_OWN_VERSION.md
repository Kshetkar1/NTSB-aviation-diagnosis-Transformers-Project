# Our Real, Data-Derived "Table 4" — P(fire | wiring, fuel)

**Companion to** `docs/TABLE4_ANALYSIS.md` (which argues Zhang's Table 4 is
illustrative). This document settles the question *concretely*: it builds a real
analogue of Zhang's Table 4 from the NTSB data, computed both with **our**
empirical counting and with **Zhang's own recreated estimator**, and shows
exactly which of Table 4's values (0.99 / 0.93 / 0.95 / 2e-9) the real estimator
can and cannot reach.

- **Script:** `tests/build_table4_analogue.py` (runnable, imports the engines
  read-only; edits nothing in `zhang_diagnosis.py` / `prognosis.py` /
  `sparse_cpt.py`).
- **Data:** `data/processed/refined_dataset_1982_2006.json` (1,742 incidents;
  102 fire accidents — reproduces Zhang's `T(fire)=102`).
- **Run:**
  `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 tests/build_table4_analogue.py`
  (add `--semantic` for the optional narrative-conditioned cell; needs network).

---

## 1. The two real parent causes (analogues of Zhang's toy x1 / x2)

Zhang's toy Table 4 uses `x1` = landing-gear brake wear and `x2` = electrical
wiring overheating as the two parents of `x3` = fire. We pick two **real**
fire-related contributory causes from the data:

| | Our parent | NTSB label(s) | Zhang toy analogue |
|---|---|---|---|
| **p1** | electrical wiring | `electrical system, electric wiring` (single label) | x2 (wiring overheating) |
| **p2** | fuel system / fuel | any contributory finding whose label mentions `fuel` (family: `fluid, fuel`; `fuel system, line`; `…, drain`; `…, nozzle`; etc.) | x1 (a real fire cause) |

"Present" uses the **`cause_factor_only` contributory convention** (only
findings flagged Cause `C` / Factor `F` count; reuses
`zhang_diagnosis._is_contributory` / `_faithful_finding_label`). Fire = the
`fire` occurrence is present in the incident.

**Why this pair.** It is the intuitive "wiring + fuel" pair *and* the only
fire-relevant pair whose four parent combinations are all populated — in
particular it is the only pair with **any** both-present incidents. Joint support
over all 1,742 incidents:

| cell (p1=wiring, p2=fuel) | incidents (denom) | of which fire (n) |
|---|---|---|
| both present | **1** | 1 |
| wiring only | 21 | 10 |
| fuel only | 45 | 23 |
| neither | 1,675 | 68 |

The both-present cell is a single incident (`20001213X25244`, which did catch
fire) — i.e. genuinely sparse, exactly the regime where smoothing / caps matter.

---

## 2. The 4-cell CPT, computed three ways

### (a) Our direct empirical counting — `P(fire | p1, p2) = n / denom`

| cell | n / denom | raw | Beta-CDF smoothed | Zhang 0.95 cap |
|---|---|---|---|---|
| wiring **Y**, fuel **Y** | 1/1 | **1.0000** | 1.0000 | **0.9500** |
| wiring **Y**, fuel **N** | 10/21 | **0.4762** | 0.7169 | 0.4762 |
| wiring **N**, fuel **Y** | 23/45 | **0.5111** | 0.7534 | 0.5111 |
| wiring **N**, fuel **N** | 68/1675 | **0.0406** | 0.0710 | 0.0406 |

Beta-CDF smoothing uses Zhang's global `ALPHA=1.04645351`, `BETA=2.02591394`
(`sparse_cpt`). Note `beta.cdf(1.0)=1.0`, so Beta-CDF does **not** tame the 1/1
both-cell — only Zhang's `if ratio==1.0: ×0.95` cap does, giving **0.95**.

### (b) Zhang's recreated estimator (Eqs. 8–11, `prognosis`)

Single-cause ratios via Eq. 9 (`zhang_baseline_cpt`, graph edge ratio
`|cause→fire| / |cause as from-or-to|`); both-present via his multi-parent
Beta-CDF rule floored by the max single cause (`zhang_baseline_multiparent`):

- wiring single-cause ratio = **11/28 = 0.3929**
- fuel-family single-cause ratio = **18/51 = 0.3529**
- both-present: contribution `(0.393+0.353)/37.56 = 0.0193`, `beta.cdf = 0.0330`,
  floored by `max(singles) = 0.3929` → **0.3929**

| cell | Zhang-recreated value | how |
|---|---|---|
| wiring **Y**, fuel **Y** | **0.3929** | Beta-CDF floored by max single |
| wiring **Y**, fuel **N** | **0.3929** | single (Eq. 9) |
| wiring **N**, fuel **Y** | **0.3529** | single (Eq. 9, fuel family) |
| wiring **N**, fuel **N** | **0.0000** | no active parent (= 0 in `constructCPT`) |

For context, the **largest** single-cause fire ratio Zhang's estimator produces
on real data is `airframe = 32/102 = 0.314` for the "all fire" node, and at most
**0.95** for sparse `1/1` cells *because of the cap* (e.g. `fuel system, primer
system` = 1/1 → 0.95). It never produces a genuine rate near 0.93 or 0.99.

### (c) Our narrative-conditioned estimate for the both-present cell (optional)

Among the 50 incidents most semantically similar to *"electrical wiring fire and
fuel system leak"*, **31/50 = 0.62** escalate to fire
(`prognosis.semantic_forward_cpt`). A neighbour-smoothed read of the sparse
both-cell: high (~0.6), but still nowhere near 0.99.

---

## 3. Side-by-side vs Zhang's illustrative Table 4

| cell | **Zhang T4 (illustrative)** | ours raw | ours 0.95-cap | Zhang-est (recreated) | ours narrative |
|---|---|---|---|---|---|
| wiring **Y**, fuel **Y** | **0.99** | 1.0000 | 0.9500 | 0.3929 | 0.62 |
| wiring **Y**, fuel **N** | **0.93** | 0.4762 | 0.4762 | 0.3929 | — |
| wiring **N**, fuel **Y** | **0.95** | 0.5111 | 0.5111 | 0.3529 | — |
| wiring **N**, fuel **N** | **2e-9** | 0.0406 | 0.0406 | 0.0000 | — |

---

## 4. Reachability: can Zhang's real estimator produce 0.99 / 0.93 / 0.95 / 2e-9?

| Table 4 value | Reachable by Zhang's real estimator? | Why |
|---|---|---|
| **0.99** (both) | **NO** | His hard ceiling is the `×0.95` cap; the multi-cause rule floors at the max single cause (~0.39 here). Even our empirical 1/1 both-cell, capped, is 0.95 — strictly below 0.99. Nothing in the method can emit 0.99. |
| **0.93** (wiring only) | **NO** | The real wiring single-cause ratio is ~0.39 (Zhang-est) / 0.48 (our empirical). 0.93 is not a ratio of the relevant counts. |
| **0.95** (fuel only) | **ONLY as a sparse-cell artifact** | 0.95 appears in the data *exclusively* as the value of a `1/1` cell after the cap — never as a genuine high rate. Our both-cell demonstrates this exactly: 1/1 → capped → **0.95**. A real "fuel only" cell is ~0.35–0.51, not 0.95. |
| **2e-9** (neither) | **NO** | Not a count ratio at all (it's a hand-picked near-zero). The real neither-cell is ~0.04 empirically and exactly 0 in Zhang's `constructCPT` (empty parent scheme). |

So **at most one** of Table 4's four numbers (the 0.95) is even *expressible* by
the estimator, and only as a degenerate single-incident cap — not as a real
probability. The headline **0.99** and the two other values are unreachable.

---

## 5. Verdict

**Confirmed: Zhang's Table 4 is illustrative, not a quantity his estimator
produces.** With two genuine fire causes (wiring + fuel) and his own recreated
estimator on the real 1982–2006 data, the CPT is approximately
**(0.39, 0.39, 0.35, 0.00)**, and our direct empirical counting gives
**(1.00→cap 0.95, 0.48, 0.51, 0.04)** — neither resembles Table 4's
**(0.99, 0.93, 0.95, 2e-9)**. The single coincidence (a 0.95) is precisely the
fingerprint of a `1/1` sparse cell hitting Zhang's cap, which is the *opposite*
of a robust estimate. This is consistent with, and quantitatively reinforces,
`docs/TABLE4_ANALYSIS.md`: Table 4's round numbers are pedagogical inputs to the
Fig. 2 / Fig. 3 belief-updating demo, not outputs of Section 4.3's method.

**The defensible artifact to show the advisor** is the table in §3, columns
"ours raw / ours 0.95-cap" and "Zhang-est" — a real, populated `P(fire | wiring,
fuel)` CPT with explicit `n/denom`, which is the genuine version of what Table 4
only *illustrates*.

---

## 6. A 3–4 sentence script for the advisor

> I built a real, data-derived version of Zhang's Table 4: the CPT
> `P(fire | electrical-wiring, fuel-system)` for all four present/absent
> combinations, computed both by our direct counting and by Zhang's own
> recreated estimator on the 1982–2006 data (102 fire accidents, reproducing his
> `T(fire)=102`). The real CPT comes out around **(0.39, 0.39, 0.35, 0.00)** with
> Zhang's estimator and **(0.95 [a capped 1/1 cell], 0.48, 0.51, 0.04)** with our
> empirical counts — nowhere near Table 4's **0.99 / 0.93 / 0.95 / 2e-9**. His
> estimator literally *cannot* produce 0.99: it caps single cells at 0.95 and
> floors the both-parent cell at the max single-cause value (~0.39), so 0.99/0.93
> are unreachable and the one 0.95 only ever appears as a single-incident
> sparse-cell artifact. So Table 4 is an illustrative teaching table (Section 3,
> Fig. 2 toy network), and the populated CPT above is the real, defensible thing
> we can actually report.

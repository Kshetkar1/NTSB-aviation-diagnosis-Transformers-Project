# Structural uncertainty in the frozen NTSB network — measured with Zhang's own CPT rules

Data, node set, CPT rule and query are held fixed. **Only the parent-cap constant varies.**

Source: `frozen_bn.json` (785 nodes / 1,438 arcs), `accident_variable_matrix.csv`
(1,742 accidents × 785 variables). Target: `personnel injury`.

## The rules used — both exact

| node type | rule | source |
|---|---|---|
| severity (4-state) | support-weighted mixture of per-parent empirical severity distributions; **all mass on "no injury" when no parent is active** | `bn_upgraded.py`, FIX 2 |
| Boolean event nodes | Beta-CDF, per-child Nelder–Mead calibration (x0 = [2,1]), floored by max active ratio | `bn_build_ours.py`, Zhang Eqs 10–14 |

Both rules' parameters were **recovered exactly from the shipped CPTs**, not re-estimated:

- a single-active cell of a severity node *is* that parent's empirical severity distribution
- a single-active cell of a Boolean node *is* that parent's edge ratio
- two-active cells give the support weights by least squares

Recovered severity parent supports (relative, ranked — this is the selection order):

```
1.000  miscellaneous/other                              0.245  fire
0.866  dragged wing, rotor, pod, float or tail/skid      0.227  near collision between aircraft
0.736  airframe/component/system failure/malfunction     0.178  in flight collision with object
0.684  on ground/water collision with terrain/water      0.164  in flight encounter with weather
0.301  on ground/water collision with object             0.138  in flight collision with terrain/water
                                                         0.123  hard landing
                                                         0.115  loss of engine power (total) — mechanical
```

---

## Result 1 — severity node, exact mixture rule

| cap | mean P(fatal) | distinct values produced |
|----:|--------------:|-------------------------:|
| 2  | 0.0025 | 4 |
| 4  | 0.0109 | 10 |
| 6  | 0.0306 | 21 |
| 8  | 0.0306 | 25 |
| 10 | 0.0394 | 41 |
| **12 (shipped)** | **0.0388** | **51** |

**Three real accidents:**

| ev_id | actual | k=2 | k=4 | k=6 | k=8 | k=10 | k=12 |
|---|---|---|---|---|---|---|---|
| 20001208X05743 | fatal | 0.000 | 0.000 | 0.530 | 0.530 | 0.388 | **0.388** |
| 20001213X32505 | fatal | 0.000 | 0.000 | 0.000 | 0.000 | 0.135 | **0.135** |
| 20001207X03623 | serious | 0.000 | 0.035 | 0.159 | 0.159 | 0.159 | **0.142** |

Across all 1,742 accidents:

- largest movement: **0.530**
- **759 / 1,742 (44%)** — P(fatal) at least **doubles** between lowest and highest cap
- 165 move by more than 0.05; 136 move by more than 0.10
- movement is **not monotonic** — more parents is not "better," it is sparser

**The sharpest point:** at low caps the rule puts *all* mass on "no injury" whenever
none of the retained parents is active. So for accidents that were actually fatal,
the model at cap 4 reports **P(fatal) = 0** — not a low number, a certainty. The cap
decides whether the model is confidently wrong.

## Result 2 — Boolean node, exact Beta-CDF rule

Target `airframe/component/system failure/malfunction` (12 parents):

| cap | 2 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|
| mean P(Yes) | 0.0038 | 0.0082 | 0.0131 | 0.0164 | 0.0240 | 0.0311 |

Mean probability rises **8×** across the cap range. Largest single-accident
movement **0.968**; 50 accidents move by more than 0.10.

So the effect is not an artifact of the severity node's mixture rule — it appears
under Zhang's Beta-CDF construction too.

---

## What this establishes

1. Structural uncertainty is real here and large, under **Zhang's own CPT rules**.
2. The cap of 12 is unjustified, and the answer depends on it materially.
3. At low caps the model reports certainty ("no injury", probability 1) for
   accidents that were fatal. The construction choice controls that.
4. **No LLM is involved.** This is the reference recipe on the reference dataset.

## What is not established

- These vary *construction* choices, not *narrative reading*. Reading variance is
  the next layer and is not measured here.
- Query is a direct CPT lookup on the target given its parents, not full-network
  propagation. Propagated effects would differ in size, not in kind.
- Parent activity is taken from the accident × variable matrix (did the occurrence
  happen), not from evidence propagated through the network.

## To redo the bootstrap under the exact rule

The resampling experiment needs each accident's **last occurrence** label and its
Zhang injury label — the severity rule is built on `support[last_occurrence]`, and
the exported matrix only records presence. A three-column CSV
(`ev_id, last_occurrence, injury_label`) would be enough.

## Reproduce

`exp_exact.py` — both experiments. Reads only the export bundle.

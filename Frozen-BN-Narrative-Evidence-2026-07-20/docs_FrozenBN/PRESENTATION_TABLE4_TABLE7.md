# Presentation Tables — Table 4 Analogue & Table 7 Reproduction

Copy these into your Tuesday slides. All numbers from validated scripts on
`refined_dataset_1982_2006.json` (102 fire accidents, 1982–2006 window).

---

## Table 4 analogue — P(fire | electrical wiring, fuel system)

**What this shows:** Zhang's published Table 4 uses hand-picked teaching values
(0.99, 0.93, 0.95). Our table uses **real NTSB data** with the same 2-parent CPT
layout. Parent 1 = electrical wiring; Parent 2 = any contributory finding
mentioning fuel.

| Wiring present? | Fuel present? | Count (fires / cell) | **Zhang Table 4** (paper) | **Our raw count** | **Our Beta-CDF** | **Zhang estimator** (recreated) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Yes | Yes | 1/1 | 0.99 | 1.0000 | 1.0000 | 0.3929 |
| Yes | No | 10/21 | 0.93 | 0.4762 | 0.7169 | 0.3929 |
| No | Yes | 23/45 | 0.95 | 0.5111 | 0.7534 | 0.3529 |
| No | No | 68/1675 | ≈0 | 0.0406 | 0.0710 | 0.0000 |

**Key line for Maha:** Zhang's 0.99 / 0.93 / 0.95 do **not** come from counting
or from his Beta-CDF on real data. Our both-present cell is literally **1 fire out
of 1 incident** (raw 100%, capped at 0.95). His recreated estimator gives **0.39**.

---

## Table 7 — P(cause | fire) — top causes (denominator = 102)

**What this shows:** Standard diagnosis — given a fire, what caused it? We reproduce
Zhang's Table 7 exactly: **85/85 causes match** (±0.0005 rounding).

| Rank | Cause | Count | **Zhang P** | **Our P** | Match? |
|:-:|---|:-:|:-:|:-:|:-:|
| 1 | Airframe/component/system failure/malfunction | 32/102 | 0.31372 | 0.31373 | ✅ |
| 2 | Loss of engine power (total) - mechanical failure/ma... | 9/102 | 0.08823 | 0.08824 | ✅ |
| 3 | Electrical system, electric wiring | 9/102 | 0.08823 | 0.08824 | ✅ |
| 4 | Fluid, fuel | 6/102 | 0.05882 | 0.05882 | ✅ |
| 5 | Auxiliary power unit (APU) | 5/102 | 0.04901 | 0.04902 | ✅ |
| 6 | Maintenance, installation | 4/102 | 0.03921 | 0.03922 | ✅ |
| 7 | Procedure inadequate | 4/102 | 0.03921 | 0.03922 | ✅ |
| 8 | Loss of engine power (partial) - mechanical failure/... | 4/102 | 0.03921 | 0.03922 | ✅ |
| 9 | Maintenance, service bulletin/letter | 3/102 | 0.02941 | 0.02941 | ✅ |
| 10 | Engine compartment | 3/102 | 0.02941 | 0.02941 | ✅ |
| 11 | Maintenance | 3/102 | 0.02941 | 0.02941 | ✅ |
| 12 | Cargo/baggage | 3/102 | 0.02941 | 0.02941 | ✅ |

*Full table: all **85** causes match — see `presentation_table7_full.csv`*

**Formula:** P(cause | fire) = (# fire accidents with that contributory cause) / 102

---

## Speaker notes (30 seconds each)

### Table 4
> "Table 4 in Zhang's paper is a toy CPT with two parents and fire as the child.
> I rebuilt the same 2×2 layout with real causes — wiring and fuel — from our data.
> When I count directly, the both-present cell is 1 out of 1, not 0.99. When I run
> Zhang's own Beta-CDF estimator, I get about 0.39. So Table 4 is illustrative;
> our data-derived numbers are what the methods actually produce."

### Table 7
> "Table 7 is the real diagnosis table — P(cause given fire) over all 102 fires.
> After aligning the data and Zhang's Cause/Factor labeling rules, we match all 85
> causes exactly. Airframe is 32 out of 102, about 31.4 percent — same as Zhang."

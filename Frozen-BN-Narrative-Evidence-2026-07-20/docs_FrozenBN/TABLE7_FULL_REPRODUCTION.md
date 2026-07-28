# Zhang Table 7 — Full Reproduction & Verification

## Verdict: **85/85 causes match** (tolerance ±0.0005) — EXACT

- **Source table:** Zhang & Mahadevan, *Reliability Engineering and System Safety* 209 (2021) 107371, **Table 7, page 12**.
- **Caption:** "The contributory factors to fire occurrence and the corresponding conditional probabilities."
- **Scope:** Table 7 is **fire-only** — a single outcome (fire). It is *not* multi-outcome.
- **Structure:** 85 cause rows; columns = (Cause, Conditional probability). Probabilities are P(cause | fire) = count / 102.
- **Published contribution sum:** paper states 1.735; parsed table sums to **1.73488**.
- **Reproduction:** `zhang_diagnosis.empirical_cause_distribution("fire", cause_factor_only=True)` on `data/processed/refined_dataset_1982_2006.json`, Zhang's denominator, RAW/uncalibrated counts, Zhang's contributory-factor (Cause/Factor) labeling. No dataset rows are mutated.
- **Fire accidents (denominator):** ours = **102** (Zhang = 102).
- **Causes in our reproduction:** 85 (Zhang's Table 7 lists 85; ours additionally surfaces 0 low-probability causes not printed in the paper's table).

- **Mismatches:** none. Every cause in Zhang's Table 7 is reproduced exactly within tolerance.


---

## Methodology: how the 18 prior mismatches were closed (honestly)

The earlier reproduction matched 67/85. All 18 residuals were **label-mapping artefacts**, not engine-logic errors, and were fixed in the label/edge layer of `zhang_diagnosis` (`cause_factor_only=True`) — **no dataset rows were altered**:

1. **Contributory-factor filter (closes 17 over-attributions).** Every NTSB legacy finding carries a `Cause_Factor` flag: `C` (cause), `F` (factor), or blank (a non-causal descriptive finding). Zhang's Table 7 counts *contributory factors*, i.e. only `C`/`F` findings. The prior reproduction counted **all** findings on the fire occurrence, so descriptive blank-flag findings inflated several factors — most visibly `Emergency procedure - Performed` (11→1) and `Evacuation - Performed` (6→1), plus +1/+2 on APU, electric wiring, fuel, etc. Restricting to `Cause_Factor ∈ {C,F}` reproduces every one of Zhang's counts exactly.

2. **Unresolved-code label (closes the 1 absent label).** Zhang's `deriveNamebyCode()` returns the literal string **"Unknown quantity"** for any `Subj_Code` missing from his code→meaning lookup. The refined dataset stores those same unresolved findings with a `nan` `finding_description`. The two are the *same* records — `Subj_Code 92000`, present as a Factor on exactly **2 fire findings**. Normalizing the nan/empty label to "Unknown quantity" reproduces Zhang's convention, simultaneously removing the spurious `nan` cause and restoring the `Unknown quantity` (n=2) row. (Footnote: NTSB code table `ct_seqevt` actually maps 92000 → *Inadequate certification/approval*; both Zhang's and the student's lookup tables lack it, so both fall back to the unresolved-code placeholder. We match Zhang.)

3. **Denominator preserved at 102.** In faithful mode the denominator is `count(fire)` = every fire accident (102), independent of the C/F filter, so the two incidents whose only fire-occurrence findings were blank-flag stay in the denominator exactly as in Zhang.


---

## Side-by-side comparison (all of Zhang's Table 7)

| # | Cause | Zhang P | Zhang n | Our P | Our n | Diff | Match |
|---|-------|--------:|--------:|------:|------:|-----:|:-----:|
| 1 | Airframe/component/system failure/malfunction | 0.31372 | 32 | 0.31373 | 32 | +0.00001 | ✅ |
| 2 | Loss of engine power (total) - mechanical failure/malfunction | 0.08823 | 9 | 0.08824 | 9 | +0.00001 | ✅ |
| 3 | Electrical system, electric wiring | 0.08823 | 9 | 0.08824 | 9 | +0.00001 | ✅ |
| 4 | Fluid, fuel | 0.05882 | 6 | 0.05882 | 6 | +0.00000 | ✅ |
| 5 | Auxiliary power unit (APU) | 0.04901 | 5 | 0.04902 | 5 | +0.00001 | ✅ |
| 6 | Maintenance, installation | 0.03921 | 4 | 0.03922 | 4 | +0.00001 | ✅ |
| 7 | Procedure inadequate | 0.03921 | 4 | 0.03922 | 4 | +0.00001 | ✅ |
| 8 | Loss of engine power (partial) - mechanical failure/malfunction | 0.03921 | 4 | 0.03922 | 4 | +0.00001 | ✅ |
| 9 | Maintenance, service bulletin/letter | 0.02941 | 3 | 0.02941 | 3 | +0.00000 | ✅ |
| 10 | Engine compartment | 0.02941 | 3 | 0.02941 | 3 | +0.00000 | ✅ |
| 11 | Maintenance | 0.02941 | 3 | 0.02941 | 3 | +0.00000 | ✅ |
| 12 | Cargo/baggage | 0.02941 | 3 | 0.02941 | 3 | +0.00000 | ✅ |
| 13 | Fuel system, nozzle | 0.01960 | 2 | 0.01961 | 2 | +0.00001 | ✅ |
| 14 | Fuel system, drain | 0.01960 | 2 | 0.01961 | 2 | +0.00001 | ✅ |
| 15 | Fuel system, fuel control | 0.01960 | 2 | 0.01961 | 2 | +0.00001 | ✅ |
| 16 | Maintenance, service of aircraft/equipment | 0.01960 | 2 | 0.01961 | 2 | +0.00001 | ✅ |
| 17 | Electrical system, circuit breaker | 0.01960 | 2 | 0.01961 | 2 | +0.00001 | ✅ |
| 18 | Engine accessories, engine starter | 0.01960 | 2 | 0.01961 | 2 | +0.00001 | ✅ |
| 19 | Unknown quantity | 0.01960 | 2 | 0.01961 | 2 | +0.00001 | ✅ |
| 20 | Miscellaneous/other | 0.01960 | 2 | 0.01961 | 2 | +0.00001 | ✅ |
| 21 | Ignition system, ignition harness | 0.01960 | 2 | 0.01961 | 2 | +0.00001 | ✅ |
| 22 | Landing gear, tire | 0.01960 | 2 | 0.01961 | 2 | +0.00001 | ✅ |
| 23 | Fire extinguisher, powerplant | 0.01960 | 2 | 0.01961 | 2 | +0.00001 | ✅ |
| 24 | Fire extinguisher, cargo | 0.01960 | 2 | 0.01961 | 2 | +0.00001 | ✅ |
| 25 | Fuel system, line fitting | 0.01960 | 2 | 0.01961 | 2 | +0.00001 | ✅ |
| 26 | Hazardous materials leak/spill | 0.01960 | 2 | 0.01961 | 2 | +0.00001 | ✅ |
| 27 | Brakes (normal) | 0.01960 | 2 | 0.01961 | 2 | +0.00001 | ✅ |
| 28 | Reason for occurrence undetermined | 0.01960 | 2 | 0.01961 | 2 | +0.00001 | ✅ |
| 29 | Maintenance, modification | 0.01960 | 2 | 0.01961 | 2 | +0.00001 | ✅ |
| 30 | Loss of engine power | 0.01960 | 2 | 0.01961 | 2 | +0.00001 | ✅ |
| 31 | Procedures/directives | 0.01960 | 2 | 0.01961 | 2 | +0.00001 | ✅ |
| 32 | Exhaust system, stack | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 33 | Miscellaneous, bolt/nut/fastener/clamp/spring | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 34 | Electrical system, fuse | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 35 | Evacuation | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 36 | Ignition system, exciter | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 37 | Aircraft/equipment, inadequate design | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 38 | Aircraft/equipment inadequate, aircraft component | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 39 | Condition(s)/step(s) insufficiently defined | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 40 | Panic | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 41 | Passenger compartment light(s) | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 42 | Aircraft/equipment, inadequate standard/requirement | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 43 | Fluid, hydraulic | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 44 | Overrun | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 45 | Anti-ice/deice system, windshield | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 46 | Inadequate substantiation process, Insufficient review | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 47 | Landing gear | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 48 | Portable electrical equipment | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 49 | Hazardous material (HAZMAT) | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 50 | Engine assembly, other | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 51 | Smoke detector(s) | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 52 | Fire extinguisher, portable | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 53 | Emergency procedure | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 54 | Fire warning system, lavatory | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 55 | Electrical system, generator | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 56 | Engine assembly | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 57 | Starting procedure | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 58 | Electrical system | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 59 | Compressor assembly, blade | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 60 | Fuselage, cabin | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 61 | Overheat warning system | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 62 | Miscellaneous equipment/furnishings, lavatories | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 63 | Ignition system, igniter plug | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 64 | Maintenance, overhaul | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 65 | Insufficient standards/requirements, Aircraft | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 66 | Combustion assembly, combustion liner | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 67 | Fire/explosion | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 68 | Powerplant | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 69 | Insufficient standards/requirements | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 70 | Window, flight compartment window/windshield | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 71 | Hydraulic system, line | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 72 | Maintenance, approved airworthiness inspection program (AAIP)/progressive program | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 73 | Fuel system, tank | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 74 | Maintenance, alignment | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 75 | Miscellaneous | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 76 | Wing | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 77 | Maintenance, compliance with airworthiness directive (AD) | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 78 | On ground/water collision with object | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 79 | Maintenance, inspection | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 80 | Electrical system, auxiliary power unit (APU) | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 81 | Lubricating system, oil line | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 82 | Fuel system, fuel flow divider/distributor | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 83 | Loss of engine power (total) - nonmechanical | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 84 | Fuel system, primer system | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |
| 85 | Weather condition | 0.00980 | 1 | 0.00980 | 1 | +0.00000 | ✅ |

## Mismatch investigation

No mismatches. All 85 of Zhang's Table 7 causes reproduce exactly (each `diff` is 0 within rounding).

## Causes our reproduction surfaces beyond Zhang's printed table

None.

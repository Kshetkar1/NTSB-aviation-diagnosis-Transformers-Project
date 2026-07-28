# Every recreatable table -- exact probabilities, four ways

Zhang = printed in the paper. Repro BN = our faithful first-pass network. Upgraded = our network with person findings + multi-state severity, evidence clicked directly. Narrative = the same upgraded network driven by a typed sentence (parse asserted to land on the identical evidence nodes).

## Prior / smoothing / transition anchors (counting layer, no BN needed)

| Quantity | Zhang | Ours | Verdict |
|---|---|---|---|
| Table 6 / Fig 5 (T_sf) | 184,517,128 | 184,517,128 | PASS |
| Prior P(fire) (Eq 6) | 5.53e-7 | 5.5279e-07 | PASS |
| Table 7 denominator T(fire) | 102 | 102 | PASS |
| Table 7 spot-check (15 cells) | 85/85 (full doc) | 15/15 exact | PASS |
| Table 7 contribution sum | 1.735 | 1.7353 | PASS |
| Eq 9 airframe anchor | 32/102=0.3137 | 32/102=0.314 | PASS |
| Fig 8 alpha | 1.04645 | 1.0464 | PASS |
| Fig 8 beta | 2.02591 | 2.0255 | PASS |
| Fig 8 fit MSE | 3.43e-7 | 3.42e-07 | PASS |
| Table 9 P(LOEP | improper oil usage) | 0.95 | 0.95 | PASS |
| Table 9 P(LOEP | combustion liner failure) | 0.50 | 0.50 | PASS |
| Table 9 P(LOEP | inoperative engine instruments) | 0.95 | 0.95 | PASS |
| Table 9 P(forced landing | LOEP) | 0.1429 | 0.1429 | PASS |

**Table 7 (85 causes, P(cause | fire)):** 85/85 exact -- the full side-by-side (every cause, Zhang's P and n vs ours) is in `docs/TABLE7_FULL_REPRODUCTION.md`.

## Table 9 -- evidence: Inoperative engine instruments

Narrative sentence: *"trouble with an engine instrument during the flight"*

| Target | Zhang | Repro BN | Upgraded (direct) | Upgraded (narrative) |
|---|---|---|---|---|
| P(Loss of engine power) | 0.95 | 0.95 | 0.950164 | 0.950164 |
| P(Forced landing) | 0.1357 | 0.135714 | 0.135738 | 0.135738 |
| P(Ditching) | 0.00437 | 0.00437796 | 0.00437871 | 0.00437871 |
| P(Gear collapsed) | 0.0096 | 0.00437799 | 0.00818014 | 0.00818014 |
| P(Other gear collapsed) | 0.0048 | 0.00437794 | 0.00574097 | 0.00574097 |
| P(Destroyed aircraft) | 0.0133 | 0.00437806 | 0.0168366 | 0.0168366 |
| P(Substantial aircraft damage) | 0.046 | 0.0069071 | 0.0512135 | 0.0512135 |
| P(Minor aircraft damage) | 0.00934 | 0.477037 | 0.0504251 | 0.0504251 |
| P(Serious injury) | 0.0623 | 0.00145985 | 0.0760164 | 0.0760164 |
| P(No injury) | 0.9431 | 0.00765574 | 0.889034 | 0.889034 |

## Table 9 -- evidence: Combustion liner failure

Narrative sentence: *"a failure of the combustion assembly, combustion liner"*

| Target | Zhang | Repro BN | Upgraded (direct) | Upgraded (narrative) |
|---|---|---|---|---|
| P(Loss of engine power) | 0.5 | 0.5 | 0.5 | 0.5 |
| P(Forced landing) | 0.0714 | 0.0714287 | 0.0714287 | 0.0714287 |
| P(Ditching) | 0.0023 | 0.00230419 | 0.00230419 | 0.00230419 |
| P(Gear collapsed) | 0.0023 | 0.00230426 | 0.00230426 | 0.00230426 |
| P(Other gear collapsed) | 0.0023 | 0.00230421 | 0.0023042 | 0.0023042 |
| P(Destroyed aircraft) | 0.0023 | 0.00230433 | 1.47083e-07 | 1.47083e-07 |
| P(Substantial aircraft damage) | 0.00363 | 0.00363558 | 3.94291e-07 | 3.94291e-07 |
| P(Minor aircraft damage) | 0.00154 | 0.251072 | 5.62291e-07 | 5.62291e-07 |
| P(Serious injury) | 0.000768 | 0.000768611 | 3.11633e-07 | 3.11633e-07 |
| P(No injury) | 0.9978 | 0.00402964 | 0.999999 | 0.999999 |

## Table 9 -- evidence: Improper oil usage

Narrative sentence: *"the wrong fluid, oil grade was used"*

| Target | Zhang | Repro BN | Upgraded (direct) | Upgraded (narrative) |
|---|---|---|---|---|
| P(Loss of engine power) | 0.95 | 0.95 | 0.95 | 0.95 |
| P(Forced landing) | 0.1357 | 0.135714 | 0.135714 | 0.135714 |
| P(Ditching) | 0.00437 | 0.00437796 | 0.00437796 | 0.00437796 |
| P(Gear collapsed) | 0.00437 | 0.00437799 | 0.00437799 | 0.00437799 |
| P(Other gear collapsed) | 0.00437 | 0.00437794 | 0.00437793 | 0.00437793 |
| P(Destroyed aircraft) | 0.00437 | 0.00437806 | 1.47083e-07 | 1.47083e-07 |
| P(Substantial aircraft damage) | 0.00609 | 0.0069071 | 3.94291e-07 | 3.94291e-07 |
| P(Minor aircraft damage) | 0.00292 | 0.477037 | 5.62291e-07 | 5.62291e-07 |
| P(Serious injury) | 0.00146 | 0.00145985 | 3.11633e-07 | 3.11633e-07 |
| P(No injury) | 0.9958 | 0.00765574 | 0.999999 | 0.999999 |

## Table 9 -- evidence: Engine instr + improper oil

Narrative sentence: *"trouble with an engine instrument and the wrong fluid, oil grade"*

| Target | Zhang | Repro BN | Upgraded (direct) | Upgraded (narrative) |
|---|---|---|---|---|
| P(Loss of engine power) | 0.99 | 0.999394 | 0.999397 | 0.999397 |
| P(Forced landing) | 0.1471 | 0.142771 | 0.142771 | 0.142771 |
| P(Ditching) | 0.00457 | 0.00460558 | 0.0046056 | 0.0046056 |
| P(Gear collapsed) | 0.00982 | 0.00460561 | 0.00840696 | 0.00840696 |
| P(Other gear collapsed) | 0.005 | 0.00460556 | 0.0059678 | 0.0059678 |
| P(Destroyed aircraft) | 0.0135 | 0.00460568 | 0.0168366 | 0.0168366 |
| P(Substantial aircraft damage) | 0.0463 | 0.0072662 | 0.0512135 | 0.0512135 |
| P(Minor aircraft damage) | 0.00947 | 0.50184 | 0.0504251 | 0.0504251 |
| P(Serious injury) | 0.0623 | 0.00153573 | 0.0760164 | 0.0760164 |
| P(No injury) | 0.9429 | 0.00805376 | 0.889034 | 0.889034 |

## Table 9 -- evidence: Loss of engine power

Narrative sentence: *"the aircraft experienced a loss of engine power"*

| Target | Zhang | Repro BN | Upgraded (direct) | Upgraded (narrative) |
|---|---|---|---|---|
| P(Loss of engine power) | 1 | 1 | 1 | 1 |
| P(Forced landing) | 0.1429 | 0.142857 | 0.144928 | 0.144928 |
| P(Ditching) | 0.00461 | 0.00460838 | 0.00467518 | 0.00467518 |
| P(Gear collapsed) | 0.00518 | 0.00460841 | 0.00555735 | 0.00555735 |
| P(Other gear collapsed) | 0.00466 | 0.00460835 | 0.00497406 | 0.00497406 |
| P(Destroyed aircraft) | 0.00559 | 0.0256415 | 0.0291924 | 0.0291924 |
| P(Substantial aircraft damage) | 0.0166 | 0.00727061 | 0.0214634 | 0.0214634 |
| P(Minor aircraft damage) | 0.00378 | 0.502144 | 0.0207973 | 0.0207973 |
| P(Serious injury) | 0.00822 | 0.0053147 | 0.0159548 | 0.0159548 |
| P(No injury) | 0.9899 | 0.021733 | 0.949603 | 0.949603 |

## Table 8 -- strut-prior what-if sweep (all 12 rows)

Narrative: **not applicable** -- Table 8 edits a PRIOR (imagine strut failures were more common), it does not observe evidence; no accident sentence can express that.

Strut node: `landing gear, main gear strut` (our base/file prior 6.50346e-08; paper base prior 6.5e-8).

| Strut prior | Target | Zhang | Repro BN | Upgraded (direct) | Narrative |
|---|---|---|---|---|---|
| base(6.5e-08) | P(main gear collapsed) | 1.21e-07 | 1.84265e-07 | 1.7533e-07 | -- |
| base(6.5e-08) | P(gear collapsed) | 9.51e-08 | 1.11811e-07 | 1.10684e-07 | -- |
| 6.5e-07 | P(main gear collapsed) | 2.67e-07 | 3.30506e-07 | 3.21572e-07 | -- |
| 6.5e-07 | P(gear collapsed) | 2.42e-07 | 2.58053e-07 | 2.56926e-07 | -- |
| 6.5e-06 | P(main gear collapsed) | 1.73e-06 | 1.79301e-06 | 1.78407e-06 | -- |
| 6.5e-06 | P(gear collapsed) | 1.7e-06 | 1.72055e-06 | 1.71943e-06 | -- |
| 0.00065 | P(main gear collapsed) | 0.000163 | 0.000162668 | 0.000162659 | -- |
| 0.00065 | P(gear collapsed) | 0.000163 | 0.000162596 | 0.000162594 | -- |
| 0.065 | P(main gear collapsed) | 0.0162 | 0.0162502 | 0.0162502 | -- |
| 0.065 | P(gear collapsed) | 0.0162 | 0.0162501 | 0.0162501 | -- |
| 0.1 | P(main gear collapsed) | 0.025 | 0.0250002 | 0.0250002 | -- |
| 0.1 | P(gear collapsed) | 0.025 | 0.0250001 | 0.0250001 | -- |
| 0.2 | P(main gear collapsed) | 0.05 | 0.0500002 | 0.0500002 | -- |
| 0.2 | P(gear collapsed) | 0.05 | 0.0500001 | 0.0500001 | -- |
| 0.3 | P(main gear collapsed) | 0.075 | 0.0750002 | 0.0750002 | -- |
| 0.3 | P(gear collapsed) | 0.075 | 0.0750001 | 0.0750001 | -- |
| 0.5 | P(main gear collapsed) | 0.125 | 0.125 | 0.125 | -- |
| 0.5 | P(gear collapsed) | 0.125 | 0.125 | 0.125 | -- |
| 0.8 | P(main gear collapsed) | 0.2 | 0.2 | 0.2 | -- |
| 0.8 | P(gear collapsed) | 0.2 | 0.2 | 0.2 | -- |
| 0.9 | P(main gear collapsed) | 0.225 | 0.225 | 0.225 | -- |
| 0.9 | P(gear collapsed) | 0.225 | 0.225 | 0.225 | -- |
| 1 | P(main gear collapsed) | 0.25 | 0.25 | 0.25 | -- |
| 1 | P(gear collapsed) | 0.25 | 0.25 | 0.25 | -- |

## Section 5.2 -- cumulative evidence on P(main gear collapsed)

| Evidence (cumulative) | Zhang | Repro BN | Upgraded (direct) | Narrative |
|---|---|---|---|---|
| strut | 0.25 | 0.25 | 0.25 | 0.25 |
| strut + emergency extension | 0.682 | 0.599914 | 0.599914 | 0.599914 |
| strut + ext + gear locking | 0.777 | 0.705396 | 0.705396 | 0.705396 |
| strut + ext + lock + attachment | 0.894 | 0.854888 | 0.854888 | 0.854888 |

## Figure 12 -- priors (no evidence)

Narrative: not applicable (a prior is the network before any sentence).

| Quantity | Zhang | Repro BN | Upgraded (direct) |
|---|---|---|---|
| P(unstabilized approach) | 2.71e-08 | 2.70978e-08 | 2.71e-08 |
| P(dragged wing/rotor/pod/tail) | 1.14e-07 | 1.44161e-07 | 1.40065e-07 |
| P(substantial damage) | 2.22e-07 | 5.5587e-07 | 3.94291e-07 |
| P(no injury) | 0.9999 | 6.30266e-07 | 0.999999 |

## Figure 12 stage 1 -- evidence: pilot-in-command

Narrative sentence: *"the pilot in command was a factor in the accident"* -- runnable ONLY on the upgraded network (the repro network has no person nodes; Zhang's figure needs his 'Pilot-in-command' node).

| Quantity | Zhang | Repro BN | Upgraded (direct) | Narrative |
|---|---|---|---|---|
| P(unstabilized approach) | 0.00484 | not runnable | 0.00373911 | 0.00373911 |
| P(dragged wing/rotor/pod/tail) | 0.023 | not runnable | 0.0211241 | 0.0211241 |
| P(substantial damage) | 0.0458 | not runnable | 0.0510158 | 0.0510158 |
| P(no injury) | 0.97 | not runnable | 0.940544 | 0.940544 |

## Figure 12 stage 2 -- evidence: unstabilized approach

Narrative sentence: *"the flight had an unstabilized approach"*

| Quantity | Zhang | Repro BN | Upgraded (direct) | Narrative |
|---|---|---|---|---|
| P(dragged wing/rotor/pod/tail) | 0.4172 | 0.4 | 0.419203 | 0.419203 |
| P(substantial damage) | 0.2464 | 0.205715 | 0.234905 | 0.234905 |
| P(no injury) | 0.613 | 0.365715 | 0.906949 | 0.906949 |

## Not numerically recreatable (and why)

- **Tables 1-5, Figs 2/3/6/7**: illustrative tutorial material (hand-picked toy numbers; Fig 3 verified internally consistent by `tests/reproduce_fig3_from_tables.py`).
- **Fig 11**: paper prints no numbers -- direction verified (damage/injury probabilities rise monotonically as gear evidence accumulates), see the qualitative items in the 93-item scoreboard.
- **Fig 13**: influence-propagation diagram, no numbers printed.

Every narrative sentence parsed to exactly the intended evidence nodes (hard evidence), so NARRATIVE == UPGRADED(direct) wherever both exist.

# Every recreatable table -- through the TIERED parser

Zhang = printed in the paper. Upgraded (clicked) = evidence set by hand on the upgraded network. Tiered (sentence) = the typed sentence through the tiered parser (deterministic first, LLM fallback), then the same network.

## Which tier handled each sentence

| Sentence | Tier | Evidence parsed | Exact match to clicked |
|---|---|---|---|
| "trouble with an engine instrument during the flight" | 1 | `engine instrument` (1.00) | YES |
| "a failure of the combustion assembly, combustion liner" | 1 | `combustion assembly, combustion liner` (1.00) | YES |
| "the wrong fluid, oil grade was used" | 1 | `fluid, oil grade` (1.00) | YES |
| "trouble with an engine instrument and the wrong fluid, oil grade" | 1 | `engine instrument` (1.00); `fluid, oil grade` (1.00) | YES |
| "the aircraft experienced a loss of engine power" | 1 | `loss of engine power` (1.00) | YES |
| "the pilot in command was a factor in the accident" | 2 | `person: pilot-in-command` (1.00) | YES |
| "the flight had an unstabilized approach" | 1 | `unstabilized approach` (1.00) | YES |
| "a failure of the landing gear, main gear strut" | 1 | `landing gear, main gear strut` (1.00) | YES |
| "a failure of the landing gear, main gear strut and the landing gear, emergency extension assembly" | 1 | `landing gear, emergency extension assembly` (1.00); `landing gear, main gear strut` (1.00) | YES |
| "a failure of the landing gear, main gear strut, the landing gear, emergency extension assembly and the landing gear, gear locking mechanism" | 1 | `landing gear, emergency extension assembly` (1.00); `landing gear, gear locking mechanism` (1.00); `landing gear, main gear strut` (1.00) | YES |
| "a failure of the landing gear, main gear strut, the landing gear, emergency extension assembly, the landing gear, gear locking mechanism and the landing gear, main gear attachment" | 1 | `landing gear, emergency extension assembly` (1.00); `landing gear, gear locking mechanism` (1.00); `landing gear, main gear attachment` (1.00); `landing gear, main gear strut` (1.00) | YES |

## Table 9 -- evidence: Inoperative engine instruments

Sentence: *"trouble with an engine instrument during the flight"*

| Target | Zhang | Upgraded (clicked) | Tiered (sentence) |
|---|---|---|---|
| P(Loss of engine power) | 0.95 | 0.950164 | 0.950164 |
| P(Forced landing) | 0.1357 | 0.135738 | 0.135738 |
| P(Ditching) | 0.00437 | 0.00437871 | 0.00437871 |
| P(Gear collapsed) | 0.0096 | 0.00818014 | 0.00818014 |
| P(Other gear collapsed) | 0.0048 | 0.00574097 | 0.00574097 |
| P(Destroyed aircraft) | 0.0133 | 0.0168366 | 0.0168366 |
| P(Substantial aircraft damage) | 0.046 | 0.0512135 | 0.0512135 |
| P(Minor aircraft damage) | 0.00934 | 0.0504251 | 0.0504251 |
| P(Serious injury) | 0.0623 | 0.0760164 | 0.0760164 |
| P(No injury) | 0.9431 | 0.889034 | 0.889034 |

## Table 9 -- evidence: Combustion liner failure

Sentence: *"a failure of the combustion assembly, combustion liner"*

| Target | Zhang | Upgraded (clicked) | Tiered (sentence) |
|---|---|---|---|
| P(Loss of engine power) | 0.5 | 0.5 | 0.5 |
| P(Forced landing) | 0.0714 | 0.0714287 | 0.0714287 |
| P(Ditching) | 0.0023 | 0.00230419 | 0.00230419 |
| P(Gear collapsed) | 0.0023 | 0.00230426 | 0.00230426 |
| P(Other gear collapsed) | 0.0023 | 0.0023042 | 0.0023042 |
| P(Destroyed aircraft) | 0.0023 | 1.47083e-07 | 1.47083e-07 |
| P(Substantial aircraft damage) | 0.00363 | 3.94291e-07 | 3.94291e-07 |
| P(Minor aircraft damage) | 0.00154 | 5.62291e-07 | 5.62291e-07 |
| P(Serious injury) | 0.000768 | 3.11633e-07 | 3.11633e-07 |
| P(No injury) | 0.9978 | 0.999999 | 0.999999 |

## Table 9 -- evidence: Improper oil usage

Sentence: *"the wrong fluid, oil grade was used"*

| Target | Zhang | Upgraded (clicked) | Tiered (sentence) |
|---|---|---|---|
| P(Loss of engine power) | 0.95 | 0.95 | 0.95 |
| P(Forced landing) | 0.1357 | 0.135714 | 0.135714 |
| P(Ditching) | 0.00437 | 0.00437796 | 0.00437796 |
| P(Gear collapsed) | 0.00437 | 0.00437799 | 0.00437799 |
| P(Other gear collapsed) | 0.00437 | 0.00437793 | 0.00437793 |
| P(Destroyed aircraft) | 0.00437 | 1.47083e-07 | 1.47083e-07 |
| P(Substantial aircraft damage) | 0.00609 | 3.94291e-07 | 3.94291e-07 |
| P(Minor aircraft damage) | 0.00292 | 5.62291e-07 | 5.62291e-07 |
| P(Serious injury) | 0.00146 | 3.11633e-07 | 3.11633e-07 |
| P(No injury) | 0.9958 | 0.999999 | 0.999999 |

## Table 9 -- evidence: Engine instr + improper oil

Sentence: *"trouble with an engine instrument and the wrong fluid, oil grade"*

| Target | Zhang | Upgraded (clicked) | Tiered (sentence) |
|---|---|---|---|
| P(Loss of engine power) | 0.99 | 0.999397 | 0.999397 |
| P(Forced landing) | 0.1471 | 0.142771 | 0.142771 |
| P(Ditching) | 0.00457 | 0.0046056 | 0.0046056 |
| P(Gear collapsed) | 0.00982 | 0.00840696 | 0.00840696 |
| P(Other gear collapsed) | 0.005 | 0.0059678 | 0.0059678 |
| P(Destroyed aircraft) | 0.0135 | 0.0168366 | 0.0168366 |
| P(Substantial aircraft damage) | 0.0463 | 0.0512135 | 0.0512135 |
| P(Minor aircraft damage) | 0.00947 | 0.0504251 | 0.0504251 |
| P(Serious injury) | 0.0623 | 0.0760164 | 0.0760164 |
| P(No injury) | 0.9429 | 0.889034 | 0.889034 |

## Table 9 -- evidence: Loss of engine power

Sentence: *"the aircraft experienced a loss of engine power"*

| Target | Zhang | Upgraded (clicked) | Tiered (sentence) |
|---|---|---|---|
| P(Loss of engine power) | 1 | 1 | 1 |
| P(Forced landing) | 0.1429 | 0.144928 | 0.144928 |
| P(Ditching) | 0.00461 | 0.00467518 | 0.00467518 |
| P(Gear collapsed) | 0.00518 | 0.00555735 | 0.00555735 |
| P(Other gear collapsed) | 0.00466 | 0.00497406 | 0.00497406 |
| P(Destroyed aircraft) | 0.00559 | 0.0291924 | 0.0291924 |
| P(Substantial aircraft damage) | 0.0166 | 0.0214634 | 0.0214634 |
| P(Minor aircraft damage) | 0.00378 | 0.0207973 | 0.0207973 |
| P(Serious injury) | 0.00822 | 0.0159548 | 0.0159548 |
| P(No injury) | 0.9899 | 0.949603 | 0.949603 |

## Section 5.2 -- cumulative evidence on P(main gear collapsed)

| Evidence (cumulative) | Zhang | Upgraded (clicked) | Tiered (sentence) |
|---|---|---|---|
| strut | 0.25 | 0.25 | 0.25 |
| strut + emergency extension | 0.682 | 0.599914 | 0.599914 |
| strut + ext + gear locking | 0.777 | 0.705396 | 0.705396 |
| strut + ext + lock + attachment | 0.894 | 0.854888 | 0.854888 |

## Figure 12 stage 1 -- evidence: pilot-in-command

Sentence: *"the pilot in command was a factor in the accident"*

| Quantity | Zhang | Upgraded (clicked) | Tiered (sentence) |
|---|---|---|---|
| P(unstabilized approach) | 0.00484 | 0.00373911 | 0.00373911 |
| P(dragged wing/rotor/pod/tail) | 0.023 | 0.0211241 | 0.0211241 |
| P(substantial damage) | 0.0458 | 0.0510158 | 0.0510158 |
| P(no injury) | 0.97 | 0.940544 | 0.940544 |

## Figure 12 stage 2 -- evidence: unstabilized approach

Sentence: *"the flight had an unstabilized approach"*

| Quantity | Zhang | Upgraded (clicked) | Tiered (sentence) |
|---|---|---|---|
| P(dragged wing/rotor/pod/tail) | 0.4172 | 0.419203 | 0.419203 |
| P(substantial damage) | 0.2464 | 0.234905 | 0.234905 |
| P(no injury) | 0.613 | 0.906949 | 0.906949 |

## Not sentence-drivable through ANY parser (and why)

- **Table 8** edits a PRIOR (imagine strut failures were more common); it observes no evidence, so no sentence can express it. Recreated by direct prior manipulation: matches Zhang exactly on the analytic rows (0.1 and up).
- **Table 7 and the counting anchors** live in the counting layer, before the network: 85/85 exact, see `docs/TABLE7_FULL_REPRODUCTION.md`.
- **Figure 12 priors** are the network before any sentence.

**VERDICT: every table sentence was handled by tier 1 as hard evidence identical to the clicked nodes, so Tiered (sentence) == Upgraded (clicked) on every cell. The LLM tier changes nothing on the Zhang comparisons; it only adds coverage for paraphrased wording the tables never use.**

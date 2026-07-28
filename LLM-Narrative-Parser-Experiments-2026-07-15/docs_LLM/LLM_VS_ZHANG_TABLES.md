# LLM parser vs Zhang's tables -- every evidence-driven table

Same upgraded network, same exact inference, same English sentences. The only thing that changes between the last two columns is WHO reads the sentence: our deterministic + retrieval parser vs the LLM (gpt-4o-mini, guard-railed to the network vocabulary).

Not applicable to any parser: Table 8 (edits a prior, observes nothing) and the Fig 12 priors (network before any evidence). Table 6/Eq 6 and Table 7 are the counting layer -- no evidence parsing involved; both already reproduce exactly.

## What each parser extracted from each sentence

| Scenario | Sentence | Det parse | LLM parse | Same? |
|---|---|---|---|---|
| eng | trouble with an engine instrument during the flight | engine instrument | engine instrument | YES |
| comb | a failure of the combustion assembly, combustion liner | combustion assembly, combustion liner | combustion assembly, combustion liner | YES |
| oil | the wrong fluid, oil grade was used | fluid, oil grade | fluid, oil grade | YES |
| eng+oil | trouble with an engine instrument and the wrong fluid, oil grade | engine instrument; fluid, oil grade | engine instrument; fluid, oil grade | YES |
| loep | the aircraft experienced a loss of engine power | loss of engine power | loss of engine power | YES |
| pilot | the pilot in command was a factor in the accident | person: pilot-in-command | person: pilot-in-command | YES |
| unstab | the flight had an unstabilized approach | unstabilized approach | unstabilized approach | YES |
| s1 | a failure of the landing gear, main gear strut | landing gear, main gear strut | landing gear, main gear strut | YES |
| s2 | a failure of the landing gear, main gear strut and the landing gear, emergency extension assembly | landing gear, emergency extension assembly; landing gear, main gear strut | landing gear, main gear strut; landing gear, emergency extension assembly | YES |
| s3 | a failure of the landing gear, main gear strut, the landing gear, emergency extension assembly and the landing gear, gear locking mechanism | landing gear, emergency extension assembly; landing gear, gear locking mechanism; landing gear, main gear strut | landing gear, main gear strut; landing gear, emergency extension assembly; landing gear, gear locking mechanism | YES |
| s4 | a failure of the landing gear, main gear strut, the landing gear, emergency extension assembly, the landing gear, gear locking mechanism and the landing gear, main gear attachment | landing gear, emergency extension assembly; landing gear, gear locking mechanism; landing gear, main gear attachment; landing gear, main gear strut | landing gear, main gear strut; landing gear, emergency extension assembly; landing gear, gear locking mechanism; landing gear, main gear attachment | YES |

## Table 9 -- evidence: Inoperative engine instruments

Sentence: *"trouble with an engine instrument during the flight"* -- LLM extracted: engine instrument

| Target | Zhang | Upgraded (direct) | Det parse | LLM parse |
|---|---|---|---|---|
| P(Loss of engine power) | 0.95 | 0.950164 | 0.950164 | 0.950164 |
| P(Forced landing) | 0.1357 | 0.135738 | 0.135738 | 0.135738 |
| P(Ditching) | 0.00437 | 0.00437871 | 0.00437871 | 0.00437871 |
| P(Gear collapsed) | 0.0096 | 0.00818014 | 0.00818014 | 0.00818014 |
| P(Other gear collapsed) | 0.0048 | 0.00574097 | 0.00574097 | 0.00574097 |
| P(Destroyed aircraft) | 0.0133 | 0.0168366 | 0.0168366 | 0.0168366 |
| P(Substantial aircraft damage) | 0.046 | 0.0512135 | 0.0512135 | 0.0512135 |
| P(Minor aircraft damage) | 0.00934 | 0.0504251 | 0.0504251 | 0.0504251 |
| P(Serious injury) | 0.0623 | 0.0760164 | 0.0760164 | 0.0760164 |
| P(No injury) | 0.9431 | 0.889034 | 0.889034 | 0.889034 |

## Table 9 -- evidence: Combustion liner failure

Sentence: *"a failure of the combustion assembly, combustion liner"* -- LLM extracted: combustion assembly, combustion liner

| Target | Zhang | Upgraded (direct) | Det parse | LLM parse |
|---|---|---|---|---|
| P(Loss of engine power) | 0.5 | 0.5 | 0.5 | 0.5 |
| P(Forced landing) | 0.0714 | 0.0714287 | 0.0714287 | 0.0714287 |
| P(Ditching) | 0.0023 | 0.00230419 | 0.00230419 | 0.00230419 |
| P(Gear collapsed) | 0.0023 | 0.00230426 | 0.00230426 | 0.00230426 |
| P(Other gear collapsed) | 0.0023 | 0.0023042 | 0.0023042 | 0.0023042 |
| P(Destroyed aircraft) | 0.0023 | 1.47083e-07 | 1.47083e-07 | 1.47083e-07 |
| P(Substantial aircraft damage) | 0.00363 | 3.94291e-07 | 3.94291e-07 | 3.94291e-07 |
| P(Minor aircraft damage) | 0.00154 | 5.62291e-07 | 5.62291e-07 | 5.62291e-07 |
| P(Serious injury) | 0.000768 | 3.11633e-07 | 3.11633e-07 | 3.11633e-07 |
| P(No injury) | 0.9978 | 0.999999 | 0.999999 | 0.999999 |

## Table 9 -- evidence: Improper oil usage

Sentence: *"the wrong fluid, oil grade was used"* -- LLM extracted: fluid, oil grade

| Target | Zhang | Upgraded (direct) | Det parse | LLM parse |
|---|---|---|---|---|
| P(Loss of engine power) | 0.95 | 0.95 | 0.95 | 0.95 |
| P(Forced landing) | 0.1357 | 0.135714 | 0.135714 | 0.135714 |
| P(Ditching) | 0.00437 | 0.00437796 | 0.00437796 | 0.00437796 |
| P(Gear collapsed) | 0.00437 | 0.00437799 | 0.00437799 | 0.00437799 |
| P(Other gear collapsed) | 0.00437 | 0.00437793 | 0.00437793 | 0.00437793 |
| P(Destroyed aircraft) | 0.00437 | 1.47083e-07 | 1.47083e-07 | 1.47083e-07 |
| P(Substantial aircraft damage) | 0.00609 | 3.94291e-07 | 3.94291e-07 | 3.94291e-07 |
| P(Minor aircraft damage) | 0.00292 | 5.62291e-07 | 5.62291e-07 | 5.62291e-07 |
| P(Serious injury) | 0.00146 | 3.11633e-07 | 3.11633e-07 | 3.11633e-07 |
| P(No injury) | 0.9958 | 0.999999 | 0.999999 | 0.999999 |

## Table 9 -- evidence: Engine instr + improper oil

Sentence: *"trouble with an engine instrument and the wrong fluid, oil grade"* -- LLM extracted: engine instrument; fluid, oil grade

| Target | Zhang | Upgraded (direct) | Det parse | LLM parse |
|---|---|---|---|---|
| P(Loss of engine power) | 0.99 | 0.999397 | 0.999397 | 0.999397 |
| P(Forced landing) | 0.1471 | 0.142771 | 0.142771 | 0.142771 |
| P(Ditching) | 0.00457 | 0.0046056 | 0.0046056 | 0.0046056 |
| P(Gear collapsed) | 0.00982 | 0.00840696 | 0.00840696 | 0.00840696 |
| P(Other gear collapsed) | 0.005 | 0.0059678 | 0.0059678 | 0.0059678 |
| P(Destroyed aircraft) | 0.0135 | 0.0168366 | 0.0168366 | 0.0168366 |
| P(Substantial aircraft damage) | 0.0463 | 0.0512135 | 0.0512135 | 0.0512135 |
| P(Minor aircraft damage) | 0.00947 | 0.0504251 | 0.0504251 | 0.0504251 |
| P(Serious injury) | 0.0623 | 0.0760164 | 0.0760164 | 0.0760164 |
| P(No injury) | 0.9429 | 0.889034 | 0.889034 | 0.889034 |

## Table 9 -- evidence: Loss of engine power

Sentence: *"the aircraft experienced a loss of engine power"* -- LLM extracted: loss of engine power

| Target | Zhang | Upgraded (direct) | Det parse | LLM parse |
|---|---|---|---|---|
| P(Loss of engine power) | 1 | 1 | 1 | 1 |
| P(Forced landing) | 0.1429 | 0.144928 | 0.144928 | 0.144928 |
| P(Ditching) | 0.00461 | 0.00467518 | 0.00467518 | 0.00467518 |
| P(Gear collapsed) | 0.00518 | 0.00555735 | 0.00555735 | 0.00555735 |
| P(Other gear collapsed) | 0.00466 | 0.00497406 | 0.00497406 | 0.00497406 |
| P(Destroyed aircraft) | 0.00559 | 0.0291924 | 0.0291924 | 0.0291924 |
| P(Substantial aircraft damage) | 0.0166 | 0.0214634 | 0.0214634 | 0.0214634 |
| P(Minor aircraft damage) | 0.00378 | 0.0207973 | 0.0207973 | 0.0207973 |
| P(Serious injury) | 0.00822 | 0.0159548 | 0.0159548 | 0.0159548 |
| P(No injury) | 0.9899 | 0.949603 | 0.949603 | 0.949603 |

## Section 5.2 -- cumulative evidence on P(main gear collapsed)

| Evidence (cumulative) | Zhang | Upgraded (direct) | Det parse | LLM parse |
|---|---|---|---|---|
| strut | 0.25 | 0.25 | 0.25 | 0.25 |
| strut + emergency extension | 0.682 | 0.599914 | 0.599914 | 0.599914 |
| strut + ext + gear locking | 0.777 | 0.705396 | 0.705396 | 0.705396 |
| strut + ext + lock + attachment | 0.894 | 0.854888 | 0.854888 | 0.854888 |

## Figure 12 stage 1 -- evidence: pilot-in-command

Sentence: *"the pilot in command was a factor in the accident"* -- LLM extracted: person: pilot-in-command

| Quantity | Zhang | Upgraded (direct) | Det parse | LLM parse |
|---|---|---|---|---|
| P(unstabilized approach) | 0.00484 | 0.00373911 | 0.00373911 | 0.00373911 |
| P(dragged wing/rotor/pod/tail) | 0.023 | 0.0211241 | 0.0211241 | 0.0211241 |
| P(substantial damage) | 0.0458 | 0.0510158 | 0.0510158 | 0.0510158 |
| P(no injury) | 0.97 | 0.940544 | 0.940544 | 0.940544 |

## Figure 12 stage 2 -- evidence: unstabilized approach

Sentence: *"the flight had an unstabilized approach"* -- LLM extracted: unstabilized approach

| Quantity | Zhang | Upgraded (direct) | Det parse | LLM parse |
|---|---|---|---|---|
| P(dragged wing/rotor/pod/tail) | 0.4172 | 0.419203 | 0.419203 | 0.419203 |
| P(substantial damage) | 0.2464 | 0.234905 | 0.234905 | 0.234905 |
| P(no injury) | 0.613 | 0.906949 | 0.906949 | 0.906949 |

## Verdict

- LLM parse landed on the identical evidence dict as the deterministic parser on **11/11** scenarios; wherever the dicts match, every posterior is identical (same network, same inference).
- Wherever they differ, the tables above show exactly how far the LLM's numbers drift from Zhang / the direct evidence.

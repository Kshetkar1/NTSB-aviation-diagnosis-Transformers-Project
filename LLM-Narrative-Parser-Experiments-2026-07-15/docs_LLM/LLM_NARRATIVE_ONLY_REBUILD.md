# Narrative-only rebuild: LLM re-codes the dataset, Zhang's analysis re-runs

gpt-4o-mini read the factual narrative of **1288** accidents (of 1742 in the 1982-2006 window; the rest have no usable free text) and reconstructed the coded record -- occurrence chain, findings, damage, injury -- from the narrative ALONE. Everything below compares the investigators' coded data (CODED) against the LLM's narrative-only records (LLM) over the SAME accidents.

## 1. Outcome extraction from the narrative

| Field | LLM vs coded record | Note |
|---|---|---|
| Aircraft damage | **558/917 = 61%** | LLM abstained (UNK) on 371 narratives that don't state damage |
| Highest injury | **1039/1282 = 81%** | default 'none' when the narrative reports nobody hurt |

## 2. Table 7 -- P(cause | fire), the paper's centerpiece table

Fire accidents found: CODED **83**, LLM narrative-only **128** (Zhang's full-window count: 102; this subset holds the accidents that have narratives).

| Cause (top 20 by combined support) | CODED n | CODED P | LLM n | LLM P |
|---|---|---|---|---|
| airframe/component/system failure/malfunction | 25 | 0.3012 | 13 | 0.1016 |
| fire warning system, powerplant | 1 | 0.0120 | 37 | 0.2891 |
| loss of engine power (total) - mechanical failure/malfunct | 7 | 0.0843 | 18 | 0.1406 |
| engine tearaway | 0 | 0.0000 | 21 | 0.1641 |
| fire extinguishing equipment | 3 | 0.0361 | 12 | 0.0938 |
| emergency procedure | 10 | 0.1205 | 4 | 0.0312 |
| aircraft/equipment inadequate, aircraft component | 1 | 0.0120 | 9 | 0.0703 |
| electrical system, electric wiring | 9 | 0.1084 | 1 | 0.0078 |
| fire warning system, airframe | 1 | 0.0120 | 7 | 0.0547 |
| auxiliary power unit (apu) | 6 | 0.0723 | 1 | 0.0078 |
| evacuation | 6 | 0.0723 | 0 | 0.0000 |
| fluid, fuel | 6 | 0.0723 | 0 | 0.0000 |
| 1 engine | 0 | 0.0000 | 6 | 0.0469 |
| aircraft/equipment inadequate, handling/performance capabi | 0 | 0.0000 | 6 | 0.0469 |
| cargo/baggage | 3 | 0.0361 | 2 | 0.0156 |
| fire warning system, cargo | 1 | 0.0120 | 4 | 0.0312 |
| loss of engine power | 1 | 0.0120 | 4 | 0.0312 |
| loss of engine power (partial) - mechanical failure/malfun | 4 | 0.0482 | 1 | 0.0078 |
| inadequate training | 0 | 0.0000 | 4 | 0.0312 |
| engine compartment | 4 | 0.0482 | 0 | 0.0000 |

- Spearman rank correlation over all 140 causes: **-0.427**
- Top-10 overlap: **2/10**
- L1 distance between the two P(cause | fire) vectors: **3.219**

## 3. Root priors (Eq. 6: occurrence count / 184,517,128 flights)

| Node | CODED n | LLM n | CODED prior | LLM prior |
|---|---|---|---|---|
| loss of engine power | 10 | 8 | 5.42e-08 | 4.34e-08 |
| fire | 83 | 128 | 4.5e-07 | 6.94e-07 |
| in flight collision with object | 36 | 42 | 1.95e-07 | 2.28e-07 |
| gear collapsed | 16 | 26 | 8.67e-08 | 1.41e-07 |
| loss of control - in flight | 41 | 266 | 2.22e-07 | 1.44e-06 |
| in flight encounter with weather | 182 | 239 | 9.86e-07 | 1.3e-06 |

- 311 nodes appear in both graphs; Spearman rank correlation of their occurrence counts: **0.436**

## 4. Full Bayesian network, Zhang's Section 4 recipe, both datasets

- CODED network: **693 nodes, 559 arcs**
- LLM narrative-only network: **334 nodes, 512 arcs**

### Table 9 forward anchors: P(loss of engine power | cause)

| Evidence | Zhang | CODED net | LLM net |
|---|---|---|---|
| engine instrument | 0.95 | 0.95 | -- |
| combustion assembly, combustion liner | 0.5 | 0.5 | -- |
| fluid, oil grade | 0.95 | -- | -- |

### Table 9 downstream column: P(target | loss of engine power)

| Target | Zhang | CODED net | LLM net |
|---|---|---|---|
| forced landing | 0.1429 | 0.2075 | 0.001769 |
| ditching | 0.00461 | 0.009433 | 0.01134 |
| gear collapsed | 0.00518 | 0.009433 | 0.004488 |


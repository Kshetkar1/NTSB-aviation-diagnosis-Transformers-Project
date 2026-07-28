# Friday deck — copyable slide text + teaching notes

Deck: `docs/friday_update.pptx` (8 slides, ~15 min). Rebuild: `python3.11 docs/build_friday_pptx.py`.
Every number below is a real computed output (scripts cited per slide).

---

## Slide 1 — Title

```
NTSB Bayesian Network — Friday Update
Your feedback → what we changed → the full pipeline, start to finish
Kanu Shetkar • Advisor: Dr. Mahadevan
```

## Slide 2 — Last week: what you flagged, and status

```
• "Evacuation is not a cause for fire — diagnosis and prognosis are mixed up" → FIXED: response events excluded from diagnosis; trees fully separated
• Sparse-data slide hard to follow → REVISED: every term (pair, n, subsamples, held-out) now defined on the slide
• Table 4: "we will only put the real example in the paper" → BUILT: real CPT P(fire | wiring, fuel) from 1,742 accidents
• Beta-CDF: "if it works fine, we can just go with that" → ADOPTED as default for sparse forward cells
• Plus two issues we found ourselves this week — both fixed and re-validated
```

## Slide 3 — The whole process, start to finish (running example: "engine caught fire during takeoff")

```
1. Detect outcome — parse query → "fire" → 102 fire accidents (Zhang's Table 7 denominator)
2. Embed + retrieve — rank all incidents by similarity → top-400 similar incidents
3. Partition — split the 102 fires into clusters: Cabin Fire (32), Electrical (21), Engine Fire (16)…
4. P(cause | K) — Zhang's Table-7 counting INSIDE each cluster (same labels, edges, denominator)
5. P(K | query) — cluster weights: the ONLY place the query enters (neutral = N_K/102; similarity = retrieval mass)
6. Chain rule — P(cause|Q) = Σ_K P(cause|K)·P(K|Q) → neutral recovers Table 7 EXACTLY
7. Build tree — recurse into causes of causes, with path probabilities

"Clusters" = the 102 fires grouped into 15 sets of similar accidents (cabin fires, electrical
fires, engine fires…); every fire is in exactly one group.
```

## Slide 4 — Everything we changed: before vs after

```
• Fire count: 38 → 102 exact (pre-2006 occurrence files were never merged)
• Prior P(fire): wrong denominator → 5.53e-7 exact to 12 decimals (BTS interpolation)
• Table 7: partial/mismatched → 85/85 cells exact (±0.0006)
• Diagnosis tree: "Evacuation", "Emergency procedure" ranked as causes → responses excluded, upstream causes only
• LTP layer: airframe 0.059 vs Zhang 0.314 (L1 = 1.96 of max 2.0) → Zhang counting inside clusters → L1 = 0, exact
• Clusters: 19 of 102 fires had no cluster (no narrative → never embedded) → backfilled from structured data → 102/102

L1 = total absolute difference between two probability distributions: 0 = identical,
2 = as far apart as mathematically possible.
```

## Slide 5 — The main fix: chain rule now provably matches Zhang

```
P(cause | query) = Σ_K P(cause | K) · P(K | query) — Zhang's Table-7 counting inside every cluster K

Query: "engine caught fire during takeoff"          OLD LTP    NEW neutral   Zhang T7   NEW similarity
Airframe/component/system failure                    0.059      0.3137        0.3137     0.3012
Electrical system, electric wiring                   —          0.0882        0.0882     0.0789
Loss of engine power (total) - mechanical            0.036      0.0882        0.0882     0.1098
L1 distance to Zhang Table 7                         1.96       0 (exact)     —          0.33 = query signal

• NEUTRAL = the query gives no information, so each cluster is weighted by its size (cabin fires: 32/102). SIMILARITY = clusters that resemble the query get more weight.
• Neutral → the clusters cancel algebraically → Table 7 EXACTLY, all 85 causes (not approximately — the algebra cancels; proven: tests/ltp_zhang_recovery.py)
• Similarity → engine query up-weights engine clusters (0.157→0.199) → LOEP rises 0.088→0.110 — the deviation IS the narrative signal, auditable per cluster
• Zhang's Table 7 is now the zero-information special case of our method
```

## Slide 6 — Full worked example: the diagnosis tree (figure: docs/figures/tree_diagnosis_fire.png)

```
• Root: FIRE — all 102 fire accidents (Table 7 denominator)
• Level 1 = Table 7 cells: wiring 9/102 = 0.088 · LOEP 9/102 = 0.088 · fluid/fuel 6/102 = 0.059 · APU 5/102 = 0.049
• Level 2 = causes of causes: e.g. wiring ← circuit breaker 2/14, maintenance/installation 2/14
• Every edge shows: probability, support n/denom, cumulative path probability
• No evacuation, no emergency procedure — responses excluded per your correction
```

## Slide 7 — Table 4: the real data-derived example

```
P(fire | electrical wiring, fuel system) — all four cells from the 1,742-accident window

wiring  fuel   n/denom    raw     Beta-CDF   final (0.95 cap)   Zhang T4 (toy)
Yes     Yes    1/1        1.000   1.000      0.950 (sparse)     0.99
Yes     No     10/21      0.476   0.717      0.476              0.93
No      Yes    23/45      0.511   0.753      0.511              0.95
No      No     68/1675    0.041   0.071      0.041              2e-9

• The 1/1 sparse cell is where the 0.95 cap / Beta-CDF machinery engages — a live demo of the sparse-data problem
• Toy values (0.99/0.93) unreachable from real counts — confirming Table 4 was illustrative, as you said
• THIS table goes in the paper
```

Source: `tests/build_table4_analogue.py` (offline).

## Slide 8 — Validation scoreboard + the ask

```
• Table 7 (85 causes, denom 102) — 85/85 exact
• Prior P(fire) — 5.53e-7 exact (12 decimals)
• Beta-CDF calibration — α=1.046, β=2.026, MSE 3.4e-7 exact
• Table 9 forward edges — 0.95 / 0.50 / 0.95 / 0.1429 exact
• Full offline parity suite — 13 PASS / 0 FAIL
• Chain rule → Table 7 (neutral weights) — EXACT, proven and tested
• Held-out diagnosis vs Zhang counting — BETTER on every metric (top-1 +5.6pp, log-loss −0.23, p ≤ 1e-8)

Held-out test in plain terms: hide each incident's answer, give both methods only the
narrative, ask them to guess the cause — ours guesses right more often, with
better-calibrated confidence, far beyond chance.

ASK: build phase is validated end to end — I'd like to start writing the paper (draft outline by next meeting)
```

---

# Teaching the diagnosis tree (colors + script)

**Colors (Streamlit app):** diagnosis = RED root + BLUE cause boxes; prognosis = PURPLE root + ORANGE event boxes.
Both drawn left→right, but the meaning differs:

- **Diagnosis:** left→right = digging BACKWARD into causes. Root (left) = what we observed. Each step right answers "why?".
- **Prognosis:** left→right = FORWARD in time. Each step right answers "what happens next?".

**Opening script (walk one path):**

> "The red box is what we observed: fire — all 102 fire accidents, the same denominator as your Table 7.
> Each blue box in the first column is a cause; the edge label is P(cause | fire) — literally a Table 7 cell:
> wiring is 9 out of 102, so 0.088. Then we recurse: of the 14 accidents where wiring contributed, 2 also had
> a circuit-breaker finding — P(circuit breaker | fire, wiring) = 2/14 = 0.143. The 'path p' under each box is
> the product along the path: 0.088 × 0.143 ≈ 0.013 — the probability of that whole causal explanation.
> Level one IS Table 7; deeper levels are Table 7's counting made recursive."

**Likely follow-ups:**
- "Why do denominators shrink to the right?" → each level conditions on the path so far (102 fires → 14 wiring-fires → …).
- "Why left-to-right if causes come first in reality?" → reading convention: we start from what's observed and explain backward; arrows mean "explained by," not "then."

# How diagnosis and prognosis are separated (Maha's point — closed)

1. **Different fields, different directions.** Diagnosis counts only findings NTSB investigators coded as
   Cause (C) or Factor (F) — `cause_factor_only=True` — upstream by definition. Prognosis follows the
   occurrence sequence in time order (`sequence_of_events`) — forward transitions, downstream.
2. **Response blocklist.** Labels that are responses, not causes — evacuation, emergency procedure,
   aborted takeoff, panic, fire extinguishing equipment (`DIAGNOSIS_RESPONSE_LABELS`) — are filtered
   out of every diagnosis tree.
3. **Regression test.** `tests/verify_diagnosis_separation.py` fails if any response label appears at
   level 1 of the fire diagnosis tree. It passes.

**One-sentence version:** "Diagnosis only counts what investigators coded as causes or factors — upstream —
and prognosis only follows the occurrence sequence forward in time; responses like evacuation are explicitly
excluded from diagnosis, and a test keeps it that way."

# Maha & Jesse — Presentation Guide

Use this with the **apps-parity** demo (`./run_app.sh`). Do **not** set
`NTSB_USE_TRAIN_INDEX=1` for the meeting — that shrinks the dataset to 177
incidents and breaks fire / Table 7.

---

## Zhang parity — test results (run 2026-08-02)

### Layer 1 — Diagnosis counting (Table 7, fire)

| Metric | Result |
|--------|--------|
| Cells vs Zhang | **85 / 85 exact** |
| Fire denominator | **102** (= Zhang) |
| Method | P(cause \| fire) counting, contributory C/F findings only |

**Demo query:** `engine caught fire during takeoff` → outcome **fire** (on full 1982–2006 window).

### Layer 2 — BN propagation (93 published cells, `bn_full_comparison.py`)

| Status | Count | Meaning |
|--------|-------|---------|
| **EXACT** | 43 | Matches Zhang to machine precision |
| **CLOSE** | 16 | Within ~15% relative or small absolute band |
| **DIFFERS** | 26 | Not close — mostly injury/damage multi-state cells |
| **QUALITATIVE** | 8 | No numeric benchmark (e.g. pilot node vocabulary) |
| **Close + Exact** | **59 / 93 (63%)** | |

### Table 9 only (what Prognosis demo shows — 5 evidence × 10 targets = 50 cells)

| Status | Count |
|--------|-------|
| EXACT | 31 |
| CLOSE | 4 |
| DIFFERS | 15 |

**Anchors Maha cares about (engine instruments evidence):**

| Target | Ours | Zhang | Verdict |
|--------|------|-------|---------|
| P(loss of engine power \| engine instruments) | 0.950 | 0.950 | EXACT |
| P(forced landing \| engine instruments) | 0.136 | 0.136 | EXACT |

**Where we differ (say this honestly):** low-probability injury/damage posteriors and some gear/ditching cells — upgraded network uses multi-state severity nodes; Zhang’s published Table 9 injury numbers don’t always match our encoding. **Lead with LOEP + forced landing**, not “no injury” row.

### Narrative → evidence caveat

Short phrase `trouble with an engine instrument during the flight` may **not** auto-parse to `engine instrument`. For the demo either:

1. Use Zhang’s wording: `The aircraft experienced inoperative engine instruments during flight.`, **or**
2. In Prognosis BN panel, manually select **`engine instrument`** in the evidence multiselect (numbers then match Table 9).

---

## Slide deck outline (12–15 slides)

### Act 1 — Problem & foundation (3 slides)

1. **Title** — Narratives as evidence for a frozen Zhang/Maha BN (1982–2006)
2. **What Zhang did** — BN from coded NTSB data; Table 7 diagnosis; Table 9 prognosis
3. **What we reproduced** — 85/85 Table 7; P(fire) exact; Table 9 anchors 0.95 / 0.136 / 0.1429

### Act 2 — Our contribution (3 slides)

4. **The new piece** — Free-text narrative → structured evidence → frozen BN (zero training)
5. **Leak-safe protocol** — Outcome phrases stripped before embedding/parsing
6. **Two modes** — Diagnosis (upstream) vs Prognosis (downstream); same evidence interface

### Act 3 — Live demo script (2 slides)

7. **Demo A — Diagnosis** — fire query; Table 7 column Ours \| Zhang \| Δ
8. **Demo B — Prognosis** — engine instrument evidence; BN P(LOEP), P(forced landing) vs Zhang

### Act 4 — Honest results (3 slides)

9. **Held-out severity (296 accidents)** — 90.9% / 77.4%; ties supervised baselines
10. **Held-out diagnosis** — 84.2% retrieval; BN path 57.7% (partial negative, reported)
11. **What BN adds** — Not accuracy — joint reasoning, what-ifs, auditable structure (Slide 8 framing)

### Act 5 — Close (2 slides)

12. **Parity scoreboard** — 59/93 close+exact; 12 structural diffs documented
13. **Ask** — Sign off estimand table + paper framing

---

## Live demo walkthrough — step by step

### Before you open Streamlit

```bash
cd Frozen-BN-Narrative-Evidence-2026-07-20/apps-parity
./run_app.sh
```

Confirm sidebar index label says **Zhang window 1982–2006**, not train-only.

---

### Demo A — Diagnosis (upstream causes)

**Say:** “You saw an outcome — what caused it?”

| Step | What you click | What to say |
|------|----------------|-------------|
| 1 | Sidebar → **Diagnosis** | “Upstream mode — Zhang Table 7 counting.” |
| 2 | Query: `engine caught fire during takeoff` | “Plain English; system detects **fire**.” |
| 3 | **Run analysis** | “Narrative embedded; similar accidents retrieved for optional tilt.” |
| 4 | **Evidence bar** | “Facts parsed for BN; diagnosis uses outcome + counting.” |
| 5 | Section 3 table | “Each row: **P(cause \| fire)**. Ours vs Zhang — should match for fire + full population.” |
| 6 | Point at airframe 32/102 = 0.314 | “Same as Zhang Table 7.” |
| 7 | Diagnosis tree | “Level 1 = Table 7; level 2+ exploratory — no Zhang benchmark.” |
| 8 | Optional BN section | “When several facts known — joint BN propagation.” |

**If outcome wrong:** use manual outcome picker → select **fire**.

---

### Demo B — Prognosis (downstream consequences)

**Say:** “Something happened — what does it lead to?”

| Step | What you click | What to say |
|------|----------------|-------------|
| 1 | Sidebar → **Prognosis** | “Downstream mode — BN propagation primary.” |
| 2 | Query: Zhang Table 9 text (see above) | “Narrative becomes evidence on BN nodes.” |
| 3 | **Run analysis** | |
| 4 | Section 2 **BN downstream** | “**P(target \| evidence)** — frozen network, not Markov tree.” |
| 5 | Evidence multiselect | Select **`engine instrument`** if not auto-parsed |
| 6 | Green banner “Matched Table 9 scenario” | “Evidence set matches Zhang column 1.” |
| 7 | Table rows LOEP + forced landing | “0.950 and 0.136 — match Zhang exactly.” |
| 8 | Expand Markov tree | “Our sequence explorer — labeled **not Zhang BN**.” |
| 9 | Epilogue audit | “Offline 93-value scoreboard — 43 exact, 16 close.” |

---

## One-sentence answers if they push back

| Question | Answer |
|----------|--------|
| Did you retrain the BN? | No — frozen on 1982–2006; narratives only set evidence at query time. |
| Do narratives enter the math directly? | No — parsed to NTSB vocabulary nodes, then propagation. |
| Why two probability types in Prognosis? | BN = Zhang-comparable; Markov tree = our exploratory forward model. |
| Why doesn’t every cell match Zhang? | 59/93 close+exact; rest mostly multi-state injury/damage encoding vs his published grid. |
| What’s your contribution? | Leak-safe narrative→evidence + honest held-out eval — not beating Zhang on counting. |

---

## Files to cite on slides

| Claim | Script |
|-------|--------|
| Table 7 85/85 | `docs_FrozenBN/table7_full_reproduction.csv` |
| BN 93-cell audit | `outputs/bn_full_comparison.json` |
| Held-out 296 | `outputs/heldout_significance.md` |
| Reproduce all | `./scripts/reproduce_foundation.sh` |

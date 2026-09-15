# Maha Meeting — Your Outline (filled in)
**Read aloud · 30 min meeting · Draft 3**

Use this structure. Each section = what Maha might ask under "what did you do?"

---

## 1. Problem — what is the issue?

| | |
|--|--|
| **What Zhang did** | Built an aviation **Bayesian network** from **coded** NTSB data (events, findings, injury/damage codes) — **not** from free-text narratives. |
| **What coded data misses** | Narratives describe **what happened in flight** and investigator context — detail codes may not capture the same way. |
| **Core issue** | No prior work (that we cite) connects **accident narratives** to a **reproduced, fixed** Zhang-style BN with **leak-safe** held-out evaluation on **both** prognosis (injury/damage) and diagnosis (causes). |
| **Scope** | FAR **Part 121** U.S. airline accidents (same scope as Zhang). |

**One sentence:** Zhang’s network reasons from **codes**; we ask whether **stories** can supply evidence on that **same locked network** without retraining on test labels.

---

## 2. Goal — what were we trying to do?

**Two jobs — both required:**

| Job | Goal |
|-----|------|
| **Job 1 — Verify** | Independently **rebuild** Zhang’s BN on **1982–2006** and **match his published numbers** (Table 7, priors, demo queries). |
| **Job 2 — Test** | On **held-out** accidents (**2007–2019**), map **redacted narratives** → evidence on the **locked** BN → predict NTSB **injury, damage, cause categories** — **without** refitting on test labels. |

**Contributions (what we added):**
1. Reproduce + verify Zhang (pyAgrum, same counting rules).
2. **Person nodes** + **4-level** injury/damage (extend graph, not replace it).
3. Narrative → evidence pipeline (phrase match + 100 similar past stories).
4. LLM **side experiments** (what works / doesn’t as label reader vs probability engine).

**What we are NOT trying to prove:** That we beat all ML baselines, or that the BN adds top-1 severity accuracy over counting neighbors.

---

## 3. Outcome — what happened?

| Job | Outcome |
|-----|---------|
| **Job 1** | **85/85** Table 7 rows match · **102** fire accidents · P(fire)=102/184,517,128 · Table 9/10 demo posteriors match Zhang |
| **Job 2 severity** | **90.9%** injury · **77.4%** damage top-1 (n=**296**) vs NTSB codes |
| **Job 2 diagnosis** | **84.2%** top cause **category** (n=**253**) — Personnel / Aircraft / Environment / **Organization** |

**Also report (honesty):**
- Same top injury/damage pick as **100-neighbor vote** on **all 296/296** — BN does **not** add top-1 severity lift.
- **TF-IDF** injury **92.2% > 90.9%** (supervised baseline on 82–06 only).
- Phrase→BN for causes **57.7%** << retrieval vote **84.2%**.
- Combined merge (phrase + neighbor severity one update) **38.5% / 41.9%** — failed by design.

---

## 4. H₀ and what actually happened

**Note:** The paper does **not** write formal statistical H₀ symbols. For Maha, say **research questions** and **what we expected vs found.**

| Question | Plain-language “null” / skeptic view | What happened |
|----------|--------------------------------------|---------------|
| **R1: Reproduce Zhang?** | Our rebuild might **not** match his tables | **Rejected** — 85/85 Table 7, demos match |
| **R2: Narratives help severity vs doing nothing?** | Stories might add **no** signal beyond prior (58.4% / 42.6%) | **Rejected** — 90.9% / 77.4% with redaction + retrieval |
| **R3: BN beats neighbor vote on severity?** | Network might secretly improve top-1 | **Not rejected — null holds** — **296/296 same** top pick |
| **R4: BN beats retrieval on causes?** | Parsed events through BN might beat neighbors | **Null holds for “beat”** — BN **57.7%**, retrieval **84.2%** |
| **R5: LLM replaces BN for P(·)?** | GPT might output good probabilities | **Rejected** — ~68% injury on 296 vs 90.9%; GPT unstable for P(·) |
| **R6: Merge all evidence paths?** | More evidence might always help | **Rejected** — fused path **38.5% / 41.9%** |

**Say to Maha:** “We state results as **questions tested**, not one global H₀. The important **negative** results — BN vs neighbors, TF-IDF, fused merge — are in the paper on purpose.”

---

## 5. Data — what we used vs what Zhang used

| | **Zhang** | **Us** |
|--|-----------|--------|
| **Accidents** | Part 121, **1982–2006** for BN build; same broad NTSB merge | Same **2,243** accidents **1982–2019** in merged dataset |
| **Build / train window** | **1982–2006** coded records → graph, CPTs, priors | **Same** — 1,742 accidents in training period |
| **Narratives for BN build** | **Not used** for network structure/CPTs | **1,288** narratives in 82–06 used only for **similarity index** (not CPT fitting) |
| **Priors** | Event counts ÷ **BTS U.S. airline departures** (184,517,128) | **Same rule** (Eq. 6) |
| **Test data** | Zhang: query **examples** on 82–06 (Table 7, 9, 10) — **not** 296 held-out narrative test | **296** accidents **2007–2019** for severity; **253** for cause eval (43 missing scorable causes) |
| **Overlap** | — | **Zero** test accidents in 82–06 training pool (leak audit) |

**One line:** Same **coded** foundation as Zhang for Job 1; we **add** narrative index + held-out **2007–2019** test Zhang did not do.

---

## 6. Did we compare to Zhang’s numbers? Same way? Did we check?

**Split Job 1 vs Job 2 — critical.**

### Job 1 — YES, compare to Zhang (same methods)

| What | How (same as Zhang) | Checked? |
|------|----------------------|----------|
| **Table 7** | Among **102 fires**, P(cause\|fire) = count/102 for **85** contributory rows | **85/85 match** · `verify_zhang_table7.py` · Appendix A |
| **P(fire) prior** | Occurrences ÷ BTS total flights | Matches Zhang |
| **Table 9 / 10 demos** | Set evidence on network → read posteriors (fire causes; engine instruments → P(loss of power)) | Match published anchors (e.g. **0.95**) |

**Say:** “Job 1 is **counting and posterior checks** on **1982–2006** — we replicated his **procedure**, not just his numbers.”

### Job 2 — NO, not compared to Zhang’s percentages

| What | Compare to | Why |
|------|------------|-----|
| **90.9% / 77.4% / 84.2%** | **NTSB coded truth** on **new** accidents | Zhang has **no** held-out narrative benchmark for 296 |
| **Baselines** | Prior, phrase-only, TF-IDF, LR, embedding classifiers | Our eval design (Table 6, 8) |

**Do not say:** “90.9% validates Zhang.” Say: “90.9% validates the **narrative pipeline** on held-out accidents.”

### Did we check evaluations?

| Check | What |
|-------|------|
| **Leak audit** | 0/296 test IDs in embedding pool |
| **Redaction probe** | Unstripped text inflates severity — why we strip |
| **Agree-on-top-pick** | bn-sev vs retrieval-sev **296/296** same injury/damage top label |
| **Scripts + outputs** | `frozenbn_heldout_narrative_bn_eval.py` → `heldout_significance.md`; `diagnosis_heldout_eval.py` → `diagnosis_heldout_eval.md` |
| **Reproduce** | `REPRODUCE.md` |

---

## 7. Pipeline — path, what each part does, why we chose it

```
STEP 1  STRIP outcome phrases (injury/damage stated in text)
           WHY: anti-leak — narratives state the answer
           PROOF: redaction_leak_probe.py

STEP 2  PHRASE MATCH  ║  STEP 3  100 SIMILAR 1982–2006 STORIES (embed + cosine)
        (same clean story, SAME TIME)
           WHY 2: Zhang vocabulary when words match
           WHY 3: narrative signal from past similar accidents without training on 296 labels
           PROOF: query_to_bn.py

STEP 4a  EVENT fractions from 100  ║  STEP 4b  CODED injury/damage counts from 100
           (SAME TIME, same neighbors)
           WHY 4a: suggest event nodes (soft evidence)
           WHY 4b: severity from past **coded** outcomes — main prognosis path
           PROOF: retrieval_facts / severity_retrieval_distributions

STEP 5  UPDATE LOCKED NETWORK (pyAgrum)
           MAIN severity (90.9%): **4b only** → injury/damage nodes
           ALT: phrase-only paths in Table 6
           FAILED: all merged → 38.5%/41.9% (double-count)
           WHY locked: 296 = exam; no CPT refit
           PROOF: Table 5, FREE_PARAMETERS.md

STEP 6  READ ANSWERS
           6A: argmax P(injury), P(damage) from Step 5
           6B: **vote** 4 cause categories from Step 3 neighbors → 84.2%
           WHY 6B vote not BN rank: beats phrase→BN; era-fair categories

STEP 7  GRADE vs NTSB codes — codes NEVER inputs
           PROOF: eval scripts load truth after prediction
```

**Why k=100?** Fixed default before test; k=25 ablation ~flat (89.9% vs 90.9%) — not tuned on 296.

**Why 4 cause categories?** 2008 taxonomy change — exact cause strings unfair across eras.

---

## 8. LLM — what we did and why it doesn’t work (as main method)

**We ran four experiments (§8.3) — not “no LLM.”**

| Experiment | What | Result |
|------------|------|--------|
| **1 — Zhang 11 texts** | GPT reads query → same nodes as phrase match | **11/11** same nodes; same posteriors as rules |
| **2 — 296 test** | GPT extracts event evidence | ~**68%** injury, ~**52%** damage vs **90.9% / 77.4%** pipeline |
| **3 — Recode 1,288 train narratives** | GPT assigns codes from prose | Injury/damage partly OK; **cause ranking fails** for fire (Spearman **−0.43**) |
| **4 — GPT outputs P(·) directly** | No BN propagation | Confident wrong numbers; small query change → big swing |

**Why it doesn’t work as the engine:**
- LLM **does not use historical CPTs** — probabilities don’t propagate through the graph.
- **Poor calibration** on 296 (ECE ~0.74 vs ~0.12 for retrieval strengths).
- **Findings/causes** in prose ≠ investigatory Cause/Factor codes.

**What we kept:** GPT can **read** short text into **labels** on Zhang’s demos; **BN always computes P(·)**.

**Future (§10.3):** Agent workflow to **draft** codes → build **new** network — **next iteration**, not this paper’s test eval.

---

## 9. Metrics — what Zhang used vs what we used

### Zhang (Job 1 — verification)

| Metric / check | Used for |
|----------------|----------|
| **P(cause \| fire)** row match | Table 7 — **85 rows** |
| **Posterior match** | Table 9, 10 style queries (e.g. P(loss of engine power \| instruments)) |
| **Prior P(event)** | Count/BTS flights |
| **No held-out narrative accuracy** | He did **not** report 296-accident top-1 narrative eval |

### Us — Job 1 (same as Zhang)

Same **counting** for Table 7; same **posterior** checks for demos.

### Us — Job 2 (our new eval — Zhang did not do this)

| Metric | Definition | Where |
|--------|------------|-------|
| **Top-1 accuracy** | Predicted class = NTSB coded class | Table 6, 8 |
| **Macro-F1** | Unweighted mean F1 over 4 injury / 4 damage / cause classes | Table 6, 8 |
| **Brier score** | Squared error of full probability vector (lower better) | `heldout_significance.md` |
| **95% bootstrap CI** | 10,000 resamples, seed 42 | `heldout_significance.md` |
| **McNemar + Holm** | Pairwise top-1 significance vs baselines | `heldout_significance.md`, REPRODUCE.md |

**Say:** “Zhang’s metrics are **replication checks** on 82–06. Our **296 test metrics** are standard classification scores vs NTSB truth — **not** comparable to a Zhang percentage because he didn’t run this test.”

---

## 10. Add these two blocks (your outline was missing them)

### What does NOT work (say before he asks)

- BN top-1 severity **≠ better** than neighbor vote (**296/296**)
- **TF-IDF > us** on injury
- **84.2% ≠ BN phrase path** (57.7%)
- **Fused merge** 38.5% / 41.9%
- **Fatal injury** rare (3/296) — weak class
- **296** used while debugging eval — slightly optimistic (not CPT leakage)

### Paper weak / what you’d do differently

- Split Job 1 vs Job 2 clearer in Results
- §6.5 should lead with **primary path (4b-only)** not “all fused”
- Abstract should say **84.2% = retrieval vote**
- Confirm on fresh **2020–24** data never used in dev

---

## 11. Am I good if I know all of this?

**Yes [Certain]** — for a **30-minute “what did you do”** meeting, this outline **plus Section 10** is enough **if you say it aloud**.

**Minimum memorized blocks:**
1. **Two jobs** + four headline numbers  
2. **Pipeline** (strip → 2∥3 → 4a∥4b → 5 → 6 → 7)  
3. **Job 1 compare Zhang / Job 2 compare NTSB** — not the same  
4. **Three failures:** neighbors tie, TF-IDF wins injury, fused merge fails  
5. **LLM:** four tests, BN does P(·)

**Skip for this meeting:** Related work details, brake-wear §4 example, every Table 7 row.

---

## Quick reference card (photo this)

| | |
|--|--|
| **Problem** | Zhang = codes only; we test narratives on fixed BN |
| **Goal** | Job 1 verify · Job 2 held-out narrative test |
| **Outcome** | 85/85 · 90.9/77.4/84.2 |
| **H₀-ish** | BN beat neighbors? **No.** Narratives help vs prior? **Yes.** |
| **Data** | Build 82–06 · Test 07–19 · Same Part 121 as Zhang |
| **vs Zhang numbers** | Job 1 yes same counting · Job 2 no — vs NTSB |
| **Pipeline** | Strip → 2∥3 → 4a∥4b → 5(4b) → 6 → 7 |
| **LLM** | Side tests; ~68% inj; BN computes P(·) |
| **Metrics** | Zhang: table match · Us: top-1, Macro-F1, Brier, CI, McNemar |

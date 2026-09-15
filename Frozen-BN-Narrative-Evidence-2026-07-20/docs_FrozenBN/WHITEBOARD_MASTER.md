# WHITEBOARD MASTER — pointer to daily wall
**Use `WHITEBOARD_PIPELINE_WALL.md` only** — plain language, 4 boards top→bottom, colors per section, flow picture, **DO NOT WRITE** jargon list, all terms explained.  
This file below is optional shorthand — **do not write both.**

---

## TODAY — 9:15–11:30 (write boards, not read draft first)

| Time | Do | Done when |
|------|-----|-----------|
| **9:15–9:25** | Print/open PDF + this file on laptop. Color key + both headers on boards. | Keys visible |
| **9:25–9:55** | **Board 1** cols 1→4 (problem → numbers → red box). Add `p.__` from PDF as you go. | Story readable left→right |
| **9:55–10:30** | **Board 2** ⓪→⑦ + blue §9 branch. Draw arrows; label k=100, redact first. | You can trace one test acc through all steps |
| **10:30–10:50** | Sidebar R1–R10 (tiny). Photo both boards. | Phone has backup |
| **10:50–11:15** | Read draft **once** with board: only §6, §7, §8.1–8.2, §9.1–9.2. Mark `p.__` gaps. | Every green number has a draft table |
| **11:15–11:30** | Say **60-sec open** aloud ×3. Drill “where is X?” table (bottom). | No stumble on pipeline order |

**Tonight (optional 20 min):** re-read §10.1–10.3 + Abstract. **Tomorrow AM (15 min):** photo boards + 60-sec open before meeting.

**Meeting tomorrow:** Topic may not be the paper — still know: (1) two jobs verify vs test, (2) nothing trained on 296, (3) honest limits in red ink, (4) 2020–24 confirmatory = **future work** (data not in repo yet).

---

## DRAFT § → WHITEBOARD → REPO (fill `p.__` from your PDF)

| Draft § / Table | Board zone | One-line “what lives here” | Repo / run (if challenged) | p.__ |
|-----------------|------------|----------------------------|------------------------------|------|
| Abstract | B1 header | Claim in 3 bullets: verify Z + narr test + honest limits | — | |
| §1 Intro | B1 Col1 | Problem, 4 contributions, roadmap | `MASTER_DRAFT_PASTE_BLOCKS.md` | |
| §2 Related work | — (skip board) | BN aviation, narr mining, virtual evidence | — | |
| §3 Data | B1 Col2 boxes | 82–06 build vs 07–19 test; n=296, n=253 dx | `shared/data/processed/refined_dataset.json` | |
| Table 1 | B1 Col2 | Cohort counts | `outputs/cohort_manifest.json` | |
| §4 BN primer | B2 R5 tiny | Tables 2–4 toy; how CPTs work | draft only | |
| §5 Build Z | B2 R5 blue | Graph, priors, CPTs, T7 counting, upgrades | `code/bn_upgraded.py`, `trees.py` | |
| §5.1–5.7 | B2 R5 | Data prep → Beta-CDF → T7 | `tests/verify_zhang_table7.py` | |
| §6.1 Redaction | B2 ① red | Strip stated inj/dmg before match/retr | `query_to_bn.py`, `tests/redaction_leak_probe.py` | |
| §6.2 Hard | B2 ② | Phrase → node YES | `parse_query_to_bn_evidence` | |
| §6.3 Retr + soft | B2 ③④a | k=100 neighbors, event strengths | `retrieval_facts.py` | |
| §6.4 Virtual sev | B2 ④b | Neighbor inj/dmg → virtual evidence | `retrieval_severity_virtual_evidence` | |
| §6.5 Propagation | B2 ⑤ | One LazyPropagation | pyAgrum | |
| Table 5 | B2 R1 / §6.6 | Fixed vs trained inventory | `FREE_PARAMETERS.md` | |
| §7 Eval protocol | B1 Col3 Job B | Grade only; metrics; discordant def | `tests/frozenbn_heldout_narrative_bn_eval.py` | |
| §8.1 + Table 6 | B1 Col4 | **90.9 / 77.4**; 0 discordant vs retr | `outputs/heldout_significance.md` | |
| §8.2 + Table 8 | B1 Col4 + R2 | **84.2** retr dx; **57.7** BN events | same eval script | |
| §8.3 LLM | B2 R3 red | 4 experiments; ECE ~0.74 | `tests/llm_*` | |
| §8.4 Summary | B1 red | Don’t oversell; TF-IDF 92.2 > pipeline | — | |
| §9.1 + Table 9 | B2 blue branch | Z fire dx demo (82–06) | `apps-parity/diagnosis_view.py` | |
| §9.2 + Table 10 | B2 blue branch | Engine instruments LOEP ~0.95 | `apps-parity/prognosis_view.py` | |
| §10.1–10.3 | B1 R8 / R9 | Limits, 2020–24 future, JAIS path | `SECTION_8_DISCUSSION.md` | |
| App A Table 7 | B2 R4 | 85/85 verify | `Appendix_A_Table7.csv` | |

---

## COLOR KEY (both boards, top corner)

| Ink | Means |
|-----|--------|
| **Black** | Structure, boxes, arrows, most text |
| **Blue** | Zhang / 82–06 / verify / his published examples |
| **Green** | Your extension / 07–19 test / pipeline / results |
| **Red** | Warnings, redaction, grade-only, don’t-claim, failures |

## ABBREV KEY

| Abbrev | Means |
|--------|--------|
| BN | Bayesian network (NOT neural net) |
| Z | Zhang & Mahadevan RESS 2021 [2] |
| acc | accidents |
| narr | narrative |
| evidence | hard / soft / inj-dmg (not “ev”) |
| retr | retrieval = k=100 similar training acc |
| sev | severity |
| dx | diagnosis (upstream causes) |
| prog | prognosis (downstream inj/dmg) |
| inj | personnel injury |
| dmg | aircraft damage |
| T7 / T9 | Table 7 / Table 9 |
| CPT | conditional probability tables |
| P(·) | probability |
| k | neighbor count (default **100**) |
| Part 121 | FAR airline ops (scope of paper) |

---

# BOARD 1 — STORY (4 columns)

## HEADER (blue)
**NTSB narr → evidence on FIXED BN**  
BN computes all P(·) · narr supplies evidence · no retrain on test labels

---

### COL 1 — PROBLEM · GOAL · OUTCOME · CONTRIBUTIONS

**PROBLEM**
- NTSB = **coded seq** (event codes, findings) + **free-text narr**
- Codes = path (eng fail → water landing → destroyed → deaths) but **lose report context**
- Z: BN from **coded data only**, window **1982–2006**
- Gap: can **narr** become **evidence** on same graph for **new** acc?

**GOAL**
1. Rebuild Z BN from coded 82–06
2. Verify Z published checks (T7, fire prior, T9-style queries)
3. Map **redacted narr** → evidence on **fixed** BN (**no retrain**)
4. Score **07–19** test acc vs NTSB codes
5. Side test: gpt-4o-mini as **text→evidence reader only** (§8.3)

**OUTCOME (deliverables)**
- Fixed BN + pyAgrum pipeline + cohort manifest + repo runs
- Verify: **85/85 T7**, **102** fires, P(fire)=**102/184,517,128**, T9 ex LOEP≈**0.95**
- Test: **n=296** inj **90.9%** dmg **77.4%** · **n=253** dx **84.2%** (4 cats)
- Draft 3 → Maha markup → JAIS path
- Streamlit demo (§9 cases)

**4 CONTRIBUTIONS (intro)**
1. Rebuild Z BN; match published query tables (pyAgrum)
2. Extend: **person-role** finding nodes + **4-level** inj/dmg (separate from dx cats)
3. Narr → evidence (phrase match + retr) **without retraining**
4. Four LLM experiments: extraction only, not P(·)

**Motivation (not proven here):** deeper understanding → root cause → safer ops. **Paper = retrospective on archived reports.**

---

### COL 2 — RQ · DATA · SCOPE

**MAIN RQ**  
Can **redacted narr** → evidence on **fixed** BN **w/o retrain** → match NTSB codes on **07–19**?

**SUB-RQ**  
Can **gpt-4o-mini** extract evidence **without** computing P(inj/dmg/cause)?

**NOT:** “LLM + BN is the main model” · “LLM does probabilities”

**SCOPE**
- **Part 121** airline acc only (14 CFR Part 121)
- Build/verify: **1982–2006**
- Test narr: **2007–2019**
- Large-aircraft NTSB refined corpus

```
┌ BLUE: 82–06 BUILD + VERIFY ─────────────┐
│ 1,742 acc in Z window                   │
│ 1,288 narr in corpus (not all in test)│
│ Network graph, CPTs, priors             │
│ Retr index built HERE only              │
│ Z checks: T7 (85 rows), fire, T9 ex     │
│ ≠ narrative generalization test         │
└─────────────────────────────────────────┘
              ↓ FIXED (never retrain from test)
┌ GREEN: 07–19 TEST ──────────────────────┐
│ 296 acc w/ narr_accf ≥100 chars         │
│ 253 w/ scorable dx findings             │
│ 43 no coded causes → out of dx score    │
│ All predictors: same 296, same redaction│
│ Text truncated 4k chars                   │
│ NTSB codes = GRADE ONLY (never inputs)  │
│ Strip stated inj/dmg phrases FIRST      │
└─────────────────────────────────────────┘
```

**RED:** Why not test narr on 82–06? Same data built BN + retr pool → **circular**, inflated scores.

---

### COL 3 — TWO EVALUATIONS · PAPER MAP

**BLUE A — VERIFY Z (Job A)**
- Same era as build (82–06)
- **Counting:** 102 fires → T7 **85 cause rows** P(cause|fire)
- **Scenario inference:** set **coded evidence** on named nodes → read posteriors (T9, gear figs)
- Q: **Did we rebuild correctly?**
- **Not** narr prediction · **Not** 90.9%

**GREEN B — TEST NARR (Job B)**
- New-era acc (07–19)
- narr → evidence → **one BN inference** → prediction
- vs NTSB coded inj/dmg/cause (grade only)
- Q: **Does text work on unseen years?**
- Z **never** did this

**RED:** A ≠ B — explain both; never merge

**PAPER § MAP (tiny)**
§2 background · §3 data · §4 BN primer · §5 build Z net · §6 narr bridge · §7 eval protocol · §8 results · §9 case studies · §10 conclusion · App T7

---

### COL 4 — NUMBERS · HONEST · DON’T CLAIM

**SEVERITY n=296 (§8.1) — top-1 accuracy**
| Predictor | Inj | Dmg |
|-----------|-----|-----|
| Prior (majority) | 58.4 | 42.6 |
| Parsed events (hard+soft) | 82.4 | 50.7 |
| Supervised LR (parsed) | 85.5 | 60.1 |
| TF-IDF LR (text) | **92.2** | 73.3 |
| Embedding LR | 91.6 | **74.0** |
| **bn-sev (primary)** | **90.9** | **77.4** |
| retr-sev (no BN) | 90.9 | 77.4 (identical 0/296 discordant) |

**Best dmg Macro-F1: 0.697 (bn-sev)** · severe screen dmg sens **75.3%**

**DIAGNOSIS n=253 (§8.2) — 4 cats P-A-E-O**
| Predictor | Top-1 |
|-----------|-------|
| Freq baseline | 45.8 |
| BN event evidence | 57.7 |
| **Retr (primary)** | **84.2** |
| Supervised emb-LR | 88.1 |
| BN lift ranking (negative) | 50.2 |

**VERIFY Z:** 85/85 T7 · 102 fires · 11/11 LLM=phrase on Z ex texts

**RED — HONEST (say aloud)**
- Sev: bn-sev **=** retr-sev **296/296** (BN adds **no** hidden accuracy)
- Dx: retr **84.2** >> BN events **57.7** (−**26.5** pts)
- Fuse event ev + retr sev in **one** update → inj **38.5%** dmg **41.9%** (worse)
- Fire cross-test: BN parsed events AUC **~0.38–0.46** · retr **~0.96–0.98**
- LLM conf ECE **~0.74** · retr strengths **~0.12**
- Fatal inj rare: 3/296 — model misses some (disclosed §8.1)
- Org category: ~0/25 top-1 for retr (rare class)
- Test acc reused in dev → may be slightly optimistic (§7/§10.3) — **not** label leakage into CPTs

**RED — DON’T CLAIM**
- LLM computes P(·)
- BN beats retr on dx
- BN adds sev accuracy beyond neighbors
- Live / in-flight prevention
- Full cause-level dx on test (only **4-category** rollup)
- “Benchmark” for Z examples (say **published examples**)
- BN cross-node inference beats retr on fire

**REAL CLAIM**
- Narr → evidence on **fixed validated** BN
- Matches supervised sev models **without training** + joint reasoning / what-if
- Retr drives dx; honest about limits

---

# BOARD 2 — HOW (pipeline + sidebar)

## HEADER (blue)
**PIPELINE §6 / Fig 2** — walk **⓪→⑦** for Maha

---

## CENTER — STEPS ⓪–⑦

```
⓪ INPUT (green)
   Test acc narr, 07–19, Part 121
   narr_accf ≥100 chars · truncate 4000 chars

① REDACT (red) §6.1
   Remove STATED inj/dmg phrases before match OR retr
   WHY: leak-safe — don’t read answer from text
   ALL later steps use redacted text ONLY

② HARD EVIDENCE (black) §6.2
   Rule-based phrase match → Z NTSB node vocabulary
   Match → node = YES, c=1.0
   Ex: "inoperative engine instruments" → engine instrument=Yes
   Negation/hypothetical clauses skipped
   No match → no hard ev (LLM fallback = §8.3 expts only)
   → step ⑤

③ RETRIEVAL INDEX (green) §6.3
   Embed redacted narr
   Search **82–06 ONLY** (test acc never in pool)
   k=**100** most similar training acc = neighbor set
   Same neighbors for ④a and ④b
   Empirical counts — NOT tuned on test labels
   (internal slice favored k=25; test plateau 89.9/77.7 vs 90.9/77.4)

④a SOFT EVIDENCE — EVENTS (green) §6.3
   For candidate event/finding nodes:
   strength = (# neighbors w/ coded label) / 100
   Ex: 71/100 landing phase → 0.71
   Pearl-style strengths on **event** nodes
   → step ⑤

④b INJ/DMG FROM NEIGHBORS (green) §6.4
   SAME 100 neighbors → read CODED inj & dmg (not events)
   Count each level ÷ 100 → fractions on inj/dmg nodes
   Ex inj: 41 none, 28 min, 24 ser, 7 fat
   Pearl virtual evidence [16] · pyAgrum addEvidence
   Drives **prog** path · Dx primary readout = retr cats (④a path), not ④b
   → step ⑤

⑤ ONE INFERENCE (blue) §6.5
   Fixed BN from §5 (82–06 build, never updated)
   Load ②+④a+④b → pyAgrum LazyPropagation
   Single joint update → posteriors
   (Inj & dmg sometimes inferred per-target separately — avoids coupling)

⑥ READ OUT (green)
   ├ PROG (downstream): P(inj levels), P(dmg levels)
   │   Levels inj: FATAL · SERIOUS · MINOR · NONE
   │   Levels dmg: DESTROYED · SUBSTANTIAL · MINOR · NONE
   └ DX (upstream): rank 4 cause categories
       Personnel · Aircraft · Environment · Organizational
       (era-fair rollup: legacy + CICTT → §8.2 keyword rules)

⑦ GRADE (red) §7–§8
   Compare pred vs NTSB coded truth
   Truth NEVER model input
   Metrics: top-1 acc, Macro-F1, Brier, bootstrap CI, McNemar+Holm
```

**RED X:** `Query? → LLM → answer` **WRONG** (skips redact, retr, BN)

**BLUE SIDE BRANCH — §9 Z PUBLISHED EXAMPLES (not test path)**
- Typed/short query → **hard evidence only** · **no k=100**
- Same step ⑤ fixed BN
- **9.1 Dx demo:** fire query → T7 cause posteriors (82–06 counts)
- **9.2 Prog demo:** engine instruments → T9 P(LOEP)=**0.95**
- Illustrates narr **can** match Z when vocabulary maps cleanly
- **Not** substitute for §8 test eval

---

## SIDEBAR — R1–R10

**R1 — THREE EVIDENCE TYPES**
| Type | Source | On nodes |
|------|--------|----------|
| Hard | Phrase in narr | event/finding = YES |
| Soft | 100 nbrs | **event** fractions |
| Inj/dmg | same 100 nbrs | **inj + dmg** fractions |

Pearl [16] · Jeffrey-style likelihoods in code · pyAgrum [10]

**R2 — DX vs PROG**
| | Question | Primary test readout |
|--|----------|---------------------|
| Prog | What inj/dmg resulted? | **90.9 / 77.4** (retr→BN) |
| Dx | What caused? (4 cats) | **84.2 retr** · **57.7 BN events** |

253 not 296: **43** lack scorable coded causes  
4 cats **P-A-E-O** because **2008 taxonomy break** (legacy vs CICTT)

**R3 — LLM §8.3 (red border) — 4 experiments**
1. **Z 11 ex texts:** LLM ev = phrase match **11/11** · same posteriors
2. **296 test narr:** LLM ev ~**68%** inj ~**52%** dmg vs **90.9/77.4** · ECE **~0.74**
3. **Recode 1288 train narr:** inj ~**81%** dmg ~**61%** · fire cause ranking **fails** (Spearman **−0.43**, top-10 overlap **2/10**)
4. **LLM P(·) w/o BN:** confident wrong numbers · no evidence propagation

**LLM role:** text → evidence labels ONLY · **BN always P(·)**

**R4 — VERIFY Z (blue) — separate from ⓪–⑦**
T7 85/85 · 102 fire · scenario posteriors  
Population + illustrative queries · **not** 90.9%

**R5 — §5 BUILD (blue, tiny)**
82–06 coded → graph · priors · CPTs · Beta-CDF smooth · parent selection · T7 counting · person-role nodes · 4-level inj/dmg · small upgrades to run Z examples · ~**12/93** full-network cells ≠ paper (tie-break; his released file also mismatches some cells)

**R6 — §8 PREDICTOR NAMES (tiny)**
bn-sev · retr-sev · hard+soft · prior · bn-fused (negative) · lr · tfidf-lr · emb-lr

**R7 — FIRE TEST (red, tiny) §7.6**
Parsed events → P(fire) through BN **fails** · retr **works**

**R8 — LIMITS (red)**
Part 121 only · phrase dict limited · test reused in dev · mapping audit ≤2 pt dx uncertainty · 296 vs **295** typo in §8 intro (fix to 296)

**R9 — NEXT (black)**
Maha edits · fresh acc 2020–24 · expand §9 · more phrases · JAIS framing · agent workflow (future, §10.3)

**R10 — REFS (tiny)**
[2] Z RESS 2021 · [16] Pearl 1988 · [10] pyAgrum · [3] LLM hallucination survey

---

# OVERFLOW STICKY (hand — if board full)

- **Preliminary mistake:** early test asked LLM for P(·) directly → unstable, wrong → pivoted to BN-only probabilities
- **Truth inj rule:** zhang_injury_code (prognosis.py)
- **Cohort file:** outputs/cohort_manifest.json
- **Reproduce:** Frozen-BN-Narrative-Evidence-2026-07-20/REPRODUCE.md
- **Acknowledgments:** you write; AI = rubric + code; numbers from verified runs

---

# 60-SEC MAHA OPEN

“Codes miss the story. We rebuilt your BN on 82–06 and verified your tables. Then we asked whether redacted narratives can feed that **fixed** network on 07–19 without retraining. **Blue** is your verification; **green** is our narrative test. **Red** is what we won’t oversell. Board 2 is the pipeline — I’ll walk ⓪ to ⑦.”

---

# QUICK POINT — “WHERE IS X?”

| He asks… | Point to… |
|----------|-----------|
| Why 07–19? | B1 Col2 green box |
| Redaction? | ① red |
| 84.2%? | ④a + B1 Col4 + R2 |
| 90.9%? | ④b + ⑤ + R2 |
| BN vs retr? | B1 Col4 red |
| LLM? | R3 |
| His T9 example? | Blue branch + R4 |
| 4 dx cats? | ⑥ + R2 P-A-E-O |
| Person-role nodes? | B1 Col1 contrib #2 |
| Why fixed? | B1 Col2 arrow “no retrain” |

---

# DRAW ORDER (~90 min)

1. Both boards: color key + headers (5 min)
2. Board 1 cols 1→4 (30 min)
3. Board 2 steps ⓪–⑦ + arrows (35 min)
4. Sidebar R1–R10 (20 min)
5. **Photo both boards** + keep this file on phone

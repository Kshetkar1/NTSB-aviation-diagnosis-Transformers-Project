# Jesse meeting — 15-minute deck (undergrad-friendly)

**Goal:** Someone who has not looked at this in months (or an undergrad new to research) can follow start → dead ends → final system → why structural mapping was dropped.

**Format:** Same as `tuesday_jul21` / `friday_update` — short bullets, one table per slide, **bold takeaway** under each table, speaker notes in italics.

**Timing:** ~10 main slides + 2 backup = **15 min** (~1.2 min/slide). Appendix only if he asks.

**Rebuild deck:** `python3.11 docs/build_jesse_jul28_pptx.py` → `docs/jesse_jul28_update.pptx`

---

## Slide 1 — Title (~30 sec)

```
NTSB Narrative Project — Full Story
From LLM-only → Zhang's network → what actually works
Kanu Shetkar  •  Jesse Spencer-Smith
July 28, 2026
```

*SAY:* "Fifteen minutes, one story: what we tried, what failed, what we kept, and especially everything we did on structural mapping and why we stopped."

---

## Slide 2 — Today's map (~1 min)

```
Where we started
  • Zhang & Mahadevan (2021): Bayesian network from coded NTSB data
  • Your idea: structural mapping — match accidents by causal chain shape, not just words

What went wrong first
  • Letting the LLM compute probabilities directly (unreliable, not reproducible)

What we built instead
  • LLM reads the story → frozen network (built from coded data) does the math

What we proved
  • Reproduce Zhang's tables; predict injury on 296 unseen accidents (~91%, leak-safe)

What we tested for you
  • Structural mapping in every form we could think of — it never beat the baseline

Where we are
  • Writing the paper; structural mapping → honest negative result in appendix
```

*SAY:* "Think of three layers: reproduce the calculator, teach it to read narratives, test your structural idea fairly."

---

## Slide 3 — The one picture (~1.5 min)

**Use figure:** `docs/figures/narrative_to_bn_architecture.png`

```
BUILD TIME (once, 1982–2006 coded data)
  • Zhang's recipe → frozen Bayesian network (validated: Table 7 exact, 77/93 network cells)
  • Narratives → embedding index + clusters (for similarity / soft evidence)

QUERY TIME (every new accident story)
  COUNTING PATH: "What cause is likely?" → retrieve similar accidents → weighted counts
  NETWORK PATH:  "What happens given these facts?" → parser → evidence → BN propagation

One rule: the narrative never retrains the network. It only enters as evidence.
```

**Takeaway:** *The LLM is the reader. The network is the calculator.*

*SAY (undergrad analogy):* "We trained a calculator from 1,742 old accidents and froze it. New stories only fill in the form — they don't change the calculator."

---

## Slide 4 — Dead end #1: LLM-only (~1 min)

```
We first asked: can GPT just read narratives and output probabilities?

What happened
  • Sounds confident but numbers are wrong (~10× off on Zhang's scenarios)
  • Change one word → whole answer flips (not reproducible)
  • No real evidence propagation (flat answers when evidence stacks up)

Lesson (Section 3.5 in the paper)
  • LLM = translator (text → structured facts)
  • Data-built network = calculator (facts → probabilities)
```

**Takeaway:** *Probabilities must come from the coded database, not from the LLM's memory.*

*SAY:* "This is the most important design rule in the project."

---

## Slide 5 — Reproduction: issues we hit fixing Zhang (~1.5 min)

| Problem we found | Fix | Result |
|------------------|-----|--------|
| Fire count 38 not 102 | Merge all occurrence files | Denominator 102 ✓ |
| Table 7 partial | Filter to Cause/Factor codes only | **85/85 exact** |
| P(fire) wrong | BTS departures denominator | **5.53×10⁻⁺ exact** |
| No person nodes | Link findings to pilot/crew | Fig. 12 pilot queries run |
| Boolean injury | 4-level injury/damage nodes | Realistic P(none); held-out eval |
| Random tie-breaks | Deterministic rebuild + 30-seed envelope | 12 gaps explained honestly |

**Takeaway:** *We can trust the network before we trust the narrative pipeline.*

*SAY:* "This took months. We didn't skip to narratives until the calculator matched Zhang."

---

## Slide 6 — The final pipeline (~1.5 min)

```
Tier 1 — Deterministic (free, exact)
  • If the narrative uses NTSB vocabulary → match directly, confidence 1.0
  • 10 of 11 Zhang scenario sentences never needed the LLM

Tier 2 — LLM fallback (only when Tier 1 finds nothing)
  • Reads described facts → maps to vocabulary
  • Strength of each fact measured from data (not LLM confidence)

Evidence types → frozen network
  • Hard: named codes set directly
  • Soft: described facts via Jeffrey conditioning
  • Stated: severity phrases in the text
```

**Takeaway:** *Deterministic first; LLM only when needed; data sets the strengths.*

---

## Slide 7 — Does it work? Held-out 296 (~1.5 min)

*Accidents 2007–2019 — never used to build the network. **Leak-safe numbers**
(outcome phrases stripped before embedding; stated-severity readout OFF —
supersedes the pre-audit 93%/81% figures, which read severity wording
from the text).*

| Method | Injury accuracy | Damage accuracy |
|--------|-----------------|-----------------|
| Network prior alone (no narrative) | 58.4% | 42.6% |
| Event evidence → frozen BN (soft-priority) | 89.9% | 55.4% |
| k-NN severity via frozen BN (bn-sev, primary) | **90.9%** | **77.4%** |
| Supervised LR, parsed features | 87.8% | 64.2% |
| Supervised LR, narrative embedding | 91.6% | 74.0% |

**Takeaway:** *With zero trained parameters, the leak-safe pipeline beats the
parsed-feature supervised baseline (McNemar p=0.02 injury / p<0.001 damage)
and statistically ties the embedding LR (p=0.50 / 0.11). Severe-outcome
screening: 93.5% sensitivity / 96.8% specificity on injury.*

*SAY:* "Coded labels arrive late; narratives exist early — this is the practical win."

---

## Slide 8 — Your idea: structural mapping (~1 min)

```
Your proposal (reasonable!)
  • LLM extracts a short causal chain from each narrative
    (who / what system / what went wrong → … → outcome)
  • Align chains with Needleman–Wunsch
  • Rerank retrieved accidents: similar structure, not just similar words
  • Hypothesis: cosine misses accidents that "look different in words" but
    "match in mechanism"
```

*SAY:* "We took this seriously — not a quick no. We wanted to falsify it properly."

---

## Slide 9 — Everything we tested (~1.5 min)

| # | Variant | What it tests |
|---|---------|---------------|
| V0 | Embedding only (A0) | Baseline — cosine similarity |
| V1 | Rerank top-50 pool by struct score | Your original rerank idea |
| V2 | Bypass clusters — pick single best match | Maybe aggregation hides signal? |
| V3 | Wider pool (200) → rerank → trim to 50 | Change pool + rerank |
| V4 | **Struct-first over ALL 1,703 accidents** | Your "don't limit to top-50" ask |
| V5 | Hybrid embedding (narrative ⊕ chain text) | Structure inside the embedding |
| V6 | Struct-weighted severity vote | Structure on the paper's main task (296 held-out) |

**Takeaway:** *Rerank, replace retrieval, and hybrid embedding — every insertion point.*

*SAY:* "~17 diagnosis variants + 14 severity variants + oracle bounds. Full report: `outputs/structmap_final_verdict/REPORT.md`"

---

## Slide 10 — Results: structural mapping (~1.5 min)

**Diagnosis (cause ranking, n≈254–296 paired)**

| Variant | vs embedding baseline |
|---------|------------------------|
| Rerank top-50 (A2) | No significant win (McNemar p ≈ 1.0 on 77-case eval; null on 296) |
| Best-match bypass (V1) | **Harmful** −1.6 to −2.6 pp (p < 0.001) |
| Struct-first all accidents (V4) | No win on diagnosis; **catastrophic on severity** |
| Hybrid embedding (V5) | Converges back to cosine — no gain |

**Severity (296 held-out — same task as Slide 7; run PRE-redaction, so the
baseline reads 92.9%/81.4% — paired comparison is still valid because every
struct variant saw the same text)**

| Method | Injury | Damage |
|--------|--------|--------|
| narr-sev (baseline, no struct) | **92.9%** | **81.4%** |
| struct-first retrieval (V4) | 81.1% (−11.8 pp) | 70.9% (−10.5 pp) |
| Best struct-fused cell (1 of 28 tests) | 92.6% | 82.8% (+1.4 pp, p=0.125) |
| Change k=50, **no structure** | 92.6% | 82.4% (+1.0 pp) | ← same noise |

**Takeaway:** *Never beat embedding baseline. Best struct cell = pool-size jitter. Struct-first actively hurts.*

---

## Slide 11 — Why it didn't work (~1 min)

```
1. Chains are too coarse
   • ~5 generic steps ("event → failure → landing → …") — many accidents match the same shape

2. Embedding already captures most of the signal
   • Structural component almost uncorrelated with being correct (ρ ≈ 0.01)

3. Where structure disagrees with embedding, it is wrong more often than right
   • Struct-first picks "same shape, wrong content" accidents → severity collapses

4. Aggregation was helping, not hiding signal
   • Picking one best incident always loses vs averaging 50 neighbors
```

**Takeaway:** *Not "we didn't try hard enough" — we tried every architectural slot. The representation is information-poor.*

*SAY to Jesse:* "Your idea was testable and worth testing. The extraction stack we have doesn't carry enough mechanism beyond what the narrative embedding already encodes."

---

## Slide 12 — Where we are + ask (~1 min)

```
Shipped
  ✓ Zhang reproduction (Table 7 exact; 77/93 network cells)
  ✓ Tiered parser validated (11/11 identity, 6/6 safety)
  ✓ 296 held-out severity evaluation
  ✓ Structural mapping: exhaustive negative result (Appendix A)

Paper (RESS / Safety Science)
  • Methods: build network + narrative bridge
  • Results: Sections 5.1–5.5 (reproduction → held-out → ablations)
  • Discussion: LLM reads, data decides; struct mapping closed

Question for you
  • Comfortable with struct mapping as appendix negative result?
  • Anything you still want rerun before we freeze the draft?
```

*SAY:* "Process question, not 'was the idea dumb.' The data closed the loop."

---

## Appendix (backup only)

- Slide A1: Scoreboard 48/29/12/4
- Slide A2: One walkthrough (`20070131X00119` Tier 1 case)
- Slide A3: McNemar table (soft+stated vs lr)
- Slide A4: Old 77-case A0 vs A2 bar chart (if he remembers early optimism)

---

## Timing cheat sheet

| Slide | Min | Cumulative |
|-------|-----|------------|
| 1 Title | 0.5 | 0.5 |
| 2 Map | 1.0 | 1.5 |
| 3 One picture | 1.5 | 3.0 |
| 4 LLM dead end | 1.0 | 4.0 |
| 5 Reproduction | 1.5 | 5.5 |
| 6 Final pipeline | 1.5 | 7.0 |
| 7 Held-out 296 | 1.5 | 8.5 |
| 8 Jesse's idea | 1.0 | 9.5 |
| 9 What we tested | 1.5 | 11.0 |
| 10 Results | 1.5 | 12.5 |
| 11 Why not | 1.0 | 13.5 |
| 12 Ask | 1.5 | 15.0 |

---

## If Jesse pushes back — short answers

**"Are you sure struct mapping doesn't help a little?"**  
→ Best cell +1.4 pp damage, p=0.125; no-structure k=50 gives +1.0 pp. Selection noise, not signal.

**"Did you try stronger α?"**  
→ α=1,2,4 in rerank and fusion; still null or harmful. V4 tried struct over all accidents.

**"Maybe better chain extraction?"**  
→ Agreed that's the remaining escape — but oracle bound on fixed pool was +0.4 pp max; ρ≈0.01 says the bottleneck is representation, not fusion formula.

**"What's the actual contribution then?"**  
→ Day-one narratives drive Zhang's frozen BN at ~91% injury / ~77% damage on leak-safe held-out data; LLM-only and struct-mapping paths falsified with evidence.

---

## Undergrad glossary (use if someone looks lost)

| Term | Plain English |
|------|---------------|
| Bayesian network | Flowchart of "if X then how likely is Y" learned from past accidents |
| Frozen | Built once; query time only updates beliefs, doesn't retrain |
| Held-out | Accidents the model never saw during construction — honest test |
| Tier 1 / Tier 2 | Dictionary match first; call LLM only if dictionary fails |
| Structural mapping | Compare accident *story shapes*, not just similar words |
| McNemar test | "Did method A beat B on the *same* accidents?" |

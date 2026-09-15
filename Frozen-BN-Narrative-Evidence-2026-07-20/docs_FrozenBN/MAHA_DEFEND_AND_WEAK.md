# Maha Meeting — Defend · Weak · Do Differently
**Plain language · Read aloud once · Pair with MAHA_MEETING_ONE_PAGE.html**

---

## PART A — If he pushes back, defend like this

**Format every time:** **Claim → How → Why → Proof**  
If stuck: *“I’d open Table 6 or Table 8 — the headline is …”*

---

### Push: “What did you actually do?”

**Defend:**
> Two jobs. **Job 1:** Rebuilt Zhang on 1982–2006 codes — **85/85 Table 7**, demos match. **Job 2:** On **296 newer accidents** (never in the build), we convert **narratives → node values** on the **fixed** network, read probabilities, grade vs **NTSB codes** only at the end — **90.9% / 77.4% / 84.2%**.

**Proof:** Table 6, Table 8, verify script, `FREE_PARAMETERS.md`

---

### Push: “Narratives don’t go in the BN — so what goes in?”

**Defend:**
> Correct — **not raw text.** Keywords set some nodes to **yes**. The **100 similar 1982–2006 accidents** set **fractions** on event nodes (4a) and **injury/damage levels** from their **official codes** (4b). Step 5 **loads those node values** and runs the fixed network math once.

**Proof:** §6, `query_to_bn.py`, Figure 2

---

### Push: “90.9% — where does that come from?”

**Defend:**
> **296** accidents, 2007–2019. After removing outcome phrases: find **100 most similar 1982–2006 narratives** → count their **NTSB-coded injury levels** → set injury node on **fixed** network → pick top level → compare to truth. **Not** Table 7. **Not** Zhang’s percentage.

**Proof:** Table 6, `heldout_significance.md`

---

### Push: “Does the BN beat counting the 100 similar accidents?”

**Defend:**
> **No.** Same top injury and damage answer on **all 296/296**. We ran both and compared. The BN still matters for **multi-node queries** (Table 10 engine example), person nodes, what-if on the graph — **not** higher top-1 severity accuracy.

**Proof:** Table 6, agree-on-top-pick check in eval

---

### Push: “84.2% — is that your BN winning on causes?”

**Defend:**
> **No.** **84.2%** = **vote** Personnel / Aircraft / Environment / Organization from the **same 100** accidents’ coded findings. Keyword → network for causes = **57.7%**. Table 9 fire demo is **Job 1**, not this score.

**Proof:** Table 8, `diagnosis_heldout_eval.md`

---

### Push: “Are you cheating? Reading the answer from the narrative?”

**Defend:**
> We **remove** stated injury/damage phrases first. **None** of the 296 are in the 1982–2006 similarity list (leak audit). NTSB codes **only for grading** — never inputs in Steps 1–6.

**Proof:** `redaction_leak_probe.py`, `heldout_leak_audit.py`, §7

---

### Push: “Did you train anything on the 296?”

**Defend:**
> **Main pipeline: no.** Network, tables, similarity list, keywords, k=100 — all fixed from **1982–2006 before** scoring. **Comparison only:** TF-IDF and other classifiers trained on **1982–2006 labels** to show what supervised methods get.

**Proof:** Table 5, `FREE_PARAMETERS.md`

**Concede:** We ran scoring code many times while fixing bugs — may be **slightly optimistic**; **not** label leakage into network tables.

---

### Push: “TF-IDF beats you — so what did you learn?”

**Defend:**
> **92.2% > 90.9%** injury — we report it. TF-IDF **learns from old labeled narratives**; our main path **does not learn from the 296**. Goal was **narrative → fixed Zhang BN**, not max ML accuracy.

**Proof:** Table 6

---

### Push: “Could Zhang have done this? / Is this just similar accidents?”

**Defend:**
> Similar past cases aren’t new. **She chose codes only.** We **reproduced** her network, defined **narrative → node evidence**, and ran a **scoring protocol** on 2007–2019 with leakage checks. Contribution = **bridge + benchmark + honest limits**, not inventing k=100.

**Proof:** Job 1 + Job 2 split, §8.4, §10.1

---

### Push: “What did you change vs Zhang?”

**Defend:**
> **Extended**, not replaced: **person nodes** (pilot queries) + **4-level** injury/damage (boolean leaves broke). **Table 7 counting unchanged** — 85/85.

**Proof:** `bn_upgraded.py`, Appendix A

---

### Push: “Why merge paths failed — did you break your pipeline?”

**Defend:**
> **On purpose.** Putting keywords + 100-accident severity **all in one network update** → **38.5% / 41.9%**. Proves **double-counting** hurts. Main path uses **4b only** for 90.9%.

**Proof:** Table 6 combined row

---

### Push: “LLM — is that your method?”

**Defend:**
> **Four side tests** in §8.3. GPT can match keywords on Zhang’s 11 short examples. On **296**, our GPT setup ~**68%** injury vs **90.9%** pipeline. GPT does **not** output final **P(·)** — network does. **Not** claiming LLMs can never work with a BN — **our tested setup** didn’t beat the main path.

**Proof:** §8.3

---

### Push: “Is this publishable / more than Zhang?”

**Defend:**
> **Not higher Table 7 numbers.** **New:** narrative evidence on **fixed** network + **scoring** on 296/253 + leakage controls + systematic negatives. Paper needs framing fix so written story matches that — **process is there**.

---

## PART B — Where it’s weak (volunteer before he asks)

Say: *“Here’s where I think the draft is weak and what I’d fix.”*

---

### Weak 1 — Paper oversells the BN

**Problem:** Abstract and §6.5 read like the BN wins on injury/causes.  
**Reality:** Severity top-1 = same as counting the 100 (**296/296**). Causes **84.2%** = vote on the 100, not BN (**57.7%**).  
**I’d fix:** Rewrite abstract + §6.5 lead so **primary paths** match Table 6/8.

---

### Weak 2 — Job 1 and Job 2 blur

**Problem:** Results opening mixes **85/85** and **90.9%**.  
**Reality:** Different jobs, different comparisons (Zhang vs NTSB).  
**I’d fix:** Separate subheads — **Verification** then **Narrative scoring**.

---

### Weak 3 — §6.5 sounds like everything merged

**Problem:** Reads like keywords + 4a + 4b always go in together.  
**Reality:** Main **90.9%** = **4b only**; merged path **failed** (38%/42%).  
**I’d fix:** First paragraph of §6.5 = primary path, then ablations.

---

### Weak 4 — 296 used while fixing scoring

**Problem:** May be slightly optimistic.  
**Reality:** We did **not** fit network tables or labels into the build from 296.  
**I’d fix:** Keep limitation visible; future confirm on **2020–2024** never used in dev.

---

### Weak 5 — Redaction tradeoff

**Problem:** Stripping outcome phrases removes cues human coders might use.  
**I’d fix:** Already in limitations — keep it; don’t hide.

---

### Weak 6 — Scope and rare classes

**Problem:** Part 121 only; Organization cause weak; fatal injury rare (3/296).  
**I’d fix:** Disclose; don’t oversell cause accuracy.

---

### Weak 7 — LLM section length

**Problem:** Can look like an LLM paper.  
**I’d fix:** Frame as **what we tested and rejected**; ask Maha if appendix is better.

---

## PART C — What I’d do differently (one paragraph for him)

> “The **experiments** match what I want the paper to say — reproduce Zhang, narrative → node evidence, score 2007–2019 honestly. The **draft** still sounds like a BN victory paper in places. After your edits, I’ll **split Job 1 and Job 2**, **rewrite abstract and §6.5** so the main path is 4b for severity and vote-on-100 for causes, and **lead with limitations** (296/296, TF-IDF, retrieval for 84.2%). I’m **not** asking for new numbers first — **alignment** first.”

---

## PART D — Never defend these (agree in one breath)

- BN beats the 100-accident count on severity  
- We beat TF-IDF on injury  
- GPT outputs P(·) in our main method  
- 90.9% proves we rebuilt Zhang  
- 84.2% comes from Table 9 or “the BN winning diagnosis”  
- We trained the main pipeline on the 296  
- 2020–2024 confirmatory is done  

---

## PART E — 30-second “weak + fix” if he’s short on time

> “Draft oversells the BN; results don’t. I need to split two jobs, clarify §6.5 primary path, and make the abstract say retrieval for causes and neighbor-agreement for severity. Science and honesty are in §8–10 — I need the rest of the paper to sound like that.”

# Maha Meeting Notes — 30 min grill prep
**Kanu · Draft 3 · Read this aloud 2× before the meeting**

**Rule:** Every claim = **Claim → How → Why → Proof**. If you can't point to proof, say: *"I'd open Table X / the eval output — headline is …"*

---

## A. If he opens with "What did you do?" (60 seconds)

> I did **two separate jobs**.
>
> **Job 1 — Verify Zhang:** Rebuilt his aviation Bayesian network on **1982–2006 coded NTSB data** and checked we match his published results — **85/85 Table 7 rows**, **102 fire** accidents, Table 9/10 demo queries.
>
> **Job 2 — Test narratives:** On **296 newer accidents (2007–2019)** the network never trained on, I asked: after **stripping cheat phrases**, can accident **stories** update the **locked** network and match **NTSB coded** injury, damage, and cause labels? Main results: **90.9% injury**, **77.4% damage**, **84.2% cause type** (on **253** with scorable causes).
>
> **Honest limits:** The network does **not** beat just counting the 100 similar past accidents on injury top-1 (**same answer 296/296**). **TF-IDF beats us** on injury (92.2% vs 90.9%). **84.2% causes** comes from **voting similar stories**, not phrase→network (57.7%). **GPT reads labels**; the **network computes P(·)**.

---

## B. Problem (why this paper exists)

| | |
|--|--|
| **Zhang did** | BN from **coded** NTSB fields only — diagnosis + prognosis |
| **Gap** | **Narratives** exist but weren't used as evidence on the network |
| **Our question** | Can **redacted** narratives supply evidence on the **fixed** BN without retraining on test labels? |
| **Scope** | FAR **Part 121** airline accidents (same as Zhang) |

---

## C. Two jobs — NEVER merge these

| | Job 1 — Verify rebuild | Job 2 — Story test |
|--|------------------------|-------------------|
| **Question** | Did we rebuild Zhang correctly? | Do new stories predict NTSB codes? |
| **Years** | 1982–2006 | 2007–2019 (held out) |
| **Headline numbers** | **85/85 Table 7**, 102 fires, P(fire)=102/184,517,128 | **90.9% / 77.4% / 84.2%** |
| **Grade against** | Zhang's published tables | **NTSB coded truth** (not Zhang) |
| **Proof** | `verify_zhang_table7.py`, App A | Table 6, Table 8, eval scripts |

**Do not say:** "90.9% proves we rebuilt Zhang."

---

## D. Pipeline — what happens to ONE test story

**Plain words only. Memorize the parallel branches.**

```
STEP 1  Strip stated injury/damage phrases (anti-leak)
           ↓
STEP 2  Phrase match (words → network nodes)     ║  STEP 3  Embed story → 100 most similar 1982–2006 accidents
           │                                              ↓
           │                                    STEP 4a  Event hints from 100  ║  STEP 4b  Count CODED injury/damage on 100
           │                                              ↓
           └──────────────→ STEP 5  Update LOCKED network ←── (main severity: 4b only)
                                    ↓
STEP 6A  Read P(injury), P(damage)     |     STEP 6B  Vote 4 cause categories from same 100 neighbors
                                    ↓
STEP 7  Grade vs NTSB codes — codes NEVER used as input
```

| Step | One line |
|------|----------|
| **1 Strip** | Remove "fatal injury", "substantial damage", etc. before anything else |
| **2 ∥ 3** | Phrase match **and** 100 similar stories on the **same** clean story — **same time** |
| **2 alone** | Can go to Step 5 **without** 4a/4b (phrase-only path) |
| **4a ∥ 4b** | Same 100 neighbors — events vs coded severity counts — **same time** |
| **5 Main severity** | **4b only** → network → 90.9% / 77.4% |
| **6B Causes** | Vote: Personnel, Aircraft, Environment, **Organization** → 84.2% |
| **7 Grade** | Compare to NTSB; never an input |

---

## E. What WORKS (say these with n and table)

| Result | n | How (one line) | Proof |
|--------|---|----------------|-------|
| **85/85 Table 7** | 102 fires, 82–06 | P(cause\|fire) counting = Zhang | App A, verify script |
| **90.9% injury top-1** | 296 | 100 similar → coded injury → network → top level | **Table 6** |
| **77.4% damage top-1** | 296 | Same path on damage node | **Table 6** |
| **84.2% cause type top-1** | 253 | Same 100 → **vote 4 categories** | **Table 8** |
| **Table 9/10 demos** | 82–06 | Phrase query → fixed BN (Zhang examples) | §9, apps-parity |
| **Strip + no leak** | 296 | 0 test IDs in search pool; codes grade-only | leak audit, redaction probe |
| **Phrase match on Zhang 11 texts** | 11 | Same nodes as rules (incl. GPT label read) | §8.3 |

---

## F. What DOES NOT work — say honestly (builds trust)

| Don't claim | Truth | Number |
|-------------|-------|--------|
| BN beats neighbor vote on injury | **Same top pick 296/296** | Table 6 |
| We beat TF-IDF on injury | **TF-IDF wins** | 92.2% > 90.9% |
| GPT outputs probabilities | **Network does** | §8.3 |
| Phrase→BN best for causes | **Retrieval vote wins** | 57.7% vs 84.2% |
| Merge all evidence in one update | **Failed on purpose** | 38.5% inj / 41.9% dmg |
| GPT primary on 296 | **Worse than pipeline** | ~68% inj vs 90.9% |
| 2020–24 confirmatory done | **Future work only** | §10.3 |

**Why we report failures:** Combined merge proves **double-counting** hurts. TF-IDF shows what **supervised ML** gets vs our **no-training** pipeline.

---

## G. What we changed vs Zhang (two upgrades)

| Upgrade | Zhang had | We added | Why |
|---------|-----------|----------|-----|
| **Person nodes** | Findings → events only | Person roles → findings | Pilot / human-factor queries (Table 10) |
| **4-level severity** | Boolean injury leaves | One injury node + one damage node, 4 levels each | Boolean broke P(no injury); matches NTSB codes |

**Say:** "We **extended** Zhang — we didn't replace the graph. Table 7 counting unchanged."

---

## H. Design choices — if he asks "why?"

| Choice | Because | Proof |
|--------|---------|-------|
| Reproduce Zhang first | Can't test narratives on unverified network | Job 1 |
| Test 07–19, not 82–06 stories | Network + search list built on 82–06 — test there = circular | leak audit |
| Strip outcome phrases | Narratives **state** outcomes = cheating | redaction_leak_probe |
| k=100 neighbors | Default at build; k=25 ablation ~flat | heldout_significance_k25 |
| 4 cause categories only | 2008 taxonomy break (legacy vs CICTT) | diagnosis_heldout_eval |
| Locked network after build | 296 = **exam** | FREE_PARAMETERS, Table 5 |
| LLM in paper | Tested; **rejected** as probability engine | §8.3 |

---

## I. Trained on 296? (Jesse question — memorize)

**Say verbatim:**

> "The **main pipeline** has **no weights fitted on the 296** test accidents. Network, CPTs, embedding index, phrase rules, and k=100 were all fixed from **1982–2006** before scoring the test set. The only **trained** models are **comparison baselines** — logistic regression, TF-IDF, embedding classifiers — trained on **1982–2006 labels only**."

**Concede:** We ran eval many times while fixing bugs — numbers may be **slightly optimistic** (§7, §10.3). **Not** label leakage into CPTs.

---

## J. Why BN if neighbors give same top injury? (hinge question)

> "On **top-1 injury/damage**, the network **does not beat** counting the 100 neighbors — **296/296 same pick**. We measured that. The BN still matters for **structured reasoning**: multi-node queries (engine instruments → P(loss of engine power)), person nodes, what-if propagation — Table 10 style. **Severity accuracy** is mostly retrieval-like; **honesty** is reporting that, not overselling the BN."

---

## K. Table 9 vs §8 test (easy to confuse)

| | Table 9 / §9 fire demo | §8 held-out test (84.2%) |
|--|------------------------|--------------------------|
| **Job** | Job 1 demo | Job 2 test |
| **Data** | 1982–2006 | 2007–2019 |
| **Method** | Phrase "fire" → count causes | 100 similar → **vote 4 categories** |
| **Proves** | We reproduce Zhang example | Generalization to new accidents |

---

## L. Paper weak spots — volunteer before he finds them

1. **§6.5** reads like all evidence fused — **primary** severity is **4b-only**; fused path is separate ablation.
2. **Results opening** mixes Job 1 (Table 7) and Job 2 (90.9%) — should split subheads.
3. **Abstract** doesn't say 84.2% = **retrieval vote**.
4. **296** used during eval debugging — slightly optimistic possible.
5. **Redaction** removes cues human coders might use.
6. **Organization** cause category weak (rare).
7. **Part 121 only** — not all aviation.

**What I'd do differently (one sentence):**

> "Same science — clearer framing: two jobs up front, primary eval path in §6, retrieval-for-diagnosis and neighbor-agreement-for-severity in the abstract."

---

## M. Grill cheat sheet — short answers

| He asks | You say |
|---------|---------|
| Two jobs? | Verify 85/85 · Test 90.9/77.4/84.2 on 296/253 |
| 90.9% how? | 100 similar → coded injury → network → top level, n=296, Table 6 |
| Cheating? | Strip phrases · 0/296 in pool · codes grade-only |
| Trained on 296? | **No** main pipeline · baselines only on 82–06 |
| BN vs neighbors? | **No gain** top-1 · 296/296 same |
| 84.2% how? | Vote 4 categories from 100 neighbors · n=253 · Table 8 |
| What failed? | Combined merge 38%/42% · GPT ~68% inj |
| LLM role? | Label reader tests · BN does P(·) · not primary |
| vs Zhang? | Person nodes + 4-level severity · Table 7 unchanged |

---

## N. Proof on laptop (if challenged)

`query_to_bn.py` · `bn_upgraded.py` · `frozenbn_heldout_narrative_bn_eval.py` · `diagnosis_heldout_eval.py` · `outputs/heldout_significance.md` · `outputs/diagnosis_heldout_eval.md` · `FREE_PARAMETERS.md` · `REPRODUCE.md`

---

## O. 10-minute cram order (if you forget everything)

1. Read **Section A** aloud 2×  
2. Read **C + D** (two jobs + pipeline)  
3. Read **E + F** (works / doesn't work)  
4. Read **I + J** (trained? / why BN?)  
5. Skim **M** table once  

**Stop.** Don't re-read the whole paper.

---

## P. Word guide — say this to Maha, not draft jargon

| Say | Not |
|-----|-----|
| phrase match | hard evidence |
| 100 similar past accidents | soft/virtual evidence, retrieval index |
| severity through network | bn-sev, virtual evidence |
| neighbor injury vote | retr-sev |
| combined test (failed) | bn-fused |
| locked after build | frozen BN |
| grade on test set | held-out eval |

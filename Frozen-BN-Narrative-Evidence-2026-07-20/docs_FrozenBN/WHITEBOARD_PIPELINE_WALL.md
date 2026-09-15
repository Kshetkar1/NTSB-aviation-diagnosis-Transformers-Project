# DAILY WALL — plain language (4 boards, top → bottom)

**PRINT VERSION (white background, flow diagram, study from paper/PDF):**  
Open **`WHITEBOARD_PRINT_STUDY.html`** in Chrome/Safari → **File → Print → Save as PDF**.

**OWN EVERY CLAIM (Maha rule — because X, proof Y):**  
**`WHITEBOARD_OWN_EVERY_CLAIM.md`** — do not say anything not in this table.

**Write each board from TOP to BOTTOM.** Then move to the next board.  
**Draft:** fill `p.__` from PDF · **Laptop file = full text · Wall = short bullets**

---

## BEFORE YOU WRITE — checklist (all your questions are answered in this file)

| Your concern | Where in this file | What to write on wall |
|--------------|-------------------|------------------------|
| No AI jargon | Word table below + **DO NOT WRITE** box | Say “phrase match”, not “hard evidence” |
| Colors per section | Each section header says **black / blue / green / red** | Copy color from header |
| 4 boards top→bottom | Boards 1–4 sections | One long rectangle = one board |
| Flow picture | Board 2 Section 1 | Draw the big diagram first |
| P(·) | Section 5 + word table | Write “P(injury)” or “probability of fire” |
| Upgrades clear | Section 5 table + Board 3 C7 | Person nodes + 4-level injury/damage |
| 4,000 chars | Section 6 + Step 0 | “No story longer — safe cap” |
| Four paths vs pipeline | Section 8 | **Same pipeline** — different scores at Step 6 |
| BTS / corpus | Section 6 + word table | Define both on Board 1 |
| Build steps C2–C7 | Board 3 | Node/edge example, parents, Beta, upgrades |
| Index / why not test stories | Step 3 | “Story search list” + exam/past-only |
| bn-sev / retr-sev | Word table | “Severity through network” / “Neighbor vote alone” |
| Combined test / virtual | Word table + Step 5 | “Combined test (failed)” / “neighbor counts into injury nodes” |
| All metrics explained | Board 4 Section 1 | Top-1, Injury %, Macro-F1, baselines |
| Locked not frozen | Word table + Board 4 §3 | “Network locked after build” |
| k-NN | Step 3 | “Find 100 most similar past stories” (optional small “k=100”) |
| 2020–24 | Section 9 footnote | **Do not put on wall** |
| Paper names vs wall | Table below | Parentheses only if needed |

### DO NOT WRITE ON WALL (draft/code words — say plain English instead)

`hard evidence` · `soft evidence` · `held-out` · `frozen` · `ablation` · `virtual evidence` ·  
`bn-sev` · `retr-sev` · `bn-fused` · `bn-post` · `emb-LR` · `leak-safe` · `ptext` · `discordant` · `corpus` alone · `index` alone

**Exception:** repo **file names** on Board 4 corner only (small).

### Draft paper name → what you say (optional small note on Board 4)

| Paper / code says | You say on wall |
|-------------------|-----------------|
| bn-sev | Severity through network (main — 90.9% / 77.4%) |
| retr-sev | Neighbor injury/damage vote (same top answer) |
| bn-fused | Combined test (failed — 38% / 42%) |
| hard + soft | Phrase match + suggestions from similar stories |
| prior | Network with no story input (guess most common) |
| TF-IDF LR | Supervised TF-IDF on text (92.2% injury) |
| embedding LR | Supervised story embedding classifier |
| LR parsed features | Supervised parser classifier |
| frequency baseline | Most common category guess |
| bn-lift | Network lift ranking (failed for causes) |

---

## COLORS — what to use where

| Color | Use for |
|-------|---------|
| **Black** | Titles, boxes, arrows, most words |
| **Blue** | Zhang / 1982–2006 / building the network / checking we rebuilt Zhang / §9 paper examples |
| **Green** | 2007–2019 test accidents / main pipeline / your result numbers |
| **Red** | Warnings · strip phrases first · truth codes for grading only · things that failed · don’t say this |

**On every board:** small color key in top corner (4 lines above).

---

## WORDS WE USE ON THE WALL (say these to Maha)

| Say this | Not this | Means |
|----------|----------|--------|
| **probability of …** or **P(injury)** | P(·) alone | The BN outputs how likely each outcome is. **P(·)** in the draft = “probability of whatever is in the blank.” |
| **phrase match** | hard evidence | Words in the story match a network node name → that node = YES |
| **suggestion from similar stories** | soft evidence | The 100 similar past accidents often had label X → suggest X with strength 0–1 |
| **neighbor injury/damage vote** | retr-sev | Count injury/damage codes on the 100 similar accidents → pick the most common level |
| **severity through the network** | bn-sev | Same neighbor counts, but entered on injury/damage nodes inside the BN → read probabilities |
| **296 test accidents** | held-out | Newer accidents (2007–2019) kept only for scoring — never used to build the network |
| **1982–2006 training years** | build window | Years used to build the network and the story search list |
| **network locked after build** | frozen | Built once from 1982–2006 codes — not updated when we score 2007–2019 stories |
| **story search list** | index / embedding index | Pre-saved vectors of every 1982–2006 narrative so we can find similar stories fast |
| **our accident dataset** | corpus | Cleaned NTSB Part 121 accidents in `refined_dataset.json` |
| **flight count table (BTS)** | BTS | Bureau of Transportation Statistics — total US airline flights (184,517,128) used as denominator for “how rare is fire” |
| **combined test (failed)** | bn-fused | We tried putting phrase match + neighbor severity in one network update — accuracy dropped to 38%/42% |
| **neighbor severity as network input** | virtual evidence | We tell the injury/damage nodes “similar accidents looked like THIS distribution” — not reading severity phrases from the text |
| **extra comparison test** | ablation | A run we did on purpose to see what breaks (e.g. combined test above) |
| **most common category guess** | frequency baseline | Always predict Personnel because it’s most common in training data |
| **supervised story classifier** | embedding LR | Logistic regression trained on 1982–2006 to map story embedding → injury/damage/category |
| **supervised parser classifier** | LR parsed features | Logistic regression trained on which nodes the phrase matcher found |

---

# BOARD 1 — WHAT + WHY (write top → bottom)

---

## SECTION 1 — TITLE STRIP · **black**, title **blue** underline

```
Using NTSB Narratives as Evidence in Zhang’s Aviation Bayesian Network
Kanu · Maha · Jesse · Vanderbilt
```

**RED (big, under title):**  
Stories never go into the network as raw text.  
Story → matched nodes + similar past accidents → network updates → **probabilities**.

---

## SECTION 2 — THE PROBLEM · **black** text · **blue** “Zhang 1982–2006” · p.__ §1

- Each NTSB report has **coded events** (engine fail, fire, landing…) + a **written factual story**
- Codes show a chain but lose wording, timing, crew context
- **Zhang & Mahadevan (2021):** built a Bayesian network from **codes only**, accidents **1982–2006**
- **Our question:** Can a **new** accident’s **story** (after removing cheat phrases) update that **same** network and predict NTSB’s coded injury, damage, and cause type?

**Our accident dataset** = cleaned NTSB Part 121 accidents in the repo (`refined_dataset.json`). *(Draft may say “corpus” — same thing.)*

---

## SECTION 3 — GOALS (5) · **black** · p.__ §1

1. Rebuild Zhang’s network from 1982–2006 codes  
2. Check we match his published tables (Table 7, fire rate, Table 9/10 examples)  
3. Turn stories (after Step 1 strip) into node updates on that **network locked after build** (no re-learning from the 296 test accidents)  
4. Score on **296** newer accidents (2007–2019) vs NTSB codes — codes used **only for grading**  
5. Side study: can GPT read text → node labels only? (network still does all probabilities)

---

## SECTION 4 — RESEARCH QUESTIONS · **green** box outline · p.__ §1

**Main:** After Step 1 (strip cheat phrases), can story → network updates match NTSB injury/damage/cause codes on 2007–2019 accidents?  
**Side:** Can GPT extract node labels without outputting probabilities itself?  
**NOT the project:** “ChatGPT predicts the accident outcome.”

---

## SECTION 5 — FOUR CONTRIBUTIONS · **black** · p.__ §1

**1.** Reproduce Zhang’s network; match his query tables in pyAgrum  

**2. Two upgrades to Zhang’s graph (what we added and why):**

| Upgrade | What Zhang had | What we added | Why it helps |
|---------|----------------|---------------|--------------|
| **Person roles** | Findings linked to events | Nodes like `person: pilot-in-command` linked to findings | Human-factor queries (“pilot error”) work — Zhang’s Fig 12 examples were missing this |
| **Four-level injury & damage** | Separate yes/no injury leaves | One node **personnel injury** with 4 levels: fatal / serious / minor / none; one node **aircraft damage**: destroyed / substantial / minor / none | Boolean leaves forced broken probabilities (e.g. P(no injury)≈0). Four levels match NTSB codes and sum to 100% |

**3.** Story → phrase match + 100 similar past stories → network updates **without** training on the 296 test accidents  

**4.** Four GPT experiments — label reader only; **P(injury)**, **P(damage)** always from the network  

**P(·) in the paper** = shorthand for **probability of something** — e.g. P(fire), P(fatal injury), P(loss of engine power). The **(·)** is the blank.

---

## SECTION 6 — DATA SPLIT · **blue** top box, **green** bottom box, **red** arrow label · p.__ §3

```
┌── BLUE: 1982–2006 (build + check Zhang) ──────────────────────┐
│ 1,742 accidents in Zhang’s year range                          │
│ 1,288 have narratives (stories) in the dataset                 │
│ Build: network structure, probability tables, story search list  │
│ Check: Table 7 85/85, 102 fires, Table 9/10 demos              │
│ NOT where 90.9% comes from                                     │
└────────────────────────────────────────────────────────────────┘
              ↓  NETWORK LOCKED — never re-learn from 296 test cases
┌── GREEN: 2007–2019 (story test) ──────────────────────────────┐
│ 296 accidents with story length ≥ 100 characters               │
│ Stories cut to 4,000 characters — NO story in our data is longer │
│ 253 have scorable cause findings (43 missing → skip cause score) │
│ NTSB injury/damage/cause codes → grading ONLY after prediction   │
└────────────────────────────────────────────────────────────────┘
```

**RED beside arrow:** If we tested stories on 1982–2006, we’d be searching stories that **built** the network — cheating / circular.

**BTS:** Bureau of Transportation Statistics flight totals — denominator **184,517,128** flights for “how rare is an event per flight.”

---

## SECTION 7 — TWO JOBS · **blue** left column · **green** right column · p.__ §7

| | **Job 1 — Check Zhang (blue)** | **Job 2 — Test stories (green)** |
|--|-------------------------------|----------------------------------|
| Years | 1982–2006 | 2007–2019 |
| Question | Did we rebuild correctly? | Do stories work on new years? |
| Numbers | 85/85 Table 7 · 102 fires | **90.9%** injury · **77.4%** damage · **84.2%** cause type |
| **RED:** | Don’t say 90.9% here | Don’t say Table 7 here |

---

## SECTION 8 — “FOUR PATHS” — same pipeline, different scores · **green** · read with Board 2

**These are NOT four different pipelines.** One pipeline (Board 2). At **Step 6** we read different answers:

| Path name | What we measure | Where in pipeline | Main number |
|-----------|-----------------|-------------------|-------------|
| **Injury & damage prediction** | Predict injury level + damage level | Steps 3–5 → neighbor vote → **severity through network** | **90.9% / 77.4%** |
| **Cause type (4 categories)** | Personnel / Aircraft / Environment / Organization | Step 3 → count cause types in same 100 similar accidents | **84.2%** |
| **Zhang’s paper examples** | Short typed query, phrase match only | Board 3 right — **no** 100 similar stories | Table 9 fire · Table 10 engine |
| **GPT side study** | Can GPT label nodes? | Off to the side — not main path | §8.3 |

**Phrase matching** = **Step 2** on Board 2.  
**100 similar stories** = **Step 3** on Board 2.  
**Steps 4a / 4b** = what we **do with** those 100 stories (event suggestions vs injury/damage counts).

---

## SECTION 9 — WHY WE DESIGNED IT THIS WAY · **black** bullets · **red** for warnings

**Why stories?** Codes miss prose; we study **archived** reports — not live in-flight prevention (**red: don’t claim real-time**).

**Why rebuild Zhang instead of a new network?** Same graph Maha published — we **extend** it, not replace it.

**What “extend” means exactly (B2):**
- Add **person:** nodes (pilot, flight crew, …) wired to findings — was in Zhang’s logic but needed for his pilot-error examples  
- Replace broken yes/no injury leaves with **one 4-level injury node + one 4-level damage node** — so probabilities behave like real NTSB severity codes  

**Why test 2007–2019, not 1982–2006?** Same years built the network **and** the story search list — testing there inflates scores.

**Why strip injury/damage phrases first?** Stories say “substantial damage” — using that to predict coded damage is reading the answer.

**Why lock the network (no re-learning on 296)?** Those 296 are our **exam** — updating the network from them would be cheating. Only **comparison classifiers** (logistic regression) learn weights from 1982–2006 labels.

**Why 100 similar stories?** Default from build; trying 25 instead changes injury accuracy only 90.9% → 89.9% — basically flat.

**Why only 4 cause categories on the test?**  
NTSB changed coding systems in **2008**. We cannot fairly match every exact cause phrase across eras. We roll causes up to **4 types:** Personnel · Aircraft · Environment · Organization.  
**Not full cause-level diagnosis on test** = we do **not** score “compressor blade retention” vs “electrical wiring” on 2007–2019 — only the **big category** (Aircraft vs Personnel, etc.).

**Why “severity through network” if same accuracy as neighbor vote?**  
- **Neighbor vote alone** = count the 100 similar accidents’ injury codes → pick most common (**90.9%**).  
- **Severity through network** = put those same counts **into** the injury/damage nodes inside the BN → read **P(injury level)** (**also 90.9%**).  
Same top answer on all 296 — but the network path lets us run **what-if** queries (§9 Table 10). **Red:** we do **not** claim the network improves top accuracy over neighbor vote.

**Why §9 demos ≠ §8 test scores?** §9 = Zhang’s short examples on 1982–2006. §8 = full real stories on 2007–2019.

**Why GPT is side only?** Early version asked GPT for probabilities — wrong and unstable. Now GPT may only suggest node names; **network computes all P(injury), P(damage), P(cause).**

*(2020–2024 confirmatory: only in draft future-work paragraph — **not on wall** unless Maha asks; you don’t claim it.)*

---

# BOARD 2 — MAIN PIPELINE + FLOW PICTURE (write top → bottom, **green**)

---

## SECTION 1 — BIG FLOW PICTURE (draw this first, use all colors)

Copy this large. Arrows **black**. Label **“AT THE SAME TIME”** between parallel boxes.

**Key:** Step 2 ∥ Step 3 (same clean story). Step 4a ∥ Step 4b (same 100 stories from Step 3). Step 2 skips 4a/4b — meets Step 5 directly.

```
                    ┌─────────────────────────────────┐
                    │  NEW ACCIDENT STORY (2007–19)   │  GREEN
                    │  ≥100 chars · max 4,000 chars     │
                    └───────────────┬─────────────────┘
                                    ▼
                    ┌─────────────────────────────────┐
                    │ STEP 1: STRIP CHEAT PHRASES     │  RED
                    │ → ONE clean story for all below   │
                    └───────────────┬─────────────────┘
                                    ▼
              ═══════ STEP 2 and STEP 3 AT THE SAME TIME ═══════
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
┌─────────────────────────┐           ┌─────────────────────────────┐
│ STEP 2: PHRASE MATCH     │           │ STEP 3: FIND 100 SIMILAR   │  GREEN
│ words → network nodes    │           │ STORIES from 1982–2006 ONLY  │
│ (outcome, person, event) │           │ (story search list)          │
│                          │           └──────────────┬──────────────┘
│  skips 4a / 4b           │                          ▼
│         │                │     ═══ 4a and 4b AT THE SAME TIME ═══
│         │                │          ┌─────────────┴─────────────┐
│         │                │          ▼                           ▼
│         │                │   ┌──────────────┐      ┌──────────────────┐
│         │                │   │ STEP 4a:     │      │ STEP 4b:         │
│         │                │   │ event        │      │ count injury &   │
│         │                │   │ suggestions  │      │ damage codes     │
│         │                │   └──────┬───────┘      └────────┬─────────┘
│         │                │          └──────────┬─────────────┘
└─────────┼────────────────┴─────────────────────┘
          └──────────────────┬──────────────────────
                             ▼
                    ┌─────────────────────────────────┐
                    │ STEP 5: UPDATE LOCKED NETWORK   │  BLUE border
                    │ ★ MAIN: Step 4b only → 90.9/77.4% │
                    │ ALT: Step 2 (+4a) → weaker        │
                    │ FAIL: Step 2+4a+4b together → 38/42%│
                    └───────────────┬─────────────────┘
                                    ▼
                    ┌─────────────────────────────────┐
                    │ STEP 6: READ ANSWERS              │  GREEN
                    │ A) injury + damage (from Step 5)  │
                    │ B) cause type (from Step 3 pool)  │
                    └───────────────┬─────────────────┘
                                    ▼
                    ┌─────────────────────────────────┐
                    │ STEP 7: GRADE vs NTSB CODES       │  RED
                    │ codes were NEVER model inputs     │
                    └─────────────────────────────────┘

     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     BLUE DASHED SIDE PATH (Zhang §9 examples only — NOT 296 test):
     Short query → Step 2 phrase match only → Step 5 → Table 9 or Table 10
     (NO Step 3 · NO 4a · NO 4b)
```

---

## SECTION 2 — STEP DETAILS (under the picture, same order)

### STEP 0 · INPUT · **green**
- Field: NTSB factual narrative (`narr_accf`)  
- Keep if story ≥ **100** characters  
- Use first **4,000** characters — **no narrative in our dataset exceeds 4,000** (safe cap, same for all methods)  
- Real injury/damage/cause codes saved separately for Step 7 only  

### STEP 1 · STRIP CHEAT PHRASES · **red** · p.__ §6.1
Remove before phrase match **or** similar-story search:  
“substantial damage”, “fatal injuries”, “killed”, “hospitalized”, NTSB full-report boilerplate, URLs, etc.  
**Output:** clean story text used in Steps 2–4.

### STEP 2 · PHRASE MATCH · **black** · p.__ §6.2 · **runs same time as Step 3**
Same **clean story** from Step 1 (not the raw story).
Match story words to **network node names** Zhang uses:
1. Main event/outcome (“loss of engine power”)  
2. People (“pilot in command”)  
3. Multi-word events (“engine instrument”) — skip “no fire” / “worried about fire”  
**Output:** list of nodes set to YES (confidence 1.0).  
**Goes to Step 5** — does **not** pass through Step 4a or 4b.  
Alone: **82.4%** injury match — not our main severity method.

### STEP 3 · FIND 100 SIMILAR STORIES · **green** · p.__ §6.3 · **runs same time as Step 2**

**Story search list (index):** Every 1982–2006 narrative converted to a number vector once and saved (`embeddings_1982_2006.npy`).  
**Why not add 2007–2019 test accidents to the list?**  
- They are the **exam** — searching similar stories must use **past** accidents only  
- Adding test stories would leak future data into the search pool and inflate scores  

**100 most similar past stories (k = 100):** Turn the clean story into a number vector → compare to the **story search list** → keep the **100** closest 1982–2006 accidents. *(Draft/code: k-NN — same idea.)*

### STEP 4a · EVENT SUGGESTIONS FROM 100 STORIES · **green** · p.__ §6.3 · **same time as 4b**
Uses **only** Step 3’s 100 stories (not Step 2).
If many similar accidents had event label X, suggest X on the network with strength = fraction (e.g. 71/100 → 0.71).  
Keep top 3 if fraction ≥ 15% and much rarer than baseline.

### STEP 4b · INJURY & DAMAGE FROM SAME 100 STORIES · **green** · p.__ §6.4 · **same time as 4a**
Uses **only** Step 3’s 100 stories (same pool as 4a, at the same time).
Count **NTSB coded** injury/damage on those 100 accidents (not words from the new story).  
**Path A — Neighbor vote alone:** pick most common level → **90.9% / 77.4%**  
**Path B — Severity through network:** feed those counts into injury/damage nodes inside BN → read **P(each level)** → **same 90.9% / 77.4%** on top pick  

**Neighbor severity as network input** = we translate neighbor counts into inputs on the **personnel injury** and **aircraft damage** nodes so the network’s **probabilities** reflect similar past outcomes.

### STEP 5 · UPDATE LOCKED NETWORK · **blue** border · p.__ §6.5

| Method name on paper | What goes in | Result |
|----------------------|--------------|--------|
| **Severity through network** ★ | Step 4b only | **90.9% / 77.4%** |
| **Neighbor vote alone** | Step 4b, skip network | Same top answer 296/296 |
| Phrase match + suggestions | Steps 2 + 4a | 82.4% inj — weaker on damage |
| **Combined test (failed)** | Steps 2+4a **and** 4b together | **38.5% / 41.9%** — same story counted twice |

**Why write “combined test (failed)”?** Proves we should **not** merge phrase match and neighbor severity in one update — they double-count the same narrative signal.

### STEP 6 · READ ANSWERS · **green** · p.__ §8

**A) Injury & damage:** Pick level with highest **P(injury)** and **P(damage)**  
Levels: fatal / serious / minor / none · destroyed / substantial / minor / none  

**B) Cause type (4 categories):** From **same 100 stories** in Step 3, vote Personnel / Aircraft / Environment / Organization → **84.2%**  
**Phrase match through network for causes:** only **57.7%** — neighbor vote wins for causes  

### STEP 7 · GRADE · **red** · p.__ §7
Compare our pick to NTSB coded answer.  
**Agree on top pick (296/296):** “Severity through network” and “neighbor vote alone” picked the **same** #1 injury and damage level every time — network did not secretly change the top answer.

**RED X:** Story → ChatGPT → injury answer (skips Steps 1–5).

---

### MARGIN EXAMPLE · trace one story · **black** on Board 2 edge

```
Story mentions “substantial damage” and “engine instruments”
Step 1: strip “substantial damage” → ONE clean story
Step 2 & 3 AT SAME TIME:
  Step 2: phrase match → engine instrument node = YES
  Step 3: find 100 similar 1982–2006 accidents
Step 4a & 4b AT SAME TIME (on those 100):
  4b: count coded injuries → e.g. 41 none, 28 minor, 24 serious, 7 fatal
Step 5: severity through network (main uses 4b; Step 2 joins only in alt/failed tests)
Step 6: A) pick injury level from Step 5 · B) cause vote from Step 3 pool
Step 7: compare to NTSB code (never used as input)
```

---

# BOARD 3 — BUILD + CHECK ZHANG (write top → bottom, mostly **blue**)

---

## SECTION 1 — BUILD NETWORK FROM 1982–2006 CODES · **blue** · p.__ §5

### C1 · Prepare data · **black**
- Part 121 airline accidents from **our accident dataset**  
- Events, findings, person on finding, injury/damage codes  
- **BTS (Bureau of Transportation Statistics):** US airline flight totals — **184,517,128** flights used as denominator for “how rare is fire per flight”  

### C2 · Graph = nodes + edges · **blue** · EXAMPLE ON WALL:

```
NODES (things that can be true/false or a level):
  [fire]  [loss of engine power]  [landing phase]
  [person: pilot-in-command]  [finding: pilot error]
  [personnel injury: fatal|serious|minor|none]
  [aircraft damage: destroyed|substantial|minor|none]

EDGES (arrows = influences):
  fire → loss of engine power → personnel injury
  person: pilot-in-command → pilot error → landing phase
  last events in chain → personnel injury & aircraft damage
```

Each **node** = one NTSB label or severity variable. Each **edge** = “this can affect that” (time order + finding links + person→finding upgrade).

### C3 · Prior probabilities · **blue**
**P(event per flight)** = (# accidents with event) / (**184,517,128** BTS flights)  
Example: **P(fire) = 102 / 184,517,128**

### C4 · Pick parents (max 12) · **blue**
Each node gets at most **12 parent** events that predict it best in the data.  
**Why 12?** Zhang’s recipe + cap keeps tables stable — too many parents = sparse empty cells.  
**Deterministic** = same data → same parents every time (no random shuffle).  
**How chosen:** rank candidate parents by how often they co-occur + Zhang’s edge-ratio rules → take top up to 12.

### C5 · Beta-CDF smoothing · **blue**
**Problem:** Some parent combinations almost never happen → raw counts give 0% or 100% nonsense.  
**Fix (Zhang’s recipe):** Smooth counts with a **Beta CDF** curve — pulls extreme probabilities toward sensible middle values.  
**We don’t “choose” beta freely** — follow Zhang Section 4 / `bn_build_ours.py` same as paper reproduction.

### C6 · Table 7 counting · **blue**
Among 102 fires in 1982–2006: count **P(each cause | fire)** → **85 rows**, must match Zhang **85/85**.

### C7 · Our two upgrades (why better) · **green** bullets on **blue** section

| Fix | Problem before | After upgrade |
|-----|----------------|---------------|
| Person nodes | Couldn’t run “pilot error given evidence” like Zhang’s figures | Person tied to findings — Fig 12 queries work |
| 4-level injury/damage | Boolean “injury yes/no” nodes broke probabilities | One injury node + one damage node with 4 levels each — matches NTSB DEST/SUBS/MINR/NONE and FATL/SERS/MINR/NONE |

### C8 · Build story search list · **blue**
Save embedding vector for every 1982–2006 story once → used in Step 3 on Board 2.  
**Never add 2007–2019 test stories** (see Board 2 Step 3).

---

## SECTION 2 — CHECK WE REBUILT ZHANG · **blue** · p.__ §9

**Job 1 — not the 90.9% story test**

1. Table 7: **85/85** cause rows match  
2. **102** fires; prior matches  
3. Table 9 demo: fire → ranked causes (phrase match, 1982–2006 counts)  
4. Table 10 demo: “inoperative engine instruments” → **P(loss of engine power) ≈ 0.95**  
5. GPT on Zhang’s 11 example texts → same nodes as phrase matcher **11/11**

**Red:** Call these **published examples**, not “benchmarks.”

---

# BOARD 4 — NUMBERS, WORDS, HONESTY (write top → bottom)

---

## SECTION 1 — RESULT TABLES · **green** numbers · p.__ §8

### Injury & damage — 296 accidents · Table 6

**Injury %** = of 296 accidents, what % of the time our **#1 injury pick** matched NTSB’s coded injury level.  
**Damage %** = same for aircraft damage level.

| Method | Injury % | Damage % | Plain English |
|--------|----------|----------|---------------|
| Network with no story input | 58.4 | 42.6 | Always guess most common level |
| Phrase match + suggestions | 82.4 | 50.7 | Events from text help injury some |
| Supervised parser classifier | 85.5 | 60.1 | ML trained on 1982–06 parsed nodes |
| Supervised TF-IDF on text | **92.2** | 73.3 | ML on word counts — **beats us on injury** |
| Supervised embedding classifier | 91.6 | **74.0** | ML on story vectors |
| **Severity through network** ★ | **90.9** | **77.4** | Our main method |
| Neighbor vote alone | 90.9 | 77.4 | Same top pick — no network needed for #1 |

**Top-1** = our single best prediction vs NTSB code — exact level match.

**Macro-F1** = average score across all four injury levels (and four damage levels), treating rare fatal and common “none” equally — not just overall accuracy.

### Cause type — 253 accidents · Table 8

**Top-1 %** = our #1 cause **category** (Personnel/Aircraft/Environment/Organization) is in NTSB’s true set.

| Method | Top-1 % |
|--------|---------|
| Most common category guess | 45.8 |
| Phrase match → network → rank categories | 57.7 |
| **100 similar stories → category vote** ★ | **84.2** |
| Supervised embedding classifier | 88.1 |
| Network “lift” ranking | 50.2 |

**Frequency baseline** = always predict the most common category (Personnel) — dumb floor.  
**Network lift ranking** = rank categories by how much phrase-match evidence **changed** network belief — tested, worked poorly (50.2%).

---

## SECTION 2 — WHAT WE LEARNED FROM EXTRA TESTS · **red** title “tests that failed or warn us”

**Say “extra tests” or “tests that failed” — not “ablation.”**

| Test | What we tried | What happened |
|------|---------------|---------------|
| Combined update | phrase match + neighbor severity together | **38.5% / 41.9%** — worse |
| Fire from phrase match only | events → P(fire) through network | AUC ~0.38 — bad |
| Fire from similar stories | neighbor vote | AUC ~0.96 — works |
| GPT labels on 296 stories | GPT reads text | ~68% injury — much worse than 90.9% |
| Rare fatal injury | only 3/296 | easy to miss — say in limits |
| Organization cause | ~25 cases | ~0% top-1 — rare category |

**Limitations (honesty, not failure):** Part 121 only · limited phrase dictionary · 4 cause categories not exact causes · 296 accidents used during project development may slightly optimistic · **not** cheating — network tables never saw test labels

---

## SECTION 3 — WHAT IS LEARNED FROM DATA vs LOCKED · p.__ Table 5

| Label on wall | Meaning |
|---------------|---------|
| **Locked after build** | Network structure, probability tables, story search list, phrase rules, k=100 — all from 1982–2006 only |
| **Zero learnable numbers** | 100-neighbor vote — just counting similar accidents |
| **Counted once from old stories (off by default)** | How often writers say “minor damage” when code says NONE — only for leak test, not main path |
| **Supervised comparison only** | Logistic regression / TF-IDF / embedding classifiers — **trained on 1982–2006 labels** to show “what ML with training achieves” vs our **no-training** pipeline |

**Jesse answer:** We did **not** train the main pipeline on the 296 test accidents. Only comparison methods learn weights — and only from 1982–2006.

**Logistic regression** = classic linear classifier: learns weights from past examples.  
**Parser classifier** = features = which nodes phrase matcher found.  
**Embedding classifier** = features = story vector from language model.  
**Why include them?** Fair comparison: “If you **are** allowed to train on old accidents, how high can you go?” (TF-IDF 92.2% injury) vs our locked network + similar stories (90.9%) **without** training on labels.

**Calibrated** = estimated probabilities from **count tables** on 1982–2006 (one-time), not iterative ML on test.  
**Fitted / trained** = comparison classifiers with learned weights.

---

## SECTION 4 — DON’T SAY / DO SAY · **red** and **green** columns

| DON’T | DO |
|-------|-----|
| Network beats similar-story vote on injury | Story → locked validated network; 90.9% without training |
| GPT outputs probabilities | Network outputs all P(injury), P(damage) |
| We beat TF-IDF | Honest: TF-IDF 92.2% > our 90.9% on injury |
| Phrase match beats similar stories for causes | Similar stories drive cause type (84.2%) |
| Real-time accident prevention | Retrospective archived NTSB reports |
| Exact cause phrase matching on 2007–2019 test | 4 category types only |

---

## SECTION 5 — 60-SECOND TALK · **black**

“We rebuilt Maha’s network on 1982–2006 and matched her tables. Then we asked: can newer accident **stories**, after stripping cheat phrases, update that **locked** network and match NTSB codes? Main results: **90.9%** injury and **77.4%** damage using 100 similar past accidents; **84.2%** on four cause types. The network gives the same top injury/damage pick as counting similar accidents — but enables her Table 10-style queries. We’re honest: TF-IDF with training scores higher on injury; GPT is a side label reader only.”

---

## SECTION 6 — REPO FILES · **black** small *(file names OK here only)*

`query_to_bn.py` · `bn_upgraded.py` · eval scripts in `tests/` · `heldout_significance.md` · `refined_dataset.json`

---

## SECTION 7 — DAILY READ ORDER · **black**

Board 1 top→bottom → Board 2 flow picture + steps → Board 4 numbers + don’t/do → (Board 3 when you need build detail)

---

# QUICK ANSWERS (your questions — also on wall as margin notes)

| You asked | Short answer |
|-----------|--------------|
| A8 vs regular path? | **Same path.** Four “paths” = four **scores** read at Step 6, not four pipelines. |
| Steps 3 & 4 on Board 1? | On Board 2: Step 3 = similar stories; Step 4a/4b = use them. A8 table points to Step 6 outputs. |
| B11 2020–24? | **Skip on wall** — future draft sentence only; you don’t claim it. |
| P(·)? | Probability of whatever is in the blank — P(injury), P(fire), etc. |
| Index? | Saved vectors of all 1982–2006 stories for fast “find similar.” |
| k-NN? | k=100 nearest neighbor stories by embedding similarity — used Step 3 & 4. |
| Virtual severity? | Say **neighbor counts fed into injury/damage nodes** — on wall Section 4b. |
| bn-fused? | Say **combined test (failed)** — both signals at once in Step 5. |

---

# WRITE ORDER TODAY (4 long boards, top → bottom)

1. **Board 1:** Sections 1–9 (~35 min)  
2. **Board 2:** Flow picture BIG, then Step 0–7 (~40 min)  
3. **Board 3:** Build C1–C8 + Check Zhang (~25 min)  
4. **Board 4:** Numbers + failed tests + locked vs trained + don’t/do + 60-sec (~25 min)  
5. **Photo all four**

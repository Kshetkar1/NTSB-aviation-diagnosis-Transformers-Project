# 6-Hour Maha Prep Plan
**Tomorrow: ~7 min project · ~8 min proposal · ~15 min discussion**

You forget when you **read**. You remember when you **say it + get quizzed**.

---

## Split your 6 hours with two tools

| Source | Own this |
|--------|----------|
| **Cursor / these files** | NTSB BN project — pipeline, numbers, defend, weak, LLM line, slides |
| **Claude** | Proposal — next steps, agent/LLM future, what you’re asking Maha |

**Do not** spend 6 hours re-reading Draft 3.

---

## Files to use (project only)

1. **`MAHA_MEETING_SLIDES_10MIN.html`** — present 6 slides in ~7 min  
2. **`MAHA_MEETING_ONE_PAGE.html`** — memory anchor (print or phone)  
3. **`MAHA_DEFEND_AND_WEAK.md`** — pushback + weak + fixes  
4. **`MAHA_MEETING_OUTLINE.md`** — if stuck on detail  

---

## THE ONE STORY (say this until automatic)

> **Two jobs.** Job 1: Rebuilt Zhang on **1982–2006 codes** — **85/85 Table 7** — **12 of 93 network cells still differ** (random tie-break; fixing in draft). Job 2: **296 newer accidents** (**2007–2019**) — never in the build — narrative → **node values** on **fixed** network → grade vs **NTSB** — **90.9% / 77.4% / 84.2%** (253 causes).  
> **Honest:** Same severity answer as counting the 100 similar accidents (**296/296**). TF-IDF **92.2 > 90.9**. **84.2%** = vote on 100, not BN (**57.7%**).  
> **LLM (~60 sec in, when describing evidence — not first words):** Main pipeline **no GPT for probabilities**. §8.3 = GPT comparison we rejected. Similar-narrative step uses **text-embedding-3-small** (fixed API) — not GPT writing answers.  
> **Weak draft:** Oversells BN — fix abstract, §6.5, split two jobs.

---

## RETRIEVAL STEP — know this (first 20 min of Hour 1)

| Piece | Answer |
|-------|--------|
| **Text in** | Factual narrative, **first 4000 characters**, **outcome phrases stripped first** |
| **Vector model** | OpenAI **`text-embedding-3-small`** (`config.py`) — number profile of text. **Not** GPT for P(·). |
| **Similarity** | **Cosine** — compare vectors, highest = closest (`main_app.find_top_matches`) |
| **Pool** | Pre-saved vectors **1982–2006 only** — `embeddings_1982_2006.npy` |
| **k = 100** | Default in `query_to_bn.py` — **not tuned on the 296** |
| **One sentence** | “Strip phrases → truncate 4000 → **text-embedding-3-small** → compare to **1982–2006** saved index → **100 closest** — 296 never in index.” |

---

# HOUR-BY-HOUR

## Hour 1 (60 min) — Stop forgetting what the project IS

**Goal:** One story in your head, not the whole draft.

| Min | Do |
|-----|-----|
| 0–20 | **RETRIEVAL STEP** table — say one sentence **3×** |
| 20–30 | **ONE STORY** — **2× aloud** |
| 30–40 | **ONE_PAGE** page 1 — pipeline Steps 1–7 aloud once |
| 40–50 | **Close everything** — ONE STORY from memory |
| 50–60 | Fix gaps · write Job 1 / Job 2 / what goes in BN on paper |

**Pass:** You can say ONE STORY without notes.

---

## Hour 2 (60 min) — WHY you made each choice

**Goal:** Maha asks “why?” — you have an answer.

| Min | Do |
|-----|-----|
| 0–15 | **DEFEND_AND_WEAK** Part A — read **Decision rows** only. For each: say **Why** aloud (verify first, strip phrases, 100 similar, 4b-only, vote causes, no learn on 296). |
| 15–30 | **What goes in BN?** Say 5×: “Narratives → **node values** (keyword yes, fractions from 100). Tables **fixed**. One probability run.” |
| 30–45 | **Keywords vs nodes:** “Network has thousands of nodes. Keywords match **some**. Not official NTSB codes — wording → node names.” |
| 45–60 | **Years:** “1982–2006 = **build**. 2007–2019 = **score**.” Not train/test. |

**Pass:** Answer “why strip?” “why 100?” “why not train on 296?” in one sentence each.

---

## Hour 3 (60 min) — Defend + weak + LLM (no gotcha)

| Min | Do |
|-----|-----|
| 0–10 | Memorize **LLM reconciliation** (say 5×): |
| | *“Main pipeline doesn’t use an LLM — keywords and 100 most similar 1982–2006 narratives; network computes all probabilities. §8.3 is GPT comparison; it didn’t replace the network. Not what I run day-to-day.”* |
| 10–25 | **DEFEND_AND_WEAK** Part B — say **weak paragraph** aloud 3× (volunteer before he asks). |
| 25–40 | **Part D “never claim”** — list from memory. |
| 40–55 | Partner or self: 8 pushback Qs from DEFEND table — **no notes**. |
| 55–60 | Rest 5 min. |

**Pass:** LLM line automatic. Weak paragraph automatic.

---

## Hour 4 (60 min) — 7-minute slides + expect interrupts

| Min | Do |
|-----|-----|
| 0–10 | Open **MAHA_MEETING_SLIDES_10MIN.html**. Read slide 1 — add **LLM sentence to open**. |
| 10–35 | Present slides **1, 2, 3, 5, 6, 7, 8, 9** only (~7 min). **Timer.** Arrow keys. |
| 35–50 | **Interrupt drill:** Stop at slide 3. Answer 3 min: “What goes in BN?” “Train on 296?” “90.9% how?” Then finish. |
| 50–60 | Second run: **5 min** — slides 1, 3, 5, 8, 9 only. |

**Pass:** Finish core slides in ≤7 min. Survive stop at slide 3.

**Slides to use:** 1 → 2 → 3 → 5 → 6 → 7 → 8 → 9 → “questions / proposal next”

---

## Hour 5 (60 min) — Proposal (Claude + you)

**Cursor doesn’t have your proposal — use Claude here.**

| Min | Do |
|-----|-----|
| 0–10 | List 4 bullets: **What I propose · Why now · What changes · What I need from Maha** |
| 10–40 | With Claude: build **8-min proposal talk** + 4–6 slides or one page. Tie to: current limits (296/296, TF-IDF, draft framing). |
| 40–55 | Say proposal **aloud once** timed (~8 min). |
| 55–60 | One line link: *“Current work = fixed BN + no LLM main path; proposal = …”* |

**Pass:** 8-min proposal without rambling.

---

## Hour 6 (60 min) — Mock meeting + sleep buffer

| Min | Do |
|-----|-----|
| 0–5 | Open: ONE STORY (no LLM first) + “interrupt me anytime.” LLM line when you hit **how evidence is extracted** (~60 sec). |
| 5–12 | **7 min** project (slides or no slides). |
| 12–20 | **8 min** proposal. |
| 20–35 | **15 min mock Q&A** — use DEFEND table; Claude can play Maha for proposal Qs. |
| 35–45 | Write **3 questions FOR Maha:** e.g. JAIS framing? §8.3 appendix? priority edits? |
| 45–50 | Photo: ONE_PAGE page 1 + LLM sentence on phone. |
| 50–60 | **Stop.** Light skim ONE STORY once. Sleep. |

**Pass:** Mock feels boring — you’re repeating, not learning. That’s ready.

---

## If you only have 4 hours

Do **Hours 1 + 3 + 4 + 5** (skip deep Hour 2 — use ONE_PAGE pipeline during Hour 4).

---

## Morning of meeting (15 min max)

1. ONE STORY **once aloud**  
2. LLM line **once**  
3. Weak paragraph **once**  
4. Walk in — **don’t** reread draft  

---

## What to ask Claude (same 6-hour question)

Paste Claude:

> “I have 6 hours. Hour 5 is proposal. Help me: (1) 8-min proposal script, (2) 4–6 slides outline, (3) how proposal connects to current fixed-BN no-LLM pipeline, (4) mock Q&A on proposal only. Current limits: 296/296 severity, TF-IDF beats us, draft oversells BN.”

---

## Remember: meeting is not a exam on the draft

Maha wants:

1. You **understand** what you did  
2. You’re **honest** where it’s weak  
3. You know **why** you chose each step  
4. **Proposal** makes sense as next step  

The draft is evidence you wrote — **tomorrow you are the expert in the room on the process**, not the PDF.

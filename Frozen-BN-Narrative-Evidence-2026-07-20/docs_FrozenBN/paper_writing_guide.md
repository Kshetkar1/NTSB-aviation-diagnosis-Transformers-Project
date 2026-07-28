# Paper Writing Guide

A checklist for writing the human version of the NTSB paper. Use this as a side panel while you write in Word. The guide gives you content prompts (what to talk about, what numbers to cite, what to avoid). It does not give you sentences. Everything you write comes from your head.

---

## Before you start

**Files to have open in a side panel:**
- `docs/full_paper_v5_draft.docx` — read once, then close. This is the AI version. It tells you the SHAPE of the paper. Don't keep it open while writing.
- `Worked_Examples/worked_examples_v2.html` — your reference for the two cases. Open it in a browser.
- `Worked_Examples/data/example_after.json`, `example_during.json`, `example_after_landing_gear.json`, `example_during_landing_gear.json` — for any number you need to quote.
- `Worked_Examples/data/aggregate_results.jsonl` — the 77-case data.

**Set a writing rule for yourself:**
- One section per sitting. Don't try to write the whole paper in one go.
- After each section, do something else for 15 min, then come back and read what you wrote out loud. If it doesn't sound like you, rewrite it.
- If you catch yourself opening v5 to "check how the AI said it," close v5. That defeats the point.

**Words to avoid** (these are AI tells):
- em dashes (—)
- en dashes used as punctuation (–)
- "leverage", "robust", "comprehensive", "delve", "tapestry"
- "moreover", "furthermore", "in particular", "in conclusion"
- "It is important to note that..."
- "This paper presents..." (use "We present...")

**Words and patterns that sound like you, based on your meeting transcripts:**
- "what we're doing here is..."
- "the goal is..."
- "I want to..." / "we want to..."
- Short sentences. Don't fight it.
- It's OK to say "this didn't work" or "we are still figuring this out"

---

## Suggested writing order

Don't write top to bottom. Write the parts you have the most evidence for first.

| Order | Section | Why |
|-------|---------|-----|
| 1 | §5 Methodology | Factual, you know how the pipeline works |
| 2 | §6 Structural Mapping | Factual, you know how A2 works |
| 3 | §7 Worked Examples | Table-driven, the data tells the story |
| 4 | §8 Aggregate Validation | Table-driven, n=77 results |
| 5 | §9 Discussion | Needs §7 and §8 in front of you |
| 6 | §10 Conclusion | Needs §9 done |
| 7 | §4 Background Concepts | Short, mostly definitions |
| 8 | §3 NTSB Data | Short |
| 9 | §2 Related Work | Needs you to know what you cite |
| 10 | §1 Introduction | Easier once you know what you're introducing |
| 11 | Abstract | Always last. You can't summarize what isn't written yet |

---

## Section-by-section guide

### Abstract (~200 words, write LAST)

**Job:** Tell the reader what the paper is, what it does, and what we found, in one paragraph.

**Claims to make:**
- The problem (NTSB analysis has used coded data only; narratives are ignored)
- Our approach in one sentence (retrieve, cluster, combine with law of total probability)
- We support two query styles (investigator full narrative, pilot voice)
- Structural mapping extension reranks retrieved candidates
- Headline aggregate number on 77 held-out test incidents
- One sentence on what this is useful for

**Numbers to drop in:**
- 2,243 incidents
- 77 held-out test
- Mean cluster cosine: A0 = 0.458, A2 = 0.466
- Solid match (cos > 0.50): 42% of 77 cases
- A2 vs A0: 17 wins, 51 ties, 9 losses

**Don't:**
- Mention Zhang or "prior work" framing
- Oversell ("transformative", "state-of-the-art")
- Use the word "leverage"

**Self-check:** Can someone who doesn't read the rest of the paper get the gist in 60 seconds?

---

### §1 Introduction (~700-900 words)

**Job:** Convince the reader the problem matters and tell them what this paper does about it.

**Claims to make:**
- Aviation accidents are studied to prevent future ones
- NTSB reports have two kinds of data: structured (coded) and unstructured (narratives)
- Statistical models of NTSB data have used the coded fields only
- Narratives have context the codes don't capture
- LLMs can now process narratives directly
- We use LLMs for retrieval + clustering, frequency counts for probabilities
- We support two practitioner views: investigator (after-event) and pilot (during-event)
- Brief roadmap of the paper

**Numbers to drop in:**
- 2,243 FAR 121 incidents, 1982-2016
- 14 NTSB files merged per incident

**Don't:**
- Cite Zhang or frame as "extending prior work"
- Start with "Aviation safety is a critical issue" (cliche)
- Use "leverage" or "robust"

**Self-check:** A reader who skips to §5 should already know what the paper is doing and why.

#### §1.1 Limitations of the Pure LLM Approach (~200 words)

**Job:** Explain why we don't just use an LLM end-to-end.

**Three problems to list:**
1. LLM-produced probabilities are not grounded in data, not reproducible
2. Hallucination (confident outputs not supported by any incident)
3. Sensitivity to query phrasing (small wording changes change the answer)

**Lesson learned:** LLMs are good at retrieving and structuring text. Frequency counts are good at producing reliable probabilities. Keep them separate.

#### §1.2 Contributions (~250 words)

**Numbered list. Four items:**
1. Hybrid pipeline: LLM for retrieval + clustering, frequency counts for probabilities via law of total probability
2. Two-query design (after-incident investigator narrative, during-incident pilot voice)
3. Structural mapping extension (causal chain extraction + Needleman-Wunsch alignment + fusion formula reranking)
4. Aggregate evaluation on 77 held-out test incidents

**Don't:** Pad the contributions list. Four real things beats six fake ones.

---

### §2 Related Work (~500-700 words total)

**Job:** Position your work against existing literature.

#### §2.1 NTSB Accident Analysis (~150 words)

**Claim:** Prior work on NTSB data has used coded fields, not narratives.

**Pitfall:** You currently have one paragraph and no citations. You need 2-3 real citations. Candidates to look up:
- Tanguy et al. on aviation incident classification with text
- Madsen et al. (2009) on NTSB text mining
- Robinson et al. on narrative analysis in aviation safety
- (Ask Maha for his preferred citations here — this is a perfect question to ask him)

**Don't:** Cite Zhang/Mahadevan (2021) unless Maha says you should.

#### §2.2 Structure-Mapping Theory (~250 words)

**Cite (in this order):**
- Gentner (1983) — structure-mapping for analogical reasoning
- Gentner and Markman (1997) — analogy vs. literal similarity
- Falkenhainer et al. (1989) — Structure-Mapping Engine algorithm
- Goldstone (1994) — alignment + feature comparison together
- Spencer-Smith and Goldstone (1997) — dynamic weights (this is your co-author Jesse's paper)

**Make this point:** We don't implement SME directly. We use a smaller, domain-tuned version (3 fields: role, system, mechanism).

#### §2.3 Two-Stage Retrieval and Reranking (~150 words)

**Cite:**
- Nogueira and Cho (2019) — passage re-ranking with BERT, two-stage pattern
- Needleman and Wunsch (1970) — sequence alignment algorithm

**Make this point:** Our fusion (embedding cosine + structural score) fits the standard two-stage retrieval pattern.

---

### §3 NTSB Accident Investigation Data (~350-450 words)

**Job:** Describe the dataset.

**Things to say:**
- NTSB is independent federal agency, all modes of transportation
- We focus on FAR 121 air carriers
- Dataset is 2,243 incidents from 1982 to 2016
- 14 separate files per incident; we merge them
- Two fields used heavily: sequence of events (with defining event anchor) and findings (cause labels)
- Held-out split: 2,166 train / 77 test

**Don't:** Say "the same dataset as prior work" — this is the stand-alone framing.

---

### §4 Background Concepts (~500-600 words)

**Job:** Define what readers need to follow §5 and §6.

Four short subsections:

#### §4.1 Vector Embeddings and Cosine Similarity
- OpenAI text-embedding-3-small, 1536 dimensions
- Equation 1: sim(q, i) = q · i / (||q|| · ||i||)
- Range 0 (unrelated) to 1 (identical)

#### §4.2 K-means Clustering
- We use k=50 on the 2,243 incident embeddings
- Each cluster gets an LLM-generated label

#### §4.3 Law of Total Probability
- Equation 2: P(C | Q) = Σ P(C | K_j) · P(K_j | Q)
- Used for both diagnosis and prognosis

#### §4.4 Needleman-Wunsch Sequence Alignment
- Equation 3: the recurrence
- Used to align causal chains (Section 6)

**Don't:** Over-explain. These are background. Readers can look up details if they want.

---

### §5 Methodology: The Hybrid Pipeline (~1500 words)

**Job:** Walk through the diagnosis and prognosis pipeline step by step.

Subsections (matching v5):
- §5.1 Merged dataset
- §5.2 Embeddings (4,360 vectors, three text types per incident)
- §5.3 Similarity search (top 50)
- §5.4 Clustering (k=50 precomputed)
- §5.5 Two query styles (NEW SECTION, important!)
- §5.6 Diagnosis (with equations 4-6)
- §5.7 Prognosis (with equations 7-8)

**For §5.5 (Two Query Styles) — this is a contribution, write it carefully:**
- After-incident query: full investigator narrative, thousands of characters, view of someone analyzing post-event
- During-incident query: one or two sentences, pilot voice, view of someone mid-event
- Same retrieval + clustering + probability steps for both
- Examples are shown in §7 (forward reference)
- Disclose that during-incident queries used in §7 are hand-written, not from a real pilot transcript corpus

**Self-check:** A reader should be able to re-implement the pipeline from this section.

---

### §6 Methodology: Structural Mapping Extension (~1300 words)

**Job:** Explain how A2 adds a second similarity signal and reranks.

Subsections:
- §6.1 Motivation (why embedding alone isn't enough)
- §6.2 Causal chain schema (role, system, mechanism fields)
- §6.3 LLM extraction (GPT-4o-mini, fixed schema, cached)
- §6.4 Step similarity (equation 9: 0.50 role + 0.30 mechanism + 0.20 system)
- §6.5 Chain alignment (Needleman-Wunsch + equation 10 for struct_sim)
- §6.6 Score fusion (equation 11: new_score = cos · exp(α · struct_sim), α=2.0)

**Key point in §6.6:** When struct_sim = 0, exp(0) = 1, so A2 reduces to A0. A2 can never hurt a candidate's raw cosine; it only boosts mechanism-matched candidates more than others.

**Don't:** Call this "Phase 3" (Zhang-paired naming).

---

### §7 Worked Examples (~1800 words including tables)

**Job:** Show two real cases end-to-end.

Two cases:
- §7.1 Engine fire (20100114X11754)
- §7.2 Landing gear (20081116X33137)

Each case has:
- A short prose intro to the incident (where, when, what happened, what NTSB found)
- §7.x.1 After-incident query (the full narrative)
- §7.x.2 During-incident query (the pilot voice version)

Each scenario has SIX tables (or three side-by-side ones if you want to consolidate):
- A0 top-5 clusters with P(K|Q)
- A2 top-5 clusters with P(K|Q)
- A0 top-5 causes from LTP
- A2 top-5 causes from LTP
- A0 top-5 next events
- A2 top-5 next events

**All tables: pull numbers from the JSON files.** Don't make up any numbers.

**§7.3 Summary table** — the four-row summary across the two cases and two query styles:
| Scenario | A0 top cluster | P(K|Q) A0 | A2 top cluster | P(K|Q) A2 |
| Engine fire, after | engine fire due to component failures | 50.4% | engine fire due to component failures | 53.5% |
| Engine fire, during (pilot) | engine fire due to component failures | 52.7% | engine fire due to component failures | 53.7% |
| Landing gear, after | nose landing gear malfunctions | 14.8% | nose landing gear malfunctions | 18.1% |
| Landing gear, during (pilot) | nose landing gear malfunctions | 38.7% | nose landing gear malfunctions | 40.7% |

**Important honest point to make in §7.3:**
- The landing gear after-incident case has lower cluster mass (15-18%) than the engine fire case (50-54%). Don't hide this. Explain it: the long narrative includes approach, runway, crew-action prose that pulls retrieval into multiple themes.
- The pilot voice query on the same incident lands at 39-41% because it's shorter and stays closer to the failure-relevant words.

---

### §8 Aggregate Validation (~900-1000 words)

**Job:** Show the worked examples aren't cherry-picked.

Subsections:
- §8.1 Setup (77 held-out, train index, never seen during pipeline)
- §8.2 Metric definition (cosine similarity between predicted top cluster and NTSB probable-cause text, in same embedding space we retrieve in)
- §8.3 Aggregate results table (the big one)
- §8.4 A2 vs A0 head-to-head
- §8.5 Secondary metric: M1 / McNemar (for the statistically inclined)
- §8.6 Limitations of the metric

**§8.3 numbers to drop:**
- Mean cluster cosine: A0 = 0.458, A2 = 0.466
- Median: A0 = 0.443, A2 = 0.467
- Solid match (cos > 0.50): A0 = 29/77 (37.7%), A2 = 32/77 (41.6%)
- Plausible match (cos > 0.40): A0 = 45/77 (58.4%), A2 = 48/77 (62.3%)
- At least related (cos > 0.30): A0 = 64/77 (83.1%), A2 = 63/77 (81.8%)
- Mean P(top cluster | Q): A0 = 27.8%, A2 = 28.6%
- Mean top-cause cosine: A0 = 0.374, A2 = 0.378

**§8.4 numbers:**
- Cluster cosine: 17 A2 wins, 51 ties, 9 A2 losses
- Top-cause cosine: 10 A2 wins, 53 ties, 14 A2 losses
- A2 no-harm rate on cluster metric: 68/77 (88%)

**§8.5 (M1 / McNemar):** Top-1 A0 = 25/77 (32.5%), A2 = 27/77 (35.1%). McNemar exact p = 0.500 (2 disagreements, both in A2's favor).

**§8.6 limitations to acknowledge:**
- Threshold (0.30/0.40/0.50) is a choice — we report three to let the reader pick
- The same embedding model is used for retrieval AND for the comparison metric (circularity)
- n=77 is small; intervals are wide

---

### §9 Discussion (~700-900 words)

**Job:** Tell the reader what to take away. This is the section Maha will mark up most heavily if it's too short.

**Three observations to develop in order:**

1. **Cluster is the right level to report a probability at.** Per-cause P(Cause | Q) spreads thin because the FAR 121 cause taxonomy is large; top-1 cause probability is in the single digits. Cluster mass P(K | Q) is concentrated and decision-useful. Both engine-fire scenarios land at 50-54% on a single cluster.

2. **Query length affects retrieval quality.** Landing gear after-incident at 15% vs. landing gear during-incident pilot voice at 39%. The pipeline is the same. The query is shorter and more focused, so retrieval concentrates on the failure family.

3. **Structural rerank is a small consistent positive, not a transformative one.** 17 wins, 51 ties, 9 losses on the cluster metric. A2 helps on specific cases where mechanism alignment breaks a near-tie.

**Things Maha will push back on if you don't address (per my earlier critique):**
- Why does A2 only help on 17 of 77? Is it that 51 cases are too easy, or 51 cases are too hard?
- What does k=50 cost or buy you? What if k=20 or k=100?
- What happens when the right answer isn't in the corpus? (No, the system still confidently returns a top cluster)

**Don't:** Make this section a victory lap. The honest moments build credibility.

---

### §10 Conclusion (~450 words)

#### §10.1 Contributions (re-state the four from §1.2, more compact)

#### §10.2 Limitations
- n=77 test set is small; intervals wide
- LLM extraction errors propagate into struct_sim
- α=2.0 set by initial experiments, not tuned
- During-incident pilot queries are hand-written, not from real pilot transcripts
- Same embedding model used for retrieval and evaluation (circularity in §8 metric)

#### §10.3 Future Work
- Validate during-incident query style against real pilot transcripts / CVR data
- Evaluate A2 on the prognosis pipeline (same head-to-head design)
- Learn the step-similarity weights (0.50, 0.30, 0.20) from data instead of fixing them

---

## After you finish writing

**Self-checks before sending to Maha:**

1. Open your version and v5 side-by-side. Spot-check 3 random paragraphs in each. Do they feel like they were written by different people? If they're too similar in structure or word choice, rewrite yours.
2. Search your doc for any of the AI tells from the top of this guide (em dashes, "leverage", etc.). Fix any that snuck in.
3. Read the abstract out loud. If you stumble on any phrase, rewrite that phrase.
4. Check that every number in the paper is traceable to a JSON file or computed quantity. No made-up examples.
5. Make sure §7.3, §8.6, and §10.2 are honest about the weak landing-gear after-incident result and the metric circularity. The credibility of the paper rests on these.

**When you send to Maha:**

Send both files. Label them clearly:
- `full_paper_v5_ai_draft.docx` (the AI version, for his content review only)
- `full_paper_v5.docx` (your version, the submission candidate)

A short cover note works: "Here are both versions you asked for. The AI draft was used as a structural reference; the v5 version is in my own words and is what I would submit."

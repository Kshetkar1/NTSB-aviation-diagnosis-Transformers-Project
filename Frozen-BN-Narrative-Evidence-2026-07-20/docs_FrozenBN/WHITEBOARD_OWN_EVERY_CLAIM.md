# OWN EVERY CLAIM — say it because X, proof is Y
**Rule from Maha:** Do not guess. Every sentence you say, you must be able to explain **how it works**, **why we did it**, and **where we tested it**.  
Use with: `WHITEBOARD_PRINT_STUDY.html` + `WHITEBOARD_PIPELINE_WALL.md`

**If you cannot point to a row in this table, do not say the claim aloud.**

---

## How to use this with Maha

For each topic, speak in **four beats**:

1. **Claim** — one number or one sentence  
2. **How** — what the pipeline actually does (Steps 1–7)  
3. **Why** — design reason (not “because AI said so”)  
4. **Proof** — script, output file, draft table, or audit we ran  

Example: *“90.9% injury top-1 on 296 test accidents — because we take 100 most similar 1982–2006 stories, count their coded injury levels, feed that into the injury node in the locked network, and pick the highest probability. We chose that path as primary because it uses the full narrative without training on test labels. Proof: `tests/frozenbn_heldout_narrative_bn_eval.py` → `outputs/heldout_significance.md` and draft Table 6.”*

---

## JOB 1 — Check we rebuilt Zhang (1982–2006)

| What I say | How it works | Why | Proof |
|------------|--------------|-----|-------|
| **85/85 Table 7 rows match** | Among 102 fires in 1982–2006, count P(each cause \| fire) for 85 cause rows — same counting as Zhang §5.6 | Job 1: verify rebuild before any story test | `tests/verify_zhang_table7.py` · `Appendix_A_Table7.csv` · draft App A · p.__ |
| **102 fire accidents** | Count accidents with fire in window dataset | Same denominator Zhang uses for fire queries | `refined_dataset_1982_2006.json` · cohort manifest |
| **P(fire) = 102 / 184,517,128** | Prior = event count ÷ BTS total US airline flights (Zhang Eq. 6) | Rare events measured per flight, not per accident only | BTS table · `bn_build_ours.py` TOTAL_FLIGHTS · draft §5.3 · p.__ |
| **Table 9 fire → ranked causes** | Phrase match “fire” on 82–06 population → count P(cause \| fire) — **no** 100 similar stories | Zhang’s published diagnosis **example** — not §8 test | `apps-parity/diagnosis_view.py` · draft §9.1 Table 9 · p.__ |
| **Table 10 engine instruments → P(loss of engine power) ≈ 0.95** | Phrase match on instrument nodes → forward propagation on locked network | Zhang’s published prognosis **example** | `apps-parity/prognosis_view.py` · draft §9.2 Table 10 · p.__ |
| **GPT on Zhang’s 11 texts = phrase match 11/11** | Same node labels as deterministic parser → same posteriors | Side check: GPT as label reader only on **his** short examples | `tests/frozenbn_tiered_parser_validation_11_scenarios.py` · draft §8.3 · p.__ |

**Do not say:** 90.9% proves we rebuilt Zhang. **Job 1 numbers ≠ Job 2 numbers.**

---

## JOB 2 — Story test (296 accidents, 2007–2019)

### Cohort — own these before any accuracy

| What I say | How it works | Why | Proof |
|------------|--------------|-----|-------|
| **n = 296** for injury/damage | Accidents in full dataset **not** in 1982–2006 window, with `narr_accf` length ≥ 100, Part 121 | Same filter for every predictor — fair comparison | `tests/build_cohort_manifest.py` → `outputs/cohort_manifest.json` · draft Table 1 · p.__ |
| **Stories truncated to 4,000 characters** | All methods see first 4000 chars of `narr_accf` | No story in our data exceeds 4000 — cap is uniform, not arbitrary trimming | Checked on `refined_dataset.json` · eval script line truncates 4000 |
| **n = 253** for cause type | Same 296 minus 43 with no scorable C/F cause findings | Cannot grade cause if NTSB has no coded cause to compare | `diagnosis_heldout_eval.py` · draft §8.2 · p.__ |
| **296 never in story search list** | Embeddings built only from 1982–2006; test IDs excluded from pool | Test accidents are the **exam** — past-only search | `tests/heldout_leak_audit.py` — **0/296** ID overlap · Step 3 in pipeline doc |
| **NTSB codes only for grading** | Steps 1–6 never read coded injury/damage/cause as input; Step 7 compares | Otherwise we predict the answer from the label | Eval scripts: truth loaded after prediction · draft §7 · p.__ |

---

### Pipeline steps — own the mechanism

| Step | What I say | How it works | Why | Proof |
|------|------------|--------------|-----|-------|
| **1 Strip** | We remove stated injury/damage phrases before anything else | Regex list strips “substantial damage”, “fatal injuries”, hospitalization boilerplate, etc. → one clean story | Narratives **state** outcomes; using that text to predict coded severity is cheating | `code/query_to_bn.py` → `redact_severity_phrases()` · `tests/redaction_leak_probe.py` · draft §6.1 · p.__ |
| **2 ∥ 3 Same time** | Phrase match and 100 similar stories both run on the **same** clean story | Step 2: words → Zhang node names. Step 3: embed story → 100 closest 1982–2006 accidents from story search list | Two independent uses of text: explicit vocabulary + similar past cases | `parse_query_to_bn_evidence()` · `main_app.find_top_matches()` · flow diagram |
| **2 Phrase match** | Words in story → network nodes set to YES | 3 passes: outcome, person aliases, multi-word events; skip negated/hypothetical clauses | Uses Zhang’s node vocabulary directly when narrative matches | `query_to_bn.py` · alone **82.4%** inj / **50.7%** dmg — draft Table 6 |
| **3 — 100 similar stories** | Embed clean story; keep 100 nearest 1982–2006 narratives | Vectors in `embeddings_1982_2006.npy`; cosine similarity; k=100 | Narrative signal from **past similar accidents** without training on 296 labels | `query_to_bn.severity_retrieval_distributions(top_k=100)` · draft §6.3 · p.__ |
| **4a ∥ 4b Same time** | Both use **same** 100 from Step 3 | 4a: weighted fraction of neighbors with event label X → suggest node. 4b: count neighbors’ **coded** injury/damage levels | Same neighbor pool; different readouts (events vs severity codes) | `retrieval_facts()` · `severity_retrieval_distributions()` · draft §6.3–6.4 |
| **4b → main severity** | Count injury/damage codes on 100 neighbors → fractions | e.g. 41/100 none, 28/100 minor — pick most common OR feed into injury/damage nodes | Severity from **past coded outcomes**, not from new story’s outcome phrases | Primary path in eval · Table 6 |
| **5 Network update** | pyAgrum updates probabilities on locked network | **Main:** neighbor severity counts → injury/damage nodes only. **Alt:** Step 2 (+4a). **Failed:** all together | Main path isolates neighbor severity; combined path double-counts same narrative | `frozenbn_heldout_narrative_bn_eval.py` → modes bn-sev, bn-fused · draft §6.5 Table 5 · p.__ |
| **6 Read** | A) argmax P(injury), P(damage). B) vote cause category on same 100 | A from Step 5. B: Personnel/Aircraft/Environment/Organization from neighbors’ findings | 2008 taxonomy break — era-fair 4 categories only on test | `diagnosis_heldout_eval.py` · draft §8.2 · p.__ |
| **7 Grade** | Compare to NTSB coded truth | Top-1 match, Macro-F1, Brier, bootstrap CI | Standard held-out evaluation | `outputs/heldout_significance.md` · `diagnosis_heldout_eval.md` |

---

### Main results — own every number

| What I say | How | Why this number (not another) | Proof |
|------------|-----|-------------------------------|-------|
| **90.9% injury top-1** | Severity through network: Step 4b → Step 5 (4b only) → pick highest P(injury level) | **Primary** predictor `bn-sev` in eval; leak-safe redaction on all runs | Table 6 · `heldout_significance.md` · n=**296** |
| **77.4% damage top-1** | Same path for damage node | Same primary path | Table 6 · n=**296** |
| **Same top pick as neighbor vote (296/296)** | Neighbor vote alone (skip network) vs severity through network | We **tested** whether network secretly changes top answer — it does not | Eval compares `retrieval-sev` vs `bn-sev` · draft §8.1 check sentence · p.__ |
| **84.2% cause type top-1** | Step 3’s 100 neighbors → vote C/F finding categories rolled to 4 types | **Primary** dx predictor = retrieval category vote, not phrase match | Table 8 · `diagnosis_heldout_eval.md` · n=**253** |
| **57.7% phrase match → network for causes** | Step 2 → Step 5 with event nodes → rank categories by posterior | Shows phrase path **weaker** than similar-story path for causes | Table 8 |
| **38.5% inj / 41.9% dmg combined test** | Step 2+4a **and** 4b in **one** network update | **We ran this on purpose** to show merging signals hurts (double-count) | Table 6 row bn-fused · say “combined test failed” |
| **92.2% injury TF-IDF** | Supervised classifier on bag-of-words, **trained on 1982–2006 labels only** | Comparison: “if you **train** on old data, you can beat us” | `tests/lr_baseline_heldout.py` or TF-IDF script · Table 6 · **say honestly we do not beat this** |
| **88.1% embedding classifier on causes** | Supervised on story embeddings, trained 1982–2006 | Same comparison for cause categories | `diagnosis_emb_lr_baseline.py` · Table 8 |
| **Prior 58.4% / 42.6%** | Network with **no** story input — guess most common level | Floor: how well you do with no narrative | Table 6 row “prior” |

---

## Design choices — own the “why”

| What I say | Why (because…) | Proof we actually checked |
|------------|----------------|---------------------------|
| **Network locked after build** | 296 are the exam; updating CPTs or embeddings from test labels would be cheating | `FREE_PARAMETERS.md` · draft Table 5 · Jesse answer: main pipeline not fitted on 296 |
| **Test 2007–2019, not 1982–2006 stories** | Same years built network **and** story search list — circular if we test there | Design doc · heldout leak audit |
| **Strip cheat phrases first** | Probe showed outcome phrases predict severity without real inference | `redaction_leak_probe.py` |
| **100 similar stories (not 25)** | Default at build; k=25 ablation: 89.9%/77.7% vs 90.9%/77.4% — flat | `outputs/heldout_significance_k25.md` · did **not** switch to 25 post-hoc |
| **4 cause categories only** | NTSB legacy vs CICTT after 2008 — exact phrase match unfair | `diagnosis_heldout_eval.py` header + keyword rollup · mapping audit sample |
| **Person nodes + 4-level injury/damage upgrades** | Boolean injury leaves broke P(no injury); pilot-error queries needed person→finding edges | `bn_upgraded.py` docstring · Fig 12 / Table 10 demos work |
| **Max 12 parents** | Zhang recipe + cap — sparse CPT cells if too many parents | `bn_build_ours.py` MAX_PARENTS · deterministic from data |
| **Beta-CDF smoothing** | Rare parent combos → raw 0%/100%; Zhang §4 recipe smooths | Faithful reproduction · `bn_build_ours.py` |
| **GPT is side only** | Early test: GPT output P(·) directly → unstable/wrong; pivoted to network-only probabilities | Draft §8.3 four experiments · ~68% inj with GPT vs 90.9% pipeline |
| **Do not claim BN beats neighbor vote on injury** | We measured 296/296 same top pick | Table 6 + agree-on-top-pick check |
| **Do not claim we beat TF-IDF** | Table 6: 92.2% > 90.9% on injury | Table 6 |

---

## Two upgrades — own exactly what you changed

**Claim:** “We extended Zhang’s graph in two ways — we did not replace it.”

| Upgrade | What Zhang had | What we changed | Why | Proof it works |
|---------|----------------|-----------------|-----|----------------|
| **Person roles** | Findings tied to events | Added `person: pilot-in-command` (etc.) → finding edges | Zhang’s human-factor queries need person nodes | Table 10 / Fig 12-style queries run in `apps-parity` |
| **Four-level severity** | Separate yes/no injury leaves | One `personnel injury` node (fatal/serious/minor/none) + one `aircraft damage` node (4 levels) | Boolean leaves forced broken probabilities; NTSB uses 4-level codes | `bn_upgraded.py` · injury from Zhang’s per-person rule (`zhang_injury_code`) |

---

## What is trained vs not — own Jesse’s question

**Say:** “The main pipeline has **no learned weights** fit on the 296 test accidents. The network, probability tables, story search list, phrase rules, and k=100 were all fixed from 1982–2006 before we scored the test set. The only **trained** models are **comparison baselines** — logistic regression, TF-IDF, embedding classifiers — trained on **1982–2006 labels only** so we can show what supervised ML achieves vs our no-training pipeline.”

**Proof:** `docs_FrozenBN/FREE_PARAMETERS.md` · draft Table 5 · p.__

**Never say:** “We trained on 296 accidents.”

---

## Honest limits — own these too (builds trust)

| Limit | Because | Proof |
|-------|---------|-------|
| Part 121 airline only | Dataset scope | draft §3 |
| Phrase dictionary limited | Only multi-word / vocabulary matches become nodes | parser code · 82.4% not 100% from phrases alone |
| TF-IDF beats us on injury | Table 6 | ran baseline scripts |
| Fatal injury rare (3/296) | Class imbalance | disclose in §8.1 |
| Organization cause ~0/25 top-1 | Rare category | diagnosis eval |
| 296 used during project development | May be slightly optimistic — **not** label leakage into CPTs | draft §10.3 · network tables never saw test labels |

**Do not put on wall unless asked:** 2020–2024 confirmatory — not in current dataset; draft future work only.

---

## Maha drill — answer template (fill in proof from table)

| If Maha asks… | You say… | Because… | Proof |
|---------------|----------|----------|-------|
| What did you do? | Rebuilt network 82–06, verified tables, tested redacted stories on 296 newer accidents | Two jobs: verify then generalize | This doc Job 1 + Job 2 |
| How does text reach the network? | Strip → phrase match **∥** 100 similar stories → update nodes → read P(·) | Narrative never raw in BN math | Flow diagram · `query_to_bn.py` |
| Where does 90.9% come from? | 100 similar stories’ **coded** injury levels → injury node → top level | Primary path bn-sev / severity through network | Table 6 · eval script |
| Does BN improve over neighbors? | **No** on top-1 injury/damage — same pick all 296 | We ran both and compared | agree 296/296 |
| Where does 84.2% come from? | Same 100 stories — vote cause **category** | Retrieval beats phrase path for causes | Table 8 |
| What did you add to Zhang? | Person nodes + 4-level injury/damage | Fix broken leaves + pilot queries | `bn_upgraded.py` |
| What is trained? | Nothing main on 296; baselines only on 82–06 | Fair test | FREE_PARAMETERS.md |
| What failed? | Combined phrase + neighbor in one update → 38%/42% | Double-count narrative | Table 6 bn-fused |
| Can GPT replace parser? | Not on 296 test (~68% inj vs 90.9%) | §8.3 experiments | draft §8.3 |
| Why 4 categories not exact causes? | 2008 taxonomy change | Era-fair rollup | diagnosis_heldout_eval.py |

---

## Before the meeting — 5-minute proof check

- [ ] I can name **two jobs** and which numbers belong to each  
- [ ] I can draw **Step 2 ∥ Step 3** and **4a ∥ 4b** without skipping  
- [ ] I can say **90.9 / 77.4 / 84.2** with **n=296 or 253** and **Table 6 or 8**  
- [ ] I can say **why redact** and point to **leak probe**  
- [ ] I can say **what we do not claim** (BN beats neighbors, beat TF-IDF, GPT does P(·))  
- [ ] I know where **`heldout_significance.md`** and **`diagnosis_heldout_eval.md`** live  

If any box fails, do not claim that topic until you re-read the row in this file.

---

## Repo proof index (open these if challenged)

| Topic | File |
|-------|------|
| Severity eval | `tests/frozenbn_heldout_narrative_bn_eval.py` |
| Severity results | `Frozen-BN-Narrative-Evidence-2026-07-20/outputs/heldout_significance.md` |
| Diagnosis eval | `tests/diagnosis_heldout_eval.py` |
| Diagnosis results | `outputs/diagnosis_heldout_eval.md` |
| Narrative → network | `code/query_to_bn.py` |
| Build network | `code/bn_upgraded.py` |
| Table 7 verify | `tests/verify_zhang_table7.py` · `Appendix_A_Table7.csv` |
| Leak audit | `tests/heldout_leak_audit.py` · `tests/redaction_leak_probe.py` |
| Cohort n=296 | `outputs/cohort_manifest.json` |
| Trained or not | `docs_FrozenBN/FREE_PARAMETERS.md` |
| Reproduce all | `REPRODUCE.md` |
| Demo Table 9/10 | `apps-parity/streamlit_app.py` |

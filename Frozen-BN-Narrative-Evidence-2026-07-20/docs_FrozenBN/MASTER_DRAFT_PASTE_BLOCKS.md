# Master Draft v2 — paste blocks (Jul 30)

**Target file:** `/Users/kanushetkar/Desktop/Master Draft NTSB Paper 7:30:26.docx`

**Do first:** Add under the title:

> **Draft v2 — Jul 30.** Results verified against `RESULTS_SECTION.md` (leak-safe, Jul 29). July 26 file superseded.

---

## Paste order (~45–90 min)

| Step | Where in Word doc | Action |
|------|-------------------|--------|
| 1 | **Abstract** (para 5) | Paste **Abstract** below |
| 2 | **§2.1** (under heading) | Paste **§2.1** |
| 3 | **§2.2, §2.3** (stubs) | Paste one line each OR skip |
| 4 | **§5.2** | **Delete** duplicate paras 43–44; replace body with **§5.2** |
| 5 | **§5.3** (under heading) | Paste **§5.3** + table |
| 6 | **§5.4** (under heading) | Paste **§5.4** + table |
| 7 | **§5.5** | Optional — paste **§5.5** or leave stub |
| 8 | **§6** (under heading) | Paste **§6** (four subsections) |
| 9 | **§7** (under heading) | Paste **§7** + insert figure |
| 10 | **§8.1** (under heading) | Paste **§8.1 Limitations** |
| 11 | **§8.2** | Keep yours OR paste **§8.2** below if shorter preferred |
| 12 | Export PDF | Send to Maha |

**Figure for §7.1:**  
`Frozen-BN-Narrative-Evidence-2026-07-20/docs_FrozenBN/figures/tree_diagnosis_streamlit.png`  
(NOT `tree_diagnosis_fire.png`)

**Banned:** 93%, 81%, “BN improves predictive accuracy,” “audit” for category mapping.

---

## Abstract

Aviation accident investigations produce both coded event records and free-text narratives. Zhang and Mahadevan (2021) built a Bayesian network for diagnosis and prognosis from the coded National Transportation Safety Board database (1982–2006); whether that model is reproducible and whether narratives can supply query-time evidence without retraining remain open. We independently reproduce the published counting layer (85/85 Table 7 rows, 102 fire occurrences) and extend the network minimally so benchmark queries run. A leak-safe pipeline converts redacted narratives into hard and soft evidence on the frozen network. On 296 held-out accidents (2007–2019), the chain reaches 90.9% injury and 77.4% damage top-1 accuracy—statistically tying strong supervised text baselines on injury while leading on damage Macro-F1 (0.697). Narrative retrieval is the primary diagnosis readout (84.2% four-category top-1). Severity accuracy equals direct neighbor voting on all 296 accidents; the network mediates that signal and enables joint reasoning at zero severity-accuracy cost. Controlled experiments show large language models work as query interfaces and usable evidence extractors but fail as replacements for the coded database or the network itself. Narratives and language models supply evidence; the data-built network supplies auditable probabilistic reasoning.

---

## §2.1 Related work

Bayesian networks have long served as structured causal models for aviation safety. Ale et al. developed the Causal Model for Air Transport Safety (CATS), linking event sequence diagrams, fault trees, and human reliability models into a single Bayesian belief network for system-level risk quantification [5]. Ancel et al. applied object-oriented Bayesian networks to infer aviation accident shaping factors and causation from historical records, enabling sensitivity analysis and mitigation prioritization for in-flight loss-of-control accidents [6]. These models integrate expert structure with probabilistic inference over coded safety variables; they do not, however, consume free-text investigation narratives at query time.

A complementary line of work applies deep learning directly to National Transportation Safety Board (NTSB) accident narratives. Zhao, Yan, and Liu proposed hierarchical tree-based sequential event prediction from aviation accident reports, using encoder–decoder models with hierarchical event embeddings to forecast rare event sequences [7]. They also formulated aviation accident event extraction as multi-label classification against the NTSB event taxonomy [8]. In follow-on work, they used hierarchical multilabel classification with BERT to extract fine-level events from NTSB accident narratives [9]. These methods automate the coding task—mapping report text to official taxonomy labels—rather than injecting narrative-derived evidence into an existing data-built network for joint diagnosis and prognosis.

Our work differs from both families. Relative to large-scale BN safety models [5,6], we reproduce Zhang and Mahadevan's data-built airline-accident network [3] and extend it with narrative-derived evidence at inference time. Relative to NTSB text-mining methods [7–9], we do not replace the coded database with predicted labels; instead, we translate narratives into evidence on the network's own variables and read posterior probabilities for causes and severity outcomes. To our knowledge, no prior study combines free-text narratives, a faithfully reproduced published Bayesian network, and leak-safe held-out evaluation of both diagnosis and prognosis.

---

## §2.2 / §2.3 (optional stubs — one line each)

**2.2 Zhang's construction recipe.** Zhang's network construction (priors, edge ratios, Beta-CDF smoothing, parent cap) is summarized in Section 3; we implement it in `bn_build_ours.py` and validate against every published quantity.

**2.3 NTSB coded records.** The event-sequence and finding structure used by Zhang is described in Section 2.4 and the NTSB Aviation Coding Manual [26].

---

## §5.2 Evaluation protocol

The network and the retrieval index are built exclusively on the 1982–2006 window (1,742 accidents). The held-out set is every accident in the corpus outside that window (2007–2019) with a factual narrative: n = 296 for severity, of which 253 carry cause findings for diagnosis.

To prevent outcome leakage, severity-stating phrases ("was destroyed," "received fatal injuries") are removed from every narrative by a deterministic redaction pass before any embedding or parsing. All predictors, including deterministic and LLM parsers, receive only redacted text; narrative-stated severity is excluded as evidence. Redaction removes explicit outcome statements, not outcome predictability—mechanism wording (stall, turbulence, gear collapse) legitimately predicts severity. A window-trained TF-IDF probe confirms the redacted text retains mechanism signal (full vs redacted differ by 0.0 pp injury / 2.4 pp damage). A held-out ID audit shows 0/296 overlaps with the build window.

At prediction time, each held-out accident is processed from its narrative text only, without using that accident's coded injury, damage, or cause fields as inputs. Retrieval uses k = 100 neighbors (pre-set default); sensitivity analysis shows a flat plateau around this choice. No component of the primary pipeline is trained: network CPTs are frozen from 1982–2006, evidence strengths are measured as fractions of retrieved neighbors, and defaults are fixed before held-out scoring.

**Limitation, disclosed:** the held-out window was scored repeatedly during development (protocol fixes, ablation design), so reported accuracies may carry selection optimism. No model parameter was fitted to held-out data; a one-shot confirmatory run on a never-scored window (e.g., 2020–2024) is planned before journal submission.

---

## §5.3 Severity results

Table X reports 4-class top-1 accuracy and Macro-F1 for injury (fatal / serious / minor / none) and damage (destroyed / substantial / minor / none) on the identical 296-accident cohort. Five comparisons per target were designated primary and Holm–Bonferroni-corrected; other contrasts are exploratory.

| Predictor | Injury acc. | Injury M-F1 | Damage acc. | Damage M-F1 |
|---|---|---|---|---|
| Majority class (prior) | 58.4% | 0.184 | 42.6% | 0.149 |
| Parsed events only (hard+soft) | 82.4% | 0.422 | 50.7% | 0.309 |
| Supervised LR (parsed features) | 85.5% | 0.440 | 60.1% | 0.408 |
| Supervised LR (TF-IDF text) | 92.2% | 0.603 | 73.3% | 0.621 |
| Supervised LR (embedding) | 91.6% | 0.474 | 74.0% | 0.543 |
| **Narrative → BN (bn-sev)** | **90.9%** | **0.470** | **77.4%** | **0.697** |

The full chain improves on the network prior by +32.5 points on injury and +34.8 on damage (both Holm-adjusted p < 1e-4). Against the strongest supervised baselines (TF-IDF and embedding logistic regression, trained on 1,286 build-window narratives), the zero-parameter chain is statistically indistinguishable on injury (Holm p = 0.87 vs TF-IDF; p = 1.0 vs embedding) and damage (p = 0.29 both). TF-IDF is numerically best on injury (92.2%); the chain is best on damage accuracy and clearly best on damage Macro-F1 (0.697 vs 0.621 / 0.543).

Two ablations locate the contribution. Reading severity directly from the 100 nearest neighbors (retrieval-sev) gives the same accuracy as the full chain on all 296 accidents (0 discordant)—the network faithfully mediates neighbor-derived severity evidence rather than adding information. Fusing event evidence and neighbor severity in one inference (bn-fused) collapses accuracy to 38.5% / 41.9%, exposing double-counting when the same narrative is used twice. **The accuracy therefore comes from the narrative-retrieval signal; the network contributes joint conditioning, what-if queries, and per-node explanations at zero severity-accuracy cost.**

Rare classes remain weak: fatal injuries (3 cases) and minor injuries (16) are never ranked top-1 by any predictor.

---

## §5.4 Diagnosis results

NTSB replaced its coding taxonomy in 2008, so exact cause-code matching across our temporal split is impossible. We evaluate at CICTT's four top-level categories (Personnel, Aircraft, Environment, Organizational). Legacy-era findings are mapped by a documented keyword rule set (97.6% coverage); contested mappings touch at most ~2 accuracy points in the same direction for every predictor. A prediction is correct if its top-ranked category is in the truth set (n = 253).

| Predictor | Top-1 | 95% CI | MRR |
|---|---|---|---|
| Category frequency baseline | 45.8% | [39.5, 52.2] | 0.685 |
| Frozen BN, event evidence | 57.7% | [51.4, 63.6] | 0.759 |
| **Narrative retrieval (zero-parameter)** | **84.2%** | **[79.4, 88.5]** | **0.915** |
| Supervised LR (embedding) | 88.1% | [84.2, 91.7] | 0.936 |

Retrieval nearly doubles the frequency baseline (Holm-adjusted p < 1e-4) and is balanced across the three common categories (68% / 62% / 72% recall). The rare Organizational class (25 cases) is effectively unrecovered. The BN event path alone reaches 57.7%—significantly above baseline (Holm p = 0.0007) but 26.5 points below retrieval. Supervised embedding LR is 3.9 points above retrieval (p = 0.041, exploratory), the expected price of training. **Retrieval (84.2%) is the primary diagnosis readout.**

---

## §5.5 Fire-node cross-inference (optional — negative result)

We scored the network's posterior on the fire occurrence node from parsed narrative event evidence only (n = 296, same redaction as §5.3). **Verdict:** the BN does not carry usable fire signal. Best BN ROC AUC ≈ 0.39–0.46; retrieval neighbor fire rate ≈ 0.97; a fire-keyword regex ≈ 0.94. The claim that the frozen BN performs cross-node inference retrieval alone cannot is not supported. The surviving value is auditable evidence composition and what-if semantics—not held-out predictive lift on unobserved nodes like fire.

---

## §6 What can the LLM do?

We ran four controlled experiments to separate roles: interface, extractor, database replacement, and network replacement. All use gpt-4o-mini unless noted.

### 6.1 Query interface — works

On Zhang's eleven benchmark English sentences (Table 9 and gear-collapse scenarios), the LLM extracts the same network nodes as our deterministic parser in every case (11/11). Posteriors from LLM-parsed evidence match hand-set and deterministic-parser evidence cell-for-cell on the upgraded frozen network. The LLM therefore serves as a robust natural-language front end when vocabulary alignment is enforced; it does not supply the probabilities.

### 6.2 Evidence extractor on real narratives — usable, not better

On held-out narratives, LLM-extracted event evidence is usable but does not beat the deterministic parser plus retrieval on severity or diagnosis. LLM self-reported confidences are poorly calibrated (ECE ≈ 0.74) compared with retrieval-derived strengths (ECE ≈ 0.12). Failures are over-extraction and miscalibration, not failure to read the text; out-of-sample paraphrase tests are the appropriate robustness check.

### 6.3 Replacement for the coded database — fails

We asked the LLM to re-code 1,288 build-window narratives from text alone (occurrences, findings, injury, damage). Outcome fields partially survive (injury 81%, damage 61% with abstentions), but **causal attribution inverts**: for fire, Spearman correlation between coded and LLM P(cause | fire) is −0.43; top-10 cause overlap is 2/10. Narratives describe events; investigative Cause/Factor findings are not recoverable from prose alone.

### 6.4 Replacement for the network — fails

When asked directly for Zhang's published conditional probabilities (no network, no dataset), the LLM's answers have median order-of-magnitude error ~10× relative to Zhang and our BN. It returns plausible-sounding numbers without evidence propagation (e.g., flat or generic answers on cumulative gear evidence). **The LLM is the reader; the data-built network is the calculator.**

---

## §7 Worked examples

This section walks through two cases a reader can follow without code: a **diagnosis** example (upstream causes given a fire query) and a **prognosis** example (downstream readouts given engine-instrument evidence). Both use the frozen upgraded network (Section 4), the 1982–2006 build window, and the narrative-to-evidence interface (Section 5).

### 7.1 Diagnosis: fire during takeoff

**Input:** *Engine caught fire during takeoff.*

**Step 1 — Detect outcome.** Vocabulary matching maps the query to **fire** (Zhang's occurrence label).

**Step 2 — Rank upstream causes.** The diagnosis tree uses Zhang's Table 7 counting over all 102 fire accidents in 1982–2006: P(cause | fire) = n(c & fire) / N(fire), counting contributory Cause/Factor findings only. Labels that appear mostly *after* fire in the event sequence (evacuation, forced landing, loss of engine power as a downstream event) are excluded so the tree shows upstream causes, not prognosis consequences.

**Step 3 — Level-1 results (Table 7).**

| Cause (contributory factor) | P(c | fire) | Count |
|---|---|---|
| Airframe/component/system failure/malfunction | 0.314 | 32/102 |
| Electrical system, electric wiring | 0.088 | 9/102 |
| Fluid, fuel | 0.059 | 6/102 |
| Auxiliary power unit (APU) | 0.049 | 5/102 |

**Step 4 — Tree readout.** Figure X shows the branching diagnosis tree for this query. Level 1 matches Table 7; deeper levels condition on co-occurring causes within the fire cohort.

**[Insert Figure X: `tree_diagnosis_streamlit.png`]**

*Interpretation.* This example validates the diagnosis path against Zhang's published fire distribution. Held-out evaluation (Section 5.4) extends retrieval logic to 2007–2019 at four-category granularity.

### 7.2 Prognosis: inoperative engine instruments

**Input:** *Trouble with an engine instrument during the flight.* (Zhang's Table 9 flagship scenario.)

**Step 1 — Parse to hard evidence.** The parser matches **engine instrument** → clamped to Yes (c = 1.0).

**Step 2 — Forward inference on the frozen BN.**

| Target | Zhang (Table 9) | Ours (narrative) |
|---|---|---|
| P(Loss of engine power) | 0.950 | 0.950 |
| P(Forced landing) | 0.136 | 0.136 |
| P(Substantial aircraft damage) | 0.046 | 0.051 |
| P(Serious injury) | 0.062 | 0.076 |
| P(No injury) | 0.943 | 0.889 |

The anchor P(LOEP | inoperative engine instruments) = 0.95 matches exactly. Minor differences in low-probability injury/damage cells reflect multi-state severity encoding in the upgraded network.

*Interpretation.* This demonstrates narrative → evidence → BN → posterior on a case where Zhang published reference numbers. It is not a held-out prediction claim.

### 7.3 Scope

These examples do not replace the 296-accident held-out evaluation (Sections 5.3–5.4). An interactive Streamlit demo is in the repository; the figures above are the paper-facing artifacts.

---

## §8.1 Limitations

**Retrospective narratives.** Every narrative is written after the investigation closed. Defensible use cases are triage, coding assistance, and what-if analysis—not real-time forecasting.

**Severity accuracy is carried by retrieval, not added by the network.** On held-out severity prediction, the full Bayesian-network readout and direct k-nearest-neighbor voting agree on all 296 accidents. We do not claim predictive superiority over neighbor voting on severity.

**Cross-node inference does not beat retrieval on held-out fire.** Event-only evidence produces network fire posteriors with ROC AUC roughly 0.38–0.46; retrieval and keyword baselines reach roughly 0.94–0.98 on the same narratives.

**Rare classes are not reliably recovered.** The held-out set contains 3 fatal-injury, 16 minor-injury, and 25 organizational-cause accidents; no predictor reliably ranks these first.

**Embedding pretraining may include test-era text.** The TF-IDF baseline (92.2% injury / 73.3% damage) provides a pretraining-free reference.

**Incomplete posterior parity with Zhang.** After upgrades, 48/93 published quantities match exactly and 29 closely; 12 differ. We claim faithfulness to method and counting-layer quantities, not every posterior cell.

**Diagnosis only at four-category granularity.** The keyword rollup is a documented rule set with a sensitivity bound (≤2 pp), not independent validation by a second coder.

**Held-out reuse during development.** The 296-accident window was scored repeatedly during protocol design; confirmatory evaluation on 2020–2024 is planned before submission.

**LLM experiments.** Single model family (gpt-4o-mini), single domain, coded NTSB records as extraction reference.

---

## §8.2 Conclusion (optional replace)

We reproduced and froze Zhang and Mahadevan's airline-accident Bayesian network, extended it minimally so published benchmark queries run, and built a leak-safe pipeline that converts free-text NTSB narratives into evidence on the network's own variables without retraining. On 296 held-out accidents, the chain achieves 90.9% injury and 77.4% damage top-1 accuracy under explicit outcome redaction, ties strong supervised text baselines on injury, and leads on damage Macro-F1; narrative retrieval remains the primary diagnosis readout at 84.2% four-category top-1 accuracy. Controlled experiments show that language models serve effectively as query interfaces and usable evidence extractors but cannot replace either the coded investigative record or the network as a probability engine. **Narratives and language models supply evidence; the data-built network supplies auditable probabilistic reasoning.** We measured every link in that chain—and both swap directions fail when tested honestly.

---

## Email to Maha (when done)

Subject: **NTSB Paper — Draft v2 (Jul 30)**

Hi Maha,

Attached is **Draft v2** of the NTSB paper (`Master Draft NTSB Paper 7:30:26.docx`). The July 26 file is superseded.

**Complete for review:** §2.1, §2.4, §3–§4, §5.1–5.4, §6, §7 (with updated diagnosis tree), §8. §5.5 optional. Abstract included.

**Please focus feedback on:** (1) diagnosis vs prognosis in §7, (2) §5 honest framing (retrieval carries severity; network for reasoning), (3) overall story.

Numbers regenerate from `REPRODUCE.md` / `RESULTS_SECTION.md`.

Best,  
Kanu

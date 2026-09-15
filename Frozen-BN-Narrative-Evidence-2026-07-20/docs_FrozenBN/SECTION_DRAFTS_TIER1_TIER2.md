# Section drafts — Tier 1 & Tier 2 (paste into Word)

Working title context: *Narrative-driven evidence for Bayesian network diagnosis and prognosis in aviation safety assessment*

**Tonight target:** paste these sections, add figure/table placeholders, read aloud once.

---

## 2.1 Related work

Bayesian networks have long served as structured causal models for aviation safety. Ale et al. developed the Causal Model for Air Transport Safety (CATS), linking event sequence diagrams, fault trees, and human reliability models into a single Bayesian belief network for system-level risk quantification [5]. Ancel et al. applied object-oriented Bayesian networks to infer aviation accident shaping factors and causation from historical records, enabling sensitivity analysis and mitigation prioritization for in-flight loss-of-control accidents [6]. These models integrate expert structure with probabilistic inference over coded safety variables; they do not, however, consume free-text investigation narratives at query time.

A complementary line of work applies deep learning directly to National Transportation Safety Board (NTSB) accident narratives. Zhao, Yan, and Liu proposed hierarchical tree-based sequential event prediction from aviation accident reports, using encoder–decoder models with hierarchical event embeddings to forecast rare event sequences [7]. They also formulated aviation accident event extraction as multi-label classification against the NTSB event taxonomy [8]. In follow-on work, they used hierarchical multilabel classification with BERT to extract fine-level events from NTSB accident narratives [9]. These methods automate the coding task—mapping report text to official taxonomy labels—rather than injecting narrative-derived evidence into an existing data-built network for joint diagnosis and prognosis.

Our work differs from both families. Relative to large-scale BN safety models [5,6], we reproduce Zhang and Mahadevan's data-built airline-accident network [3] and extend it with narrative-derived evidence at inference time. Relative to NTSB text-mining methods [7–9], we do not replace the coded database with predicted labels; instead, we translate narratives into evidence on the network's own variables and read posterior probabilities for causes and severity outcomes. To our knowledge, no prior study combines free-text narratives, a faithfully reproduced published Bayesian network, and leak-safe held-out evaluation of both diagnosis and prognosis.

---

## 2.4 Dataset and temporal split

We work from a merged NTSB corpus spanning commercial airline accidents from 1982 through 2019. The analysis window matches Zhang and Mahadevan [3]: **1,742 accidents from 1982–2006** form the build set from which network structure, conditional probability tables, and the narrative retrieval index are constructed. Of these, **1,288 accidents carry usable factual narratives** (the subset used for embedding-based retrieval and for supervised baseline training where applicable).

All out-of-sample evaluation uses accidents **outside the build window**. The held-out set comprises **296 accidents from 2007–2019** with factual narratives available; **253** of these carry coded cause findings usable for diagnosis evaluation. A held-out identifier audit confirms **zero overlap** between build and evaluation accidents. The temporal split is deliberate: the network parameters are frozen after 2006, and no component of the primary narrative pipeline is fitted on held-out years. Supervised comparison models (logistic regression on parsed features, TF-IDF, or embeddings) are trained only on build-window accidents with narratives and scored on the identical 296-accident cohort. Processed artifacts are versioned in the repository (`data/processed/refined_dataset_1982_2006.json`; full corpus manifest in `outputs/cohort_manifest.json`).

---

## 3. Faithful reproduction of Zhang's network

We first ask whether Zhang and Mahadevan's published methodology can be reproduced independently from the same NTSB coded records. Zhang's construction (Section 4 of [3]) proceeds in stages: empirical priors from occurrence counts (Eq. 6), parent selection by edge-strength ratios (Eq. 8), Beta-CDF smoothing of sparse conditional probabilities (Eqs. 10–14), and a cap of twelve parents per node. We implement this recipe in `code/bn_build_ours.py` and validate it against every quantity the original paper reports.

At the counting layer—before full network inference—we recover Zhang's published anchors exactly. Total performed flights 1982–2006 sum to **184,517,128** (Table 6). Fire occurrences in the window total **102**, yielding the prior **P(fire) = 102/184,517,128 = 5.53×10⁻⁷** (Eq. 6). All **85 rows** of Table 7 (contributory factors to fire, conditional on fire occurrence) match the published values within tolerance ±0.0005 (**85/85 exact**; full row-by-row verification in supplementary documentation). Beta-CDF smoothing parameters (α ≈ 1.046, β ≈ 2.026) and spot-checked transition probabilities in Table 9 also agree.

Full Bayesian-network inference introduces additional variance. On a 93-item scoreboard comparing our first-pass faithful build to Zhang's published posteriors and worked examples, we obtain **43 exact matches, 16 close, 26 differ, and 8 qualitative-only comparisons** before any methodological extension. The dominant source of residual disagreement is not data error but **stochastic parent selection**: Zhang's edge-ratio rule breaks ties at random when multiple candidate parents have equal strength, so his published numbers represent one realization of a legitimate construction procedure. Repeating the build under fixed random seeds produces a variance envelope (`outputs/bn_variance_envelope.json`) that contains many of the published values; Zhang's own released network file (`NTSB.xdsl`) likewise fails to reproduce every cell of his Table 8 exactly. We therefore claim faithfulness to his **method and data-derived quantities** (Table 7, fire counts, prior formula), not cell-for-cell identity of every posterior under a single random draw.

An internal-consistency check on Zhang's tutorial figures (Tables 3–5, Figure 3) confirms that those values are illustrative rather than recomputed from the 1982–2006 window—a clarification relevant when interpreting minor discrepancies in worked examples versus counting-layer reproduction.

**[Figure placeholder: 93-item scoreboard, faithful build vs Zhang — before upgrades]**

---

## 4. Two methodological upgrades

A strictly faithful first-pass build cannot run several queries Zhang demonstrates. Two targeted extensions recover published behavior without altering the core data or counting rules.

**Person-finding nodes.** Zhang's Figure 12 (pilot-error scenario) requires evidence on person nodes such as "Pilot-in-command." The original occurrence–finding structure links mechanical and procedural findings to events but does not expose person entities as first-class evidence nodes. We add person nodes and connect findings to the responsible party where NTSB coding identifies one (`code/bn_upgraded.py`). This unlocks pilot-centric queries that were previously non-runnable on our reproduction.

**Multi-state severity nodes.** Zhang's published examples include injury and aircraft-damage outcomes with four severity levels each. A binary collapse of these nodes produces impossible conditional probability cells (e.g., P(no injury) ≈ 0 under evidence that should permit serious injury). We restore four-state injury (fatal, serious, minor, none) and four-state damage (destroyed, substantial, minor, none) nodes with CPTs estimated from the 1982–2006 window, matching the granularity of Zhang's Table 9 prognostic readouts.

After both upgrades, the 93-item scoreboard improves to **48 exact, 29 close, 12 differ, and 4 qualitative** (`tests/bn_upgraded_full.py`). Each remaining gap is traced to tie-breaking variance, a changed estimand (empirical reachability versus full BN marginal), or a quantity Zhang's released model also fails to reproduce. Side-by-side probabilities for every recreatable Table 9 scenario—Zhang's values, our faithful build, our upgraded network with direct evidence, and the same network driven by a typed English sentence—appear in supplementary tables; narrative and direct-evidence posteriors agree when parsing lands on the same evidence nodes.

The **frozen operational network** used for all narrative experiments in Sections 5–6 is this upgraded build. Upgrades are fixed before any held-out evaluation; they are not tuned on 2007–2019 data.

**[Figure placeholder: scoreboard before/after upgrades]**

---

## 7. Demonstration system

To show that the pipeline is usable beyond batch evaluation, we provide an interactive demonstration built in Streamlit (`apps/frozenbn_streamlit_diagnosis_prognosis_demo.py`). The interface supports three modes aligned with the paper's claims:

1. **Diagnosis tree.** A user types or selects a natural-language query (e.g., "engine caught fire during takeoff"). The system retrieves similar 1982–2006 accidents, aggregates coded cause distributions, and displays a hierarchical diagnosis tree whose level-1 categories match Zhang's Table 7 counting layer exactly.

2. **Prognosis tree.** Given the same query, the system returns injury and aircraft-damage probability distributions derived from the frozen upgraded network, including multi-evidence conditioning when both event facts and severity-related context are available.

3. **Narrative pre-parsing.** Optional LLM-assisted parsing converts free text into structured evidence on named network nodes before inference, reproducing hand-set evidence on Zhang's benchmark queries.

The demo uses the same frozen network parameters and retrieval index as the held-out evaluation pipeline; it is evidence of deployability, not an additional performance claim. Launch instructions and dependencies are documented in the public repository (`REPRODUCE.md`).

**[Figure placeholder: screenshot — diagnosis tree + prognosis panel for one example query]**

---

## 8. Discussion

### 8.1 Limitations

We state limitations without hedging; each constrains what the results support.

1. **Retrospective narratives.** Every narrative is written after the investigation closed. Defensible use cases are triage of an existing report, coding assistance, and what-if analysis—not real-time prediction of an unfolding accident.

2. **The network adds no severity accuracy, by construction.** The primary severity predictor (`bn-sev`) and direct neighbor voting (`retrieval-sev`) agree on all 296 held-out accidents (0 discordant). The network faithfully mediates k-nearest-neighbor severity evidence rather than adding information to it. Severity accuracy figures reflect the narrative-retrieval signal; the network's contribution is joint probabilistic reasoning at zero accuracy cost. We do not claim predictive superiority over retrieval on severity.

3. **Cross-node inference does not beat retrieval on held-out fire.** A designed negative experiment scores the network's posterior on the fire occurrence node from parsed event evidence only. Bayesian-network ROC AUC reaches at most ~0.46 (below chance on one definition); retrieval neighbor fire rate reaches ~0.97; a single fire-keyword regex reaches ~0.94. The surviving claim for cross-node reasoning is auditable evidence composition and what-if semantics—not held-out predictive lift on unobserved nodes.

4. **Rare classes are not reliably recovered.** The held-out set contains 3 fatal-injury accidents, 16 minor-injury accidents, and 25 organizational-cause accidents; no predictor reliably ranks these rare classes first. We report per-class recall and full confusion matrices rather than hiding misses inside headline accuracy.

5. **Embedding pretraining may include test-era reports.** The embedding model's training corpus is undisclosed and may contain 2007–2019 NTSB text; redaction cannot reach inside a pretrained encoder. Comparisons remain internally fair because all embedding-based methods share the same channel, and the TF-IDF baseline (92.2% injury / 73.3% damage) provides a pretraining-free reference.

6. **Twelve of ninety-three published BN quantities still differ after upgrades.** Tie-breaking randomness in Zhang's construction, plus discrepancies between his published tables and his released model file, mean we claim faithfulness to method and counting-layer quantities—not every posterior cell.

7. **Diagnosis is evaluated at four-category granularity only.** NTSB's 2008 taxonomy change (legacy subject codes to CICTT) prevents exact code matching across our temporal split. We roll legacy and modern codes to CICTT's four top-level categories (Personnel, Aircraft, Environment, Organizational) via a documented keyword rule set. A sensitivity review bounds the mapping's influence at ≤2 accuracy points in the same direction for every predictor; this is a bound, not independent validation.

8. **Held-out reuse during development.** The 296-accident window was scored repeatedly while protocol fixes and ablations were designed. No model parameter was fitted to held-out data, but design decisions saw held-out results; reported accuracies may carry selection optimism. A one-shot confirmatory evaluation on a never-scored window (e.g., 2020–2024) is required before submission.

### 8.2 Conclusion

We set out to reproduce Zhang and Mahadevan's airline-accident Bayesian network and to ask whether free-text NTSB narratives can serve as query-time evidence for diagnosis and prognosis. We show that the published counting layer and methodology reproduce exactly at the data-derived quantities that anchor the model (85/85 Table 7 rows, 102 fire occurrences, prior formula), while full-network posteriors vary legitimately with tie-breaking randomness. Two minimal upgrades—person nodes and multi-state severity—make the reproduced network runnable on Zhang's own benchmark queries. A narrative-to-evidence bridge translates investigation text into hard and soft evidence on the network's variables; on 296 held-out accidents, this chain reaches 90.9% injury and 77.4% damage top-1 accuracy under a leak-safe protocol, statistically tying strong supervised text baselines on injury while leading on damage Macro-F1. Controlled experiments further show that large language models serve effectively as a query interface and usable evidence extractor but fail as replacements for either the coded database or the network itself. The architecture we advocate is therefore modular: narratives and language models supply evidence; the data-built network supplies auditable probabilistic reasoning. We measured every link—and both swap directions fail when tested honestly.

---

## Tonight checklist

| Done? | Section | Action |
|:-----:|---------|--------|
| ☐ | **§2.4** | Paste §2.4 above; verify 1,742 / 1,288 / 296 / 253 numbers |
| ☐ | **§7** | Paste §7; add one Streamlit screenshot |
| ☐ | **§8** | Paste §8; trim if Maha wants shorter limitations |
| ☐ | **§3** | Paste §3; insert scoreboard figure placeholder |
| ☐ | **§4** | Paste §4; insert before/after figure placeholder |
| ☐ | **§2.1** | Paste §2.1; confirm refs [5]–[9] match bibliography |

**Tomorrow:** §2.2–2.3 (Zhang recipe detail + NTSB background), §5 (narrative bridge + held-out table), §6 (LLM four experiments), §1 polish, abstract last.

**Do NOT use in these sections:** 93%/81% (pre-leakage), "BN improves predictive accuracy," "audit" for category mapping.

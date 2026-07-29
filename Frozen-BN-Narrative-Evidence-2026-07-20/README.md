# Frozen BN + Narrative Evidence (2026-07-20)

**Status:** ACCEPTED — main paper path  
**Created:** 2026-07-20  
**Modes:** Diagnosis **and** prognosis (injury + damage)  
**Readiness:** Phase C — repo paths + foundation scripts verified via `./scripts/reproduce_foundation.sh` (Jul 2026); full held-out eval requires `REPRODUCE.md` §2–2b

Reproduce Zhang & Mahadevan (2021) BN from coded NTSB 1982–2006, **freeze**
it, then at query time map free-text narratives to evidence on the network
(no retraining). The **operational frozen network** is the upgraded build in
`code/bn_upgraded.py` (Zhang §4 recipe + person→finding edges + multi-state
severity nodes); the baseline-only scoreboard (43/16/26/8 on 93 published
numbers) documents pre-upgrade reproduction — see `outputs/BN_COMPARISON_REPORT.md`.

## The architecture (one sentence)

Narrative → leak-safe redaction → (a) event evidence via deterministic parse +
retrieval soft facts, and (b) k-NN severity likelihoods from the 100 most
similar training accidents → **both enter the frozen BN** (Pearl virtual
evidence / Jeffrey conditioning) → posteriors for diagnosis (causes) and
prognosis (injury + damage).

**Primary severity predictor: `bn-sev`** — the k-NN severity distribution
enters the frozen network as virtual evidence on the multi-state severity
nodes (per-target Jeffrey conditioning) and the posterior is read out of the
BN. Verified lossless: the BN-mediated posterior reproduces the k-NN evidence
distribution exactly (self-test in the eval, 0/296 discordant vs `retrieval-sev`).

**What the BN does and does not contribute (stated plainly):** the severity
*accuracy* belongs to the k-NN retrieval signal — `bn-sev` and
`retrieval-sev` are identical by construction, so the BN adds **zero
predictive lift** on this task. What BN mediation buys is (a) a single
coherent probabilistic model where narrative evidence, event evidence, and
what-if interventions compose (joint inference the raw k-NN readout cannot
do), (b) auditability — every posterior is explainable by which evidence
moved which node — and (c) zero accuracy cost for those capabilities
(`bn-fused` shows naive fusion *loses* accuracy, which is itself a finding
about violated conditional independence). Claims of BN predictive superiority
over retrieval would be false and are made nowhere.

## Key results (296 held-out, 2007–2019; leak-safe)

All baselines run on the IDENTICAL 296-accident cohort (same narrative
filter, truncation, and redaction; `outputs/cohort_manifest.json` lists the
IDs). Pairwise p-values are Holm-Bonferroni-corrected within the five
designated primary comparisons per target
(`outputs/heldout_significance.md`).

| Predictor | Injury top-1 | Damage top-1 | Notes |
|-----------|--------|--------|-------|
| majority class / BN prior | 58.4% | 42.6% | baseline |
| soft-priority (BN event path) | 89.9% | 55.4% | diagnosis evidence only |
| **bn-sev (primary)** | **90.9%** | **77.4%** | k-NN severity through frozen BN |
| LR on parsed features (supervised) | 85.5% | 60.1% | bn-sev better: Holm p=0.006 / p<0.0001 |
| LR on narrative embedding (supervised) | 91.6% | 74.0% | vs bn-sev n.s.: Holm p=1.0 / p=0.29 |
| LR on TF-IDF text (supervised) | 92.2% | 73.3% | vs bn-sev n.s.: Holm p=0.87 / p=0.29 |
| bn-fused (event + severity evidence) | 38.5% | 41.9% | **negative ablation** — same-narrative double counting |

The TF-IDF row is the leak-probe model promoted to a first-class baseline
(`tests/tfidf_lr_baseline_heldout.py`, probe configuration unchanged): it
is numerically the best injury predictor (statistical tie, 8 discordant
accidents) and below bn-sev on damage. bn-sev keeps the best damage
Macro-F1 (0.697 vs 0.621) and damage severe-screen sensitivity (75.3% vs
58.0%), and requires zero supervised training. The honest headline is that
supervised text models and the zero-parameter chain sit in one statistical
tie on accuracy; the chain's contribution is the frozen-BN reasoning layer,
not accuracy dominance.

Binary severe-outcome screening (bn-sev): severe injury sensitivity 93.5% /
specificity 96.8%; severe damage 75.3% / 89.8%. Honest miss, disclosed: of
the 3 fatal accidents, only 1 is flagged severe on injury (predicted SERS);
the other 2 are predicted NONE on injury, though both are flagged severe on
damage (DEST/SUBS). Per-class recall and full confusion matrices in
`outputs/heldout_significance.md`.

LLM parsing tiers under the leak-safe protocol (`--llm gpt-4.1`, full redaction,
outputs `*_llm.json`, rerun 2026-07-29): llm-tier **68.2% / 51.7%**, llm-first
**66.9% / 57.8%** — both far below bn-sev (90.9% / 77.4%), confirming the LLM is a
text reader, not the predictive signal. Tier usage on 296 narratives: 176 deterministic,
120 LLM fallback, 0 hard failures; redaction leaves 0 stated-severity detections.

**Fire-node cross-inference (negative):** feeding only parsed event evidence and reading
P(fire) from the frozen BN scores ROC AUC ~0.38–0.46 (below chance) vs retrieval
~0.96–0.98 and a fire-word regex ~0.94 on the same held-out cohort — the BN does not
beat retrieval on held-out fire prediction. See `outputs/fire_node_cross_inference.md`.

## Diagnosis (cause-category, era-fair; 253 held-out)

NTSB switched coding taxonomies in 2008, so exact-code matching across the
split is impossible by design; both eras are rolled up to CICTT top-level
cause categories (97.6% of window C/F findings mapped by a published keyword
rule set, not a coded judgement exercise). The rollup's standing is a
**sensitivity bound, not a validation**: a 75-row stratified sample was reviewed
for internal consistency against CICTT conventions (2 rule defects found and
corrected, 5 genuine boundary rows kept), and the contested mass — 316/5,062
window findings, 6.2% — bounds the mapping's effect at **<= 2 pp on absolute
accuracy, identical in direction for every predictor**, so the reported ordering
is insensitive to mapping choices. Observed effect of the corrections: retrieval
+0.4 pp, nothing else moved. The review was an AI pass plus first-author check —
**not independent dual coding**, and not claimed as such
(`outputs/mapping_audit_summary.md`). Truth = the category set of the
accident's C/F findings; metric definitions (set-membership top-1 vs strict
per-category recall) are spelled out in `outputs/diagnosis_heldout_eval.md`.

| Predictor | Top-1 | MRR | Notes |
|-----------|-------|-----|-------|
| frequency baseline | 45.8% | 0.685 | always guesses Personnel |
| **narrative retrieval (primary, zero-parameter)** | **84.2%** | **0.915** | vs freq: Holm p<0.0001 |
| emb-LR (supervised, needs coded labels) | 88.1% | 0.936 | beats retrieval p=0.041 (exploratory) |
| BN event path (posterior) | 57.7% | 0.759 | beats freq (Holm p=0.0007) |
| BN event path (lift) | 50.2% | 0.723 | negative result: max-lift is noisy |

Narratives carry strong diagnostic signal, and **retrieval (84.2%) is the
primary readout** — not the BN. Supervised emb-LR is 3.9 pp better (88.1%,
p = 0.041, exploratory), which is the disclosed price of needing zero labels.
The BN event path (57.7%) is reported as a structured-inference result *and* a
partial negative result: it clears the frequency baseline by 11.9 pp
(Holm p = 0.0007), which is real evidence that parsed narrative facts propagate
to cause nodes nothing pointed at, but it trails retrieval by 26.5 pp and is
not a competitive predictor. Retrieval is balanced across
Personnel/Aircraft/Environment (68/62/72%); the rare Organizational class is
effectively unrecoverable — retrieval, freq, and bn-lift never rank it first
(0/25) and the BN event path does so exactly once (1/25).
Full reports: `outputs/diagnosis_heldout_eval.md`,
`outputs/diagnosis_emb_lr.md`.

## Leakage protocol (Jesse's audit, all measured)

1. **Train/test:** 0 of 296 held-out IDs in the BN window or embedding index
   (`tests/heldout_leak_audit.py`).
2. **Outcome-in-text:** stated injury/damage phrases, death/medical wording,
   AND NTSB full-report boilerplate (which marks fatal investigations) are
   stripped before **any use of the text** — embedding, retrieval, AND the
   deterministic/LLM parsers all receive only the redacted narrative
   (`query_to_bn.redact_severity_phrases`; verified: rerun under full
   redaction left every headline number unchanged).
3. **Residual leak probe, two-tier** (`tests/redaction_leak_probe.py`):
   (a) a TF-IDF+LR probe trained ONLY on 1982–2006 window narratives and
   scored once on the held-out set — full vs redacted text differ by 0.0 pp
   (injury) / 2.4 pp (damage); (b) an in-sample CV probe kept as a
   worst-case upper bound. Scope stated precisely: redaction removes
   **explicit outcome statements**, not outcome *predictability* — the
   redacted text still supports ~92% injury accuracy because crash-mechanism
   wording (turbulence, landing gear, tug, fuselage) legitimately predicts
   severity; the probe's top-weight tokens are checked to be mechanism
   words, not outcome words. That probe is itself reported as the tfidf-lr
   baseline above, so its strength is in the main table, not buried here.
4. Stated-severity readout exists only as an OFF-by-default ablation
   (`LEAK_SAFE_SEVERITY`).

## Free parameters / "what is trained?"

Nothing in the primary pipeline is fitted; see
`docs_FrozenBN/FREE_PARAMETERS.md` for the full inventory (frozen /
calibrated / selected / fitted) and `outputs/hyperparam_sensitivity.md` for
the internal-validation sweep (results stable across top_k 25–200; k=25
variant: 89.9% / 77.7%, see `outputs/heldout_significance_k25.md`).

## Limitations (canonical list: `docs_FrozenBN/RESULTS_SECTION.md` §5.6)

Seven, none of them fixable by better engineering: (1) narratives are
retrospective, so this is triage/coding-assist, not real-time prediction;
(2) the BN adds **no** severity accuracy by construction (`bn-sev` =
`retrieval-sev`, 0/296 discordant); (2b) cross-node inference does **not**
beat retrieval on held-out fire (BN ROC AUC ~0.38–0.46 vs retrieval
~0.96–0.98; §5.5); (3) rare classes are hopeless at this n
(3 fatal injuries, 16 minor, Organizational recovered 1/25 at best); (4) the
embedding model may
have been pretrained on post-2006 NTSB text — TF-IDF LR is the pretraining-free
reference and matches it; (5) 12 of Zhang's 93 published BN numbers still
differ, attributed to his randomized tie-breaking and the fact that his own
released `NTSB.xdsl` also fails his published Table 8; (6) diagnosis is scored
only at four-category granularity, forced by the 2008 taxonomy break, on a rule
set no independent coder has checked (bounded at ≤ 2 pp, not validated);
(7) the held-out window was scored repeatedly during development — **a one-shot
2020–2024 confirmatory run is a prerequisite for submission.**

## Start here

1. `REPRODUCE.md` — run commands  
2. `FILE_INDEX.md` — every important file  
3. `../PROJECT_MAP.md` — whole repo map  

## Depends on `shared/`

- `shared/code/main_app.py` — embeddings + retrieval  
- `shared/code/config.py` — paths  
- `shared/data/processed/` — embeddings + merged JSON  

## Rejected related work (other folders)

- `../LLM-Narrative-Parser-Experiments-2026-07-15/` — LLM-first parser  
- `../Structural-Mapping-Retrieval-2026-07-25/` — struct reranking  

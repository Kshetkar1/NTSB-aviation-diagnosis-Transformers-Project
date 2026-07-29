# Frozen BN + Narrative Evidence (2026-07-20)

**Status:** ACCEPTED — main paper path  
**Created:** 2026-07-20  
**Modes:** Diagnosis **and** prognosis (injury + damage)  
**Readiness:** Phase C — repo paths + `./scripts/reproduce_paper.sh` verified (Jul 2026)

Reproduce Zhang & Mahadevan (2021) BN from coded NTSB 1982–2006, **freeze**
it, then at query time map free-text narratives to evidence on the network
(no retraining).

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

LLM parsing tiers under the leak-safe protocol (`--llm gpt-4.1`, outputs
`*_llm.json`): tiered parser 68.9% / 51.4%, LLM-first 69.9% / 57.8% -- both
far below bn-sev, confirming the LLM is a text reader, not the predictive
signal. Tier usage on 296 narratives: 176 deterministic, 120 LLM fallback,
0 hard failures; redaction leaves 0 stated-severity detections. (Protocol
note: these ablation numbers were run before parser inputs were also
redacted; the deterministic-path rerun under full redaction left every
number unchanged, so the conclusion is unaffected. Rerun with `--llm --tag
_llm` to refresh under the current protocol; requires `OPENAI_API_KEY`.)

## Diagnosis (cause-category, era-fair; 253 held-out)

NTSB switched coding taxonomies in 2008, so exact-code matching across the
split is impossible by design; both eras are rolled up to CICTT top-level
cause categories (97.6% of window C/F findings mapped by auditable rules;
dual-coded audit adjudicated by the first author 2026-07-29: 68/75 rows ok,
2 overturned rows corrected in the rules, rerun moved retrieval +0.4 pp only,
`outputs/mapping_audit_summary.md`). Truth = the category set of the
accident's C/F findings; metric definitions (set-membership top-1 vs strict
per-category recall) are spelled out in `outputs/diagnosis_heldout_eval.md`.

| Predictor | Top-1 | MRR | Notes |
|-----------|-------|-----|-------|
| frequency baseline | 45.8% | 0.685 | always guesses Personnel |
| **narrative retrieval (primary, zero-parameter)** | **84.2%** | **0.915** | vs freq: Holm p<0.0001 |
| emb-LR (supervised, needs coded labels) | 88.1% | 0.936 | beats retrieval p=0.041 (exploratory) |
| BN event path (posterior) | 57.7% | 0.759 | beats freq (Holm p=0.0007) |
| BN event path (lift) | 50.2% | 0.723 | negative result: max-lift is noisy |

Narratives carry strong diagnostic signal: both readouts crush the
frequency baseline; a supervised readout adds ~4 pp over zero-parameter
retrieval. Retrieval is balanced across Personnel/Aircraft/Environment
(68/62/72%); no predictor catches the rare Organizational class (0/25).
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

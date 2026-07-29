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

## Key results (296 held-out, 2007–2019; leak-safe)

| Predictor | Injury top-1 | Damage top-1 | Notes |
|-----------|--------|--------|-------|
| majority class / BN prior | 58.4% | 42.6% | baseline |
| soft-priority (BN event path) | 89.9% | 55.4% | diagnosis evidence only |
| **bn-sev (primary)** | **90.9%** | **77.4%** | k-NN severity through frozen BN |
| LR on parsed features (supervised) | 87.8% | 64.2% | McNemar vs bn-sev p=0.02 / p<0.001 |
| LR on narrative embedding (supervised) | 91.6% | 74.0% | vs bn-sev p=0.50 / p=0.11 (n.s.) |
| bn-fused (event + severity evidence) | 38.5% | 41.9% | **negative ablation** — same-narrative double counting |

Binary severe-outcome screening (bn-sev): severe injury sensitivity 93.5% /
specificity 96.8%; severe damage 75.3% / 89.8%. All 3 fatal accidents are
flagged severe (at 4-class granularity they land on the adjacent SERS class;
per-class recall disclosed in `outputs/heldout_significance.md`).

## Diagnosis (cause-category, era-fair; 253 held-out)

NTSB switched coding taxonomies in 2008, so exact-code matching across the
split is impossible by design; both eras are rolled up to CICTT top-level
cause categories (98.3% of window C/F findings mapped by auditable rules).
Truth = the category set of the accident's C/F findings.

| Predictor | Top-1 | MRR | Notes |
|-----------|-------|-----|-------|
| frequency baseline | 45.8% | 0.685 | always guesses Personnel |
| **narrative retrieval (primary, zero-parameter)** | **83.8%** | **0.912** | vs freq: McNemar p<0.0001 |
| emb-LR (supervised, needs coded labels) | 88.1% | 0.936 | beats retrieval p=0.027 |
| BN event path (posterior) | 57.7% | 0.759 | beats freq (p=0.0004) |
| BN event path (lift) | 49.8% | 0.721 | negative result: max-lift is noisy |

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
   stripped before any embedding (`query_to_bn.redact_severity_phrases`).
3. **Residual leak probe:** TF-IDF diagnostic on redacted text
   (`tests/redaction_leak_probe.py`) — remaining predictive tokens are
   crash-mechanism words (turbulence, landing gear, postcrash fire), not
   outcome statements.
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

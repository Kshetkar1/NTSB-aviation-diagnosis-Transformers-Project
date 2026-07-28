# Frozen BN + Narrative Evidence (2026-07-20)

**Status:** ACCEPTED — main paper path  
**Created:** 2026-07-20  
**Modes:** Diagnosis **and** prognosis (injury + damage)  
**Readiness:** Phase C — repo paths + `./scripts/reproduce_paper.sh` verified (Jul 2026)

Reproduce Zhang & Mahadevan (2021) BN from coded NTSB 1982–2006, **freeze** it, then at query time map free-text narratives to hard / soft / stated evidence on existing BN nodes (no retraining).

## Key results (296 held-out, 2007–2019)

| Predictor | Injury | Damage |
|-----------|--------|--------|
| prior | ~58.4% | ~42.6% |
| full BN + narrative | ~88.5% | ~64.2% |
| LR (train on build set) | ~87.5% | ~64.5% |

See `outputs/heldout_significance.md`.

## Start here

1. `REPRODUCE.md` — run commands  
2. `FILE_INDEX.md` — every important file  
3. `../PROJECT_MAP.md` — whole repo map  

## Depends on `shared/`

- `shared/code/main_app.py` — soft evidence neighbors  
- `shared/code/config.py` — paths  
- `shared/data/processed/` — embeddings + merged JSON  

## Rejected related work (other folders)

- `../LLM-Narrative-Parser-Experiments-2026-07-15/` — LLM-first parser  
- `../Structural-Mapping-Retrieval-2026-07-25/` — struct reranking  

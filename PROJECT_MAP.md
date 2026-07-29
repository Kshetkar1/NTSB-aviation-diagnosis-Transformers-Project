# Project map — NTSB aviation research

**Phase C readiness (2026-07-28):** Root is clean; data in `shared/data/processed/`; Zhang foundation verified via `./scripts/reproduce_foundation.sh` (3 scripts; 13/13 PASS on `reproduce_all_examples.py`). Full paper eval (held-out, diagnosis, audits): `Frozen-BN-Narrative-Evidence-2026-07-20/REPRODUCE.md`.

**Start here for the current paper:**  
`Frozen-BN-Narrative-Evidence-2026-07-20/README.md`  
**Reproduce commands:** `Frozen-BN-Narrative-Evidence-2026-07-20/REPRODUCE.md`

## Approach folders

| Folder | Status | Notes |
|--------|--------|-------|
| `Frozen-BN-Narrative-Evidence-2026-07-20/` | **ACCEPTED** | Main paper; frozen Zhang BN + query-time narrative evidence |
| `LLM-Narrative-Parser-Experiments-2026-07-15/` | REJECTED | LLM-first parser (~69% injury); tier-2 only in current |
| `Structural-Mapping-Retrieval-2026-07-25/` | REJECTED (main) | A0 vs A2 struct rerank; appendix / worked examples |
| `Embedding-Similarity-Counting-Path-2026-03-25/` | SUPERSEDED | Cosine + k-means counting path |
| `Zhang-Replication-Foundation-2026-06-04/` | FOUNDATION | PySMILE runner + Zhang reference |
| `BN-CPT-Upgrade-Experiments-2026-06-19/` | EXPLORATORY | June empirical CPT lane |
| `shared/` | INFRASTRUCTURE | Data + `main_app`, `config`, `zhang_diagnosis` |
| `archive/` | DO NOT USE | Old bundles, legacy scratch tests |

## Shared dependencies (important)

The **current** approach imports from `shared/code/`:

- `main_app.py` — embedding retrieval (soft evidence neighbors at query time)
- `config.py` — paths; `REPO_ROOT` = repository root
- `zhang_diagnosis.py` — Zhang counting / diagnosis helpers

This is intentional: one copy, documented in `shared/code/README.md`.

## For AI / new readers

1. Read this file  
2. Read `Frozen-BN-Narrative-Evidence-2026-07-20/FILE_INDEX.md`  
3. Do **not** edit `archive/` or rejected folders for the main paper unless running ablations

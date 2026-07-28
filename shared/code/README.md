# shared/code

Infrastructure used by multiple approach folders.

| Module | Role | Used by |
|--------|------|---------|
| `config.py` | `REPO_ROOT`, data paths, outputs | All approaches |
| `main_app.py` | Embedding retrieval + similarity diagnosis | Embedding path, struct mapping A0/A2, **Frozen-BN soft evidence** |
| `zhang_diagnosis.py` | Zhang counting / outcome detection | Frozen-BN, Embedding, Zhang replication |

**Do not duplicate these files** into approach folders. Import from `shared/code` (see root `pytest.ini` / `PYTHONPATH`).

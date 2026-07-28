# NTSB shift to new project approach (2026-03-25)

Use this document when creating the **new evaluation project**. It lists everything to copy from **NTSB_Shivy**, what to **omit**, what is **generated**, and one-shot shell commands.

**Important:** This file does **not** embed huge artifacts (`refined_dataset.json`, `embeddings.npy`, full `main_app.py`). Copy those with `cp`/`rsync` from paths below. Small reference copies of `config.py` and `requirements.txt` are included at the bottom for quick bootstrap.

---

## 1. Core application (copy as-is, then edit paths in `config.py`)

| Path | Notes |
|------|--------|
| `config.py` | **Edit** in new repo: `DATA_DIR`, train/test artifact paths, optional `RAW_DIR`. |
| `main_app.py` | Large; copy whole file. Add `exclude_ev_ids` to retrieval for eval if needed. |
| `streamlit_app.py` | Optional: UI demo only. |
| `run_app.sh` | Optional: if you use it to launch Streamlit. |

**Variants (copy only if you still need them)**

- `main_app_withModernBert.py`
- `streamlit_app_withModernBert.py`
- `streamlit_app_1_31_26.py`

---

## 2. Preprocessing pipeline (copy all four)

| Path |
|------|
| `data/preprocessing/01_create_refined_dataset.py` |
| `data/preprocessing/2_generate_embeddings.py` |
| `data/preprocessing/2b_precompute_diagnostic_data.py` |
| `data/preprocessing/3_precompute_clusters.py` |

**Note:** Align raw file locations with `config` (`01` currently reads from `DATA_DIR` in this repo; you also have `data/raw/` with the xlsx/txt sources—point one consistent layout in the new project).

---

## 3. Dependencies

| Path |
|------|
| `requirements.txt` |

Create `.env` in the new project (do **not** commit secrets); set at least `OPENAI_API_KEY`. If you add `.env.example` later, copy the variable names only.

---

## 4. Raw NTSB inputs (copy or symlink)

These live under **`data/raw/`** in this repo:

- `aircraft.xlsx`
- `engines.xlsx`
- `events.xlsx`
- `findings.xlsx`
- `injury.xlsx`
- `narratives.xlsx`
- `Events_Sequence.txt`
- `Occurrences.txt`
- `seq_of_events.txt`
- `ct_seqevt.txt`

---

## 5. Generated / large artifacts (do not paste into this doc—copy with disk)

**Typical locations:** `data/processed/`

| File | Purpose |
|------|---------|
| `refined_dataset.json` | Merged incident records |
| `embeddings.npy` | Vector index |
| `embeddings_map.json` | Row alignment + `diagnostic_data` |
| `embeddings_map.json.backup` | Optional backup |
| `cause_statistics.json` | If produced by `2b` |
| `embeddings_checkpoint.npz` | Optional resume checkpoint for embeddings |
| `modernbert/` or `openai/` subfolders | If you use alternate embedding outputs |

For **strict no-leakage eval**, you will **regenerate** train-only versions (e.g. `refined_dataset_train.json`, `embeddings_train.npy`, `embeddings_map_train.json`) in the new repo—not copy the full-corpus files as your only index.

---

## 6. Optional: tests / diagnostics to copy

Only if useful for debugging (not required for a minimal eval repo):

- `tests/test_chain_rule_diagnosis.py`
- `tests/06_transition_probability_investigation.py`

---

## 7. Do **not** copy (unless you need them)

- `docs/` (large; eval repo can link back)
- `experiments/`
- `travis.ipynb`
- Duplicate PDFs under `docs/`
- `.venv/` (recreate in new project)

---

## 8. Suggested new-repo folders to **create** (empty structure)

```
evaluation/splits/train_ev_ids.txt
evaluation/splits/test_ev_ids.txt
evaluation/scripts/evaluate_diagnosis.py   # new
evaluation/README.md                       # protocol: τ, Cause_Factor, query text
```

---

## 9. One-shot copy commands (run from **this repo root**)

From `NTSB_Shivy` parent directory, adjust `DEST` to your new project path:

```bash
DEST="/path/to/your/new/ntsb-eval-project"
mkdir -p "$DEST/data/preprocessing" "$DEST/data/raw" "$DEST/data/processed" "$DEST/evaluation/splits" "$DEST/evaluation/scripts"

# Core code
cp config.py main_app.py requirements.txt "$DEST/"
cp streamlit_app.py "$DEST/" 2>/dev/null || true

# Preprocessing
cp data/preprocessing/*.py "$DEST/data/preprocessing/"

# Raw data
cp data/raw/* "$DEST/data/raw/"

# Large processed artifacts (optional—prefer regenerating train-only in eval repo)
# cp data/processed/refined_dataset.json "$DEST/data/processed/"
# cp data/processed/embeddings.npy "$DEST/data/processed/"
# cp data/processed/embeddings_map.json "$DEST/data/processed/"
```

---

## 10. Inline reference copies (small files)

### `config.py` (snapshot from 2026-03-25 — verify against repo before relying)

```python
"""
Configuration file for the Aviation Incident Diagnosis Engine.

This file centralizes all configuration settings including API keys,
file paths, and model settings. API keys should be set via environment
variables for security.
"""

import os
from pathlib import Path

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, that's okay - user can use environment variables
    pass

# --- Project Root Directory ---
# Automatically detect project root (directory containing this config.py file)
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# --- OpenAI API Configuration ---
# Get API key from environment variable (recommended)
# Note: Validation happens when API is actually used, not on import
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_openai_api_key():
    """Get OpenAI API key with validation. Call this when actually using the API."""
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it using: export OPENAI_API_KEY='your-key-here' "
            "or create a .env file (see .env.example)"
        )
    return OPENAI_API_KEY

# --- Model Configuration ---
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # Default to gpt-4o-mini
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # Default embedding model

# --- Data File Paths ---
# All paths are relative to PROJECT_ROOT
REFINED_DATA_PATH = DATA_DIR / "refined_dataset.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
EMBEDDINGS_MAP_PATH = DATA_DIR / "embeddings_map.json"
CAUSE_STATS_PATH = DATA_DIR / "cause_statistics.json"

# --- Output Paths ---
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)  # Create output directory if it doesn't exist

# --- Validation ---
def validate_paths():
    """Validate that required data files exist."""
    required_files = [
        REFINED_DATA_PATH,
        EMBEDDINGS_PATH,
        EMBEDDINGS_MAP_PATH,
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Required data files not found: {[str(f) for f in missing_files]}. "
            f"Please run the data processing scripts first."
        )

# Optional: Validate paths on import (comment out if data files don't exist yet)
# validate_paths()
```

### `requirements.txt` (snapshot)

```
pandas
numpy
openai
tqdm
scikit-learn
streamlit
python-dotenv
```

---

## 11. Cursor bundle: `cursor 3/` (in this repo)

Your friend’s Cursor material is in **`cursor 3/`** at the project root (not inside `.cursor` yet). It includes:

| Path | Role |
|------|------|
| `cursor 3/index.mdc` | Global planning directive (`alwaysApply: true`) |
| `cursor 3/rules/planning_directive.mdc` | Same planning loop (audit → plan → critique → questions) |
| `cursor 3/rules/github_workflow.mdc` | Git/GitHub workflow |
| `cursor 3/commands/*.md` | Slash commands: `planning-workflow`, `check-plan`, `smart-commit`, etc. |
| `cursor 3/docs/cursor_planning_workflow.md` | Longer reference |

### Install into the **new** project (recommended layout)

Run from **this repo root** (`NTSB_Shivy`) so `cursor 3/` resolves:

```bash
cd /path/to/NTSB_Shivy   # parent of `cursor 3/`
NEW="/path/to/new/ntsb-eval-project"
mkdir -p "$NEW/.cursor/rules" "$NEW/.cursor/commands"

# Rules → Cursor’s rules folder
cp "cursor 3/rules/"*.mdc "$NEW/.cursor/rules/"

# Optional: also copy index if you want it as an extra rule (dedupe vs planning_directive.mdc)
# cp "cursor 3/index.mdc" "$NEW/.cursor/rules/shivam_index.mdc"

# Commands (Cursor CLI / editor picks these up from .cursor/commands)
cp "cursor 3/commands/"*.md "$NEW/.cursor/commands/"

# Optional reference doc in new repo
mkdir -p "$NEW/docs/cursor"
cp "cursor 3/docs/cursor_planning_workflow.md" "$NEW/docs/cursor/" 2>/dev/null || true
```

**Deduping:** `index.mdc` and `rules/planning_directive.mdc` are very similar—usually install **`rules/*.mdc`** only, and skip copying `index.mdc` twice unless you rename one.

### NTSB eval rule to add (new file in new repo)

Create **`$NEW/.cursor/rules/ntsb_eval_protocol.mdc`** with your evaluation contract, for example:

- Frozen **`test_ev_ids`**; no tuning τ on test.
- Model **input:** narrative text only; **never** pass `findings` into diagnosis.
- **Labels:** structured findings for scoring; match rule (embedding ≥ τ).
- **Artifacts:** train-only `embeddings` / `embeddings_map` for strict runs; **exclude-self** retrieval when the query is from the test incident.
- **Metrics:** Recall@K, MRR (document K and τ in `evaluation/README.md`).

### Your own Cursor **user** rules

If your global user rules say “run commands yourself / execute,” that can conflict with Shivam’s directive (“stop and wait for user answers” in §4). For **bugfixes and implementation**, the directive itself says to skip the long loop. For **plans**, you decide which rule wins; add a line in `ntsb_eval_protocol.mdc` if you want: *“When the user says implement or execute, skip wait-for-answers.”*

---

## 12. Checklist before you run eval in the new project

- [ ] `OPENAI_API_KEY` set
- [ ] Raw inputs present and paths in `config` / `01` match
- [ ] `train_ev_ids.txt` / `test_ev_ids.txt` created
- [ ] Train-only `embeddings` + map built (if strict eval)
- [ ] `find_top_matches` excludes test `ev_id` when scoring that incident
- [ ] τ and Cause_Factor rule for matching predictions → findings written in `evaluation/README.md`

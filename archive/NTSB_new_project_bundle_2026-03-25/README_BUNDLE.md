# NTSB new project bundle (2026-03-25)

This folder is a **portable copy** of the code, raw NTSB inputs, Cursor (`cursor_3`), and the shift checklist—ready to copy into a **new git repo** root.

## What is included

- `config.py`, `main_app.py`, `streamlit_app.py`, `requirements.txt`, `run_app.sh`
- `main_app_withModernBert.py`, `streamlit_app_withModernBert.py`, `streamlit_app_1_31_26.py` (if present in source)
- `data/preprocessing/*.py` (all four scripts)
- `data/raw/*` (all tabular / sequence sources from this project)
- `cursor_3/` (rules + commands; rename/install per `NTSB_shift_to_new_project_approach_2026-03-25.md` §11)
- `NTSB_shift_to_new_project_approach_2026-03-25.md`

Empty placeholders: `data/processed/`, `evaluation/splits/`, `evaluation/scripts/` (fill after copy).

## What is **not** included (too large or regenerate)

Copy from the original `NTSB_Shivy` project **after** you paste this bundle, **or** regenerate with preprocessing:

| File | Action |
|------|--------|
| `data/processed/refined_dataset.json` | `cp` from old repo **or** run `01_create_refined_dataset.py` (fix paths first) |
| `data/processed/embeddings.npy` | `cp` **or** run `2_generate_embeddings.py` |
| `data/processed/embeddings_map.json` | `cp` **or** from step 2 + `2b_precompute_diagnostic_data.py` |
| `data/processed/cause_statistics.json` | optional; from `2b` |
| `*.backup` files | optional |

### One-liner from old project (run on your machine)

```bash
OLD="/path/to/NTSB_Shivy"
NEW="/path/to/your/new/repo"   # after you copy this bundle there as the repo root
cp "$OLD/data/processed/refined_dataset.json" "$NEW/data/processed/" 2>/dev/null || true
cp "$OLD/data/processed/embeddings.npy" "$NEW/data/processed/" 2>/dev/null || true
cp "$OLD/data/processed/embeddings_map.json" "$NEW/data/processed/" 2>/dev/null || true
```

For **strict train-only eval**, do **not** copy full embeddings; build `refined_dataset_train.json` + train-only embeddings instead (see shift doc).

## Copy this bundle to a new repo

```bash
cp -R NTSB_new_project_bundle_2026-03-25/* /path/to/new/empty-repo/
cd /path/to/new/empty-repo
git init   # if new
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# add .env with OPENAI_API_KEY
```

Then install Cursor rules from `cursor_3/` into `.cursor/rules` and `.cursor/commands` (see §11 in the shift markdown).

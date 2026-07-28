# Track B — Preprocessing hooks (train artifacts)

## Goal

Let embedding and diagnostic steps run against **train-only merged** JSON and write **train-labeled outputs** (`*_train.*`) without breaking the existing **full-corpus** flow. Downstream **Track C** reads **train** embeddings + **augmented train map** to score test incidents (narrative-only query, findings labels, exclude-self).

> **Golden rule:** Preprocessing **`--train`** reads **`merged_dataset_train.json`** only — never the full `merged_dataset.json` for those outputs. **Track C / `main_app` strict mode** must also load **train** merged + **`embeddings_train.*`** (set **`NTSB_USE_TRAIN_INDEX=1`**). Using the full merge + full embeddings for strict metrics **invalidates** the protocol.

## Paths (canonical)

| Role | Full corpus | Train-only (strict eval) |
|------|-------------|---------------------------|
| Merged incidents | `config.MERGED_DATA_PATH` → `data/processed/merged_dataset.json` | `config.TRAIN_MERGED_DATA_PATH` → `merged_dataset_train.json` |
| Embeddings | `config.EMBEDDINGS_PATH` | `config.EMBEDDINGS_TRAIN_PATH` |
| Map | `config.EMBEDDINGS_MAP_PATH` | `config.EMBEDDINGS_MAP_TRAIN_PATH` |
| Checkpoint (resume) | `data/processed/embeddings_checkpoint.npz` | `config.EMBEDDINGS_CHECKPOINT_TRAIN_PATH` |
| Cause stats (`2b`) | `config.CAUSE_STATS_PATH` | `config.CAUSE_STATS_TRAIN_PATH` |

Train filenames match [`Project_Outline_For_Cursor.md`](../Project_Outline_For_Cursor.md) §7.

## Run order

1. **`2_generate_embeddings.py`** — builds `.npy` + **raw** map (no `diagnostic_data` yet). **Train:** run with `--train` **after** `merged_dataset_train.json` exists.
2. **`2b_precompute_diagnostic_data.py`** — reads merged + map, augments map entries. **Train:** `--train` expects `embeddings_map_train.json` from step 1.
3. **`3_precompute_clusters.py`** (optional) — clusters in embedding space, writes `cluster_label` **into the merged JSON** it loaded. **Train:** `--train` updates **`merged_dataset_train.json` only**, not the full merged file.

**Dictionary rows:** `2` still embeds **`ct_seqevt.txt`** dictionary lines in both modes (same as full run). Those rows are not test incidents; omitting them in train-only mode would be an optional cost-saving change—document if you ever do it.

## Scope (you own)

- `data/preprocessing/2_generate_embeddings.py` — **`--train`** uses `TRAIN_MERGED_DATA_PATH` and writes `EMBEDDINGS_TRAIN_PATH` / `EMBEDDINGS_MAP_TRAIN_PATH` / train checkpoint.
- `data/preprocessing/2b_precompute_diagnostic_data.py` — **`--train`** reads train merged + train map; writes `embeddings_map_train.json` and `cause_statistics_train.json` (no overwrite of full `embeddings_map.json`).
- `data/preprocessing/3_precompute_clusters.py` — **`--train`** reads train embeddings/map/merged; writes cluster labels only to **`merged_dataset_train.json`**.
- **`config.py`** — train path constants (see table above). No separate `config_eval.py` required unless you outgrow `config.py`.

## Dependency

- **`merged_dataset_train.json`** and split lists from **Track A**. If artifacts are missing, `2 --train` will fail at file open—fix Track A first.

## Do not touch (other tracks)

- `find_top_matches` / eval loop (**Track C**).
- **Streamlit** (**Track D**).

## Requirements

- **Default = full corpus:** no flags → `MERGED_DATA_PATH`, `embeddings.npy`, `embeddings_map.json`, full checkpoint path.
- **Train mode = explicit:** **`--train`** only (no inferring from file existence).
- **No clobber:** `--train` **never** writes `embeddings.npy` / `embeddings_map.json` / full `cause_statistics.json`.
- **Shape lock:** embedding row count equals `len(embeddings_map)` JSON list. Enforced by [`tests/test_train_preprocessing_artifacts.py`](../../tests/test_train_preprocessing_artifacts.py) when train files exist.
- **Extra outputs:** `2b` train mode also writes **`cause_statistics_train.json`**. List any new artifacts in `evaluation/README.md` if you add them.

## Example commands

From repo root (requires `OPENAI_API_KEY` for `2` and `3` naming step):

```bash
# Full corpus (production / demo index)
python data/preprocessing/2_generate_embeddings.py
python data/preprocessing/2b_precompute_diagnostic_data.py
python data/preprocessing/3_precompute_clusters.py

# Strict eval — train-only index (after Track A)
python data/preprocessing/2_generate_embeddings.py --train
python data/preprocessing/2b_precompute_diagnostic_data.py --train
python data/preprocessing/3_precompute_clusters.py --train
```

## After implementation — verify

### Automated

- **Module:** [`tests/test_train_preprocessing_artifacts.py`](../../tests/test_train_preprocessing_artifacts.py)

  ```bash
  pytest tests/test_train_preprocessing_artifacts.py -q -m "not integration"
  pytest tests/test_train_preprocessing_artifacts.py -q
  ```

### Manual

```bash
python data/preprocessing/2_generate_embeddings.py --help
python data/preprocessing/2b_precompute_diagnostic_data.py --help
python data/preprocessing/3_precompute_clusters.py --help
# After train pipeline run:
pytest tests/test_train_preprocessing_artifacts.py -q
```

## Definition of done

- [ ] Full pipeline **unchanged** without `--train` (smoke on a tiny test merge if needed).
- [ ] **`--train`** runbook works: `2` → `2b` → optional `3`; only `*_train.*` and train merged JSON touched.
- [ ] **Track C** can point `embeddings` / `map` to train paths (document in outline or eval README).
- [ ] **`pytest tests/test_train_preprocessing_artifacts.py`** passes (integration once train artifacts exist).

## End-of-chat report

- Flags/env vars added:
- Exact paths for train embeddings, map, `cause_statistics_train.json`, and optional `3` cluster labels:
- Example commands: **full** vs **train**:
- Pytest result:

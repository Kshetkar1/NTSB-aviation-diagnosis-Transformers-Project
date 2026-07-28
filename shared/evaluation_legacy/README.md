# Evaluation (strict train-only index)

For **leak-safe** scoring, the retrieval index must **not** contain test incidents. After Track A + Track B:

1. Build train-only embeddings: `python data/preprocessing/2_generate_embeddings.py --train` (then `2b --train`, optional `3 --train`).
2. Run the app or any script that imports `main_app` **with the train index**:

```bash
export NTSB_USE_TRAIN_INDEX=1
streamlit run streamlit_app.py
# or: python -c "import main_app; ..."
```

When `NTSB_USE_TRAIN_INDEX=1`, `main_app` loads:

- `data/processed/merged_dataset_train.json`
- `data/processed/embeddings_train.npy`
- `data/processed/embeddings_map_train.json`

Unset the variable or set it to `0` for the **full 2,243** corpus (demo / production-style index).

See `config.USE_TRAIN_INDEX`, `ACTIVE_INCIDENT_DATA_PATH`, and [`DOCS_NEW_FILE/agent_tasks/02_Track_Preprocessing_Hooks.md`](../DOCS_NEW_FILE/agent_tasks/02_Track_Preprocessing_Hooks.md).

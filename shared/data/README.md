# Shared processed data

Canonical location for NTSB incident JSON, embeddings, and train/window splits.

## Required for Frozen-BN reproduction

| File | Purpose |
|------|---------|
| `processed/refined_dataset.json` | Full corpus 1982–2019 |
| `processed/refined_dataset_1982_2006.json` | Zhang training window |
| `processed/embeddings*.npy` + `embeddings_map*.json` | Retrieval index |
| `processed/cause_statistics.json` | Cause counts |

Zhang BTS departures table and `NTSB.xdsl` live in
`Zhang-Replication-Foundation-2026-06-04/reference/`.

Set `PYTHONPATH=shared/code:Frozen-BN-Narrative-Evidence-2026-07-20/code` from repo root.

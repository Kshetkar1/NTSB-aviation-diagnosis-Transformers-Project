"""Persist a dedicated 1982-2006 retrieval index for Zhang comparisons.

Filters the (already corrected) embeddings.npy + embeddings_map.json down to the
1982-2006 incidents, writing:
  - embeddings_1982_2006.npy
  - embeddings_map_1982_2006.json
The window dataset (refined_dataset_1982_2006.json) already exists.
Vectors are reused unchanged; bayesian_data was refreshed in update_engine_dataset.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DP = Path(__file__).resolve().parents[1] / "data" / "processed"


def main() -> None:
    window = set(json.loads((DP / "refined_dataset_1982_2006.json").read_text()).keys())
    emb = np.load(DP / "embeddings.npy")
    emap = json.loads((DP / "embeddings_map.json").read_text())

    sel = [i for i, c in enumerate(emap) if c.get("ev_id") in window]
    new_emb = emb[sel]
    new_map = [emap[i] for i in sel]

    np.save(DP / "embeddings_1982_2006.npy", new_emb)
    (DP / "embeddings_map_1982_2006.json").write_text(json.dumps(new_map))
    print(f"window index written: {new_emb.shape[0]} vectors, {len(new_map)} chunks")
    # sanity: how many have non-empty causes
    nonempty = sum(1 for c in new_map if (c.get("bayesian_data") or {}).get("all_causes"))
    print(f"chunks with non-empty causes: {nonempty}/{len(new_map)}")


if __name__ == "__main__":
    main()

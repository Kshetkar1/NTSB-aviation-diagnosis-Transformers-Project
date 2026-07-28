#!/usr/bin/env python3
"""
Smoke test for query-aligned prognosis (predict_future_events_aligned).

Requires: data/processed/refined_dataset.json, embeddings.npy, embeddings_map.json, OPENAI_API_KEY.

Run from repo root:
  python tests/smoke_prognosis_transitions.py
  python tests/smoke_prognosis_transitions.py "bird strike on approach"
"""

from __future__ import annotations

import os
import sys

# Repo root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.pop("NTSB_USE_TRAIN_INDEX", None)


def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else "engine fire during takeoff"

    import main_app  # noqa: E402

    if not main_app.DATA_LOADED:
        print("FAIL: knowledge base not loaded (run preprocessing / check config paths).")
        return 1

    r = main_app.predict_future_events_aligned(query, top_n_incidents=20, max_chain_steps=5)
    if r.get("error"):
        print("ERROR:", r["error"])
        return 1

    print("Query:", repr(query))
    print("aligned_incidents:", len(r.get("aligned_incidents") or []))
    print("downstream_incident_count:", r.get("downstream_incident_count"))
    print("terminal_only_count:", r.get("terminal_only_count"))
    print("multi_step blocks:", len(r.get("multi_step") or []))
    for block in r.get("multi_step") or []:
        print(
            f"  Step {block['step']}: {len(block.get('distribution') or [])} outcomes, "
            f"weight={block.get('total_weight', 0):.4f}"
        )
    fe = r.get("future_events") or []
    if fe:
        print("top step-1 probability:", round(fe[0]["probability"], 4))
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

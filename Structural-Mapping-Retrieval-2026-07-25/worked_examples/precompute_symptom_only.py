"""
Stress test: run the same diagnosis + prognosis pipeline (A0 + A2) on
**symptom-only** queries that intentionally remove diagnosis-leading words
("fire", "engine", "gear", "failure", etc.). The intent is to show whether the
pipeline can still land on the right cluster family from leading indicators
alone -- and not just by keyword overlap with the cause taxonomy.

Reuses helpers from precompute_examples (same retrieval / clustering / LTP /
structural-rerank code path).

Outputs:
  data/example_symptom_engine_fire.json
  data/example_symptom_landing_gear.json
"""

from __future__ import annotations

import os
from pathlib import Path

# Held-out mode must be set BEFORE import main_app.
os.environ["NTSB_USE_TRAIN_INDEX"] = "1"

from precompute_examples import (  # noqa: E402
    OUT_DIR,
    _load_full_narrative_and_truth,
    build_code_embeddings,
    load_query_struct_jsonl,
    load_struct_jsonl,
    main_app,
    run_example,
)
from paths import DEFAULT_QUERY_CACHE_V2_PATH, DEFAULT_STRUCT_CACHE_V2_PATH  # noqa: E402

INCIDENTS = [
    {
        "ev_id": "20100114X11754",
        "slug": "engine_fire",
        "label": "ATR72 left-engine fire on takeoff (St. Croix, 2010) - symptom only",
        "symptom_query": (
            "We just lifted off. Strong yaw to the left, master warning sounding, "
            "I'm losing power on one side and I smell something burning. "
            "What am I looking at?"
        ),
    },
    {
        "ev_id": "20081116X33137",
        "slug": "landing_gear",
        "label": "DHC-8 nose-gear retracted landing (Philadelphia, 2008) - symptom only",
        "symptom_query": (
            "On final. Two greens, one red on my configuration panel. "
            "Tower can see my doors are open but can't see what should be "
            "visible below. Going around."
        ),
    },
]


def main() -> None:
    if not main_app.DATA_LOADED:
        raise RuntimeError("main_app failed to load data.")

    print(f"Mode: HELD-OUT (train-only index, {len(main_app.refined_dataset)} incidents)")

    struct_by_eid = load_struct_jsonl(DEFAULT_STRUCT_CACHE_V2_PATH)
    query_cache = load_query_struct_jsonl(DEFAULT_QUERY_CACHE_V2_PATH)
    print(f"Loaded struct cache: {len(struct_by_eid)} incidents, {len(query_cache)} cached queries")

    codes_list, labels_list, code_embs = build_code_embeddings()
    print(f"Loaded code embeddings ({len(codes_list)} codes)")

    for cfg in INCIDENTS:
        ev_id = cfg["ev_id"]
        slug = cfg["slug"]
        label = cfg["label"]
        symptom_query = cfg["symptom_query"]

        out_path = OUT_DIR / f"example_symptom_{slug}.json"

        print(f"\n{'#' * 80}")
        print(f"# Incident: {label}")
        print(f"# ev_id   : {ev_id}")
        print(f"# symptom : {symptom_query}")
        print(f"{'#' * 80}")

        _, truth = _load_full_narrative_and_truth(ev_id)

        run_example(
            scenario="symptom_only",
            diagnosis_query=symptom_query,
            prognosis_query=symptom_query,
            ev_id=ev_id,
            truth=truth,
            out_path=out_path,
            struct_by_eid=struct_by_eid,
            query_cache=query_cache,
            codes_list=codes_list,
            labels_list=labels_list,
            code_embs=code_embs,
        )

    print("\n" + "=" * 80)
    print("DONE - symptom-only stress test finished")
    print("=" * 80)


if __name__ == "__main__":
    main()

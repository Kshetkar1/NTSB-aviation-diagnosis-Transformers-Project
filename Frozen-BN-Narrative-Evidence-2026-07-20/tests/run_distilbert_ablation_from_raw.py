#!/usr/bin/env python3
"""Build a minimal incident JSON from shared/data/raw and run retrieval ablation.

Used when refined_dataset.json is not checked in. Produces the same severity
k-NN metric as embedding_model_ablation.py for OpenAI vs DistilBERT encoders.

  PYTHONPATH=shared/code:Frozen-BN-Narrative-Evidence-2026-07-20/code \
  python3 Frozen-BN-Narrative-Evidence-2026-07-20/tests/run_distilbert_ablation_from_raw.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "shared" / "data" / "raw"
PROC = REPO / "shared" / "data" / "processed"
FULL = PROC / "refined_dataset.json"
WINDOW = PROC / "refined_dataset_1982_2006.json"


def json_safe(obj):
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(x) for x in obj]
    if isinstance(obj, float) and pd.isna(obj):
        return None
    return obj


def year_from_ev_date(s) -> int | None:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    t = str(s).strip()
    if len(t) >= 4 and t[:4].isdigit():
        return int(t[:4])
    return None


def build_minimal_json() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    print("Loading raw tables ...")
    events = pd.read_excel(RAW / "events.xlsx", usecols=["ev_id", "ev_date"])
    aircraft = pd.read_excel(RAW / "aircraft.xlsx", usecols=["ev_id", "damage"])
    narratives = pd.read_excel(RAW / "narratives.xlsx", usecols=["ev_id", "narr_accf"])
    injury = pd.read_excel(RAW / "injury.xlsx")

    injury = injury.groupby("ev_id").apply(
        lambda g: g[["inj_person_category", "injury_level", "inj_person_count"]]
        .to_dict("records")
    ).reset_index(name="injuries")

    base = events.drop_duplicates("ev_id").merge(
        narratives.drop_duplicates("ev_id"), on="ev_id", how="inner"
    )
    base = base.merge(aircraft.drop_duplicates("ev_id"), on="ev_id", how="left")
    base = base.merge(injury, on="ev_id", how="left")
    base["injuries"] = base["injuries"].apply(lambda x: x if isinstance(x, list) else [])

    out: dict = {}
    for _, row in base.iterrows():
        ev = str(row["ev_id"])
        out[ev] = {
            "ev_date": row.get("ev_date"),
            "narr_accf": str(row.get("narr_accf") or ""),
            "damage": str(row.get("damage") or ""),
            "injuries": row["injuries"],
            "sequence_of_events": [],
            "findings": [],
        }

    out = json_safe(out)
    print(f"Writing {FULL} ({len(out)} incidents) ...")
    FULL.write_text(json.dumps(out))

    window = {}
    for ev, inc in out.items():
        y = year_from_ev_date(inc.get("ev_date"))
        if y is not None and 1982 <= y <= 2006:
            window[ev] = inc
    print(f"Writing {WINDOW} ({len(window)} incidents) ...")
    WINDOW.write_text(json.dumps(window))


def main() -> int:
    if not (RAW / "narratives.xlsx").is_file():
        print(f"Missing raw data under {RAW}")
        return 1

    if not FULL.is_file() or not WINDOW.is_file():
        build_minimal_json()

    ablation = Path(__file__).resolve().parent / "embedding_model_ablation.py"
    import os

    distil = "sentence-transformers/distilbert-base-nli-mean-tokens"
    if os.getenv("OPENAI_API_KEY"):
        models = f"text-embedding-3-small,{distil}"
    else:
        print("OPENAI_API_KEY not set — comparing DistilBERT vs shipped 3-small index only.")
        models = distil
    cmd = [sys.executable, str(ablation), "--models", models]
    if "--limit" in sys.argv:
        i = sys.argv.index("--limit")
        cmd.extend(["--limit", sys.argv[i + 1]])
    print("Running", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Step 1: Label all incidents (yes/no/unknown) for Table 4/5 variables."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cpt_config import OUTPUT_DIR, REFINED_DATA_PATH  # noqa: E402
from variables import label_incident  # noqa: E402


def main() -> None:
    with open(REFINED_DATA_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    rows = [label_incident(ev_id, inc) for ev_id, inc in dataset.items()]
    rows.sort(key=lambda r: r["ev_id"])

    out_csv = OUTPUT_DIR / "incident_labels.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "n_incidents": len(rows),
        "dataset_path": str(REFINED_DATA_PATH),
        "table4_relevant": sum(1 for r in rows if r.get("table4_relevant") == "1"),
        "brake_wear": dict(Counter(r["brake_wear"] for r in rows)),
        "electrical_overheat": dict(Counter(r["electrical_overheat"] for r in rows)),
        "fire": dict(Counter(r["fire"] for r in rows)),
        "x4_damage": dict(Counter(r["x4_damage"] for r in rows)),
    }
    out_json = OUTPUT_DIR / "label_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Labeled {len(rows)} incidents → {out_csv}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

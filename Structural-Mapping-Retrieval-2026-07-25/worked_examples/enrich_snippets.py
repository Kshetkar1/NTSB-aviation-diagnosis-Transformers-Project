"""
Add narrative snippets to retrieval rows in both a0.json and a2.json.

The precompute scripts leave the `snippet` field empty because the `text`
field on main_app match dicts is empty for incident rows. This script reads
the full refined dataset (data/processed/refined_dataset.json) and fills in
a short narrative snippet for each retrieval ev_id.
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
REFINED = PROJECT_ROOT / "data" / "processed" / "refined_dataset.json"

A0 = _HERE / "data" / "a0.json"
A2 = _HERE / "data" / "a2.json"


def _short(text: str, n: int = 180) -> str:
    return (text or "").strip().replace("\n", " ").replace("  ", " ")[:n]


def _snippet_for(rec: dict) -> str:
    for k in ("narr_accp", "narr_accf", "narrative_summary", "narr_cause"):
        v = rec.get(k)
        if v:
            return _short(v)
    return ""


def enrich(path: Path, refined: dict) -> None:
    data = json.loads(path.read_text())
    for row in data.get("retrieval", []):
        ev_id = row.get("ev_id")
        rec = refined.get(ev_id) or {}
        if not row.get("snippet"):
            row["snippet"] = _snippet_for(rec)
    path.write_text(json.dumps(data, indent=2))
    print(f"  Enriched {path.name}: {len(data.get('retrieval', []))} rows")


def main() -> None:
    refined = json.loads(REFINED.read_text())
    print(f"Loaded {len(refined)} incidents from refined_dataset.json")
    for p in (A0, A2):
        if p.exists():
            enrich(p, refined)
        else:
            print(f"  (skip {p.name}: not found)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build full refined_dataset.json + corrected 1982-2006 window for Frozen-BN.

Steps:
  1. 01_create_refined_dataset.py  -> merged + refined (all tables from raw/)
  2. rebuild_1982_2006.py          -> legacy Occurrences sequences on window
  3. add_findings_1982_2006.py     -> legacy seq_of_events findings on window
  4. Merge window sequence + findings back into full refined_dataset.json

Run from repo root:
  python3 shared/data/preprocessing/build_refined_complete.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PROC = REPO / "shared" / "data" / "processed"
FULL = PROC / "refined_dataset.json"
WINDOW = PROC / "refined_dataset_1982_2006.json"


def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=REPO)


def merge_window_into_full() -> None:
    full = json.loads(FULL.read_text(encoding="utf-8"))
    window = json.loads(WINDOW.read_text(encoding="utf-8"))
    merged = 0
    for ev_id, winc in window.items():
        if ev_id not in full:
            continue
        full[ev_id]["sequence_of_events"] = winc.get("sequence_of_events", [])
        full[ev_id]["findings"] = winc.get("findings", [])
        merged += 1
    FULL.write_text(json.dumps(full, indent=2), encoding="utf-8")
    print(f"merged window corrections into full refined: {merged} incidents")


def validate() -> None:
    full = json.loads(FULL.read_text(encoding="utf-8"))
    window = json.loads(WINDOW.read_text(encoding="utf-8"))

    def yr(inc):
        s = str(inc.get("ev_date") or "")[:4]
        return int(s) if s.isdigit() else None

    narr100 = sum(
        1 for v in full.values()
        if len(str(v.get("narr_accf") or "").strip()) >= 100
    )
    w_narr100 = sum(
        1 for v in window.values()
        if len(str(v.get("narr_accf") or "").strip()) >= 100
    )
    w_seq = sum(1 for v in window.values() if v.get("sequence_of_events"))
    w_find = sum(1 for v in window.values() if v.get("findings"))
    full_seq = sum(1 for v in full.values() if v.get("sequence_of_events"))
    full_find = sum(1 for v in full.values() if v.get("findings"))

    held = sum(
        1 for k, v in full.items()
        if k not in window and len(str(v.get("narr_accf") or "").strip()) >= 100
        and (y := yr(v)) is not None and y >= 2007
    )

    print("\n=== Validation ===")
    print(f"full refined incidents:     {len(full)}")
    print(f"window 1982-2006:         {len(window)}  (target ~1742)")
    print(f"narr_accf >= 100 (full):    {narr100}")
    print(f"narr_accf >= 100 (window):  {w_narr100}  (index target ~1363)")
    print(f"with sequence (window):     {w_seq}")
    print(f"with findings (window):     {w_find}")
    print(f"with sequence (full):       {full_seq}")
    print(f"with findings (full):       {full_find}")
    print(f"held-out narr>=100 2007+:   {held}  (eval target ~296)")


def main() -> int:
    py = sys.executable
    run([py, "shared/data/preprocessing/01_create_refined_dataset.py"])
    run([py, "Frozen-BN-Narrative-Evidence-2026-07-20/tests/rebuild_1982_2006.py"])
    run([py, "Frozen-BN-Narrative-Evidence-2026-07-20/tests/add_findings_1982_2006.py"])
    merge_window_into_full()
    validate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

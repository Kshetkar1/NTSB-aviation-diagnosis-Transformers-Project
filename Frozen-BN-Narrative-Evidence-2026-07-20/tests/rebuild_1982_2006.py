"""Phase 1 rebuild: 1982-2006 dataset with legacy event sequences restored.

- filter refined_dataset.json to 1982-2006 (Zhang's analysis window)
- repopulate each accident's `sequence_of_events` from data/raw/Occurrences.txt
  (Occurrence_Code -> meaning via metaData.xlsx)
- write data/processed/refined_dataset_1982_2006.json
- verify fire occurrences == 102
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
_SHARED = REPO_ROOT / "shared" / "code"
_FROZEN_CODE = FROZEN_DIR / "code"
for _p in (_SHARED, _FROZEN_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
ROOT = REPO_ROOT
SRC = ROOT / "shared" / "data" / "processed" / "refined_dataset.json"
OCC = ROOT / "data" / "raw" / "Occurrences.txt"
META = ROOT / "Zhang-Replication-Foundation-2026-06-04" / "reference" / "data" / "metaData.xlsx"
OUT = ROOT / "shared" / "data" / "processed" / "refined_dataset_1982_2006.json"

FIRE_MEANING = "Fire"


def year(v) -> int | None:
    s = str(v.get("ev_date") or "")[:4]
    return int(s) if s.isdigit() else None


def main() -> None:
    ref = json.loads(SRC.read_text(encoding="utf-8"))
    pre = {k: v for k, v in ref.items() if (y := year(v)) is not None and y <= 2006}
    print(f"refined_dataset: {len(ref)} total -> {len(pre)} in 1982-2006")

    md = pd.read_excel(META, dtype=str)
    md["clean"] = md["meaning"].astype(str).map(lambda s: re.sub("[^a-zA-Z]+", "", s))
    code2meaning = dict(zip(md["code_iaids"].astype(str), md["meaning"].astype(str)))
    fire_codes = set(md[md["clean"] == FIRE_MEANING]["code_iaids"].astype(str))

    occ = pd.read_csv(OCC, sep=",", dtype=str, on_bad_lines="skip")
    occ["Occurrence_Code"] = occ["Occurrence_Code"].astype(str)

    by_ev: dict[str, list] = defaultdict(list)
    for r in occ[occ["ev_id"].isin(pre)].itertuples(index=False):
        by_ev[r.ev_id].append(r)

    n_pop = 0
    for ev_id, inc in pre.items():
        rows = sorted(by_ev.get(ev_id, []), key=lambda r: int(str(r.Occurrence_No) or 0))
        seq = []
        for r in rows:
            code = str(r.Occurrence_Code)
            seq.append({
                "ev_id": ev_id,
                "Aircraft_Key": r.Aircraft_Key,
                "Occurrence_No": r.Occurrence_No,
                "Occurrence_Code": code,
                "Occurrence_Description": code2meaning.get(code, "Unknown"),
                "phase_no": r.Phase_of_Flight,
                "phase_description": code2meaning.get(str(r.Phase_of_Flight), ""),
                "eventsoe_no": None,
                "source": "legacy_occurrences",
            })
        inc["sequence_of_events"] = seq
        if seq:
            n_pop += 1

    OUT.write_text(json.dumps(pre, indent=2), encoding="utf-8")
    print(f"wrote {OUT.name}: {n_pop}/{len(pre)} accidents now have sequence_of_events")

    # verify fire == 102
    fire = {k for k, v in pre.items()
            if any(s["Occurrence_Code"] in fire_codes for s in v["sequence_of_events"])}
    print(f"\nFire occurrences in rebuilt dataset: {len(fire)}  (target 102)")
    print("MATCH" if len(fire) == 102 else "MISMATCH")


if __name__ == "__main__":
    main()

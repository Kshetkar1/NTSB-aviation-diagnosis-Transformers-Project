"""Phase 2 (write): add legacy findings (causes) to the 1982-2006 dataset.

Populates each accident's `findings` from seq_of_events.txt:
  Subj_Code -> meaning (the cause label used in Zhang's Table 7),
  plus modifier / person / cause_factor detail.
"""
from __future__ import annotations

import json
import re
import sys
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
RAW = ROOT / "shared" / "data" / "raw"
META = ROOT / "Zhang-Replication-Foundation-2026-06-04" / "reference" / "data" / "metaData.xlsx"
DS = ROOT / "shared" / "data" / "processed" / "refined_dataset_1982_2006.json"


def code_name(code, m):
    try:
        c = str(int(float(code)))
    except (ValueError, TypeError):
        return ""
    return m.get(c, "")


def main() -> None:
    md = pd.read_excel(META, dtype=str)
    code2mean = dict(zip(md["code_iaids"].astype(str), md["meaning"].astype(str)))

    seq = pd.read_csv(RAW / "seq_of_events.txt", sep="\t", dtype=str, on_bad_lines="skip")

    ds = json.loads(DS.read_text(encoding="utf-8"))
    ids = set(ds.keys())
    seq = seq[seq["ev_id"].isin(ids)]

    by_ev = defaultdict(list)
    for r in seq.itertuples(index=False):
        subj = code_name(r.Subj_Code, code2mean)
        if not subj:
            continue
        by_ev[r.ev_id].append({
            "ev_id": r.ev_id,
            "Occurrence_No": r.Occurrence_No,
            "seq_event_no": r.seq_event_no,
            "Subj_Code": r.Subj_Code,
            "finding_description": subj,
            "modifier_description": code_name(r.Modifier_Code, code2mean),
            "person_description": code_name(r.Person_Code, code2mean),
            "Cause_Factor": r.Cause_Factor,
            "source": "legacy_seq_of_events",
        })

    n = 0
    for ev_id, inc in ds.items():
        f = by_ev.get(ev_id, [])
        inc["findings"] = f
        if f:
            n += 1

    DS.write_text(json.dumps(ds, indent=2), encoding="utf-8")
    print(f"updated {DS.name}: {n}/{len(ds)} accidents now have findings (causes)")
    # quick integrity: how many fire accidents have airframe-malfunction finding
    fire_airframe = sum(
        1 for v in ds.values()
        if any(s.get("Occurrence_Code") == "171" for s in v.get("sequence_of_events", []))
        and any("Airframe/component/system failure" in str(fd.get("finding_description") or "")
                for fd in v.get("findings", []))
    )
    print(f"fire accidents with an airframe-malfunction finding: {fire_airframe}")


if __name__ == "__main__":
    main()

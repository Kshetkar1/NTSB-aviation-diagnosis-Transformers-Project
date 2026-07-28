"""Persist the corrected legacy data into the engine's live files.

1. Merge corrected pre-2007 sequence_of_events + findings into the FULL
   refined_dataset.json (post-2007 entries untouched).
2. Refresh embeddings_map.json: for pre-2007 chunks, rebuild bayesian_data from
   the corrected findings (subject-level causes) + keep the narrative cause.
   Vectors (embeddings.npy) are reused unchanged.

Run AFTER backing up refined_dataset.json and embeddings_map.json.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
_SHARED = REPO_ROOT / "shared" / "code"
_FROZEN_CODE = FROZEN_DIR / "code"
for _p in (_SHARED, _FROZEN_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
ROOT = REPO_ROOT
DP = ROOT / "shared" / "data" / "processed"
FULL = DP / "refined_dataset.json"
CORR = DP / "refined_dataset_1982_2006.json"
EMAP = DP / "embeddings_map.json"


def causes_from_findings(inc: dict):
    finding_strs, causal, narr = [], [], []
    for fd in inc.get("findings", []):
        s = (fd.get("finding_description") or "").strip()
        if not s:
            continue
        mod = (fd.get("modifier_description") or "").strip()
        label = f"{s}, {mod}" if mod else s
        finding_strs.append(label)
        if str(fd.get("Cause_Factor") or "").strip().upper().startswith("C"):
            causal.append(label)
    all_causes = []
    for c in finding_strs + narr:
        if c and c not in all_causes:
            all_causes.append(c)
    return finding_strs, (causal or finding_strs), all_causes


def main() -> None:
    full = json.loads(FULL.read_text(encoding="utf-8"))
    corr = json.loads(CORR.read_text(encoding="utf-8"))

    # 1) merge corrected pre-2007 entries into full dataset
    merged = 0
    for ev_id, cinc in corr.items():
        if ev_id in full:
            full[ev_id]["sequence_of_events"] = cinc.get("sequence_of_events", [])
            full[ev_id]["findings"] = cinc.get("findings", [])
            merged += 1
    FULL.write_text(json.dumps(full, indent=2), encoding="utf-8")
    print(f"merged corrected data into {merged} pre-2007 incidents of full dataset")

    # 2) refresh index bayesian_data for pre-2007 chunks
    emap = json.loads(EMAP.read_text(encoding="utf-8"))
    pre_ids = set(corr.keys())
    refreshed = 0
    for chunk in emap:
        ev_id = chunk.get("ev_id")
        if ev_id not in pre_ids:
            continue
        inc = full.get(ev_id, {})
        finding_strs, causal, all_causes = causes_from_findings(inc)
        old_bd = chunk.get("bayesian_data") or {}
        narr_causes = list(old_bd.get("narrative_causes", []))
        for c in narr_causes:
            if c and c not in all_causes:
                all_causes.append(c)
        bd = {
            "findings": finding_strs,
            "causal_findings": causal,
            "narrative_causes": narr_causes,
            "all_causes": all_causes,
            "narr_cause": old_bd.get("narr_cause"),
            "has_diagnostic_data": bool(all_causes),
        }
        chunk["bayesian_data"] = bd
        chunk["diagnostic_data"] = bd
        refreshed += 1
    EMAP.write_text(json.dumps(emap), encoding="utf-8")
    print(f"refreshed bayesian_data in {refreshed} pre-2007 index chunks")

    # integrity
    with_seq = sum(1 for v in full.values() if v.get("sequence_of_events"))
    with_find = sum(1 for v in full.values() if v.get("findings"))
    print(f"full dataset: {len(full)} incidents | with sequence_of_events={with_seq} | with findings={with_find}")


if __name__ == "__main__":
    main()

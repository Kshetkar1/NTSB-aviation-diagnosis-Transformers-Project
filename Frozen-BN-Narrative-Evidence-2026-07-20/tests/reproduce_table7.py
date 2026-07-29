"""Phase 2 (EARLY VARIANT): reproduce Zhang's Table 7 fire-cause contributions.

Zhang (buildOneGraphRep, pre-2006): an edge cause->Fire exists when
  (a) a subject (Subj_Code meaning) is attached to the fire occurrence
      (subject.Occurrence_No == fire occurrence's position), or
  (b) the occurrence immediately preceding Fire in the chain.
Table 7 value for a cause = (# fire-accidents with that cause->Fire edge) / 102.

NOTE: this script counts ALL findings on the fire occurrence and therefore
over-counts several causes relative to Zhang's published table (e.g. electric
wiring 0.1078 vs Zhang 0.0882) -- Zhang counts only contributory (Cause/Factor
flagged) findings. The paper-faithful reproduction (85/85 exact) is
`zhang_diagnosis.empirical_cause_distribution("fire", cause_factor_only=True)`,
documented in docs_FrozenBN/TABLE7_FULL_REPRODUCTION.md. This script is kept
as the historical first-pass counting check (denominator = 102 verified).
"""
from __future__ import annotations

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

# A few published Table 7 anchors (cause -> Zhang conditional prob)
ZHANG_T7 = {
    "Airframe/component/system failure/malfunction": 0.31372,
    "Electrical system, electric wiring": 0.08823,
    "Loss of engine power (total) - mechanical failure/malfunction": 0.08823,
    "Fluid, fuel": 0.05882,
    "Auxiliary power unit (APU)": 0.04901,
    "Maintenance, installation": 0.03921,
    "Brakes (normal)": 0.01960,
}


def name(code, code2mean):
    return code2mean.get(str(code), None)


def main() -> None:
    md = pd.read_excel(META, dtype=str)
    md["clean"] = md["meaning"].astype(str).map(lambda s: re.sub("[^a-zA-Z]+", "", s))
    code2mean = dict(zip(md["code_iaids"].astype(str), md["meaning"].astype(str)))
    fire_codes = set(md[md["clean"] == "Fire"]["code_iaids"].astype(str))

    ev = pd.read_excel(RAW / "events.xlsx", dtype=str)
    ev_year = dict(zip(ev["ev_id"], ev.get("ev_year", ev["ev_id"])))
    import json
    ref = json.loads((ROOT / "shared" / "data" / "processed" /
                      "refined_dataset.json").read_text())
    def yr(k):
        s = str(ref.get(k, {}).get("ev_date") or "")[:4]
        return int(s) if s.isdigit() else None
    pre_ids = {k for k in ref if (y := yr(k)) is not None and y <= 2006}

    occ = pd.read_csv(RAW / "Occurrences.txt", sep=",", dtype=str, on_bad_lines="skip")
    occ["Occurrence_No_i"] = pd.to_numeric(occ["Occurrence_No"], errors="coerce")
    occ = occ[occ["ev_id"].isin(pre_ids)]

    seq = pd.read_csv(RAW / "seq_of_events.txt", sep="\t", dtype=str, on_bad_lines="skip")
    seq = seq[seq["ev_id"].isin(pre_ids)]

    # group
    occ_by_ev = {k: g.sort_values("Occurrence_No_i") for k, g in occ.groupby("ev_id")}
    subj_by_ev = defaultdict(list)
    for r in seq.itertuples(index=False):
        try:
            sc = str(int(float(r.Subj_Code)))
        except (ValueError, TypeError):
            continue
        nm = name(sc, code2mean)
        if nm:
            subj_by_ev[r.ev_id].append((str(r.Occurrence_No), nm))

    fire_accidents = []
    cause_count = defaultdict(set)  # cause -> set(ev_id)

    for ev_id, g in occ_by_ev.items():
        codes = list(g["Occurrence_Code"].astype(str))
        nos = list(g["Occurrence_No"].astype(str))
        names = [name(c, code2mean) or "Unknown" for c in codes]
        fire_pos = [i for i, c in enumerate(codes) if c in fire_codes]
        if not fire_pos:
            continue
        fire_accidents.append(ev_id)
        for i in fire_pos:
            fire_occ_no = nos[i]
            # (a) subjects attached to the fire occurrence
            for s_occ_no, s_name in subj_by_ev.get(ev_id, []):
                if s_occ_no == fire_occ_no:
                    cause_count[s_name].add(ev_id)
            # (b) the occurrence immediately before fire in the chain
            if i >= 1:
                cause_count[names[i - 1]].add(ev_id)

    total = len(fire_accidents)
    print(f"Fire accidents (pre-2006): {total}  (target 102)\n")
    ranked = sorted(cause_count.items(), key=lambda kv: -len(kv[1]))
    print(f"{'cause':60} {'n':>4} {'P(fire|cause)':>13} {'Zhang':>8}")
    print("-" * 92)
    for cause, evs in ranked[:25]:
        n = len(evs)
        p = n / total
        z = ZHANG_T7.get(cause)
        zs = f"{z:.4f}" if z is not None else ""
        print(f"{cause[:60]:60} {n:>4} {p:>13.4f} {zs:>8}")

    print("\nAnchor check vs Zhang Table 7:")
    for cause, z in ZHANG_T7.items():
        n = len(cause_count.get(cause, set()))
        print(f"  {cause[:55]:55} ours={n/total:.4f}  zhang={z:.4f}")


if __name__ == "__main__":
    main()

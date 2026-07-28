"""Apply YOUR semantic smoothing to Zhang's Table 9 backward-inference cells.
For each evidence query: retrieve top-K similar incidents, then estimate
P(outcome) as the fraction of those incidents matching the outcome (reading
occurrence / damage / injury fields). Compare to Zhang's BN."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
_SHARED = REPO_ROOT / "shared" / "code"
_FROZEN_CODE = FROZEN_DIR / "code"
for _p in (_SHARED, _FROZEN_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
ROOT = REPO_ROOT
import main_app  # noqa: E402

OUTCOMES = [
    "loss of engine power", "forced landing",
    "substantial aircraft damage", "serious injury", "no injury",
]

ZHANG = {
    "Inoperative engine instruments": {
        "loss of engine power": 0.95, "forced landing": 0.1357,
        "substantial aircraft damage": 0.046, "serious injury": 0.0623, "no injury": 0.9431},
    "Combustion liner failure": {
        "loss of engine power": 0.50, "forced landing": 0.0714,
        "substantial aircraft damage": 0.00363, "serious injury": 0.000768, "no injury": 0.09978},
    "Improper oil usage": {
        "loss of engine power": 0.95, "forced landing": 0.1357,
        "substantial aircraft damage": 0.00609, "serious injury": 0.00146, "no injury": 0.09958},
}
QUERIES = {
    "Inoperative engine instruments":
        "The aircraft experienced inoperative engine instruments during flight. "
        "The engine instruments failed and were not providing accurate readings.",
    "Combustion liner failure":
        "The aircraft experienced a combustion liner failure during flight. "
        "The combustion assembly liner in the engine cracked and failed.",
    "Improper oil usage":
        "The aircraft had improper oil usage. The engine oil was of improper "
        "grade or contaminated, affecting engine performance.",
}


def match_outcome(inc, outcome):
    o = outcome.lower()
    occ = [(s.get("Occurrence_Description") or "").lower() for s in inc.get("sequence_of_events", [])]
    dmg = (inc.get("damage") or "").upper()
    inj = (inc.get("ev_highest_injury") or "").upper()
    if "loss of engine power" in o:
        return any("loss of engine power" in x for x in occ)
    if "forced landing" in o:
        return any("forced landing" in x for x in occ)
    if "substantial" in o:
        return dmg == "SUBS"
    if "destroyed" in o:
        return dmg == "DEST"
    if "minor aircraft damage" in o:
        return dmg == "MINR"
    if "serious injury" in o:
        return inj == "SERS"
    if "no injury" in o:
        return inj == "NONE"
    return False


def topk_incidents(query, k=50):
    q = main_app.get_embedding(query)
    _, matches = main_app.find_top_matches(q)
    seen, ev_ids = set(), []
    for m in matches:
        if m.get("source") != "incident":
            continue
        ev = m.get("ev_id")
        if ev and ev not in seen:
            seen.add(ev)
            ev_ids.append(ev)
        if len(ev_ids) >= k:
            break
    return ev_ids


def main() -> None:
    ds = main_app.refined_dataset
    K = 50
    for ev_label, query in QUERIES.items():
        ev_ids = topk_incidents(query, K)
        incs = [ds.get(e, {}) for e in ev_ids]
        print("=" * 78)
        print(f"EVIDENCE: {ev_label}   (K={len(incs)} semantic neighbors)")
        print("=" * 78)
        print(f"{'outcome':30} {'semantic':>14} {'Zhang BN':>10}")
        print("-" * 60)
        for oc in OUTCOMES:
            n = sum(1 for inc in incs if match_outcome(inc, oc))
            sem = n / len(incs) if incs else 0.0
            z = ZHANG[ev_label].get(oc, float("nan"))
            print(f"{oc:30} {sem:>7.3f}({n:>2}/{len(incs):>2}) {z:>10.4f}")
        print()


if __name__ == "__main__":
    main()

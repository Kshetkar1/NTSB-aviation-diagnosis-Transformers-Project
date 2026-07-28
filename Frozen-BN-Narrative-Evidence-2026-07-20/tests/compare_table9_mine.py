"""Show our engine's probabilities vs Zhang's Table 9, for the single-evidence
scenarios. Reports: our retrieval value, raw-count forward conditional, Zhang BN.
No Beta-CDF yet -- this is the "where do we stand" snapshot."""
from __future__ import annotations

import json
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
DP = ROOT / "shared" / "data" / "processed"

OUTCOMES = ["loss of engine power", "forced landing", "substantial aircraft damage", "no injury"]

# Zhang BN values (table9_engine_power.json, single-evidence columns)
ZHANG = {
    "Inoperative engine instruments": {
        "loss of engine power": 0.95, "forced landing": 0.1357,
        "substantial aircraft damage": 0.046, "no injury": 0.9431},
    "Combustion liner failure": {
        "loss of engine power": 0.50, "forced landing": 0.0714,
        "substantial aircraft damage": 0.00363, "no injury": 0.09978},
    "Improper oil usage": {
        "loss of engine power": 0.95, "forced landing": 0.1357,
        "substantial aircraft damage": 0.00609, "no injury": 0.09958},
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
# substrings to find the evidence cause node in the data (for raw counting)
EVIDENCE_FIND = {
    "Inoperative engine instruments": lambda f: "engine instrument" in f,
    "Combustion liner failure": lambda f: "combustion" in f,
    "Improper oil usage": lambda f: "oil" in f and ("grade" in f or "improper" in f or "contam" in f),
}


def best_match(items, key_label, key_prob, target):
    best = 0.0
    for it in items:
        if target in str(it.get(key_label, "")).lower():
            best = max(best, float(it.get(key_prob, 0.0)))
    return best


def raw_count(ds, ev_pred, outcome):
    def findings(v): return [(f.get("finding_description") or "").strip().lower() for f in v.get("findings", [])]
    def occs(v): return [(s.get("Occurrence_Description") or "").strip().lower() for s in v.get("sequence_of_events", [])]
    cause = [k for k, v in ds.items() if any(ev_pred(f) for f in findings(v))]
    if not cause:
        return 0.0, 0, 0
    co = sum(1 for k in cause if any(outcome in o for o in occs(ds[k])))
    return co / len(cause), co, len(cause)


def main() -> None:
    import main_app
    ds = main_app.refined_dataset

    for ev_label, query in QUERIES.items():
        print("=" * 92)
        print(f"EVIDENCE: {ev_label}")
        print("=" * 92)
        diag = main_app.diagnose_with_conditional_probabilities(query, top_n=400, top_n_incidents=50)
        try:
            prog = main_app.predict_future_events(query, top_n_incidents=50, max_chain_steps=1)
            prog_events = prog.get("future_events") or []
        except Exception:
            prog_events = []
        causes = diag.get("weighted_causes") or []

        print(f"{'outcome':32} {'ours(retr)':>11} {'raw-count':>12} {'Zhang BN':>9}")
        print("-" * 70)
        for oc in OUTCOMES:
            ours = max(
                best_match(causes, "cause", "probability", oc),
                best_match(prog_events, "event", "probability", oc),
            )
            rc, co, n = raw_count(ds, EVIDENCE_FIND[ev_label], oc)
            z = ZHANG[ev_label].get(oc, float("nan"))
            rc_s = f"{rc:.3f}({co}/{n})"
            print(f"{oc:32} {ours:>11.4f} {rc_s:>12} {z:>9.4f}")
        print()


if __name__ == "__main__":
    main()

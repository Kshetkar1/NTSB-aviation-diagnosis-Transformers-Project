#!/usr/bin/env python3
"""EXACT-FILTER DIAGNOSIS VALIDATION  (a.k.a. the live `diagnose_conditional`).

This mode answers: given an OUTCOME plus one or more KNOWN FACTS, hard-filter the
corpus to the accidents whose node-set contains the outcome AND every stated fact,
then rank causes by Zhang's count(cause & outcome)/count(outcome) over that
filtered cohort.  It is a deterministic, no-network, set-intersection query
(mode="global") -- distinct from the embedding-retrieval lane.

We do NOT rename the live function. This script:
  1. Runs several REAL outcome+condition combinations drawn from the data.
  2. Prints the filtered cohort size n and the resulting cause ranking.
  3. Demonstrates graceful degradation as the filter narrows toward tiny n.
  4. Emits a machine-readable JSON for the report.

Run (framework python 3.11, no network):
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/exact_filter_validation.py
"""
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
import zhang_diagnosis as zd  # noqa: E402

DATASET = ROOT / "shared" / "data" / "processed" / "refined_dataset_1982_2006.json"
OUT_JSON = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "docs_FrozenBN" / "exact_filter_validation.json"

# (outcome query, explicit condition phrases). Conditions are matched to dataset
# labels by zhang_diagnosis.match_cause, mirroring the live engine.
CASES = [
    ("fire", ["wiring"]),
    ("fire", ["fuel"]),
    ("fire", ["auxiliary power unit"]),
    ("fire", ["maintenance"]),
    ("loss of engine power", ["fuel"]),
    ("loss of engine power", ["carburetor"]),
    ("gear collapsed", ["landing gear"]),
    # progressively narrower -> should degrade to tiny n sensibly
    ("fire", ["electric wiring", "maintenance"]),
    ("fire", ["wiring", "fuel", "maintenance"]),
]


def run_case(ds, outcome_query, conditions):
    res = zd.diagnose_conditional(outcome_query, conditions=conditions,
                                  mode="global", top_n=8, dataset=ds)
    return res


def main():
    ds = json.loads(DATASET.read_text(encoding="utf-8"))
    records = []
    print("=" * 92)
    print("EXACT-FILTER DIAGNOSIS  --  outcome + known facts -> filtered cohort -> cause ranking")
    print("=" * 92)
    for outcome_query, conditions in CASES:
        res = run_case(ds, outcome_query, conditions)
        if res.get("error"):
            print(f"\n[{outcome_query!r} | {conditions}] ERROR: {res['error']}")
            records.append({"outcome_query": outcome_query, "conditions": conditions,
                            "error": res["error"]})
            continue
        n = res.get("eligible_with_outcome")
        matched = res.get("matched_conditions")
        unmatched = res.get("unmatched_conditions")
        print(f"\nOUTCOME={res['outcome']!r}  CONDITIONS={conditions}")
        print(f"  matched->labels: {matched}   unmatched: {unmatched}")
        print(f"  filtered cohort size n = {n}  (accidents with outcome AND all facts)")
        if not n:
            print("  -> empty cohort: degrades to 'no matching incidents' (safe).")
        else:
            top = res["causes"][:6]
            for c in top:
                print(f"     {c['probability']*100:5.1f}%  (n={c['n']:>2})  {c['cause'][:60]}")
        records.append({
            "outcome_query": outcome_query, "outcome": res.get("outcome"),
            "conditions": conditions, "matched_conditions": matched,
            "unmatched_conditions": unmatched, "cohort_n": n,
            "top_causes": [{"cause": c["cause"], "p": c["probability"], "n": c["n"]}
                           for c in res["causes"][:6]],
        })

    OUT_JSON.write_text(json.dumps(records, indent=2))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()

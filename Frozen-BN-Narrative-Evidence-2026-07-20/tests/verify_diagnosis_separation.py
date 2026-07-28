#!/usr/bin/env python3
"""Verify diagnosis tree excludes downstream response labels at level 1."""
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
import trees
import zhang_diagnosis as zd

DATASET = ROOT / "shared" / "data" / "processed" / "refined_dataset_1982_2006.json"


def main():
    ds = json.loads(DATASET.read_text(encoding="utf-8"))
    result = trees.build_diagnosis_tree(
        "engine caught fire during takeoff",
        dataset=ds, cause_factor_only=True, exclude_responses=True,
        full_population=True, branching=4, depth=2, min_prob=0.02,
    )
    root = result["tree"]
    assert root is not None, result["meta"]
    l1 = [c["label"].lower() for c in root["children"]]
    leaked = [lab for lab in l1 if lab in zd.DIAGNOSIS_RESPONSE_LABELS]
    assert not leaked, f"response labels in L1: {leaked}"
    assert "electrical system, electric wiring" in l1
    assert result["meta"]["outcome_count_in_pool"] == 102
    print("PASS: diagnosis L1 has upstream causes only (no evacuation/emergency procedure)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

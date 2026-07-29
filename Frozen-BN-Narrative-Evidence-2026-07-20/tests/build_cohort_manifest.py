#!/usr/bin/env python3
"""Write outputs/cohort_manifest.json: the exact accident IDs in every
evaluation cohort, so every "n = ..." in the paper is an auditable claim.

Cohorts:
  * severity      -- held-out accidents with narr_accf >= 100 chars
                     (frozenbn_heldout_narrative_bn_eval.py + lr_baseline)
  * diagnosis     -- severity cohort members that also have >= 1 mapped
                     C/F cause finding (diagnosis_heldout_eval.py)
Cross-checked against the per-item output files when they exist.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/build_cohort_manifest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
OUT = FROZEN_DIR / "outputs" / "cohort_manifest.json"

FULL = REPO_ROOT / "shared" / "data" / "processed" / "refined_dataset.json"
WINDOW = (REPO_ROOT / "shared" / "data" / "processed" /
          "refined_dataset_1982_2006.json")


def main() -> int:
    full = json.loads(FULL.read_text())
    window_ids = set(json.loads(WINDOW.read_text()).keys())

    severity = []
    for k, inc in sorted(full.items()):
        if k in window_ids:
            continue
        narr = str(inc.get("narr_accf") or "").strip()
        if len(narr) >= 100:
            severity.append(k)

    # cross-check against the eval's own per-item file
    checks = {}
    per = FROZEN_DIR / "outputs" / "heldout_per_item.json"
    if per.exists():
        ids = sorted(r["id"] for r in json.loads(per.read_text())["items"])
        checks["severity_matches_heldout_per_item"] = (ids == sorted(severity))
    diag = FROZEN_DIR / "outputs" / "diagnosis_heldout_eval.json"
    diagnosis = []
    if diag.exists():
        diagnosis = sorted(r["id"] for r in
                           json.loads(diag.read_text())["items"])
        checks["diagnosis_subset_of_severity"] = (
            set(diagnosis) <= set(severity))
    lr = FROZEN_DIR / "outputs" / "lr_per_item.json"
    if lr.exists():
        lr_ids = sorted(r["id"] for r in json.loads(lr.read_text())["items"])
        checks["lr_cohort_equals_severity"] = (lr_ids == sorted(severity))

    manifest = {
        "window_n": len(window_ids),
        "severity_cohort": {"n": len(severity), "ids": severity,
                            "filter": "not in window; narr_accf >= 100 chars"},
        "diagnosis_cohort": {"n": len(diagnosis), "ids": diagnosis,
                             "filter": "severity cohort AND >= 1 mapped "
                                       "C/F cause finding"},
        "window_overlap": sorted(set(severity) & window_ids),
        "cross_checks": checks,
    }
    OUT.write_text(json.dumps(manifest, indent=1))
    print(f"severity cohort n={len(severity)}  diagnosis n={len(diagnosis)}")
    print(f"window overlap: {len(manifest['window_overlap'])} (must be 0)")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print(f"wrote {OUT}")
    return 0 if not manifest["window_overlap"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

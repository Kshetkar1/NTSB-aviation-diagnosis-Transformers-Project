"""Derive the DEFAULT selective margin threshold from the validated experiment.

Reuses tests/loo_selective.eval_outcome (the harness that showed margin-gating
lift for fire / gear collapse at ~top-25% coverage) and reads off the margin
value at the top-25%-coverage operating point: the smallest margin still kept
when we commit on only the most-confident 25% of incidents (= 75th percentile of
the per-incident margin distribution).

Run (framework python, network for embeddings):
  python tests/derive_margin_threshold.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loo_selective import eval_outcome  # noqa: E402

COVERAGE = 0.25


def margin_at_coverage(records, coverage=COVERAGE):
    """Smallest margin still committed at the given top-coverage (descending)."""
    margins = sorted((r["margin"] for r in records), reverse=True)
    if not margins:
        return None
    m = max(1, round(len(margins) * coverage))
    return margins[m - 1]


def main():
    rows = {}
    for word in ("fire", "gear collapse"):
        name, records = eval_outcome(word)
        rows[name] = records
        thr = margin_at_coverage(records)
        print(f"{name:18} n={len(records):>3}  margin@top-25% = "
              f"{thr if thr is None else round(thr, 4)}")

    combined = [r for recs in rows.values() for r in recs]
    thr_all = margin_at_coverage(combined)
    print(f"{'fire+gear':18} n={len(combined):>3}  margin@top-25% = "
          f"{thr_all if thr_all is None else round(thr_all, 4)}")


if __name__ == "__main__":
    main()

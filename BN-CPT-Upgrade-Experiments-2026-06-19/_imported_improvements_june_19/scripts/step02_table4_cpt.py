#!/usr/bin/env python3
"""Step 2: Table 4 CPT — full dataset (all incidents, unfiltered)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from table4 import load_labels, run_table4  # noqa: E402


def main() -> None:
    rows = load_labels()
    cells, tier_a, _ = run_table4(
        rows,
        stem="table4_cpt_full",
        title="Table 4 comparison — FULL cohort (unfiltered)",
        method_note=(
            "**Method:** Count P(fire=yes) where brake wear and overheat labels match the cell. "
            "**Not a Bayesian network.**"
        ),
        cohort_note=(
            "**Cohort:** All 2,243 incidents. Parent `no` includes incidents that never mention "
            "brake/electrical (inflates both-no cell). See `table4_cpt_restricted` for Maha comparison."
        ),
    )
    print(f"Table 4 FULL → outputs/table4_cpt_full.csv")
    print(f"Tier A: {'PASS' if tier_a else 'FAIL'}")
    for c in cells:
        p = c["p_fire_yes"]
        p_s = f"{p:.4f}" if p == p else "n/a"
        print(f"  {c['cell_label']}: n={c['n_cohort']} P(fire)={p_s}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Step 5: Table 4 CPT — restricted cohort (Maha comparison)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohort import row_eligible_for_cell, row_in_restricted_pool  # noqa: E402
from table4 import load_labels, run_table4  # noqa: E402


def main() -> None:
    rows = load_labels()
    n_relevant = sum(1 for r in rows if row_in_restricted_pool(r))

    cells, tier_a, summary = run_table4(
        rows,
        stem="table4_cpt_restricted",
        title="Table 4 comparison — RESTRICTED cohort (Maha)",
        method_note=(
            "**Method:** Empirical CPT — P(fire=yes | brake wear × electrical overheat) "
            "from NTSB case counts. **Not a Bayesian network.**"
        ),
        cohort_note=(
            f"**Cohort:** {n_relevant} incidents that mention landing gear/brake, electrical/wiring, "
            "overheat, or fire. Parent **`no`** requires that domain to appear in text "
            "(both-no = gear/brake + electrical discussed, neither fault yes)."
        ),
        pool_filter=row_in_restricted_pool,
        row_filter=row_eligible_for_cell,
    )

    # Per-cell eligible counts in restricted pool (before fire outcome)
    diag = {"n_pool": n_relevant, "cells_in_pool": {}}
    pool = [r for r in rows if row_in_restricted_pool(r)]
    for bw, eo in [("yes", "yes"), ("yes", "no"), ("no", "yes"), ("no", "no")]:
        eligible = [r for r in pool if row_eligible_for_cell(r, bw, eo)]
        diag["cells_in_pool"][f"brake={bw}, overheat={eo}"] = {
            "n_eligible": len(eligible),
            "n_fire_yes": sum(1 for r in eligible if r["fire"] == "yes"),
        }
    from cpt_config import OUTPUT_DIR  # noqa: E402

    (OUTPUT_DIR / "table4_restricted_cohort_diag.json").write_text(
        __import__("json").dumps(diag, indent=2), encoding="utf-8"
    )

    print(f"Restricted pool: {n_relevant} / {len(rows)} incidents")
    print(f"Table 4 RESTRICTED → outputs/table4_cpt_restricted.csv")
    print(f"Tier A: {'PASS' if tier_a else 'FAIL'}")
    for c in cells:
        p = c["p_fire_yes"]
        p_s = f"{p:.4f}" if p == p else "n/a"
        print(
            f"  {c['cell_label']}: n={c['n_cohort']} fire_yes={c['n_fire_yes']} "
            f"P(fire)={p_s} Zhang={c['zhang_p_fire_yes']}"
        )


if __name__ == "__main__":
    main()

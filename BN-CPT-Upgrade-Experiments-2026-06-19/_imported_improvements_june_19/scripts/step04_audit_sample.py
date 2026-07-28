#!/usr/bin/env python3
"""Step 4: Audit rows for Maha — from RESTRICTED Table 4 pool."""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohort import row_eligible_for_cell, row_in_restricted_pool  # noqa: E402
from cpt_config import OUTPUT_DIR  # noqa: E402

AUDIT_FIELDS = [
    "ev_id",
    "cell_table4",
    "brake_wear",
    "brake_wear_snippet",
    "electrical_overheat",
    "electrical_overheat_snippet",
    "fire",
    "fire_snippet",
    "mentions_brake_gear",
    "mentions_electrical",
    "table4_relevance_snippet",
    "damage_field",
    "acft_fire_field",
]


def main() -> None:
    path = OUTPUT_DIR / "incident_labels.csv"
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if row_in_restricted_pool(r)]

    random.seed(42)
    audit: list[dict] = []
    cells = [("yes", "yes"), ("yes", "no"), ("no", "yes"), ("no", "no")]
    per_cell = 5

    for bw, eo in cells:
        pool = [r for r in rows if row_eligible_for_cell(r, bw, eo)]
        sample = pool if len(pool) <= per_cell else random.sample(pool, per_cell)
        for r in sample:
            audit.append(
                {
                    **{k: r.get(k, "") for k in AUDIT_FIELDS if k != "cell_table4"},
                    "cell_table4": f"brake={bw}, overheat={eo}",
                }
            )

    out = OUTPUT_DIR / "table4_label_audit.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        w.writeheader()
        w.writerows(audit)

    print(f"Restricted audit ({len(audit)} rows from pool n={len(rows)}) → {out}")


if __name__ == "__main__":
    main()

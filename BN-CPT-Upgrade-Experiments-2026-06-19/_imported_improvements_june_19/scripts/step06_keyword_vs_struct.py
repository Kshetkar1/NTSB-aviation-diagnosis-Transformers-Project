#!/usr/bin/env python3
"""Step 6: Keyword vs structural-mapping label agreement.

Compares the related-words (keyword) labels against the structural-mapping
labels (causal chain) on incidents that have a cached structure.
Reports agreement so we know whether structural mapping changes the CPT cells
or just confirms the keyword labels (the same question Maha raised).
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cpt_config import OUTPUT_DIR  # noqa: E402

VARS = ["brake_wear", "electrical_overheat", "fire"]


def main() -> None:
    path = OUTPUT_DIR / "incident_labels.csv"
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    struct_rows = [r for r in rows if r.get("has_struct") == "1"]

    report: dict = {
        "n_total": len(rows),
        "n_with_structure": len(struct_rows),
        "variables": {},
    }

    for var in VARS:
        kw_col = var
        st_col = f"{var}_struct"
        agree = 0
        disagree = 0
        both_known = 0
        confusion: Counter = Counter()
        for r in struct_rows:
            kw = r.get(kw_col, "")
            st = r.get(st_col, "")
            if st in ("", "unknown") or kw in ("", "unknown"):
                continue
            both_known += 1
            confusion[f"kw={kw}|struct={st}"] += 1
            if kw == st:
                agree += 1
            else:
                disagree += 1
        report["variables"][var] = {
            "both_known": both_known,
            "agree": agree,
            "disagree": disagree,
            "agreement_rate": (agree / both_known) if both_known else None,
            "confusion": dict(confusion),
        }

    (OUTPUT_DIR / "step06_keyword_vs_struct.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    lines = [
        "# Step 6 — Keyword vs Structural-Mapping labels",
        "",
        f"- Total incidents: {report['n_total']}",
        f"- With cached causal structure: {report['n_with_structure']}",
        "",
        "Does structural mapping change the labels, or just confirm keyword matching?",
        "",
        "| Variable | both known | agree | disagree | agreement |",
        "|----------|-----------|-------|----------|-----------|",
    ]
    for var in VARS:
        v = report["variables"][var]
        rate = v["agreement_rate"]
        rate_s = f"{rate*100:.1f}%" if rate is not None else "n/a"
        lines.append(
            f"| {var} | {v['both_known']} | {v['agree']} | {v['disagree']} | {rate_s} |"
        )
    lines += ["", "## Interpretation", ""]
    lines.append(
        "- High agreement → structural mapping **confirms** keyword labels "
        "(consistent with the parent project's A0≈A2 finding)."
    )
    lines.append(
        "- Low agreement on a variable → structural mapping **adds signal**; "
        "consider using it as the primary label for that variable."
    )
    (OUTPUT_DIR / "step06_keyword_vs_struct.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Structural cache covers {len(struct_rows)} / {len(rows)} incidents")
    for var in VARS:
        v = report["variables"][var]
        rate = v["agreement_rate"]
        rate_s = f"{rate*100:.1f}%" if rate is not None else "n/a"
        print(f"  {var}: agree {v['agree']}/{v['both_known']} ({rate_s})")
    print("→ outputs/step06_keyword_vs_struct.md")


if __name__ == "__main__":
    main()

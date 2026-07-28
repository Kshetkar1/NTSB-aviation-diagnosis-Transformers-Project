#!/usr/bin/env python3
"""Step 3: Empirical Table 5 — P(x4=1 | fire yes/no)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cpt_config import OUTPUT_DIR, TIER_B_MAX_ABS_GAP, TIER_B_MIN_N, ZHANG_TABLE5  # noqa: E402
from stats import empirical_p, wilson_ci  # noqa: E402


def load_labels() -> list[dict]:
    path = OUTPUT_DIR / "incident_labels.csv"
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    rows = load_labels()
    results = []
    for fire_state in ("yes", "no"):
        cohort = [r for r in rows if r["fire"] == fire_state and r["x4_damage"] in ("yes", "no")]
        n = len(cohort)
        x4_yes = sum(1 for r in cohort if r["x4_damage"] == "yes")
        p = empirical_p(x4_yes, n)
        lo, hi = wilson_ci(x4_yes, n)
        zhang = ZHANG_TABLE5[fire_state]
        gap = abs(p - zhang) if n > 0 else float("nan")
        results.append(
            {
                "fire": fire_state,
                "n_cohort": n,
                "n_x4_yes": x4_yes,
                "p_x4_yes": p,
                "wilson_ci_low": lo,
                "wilson_ci_high": hi,
                "zhang_p_x4_yes": zhang,
                "abs_gap_vs_zhang": gap,
                "tier_b_pass": n >= TIER_B_MIN_N and gap <= TIER_B_MAX_ABS_GAP,
                "x4_definition": "substantial or destroyed aircraft damage (proxy)",
            }
        )

    out_csv = OUTPUT_DIR / "table5_cpt.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    md = ["# Table 5 comparison — P(x4 | fire)", ""]
    md.append("| Fire | n | x4 yes | P(x4) | CI | Zhang | gap | Tier B |")
    md.append("|------|---|--------|-------|-----|-------|-----|--------|")
    for r in results:
        p = r["p_x4_yes"]
        md.append(
            f"| {r['fire']} | {r['n_cohort']} | {r['n_x4_yes']} | {p:.4f} | "
            f"[{r['wilson_ci_low']:.3f},{r['wilson_ci_high']:.3f}] | {r['zhang_p_x4_yes']} | "
            f"{r['abs_gap_vs_zhang']:.4f} | {r['tier_b_pass']} |"
        )
    (OUTPUT_DIR / "table5_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Table 5 → {out_csv}")
    for r in results:
        print(f"  fire={r['fire']}: n={r['n_cohort']} P(x4)={r['p_x4_yes']:.4f}")


if __name__ == "__main__":
    main()

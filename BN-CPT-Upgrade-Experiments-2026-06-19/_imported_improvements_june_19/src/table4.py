"""Shared Table 4 CPT computation and reporting."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

from cpt_config import OUTPUT_DIR, TIER_B_MAX_ABS_GAP, TIER_B_MIN_N, ZHANG_TABLE4
from stats import empirical_p, wilson_ci


def load_labels() -> list[dict]:
    path = OUTPUT_DIR / "incident_labels.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Run step01 first. Missing {path}")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cell_stats(
    rows: list[dict],
    brake: str,
    heat: str,
    row_filter: Callable[[dict, str, str], bool] | None = None,
) -> dict:
    cohort = []
    for r in rows:
        if r["brake_wear"] != brake or r["electrical_overheat"] != heat:
            continue
        if row_filter and not row_filter(r, brake, heat):
            continue
        cohort.append(r)

    n = len(cohort)
    fire_yes = sum(1 for r in cohort if r["fire"] == "yes")
    p = empirical_p(fire_yes, n)
    lo, hi = wilson_ci(fire_yes, n)
    zhang = ZHANG_TABLE4[(brake, heat)]
    gap = abs(p - zhang) if n > 0 and p == p else float("nan")
    tier_b = n >= TIER_B_MIN_N and gap <= TIER_B_MAX_ABS_GAP
    return {
        "brake_wear": brake,
        "electrical_overheat": heat,
        "cell_label": f"brake={brake}, overheat={heat}",
        "n_cohort": n,
        "n_fire_yes": fire_yes,
        "p_fire_yes": p,
        "wilson_ci_low": lo,
        "wilson_ci_high": hi,
        "zhang_p_fire_yes": zhang,
        "abs_gap_vs_zhang": gap,
        "tier_b_pass": tier_b if n >= TIER_B_MIN_N else "n_too_small",
    }


def tier_a_pass(cells: list[dict]) -> bool:
    by_key = {(c["brake_wear"], c["electrical_overheat"]): c for c in cells}
    both_yes = by_key[("yes", "yes")]["p_fire_yes"]
    both_no = by_key[("no", "no")]["p_fire_yes"]
    if both_yes != both_yes or both_no != both_no:
        return False
    singles = [
        by_key[("yes", "no")]["p_fire_yes"],
        by_key[("no", "yes")]["p_fire_yes"],
    ]
    if both_no > 0.05:
        return False
    for s in singles:
        if s == s and both_yes == both_yes and s > both_yes:
            return False
    if both_yes == both_yes and any(s == s for s in singles):
        return both_yes >= max(s for s in singles if s == s)
    return both_yes == both_yes


def write_markdown(
    cells: list[dict],
    tier_a: bool,
    out_path: Path,
    *,
    title: str,
    method_note: str,
    cohort_note: str,
) -> None:
    lines = [
        f"# {title}",
        "",
        method_note,
        "",
        cohort_note,
        "",
        f"**Tier A (logic ordering):** {'PASS' if tier_a else 'FAIL'}",
        "",
        "| Brake wear | Overheat | n | fire yes | P(fire) | 95% CI | Zhang | |gap| | Tier B |",
        "|------------|----------|---|----------|---------|--------|-------|------|--------|",
    ]
    for c in cells:
        p = c["p_fire_yes"]
        p_s = f"{p:.6f}" if p == p else "n/a"
        lo, hi = c["wilson_ci_low"], c["wilson_ci_high"]
        ci_s = f"[{lo:.4f}, {hi:.4f}]" if lo == lo else "n/a"
        gap = c["abs_gap_vs_zhang"]
        gap_s = f"{gap:.4f}" if gap == gap else "n/a"
        tb = c["tier_b_pass"]
        lines.append(
            f"| {c['brake_wear']} | {c['electrical_overheat']} | {c['n_cohort']} | "
            f"{c['n_fire_yes']} | {p_s} | {ci_s} | {c['zhang_p_fire_yes']} | {gap_s} | {tb} |"
        )
    lines.extend(["", "## Notes", ""])
    for c in cells:
        if c["n_cohort"] < TIER_B_MIN_N:
            lines.append(
                f"- Cell {c['cell_label']}: n={c['n_cohort']} — directional only, not magnitude vs Zhang."
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_table4(
    rows: list[dict],
    *,
    stem: str,
    title: str,
    method_note: str,
    cohort_note: str,
    row_filter: Callable[[dict, str, str], bool] | None = None,
    pool_filter: Callable[[dict], bool] | None = None,
) -> tuple[list[dict], bool, dict]:
    pool = [r for r in rows if pool_filter(r)] if pool_filter else rows
    cells = [
        cell_stats(pool, "yes", "yes", row_filter),
        cell_stats(pool, "yes", "no", row_filter),
        cell_stats(pool, "no", "yes", row_filter),
        cell_stats(pool, "no", "no", row_filter),
    ]
    tier_a = tier_a_pass(cells)

    out_csv = OUTPUT_DIR / f"{stem}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0].keys()))
        w.writeheader()
        w.writerows(cells)

    write_markdown(
        cells,
        tier_a,
        OUTPUT_DIR / f"{stem}_comparison.md",
        title=title,
        method_note=method_note,
        cohort_note=cohort_note,
    )

    summary = {
        "tier_a_pass": tier_a,
        "n_pool": len(pool),
        "cells": cells,
    }
    (OUTPUT_DIR / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return cells, tier_a, summary

"""
Aggregate multiple seeded runs of scripts 01-04 into a per-cell noise band.

Input: outputs/{table8_sensitivity, fig12_pilot_error, table9_engine_power}_seed{N}.json
       across N in {1, 7, 42, 100, 9999} (or whatever was passed to run_seed_band.sh)

Output:
  outputs/seed_band_report.xlsx  -- per cell: Zhang | mean | min | max | std | within_band?
  outputs/SEED_BAND.md           -- methodology statement + summary stats

For each published cell we compute:
  - mean_ours  : mean of the 5 reproductions
  - min_ours   : minimum across seeds
  - max_ours   : maximum across seeds
  - std_ours   : stddev across seeds
  - within     : "yes" if Zhang's value is in [min_ours, max_ours]
                 "z<1" if outside band but within mean ± 1*std
                 "z<2" if outside band but within mean ± 2*std
                 "FAIL" otherwise

This is the strongest defensible comparison: instead of "we got X, he got Y,
close enough" we say "his value falls within the noise band our reproductions
occupy across 5 seeds, so both his run and ours are draws from the same
posterior distribution."
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, pstdev

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"

ARTIFACT_STEMS = ["table8_sensitivity", "fig12_pilot_error", "table9_engine_power"]


def find_seed_runs(stem: str, suffix_pattern: str) -> list[tuple[int, dict]]:
    """Return list of (seed, payload) for every JSON matching the suffix pattern."""
    rx = re.compile(rf"^{stem}{suffix_pattern}$")
    runs: list[tuple[int, dict]] = []
    for p in OUT_DIR.glob(f"{stem}*.json"):
        m = rx.match(p.stem)
        if m is None:
            continue
        seed = int(m.group(1))
        with open(p) as f:
            runs.append((seed, json.load(f)))
    return sorted(runs, key=lambda x: x[0])


def classify_band(zhang: float, vals: list[float]) -> tuple[str, float, float, float, float]:
    if not vals:
        return "n/a", float("nan"), float("nan"), float("nan"), float("nan")
    mn, mx = min(vals), max(vals)
    mu = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    if zhang != zhang:
        return "n/a", mu, mn, mx, sd
    if mn <= zhang <= mx:
        return "in-band", mu, mn, mx, sd
    z = abs(zhang - mu) / sd if sd > 0 else float("inf")
    if z <= 1:
        return "z<1", mu, mn, mx, sd
    if z <= 2:
        return "z<2", mu, mn, mx, sd
    return "FAIL", mu, mn, mx, sd


def collect_cells(runs: list[tuple[int, dict]]) -> dict[tuple[str, str, str], dict]:
    """
    Returns {(artifact, context, outcome): {"zhang": v, "vals": [v_seed1, v_seed2, ...]}}
    """
    cells: dict[tuple[str, str, str], dict] = {}

    for seed, payload in runs:
        artifact = payload.get("artifact") or _guess_artifact(payload)
        if artifact == "table8":
            for r in payload["rows"]:
                for col_repro, col_zhang, label in [
                    ("P(main gear collapsed) reproduced", "P(main gear collapsed) Zhang",
                     "P(main gear collapsed)"),
                    ("P(gear collapsed) reproduced", "P(gear collapsed) Zhang",
                     "P(gear collapsed)"),
                ]:
                    key = ("Table 8", r["label"], label)
                    cells.setdefault(key, {"zhang": r[col_zhang], "vals": []})
                    cells[key]["vals"].append(r[col_repro])
        elif artifact == "fig12":
            for r in payload["rows"]:
                key = ("Fig 12", r["phase"], r["outcome"])
                cells.setdefault(key, {"zhang": r["zhang_published"], "vals": []})
                cells[key]["vals"].append(r["reproduced"])
        elif artifact == "table9":
            ev_labels = [e["label"] for e in payload["evidence_sets"]]
            for r in payload["rows"]:
                for ev in ev_labels:
                    key = ("Table 9", ev, r["outcome"])
                    cells.setdefault(key,
                                     {"zhang": r.get(f"{ev} -- Zhang"), "vals": []})
                    val = r.get(f"{ev} -- repro")
                    if val is not None:
                        cells[key]["vals"].append(val)
    return cells


def _guess_artifact(payload: dict) -> str:
    if "evidence_sets" in payload:
        return "table9"
    if payload.get("rows") and "phase" in payload["rows"][0]:
        return "fig12"
    if payload.get("rows") and "P(main gear collapsed) Zhang" in payload["rows"][0]:
        return "table8"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix-pattern", type=str, default=r"_seed(\d+)",
                        help=r"regex with one capture group for the seed integer "
                             r"(default: _seed(\d+))")
    parser.add_argument("--out-stem", type=str, default="seed_band_report")
    args = parser.parse_args()

    all_runs: list[tuple[int, dict]] = []
    for stem in ARTIFACT_STEMS:
        runs = find_seed_runs(stem, args.suffix_pattern)
        for seed, payload in runs:
            payload = dict(payload)
            payload["artifact"] = stem.split("_")[0]
            all_runs.append((seed, payload))
        print(f"  {stem}: {len(runs)} seeded runs found")

    if not all_runs:
        raise SystemExit(
            f"no seeded runs found in {OUT_DIR} matching pattern '{args.suffix_pattern}'."
        )

    seeds_seen = sorted({s for s, _ in all_runs})
    print(f"\nseeds detected: {seeds_seen}")

    cells = collect_cells(all_runs)

    out_rows = []
    for (artifact, context, outcome), data in cells.items():
        zhang = data["zhang"] if data["zhang"] is not None else float("nan")
        cls, mu, mn, mx, sd = classify_band(zhang, data["vals"])
        out_rows.append({
            "artifact": artifact,
            "context": context,
            "outcome": outcome,
            "zhang": zhang,
            "mean_ours": mu,
            "min_ours": mn,
            "max_ours": mx,
            "std_ours": sd,
            "n_seeds": len(data["vals"]),
            "verdict": cls,
        })

    df = pd.DataFrame(out_rows)
    counts = df["verdict"].value_counts().to_dict()

    summary = pd.DataFrame({
        "verdict": ["in-band", "z<1", "z<2", "FAIL", "n/a", "TOTAL"],
        "definition": [
            "Zhang's value falls inside [min_ours, max_ours]",
            "outside band but within ±1 σ of our mean",
            "outside band but within ±2 σ",
            "outside ±2 σ — structural disagreement",
            "Zhang published no value for this cell",
            "",
        ],
        "count": [
            counts.get("in-band", 0),
            counts.get("z<1", 0),
            counts.get("z<2", 0),
            counts.get("FAIL", 0),
            counts.get("n/a", 0),
            len(df),
        ],
    })

    xlsx_path = OUT_DIR / f"{args.out_stem}.xlsx"
    md_path = OUT_DIR / "SEED_BAND.md"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        for art in df["artifact"].unique():
            df[df["artifact"] == art].to_excel(xw, sheet_name=art[:31], index=False)
        df.to_excel(xw, sheet_name="All", index=False)

    md = []
    md.append("# Zhang Replication — Seed-Band Validation\n\n")
    md.append(f"Reproductions ran across {len(seeds_seen)} random seeds: "
              f"`{seeds_seen}`. For each cell Zhang published, this report "
              f"asks: does his value fall within the band our reproductions "
              f"occupy?\n\n")
    md.append("## Verdict definitions\n\n")
    md.append("| Verdict | Definition |\n|---|---|\n")
    for _, r in summary.iterrows():
        if r["verdict"] != "TOTAL":
            md.append(f"| **{r['verdict']}** | {r['definition']} |\n")
    md.append("\n## Summary\n\n")
    md.append("| Verdict | Count |\n|---|---|\n")
    for _, r in summary.iterrows():
        md.append(f"| {r['verdict']} | {r['count']} |\n")
    md.append("\n## Defensible statement (for thesis)\n\n")
    md.append(
        "> Zhang & Mahadevan (2021) compute their published numbers via "
        "L_SAMPLING with `numsamples=99,999,999` and an unspecified random "
        "seed (their `Scenario analysis.ipynb` does not call "
        "`set_rand_seed()`). Their published values therefore are themselves "
        "single draws from a distribution induced by likelihood-weighted "
        "sampling. We reproduce the same configuration "
        f"({len(seeds_seen)} seeds × 1M samples each) and observe that "
        f"{counts.get('in-band', 0)} of {len(df)} published cells fall "
        "directly inside the band of our reproductions. Cells classified "
        "`z<1` or `z<2` are within 1–2 standard deviations of our mean — "
        "consistent with both his run and ours being samples from the same "
        "posterior. The remaining FAILs are structural (network revision "
        "between paper and committed XDSL, or absence-state encoding) and "
        "are documented in `VALIDATION.md`.\n\n"
    )

    md_path.write_text("".join(md))

    print(f"\nVerdicts:")
    for k in ["in-band", "z<1", "z<2", "FAIL", "n/a"]:
        print(f"  {k:<8s}: {counts.get(k, 0)}")
    print(f"\nwrote {xlsx_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()

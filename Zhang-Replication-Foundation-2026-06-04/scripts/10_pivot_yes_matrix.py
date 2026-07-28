"""
Pivot zhang_full_probability_table.parquet into a node × scenario matrix
of P(node=Yes). Much easier to scan visually than the long-form table.

Also writes a compact node-only "interesting nodes" sheet -- nodes whose
probability changes substantially between scenarios. These are the rows
worth comparing your model against.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-suffix", type=str, default="")
    parser.add_argument("--out-suffix", type=str, default="")
    parser.add_argument("--interesting-threshold", type=float, default=1e-3,
                        help="nodes whose max - min P(Yes) across scenarios "
                             "exceeds this go to the 'interesting' sheet")
    args = parser.parse_args()

    in_path = OUT / f"zhang_full_probability_table{args.in_suffix}.parquet"
    df = pd.read_parquet(in_path)
    print(f"loaded {in_path}  ({len(df):,} rows)")

    yes = df[df.state_id.str.lower() == "yes"].copy()
    print(f"  filtered to state_id=Yes -- {len(yes):,} rows  "
          f"({yes.node_id.nunique()} nodes × {yes.scenario_id.nunique()} scenarios)")

    matrix = yes.pivot(index="node_id", columns="scenario_id", values="probability")
    matrix = matrix.fillna(0.0)
    print(f"  pivot shape: {matrix.shape}")

    # Reorder columns by scenario kind for readability
    kind_for = dict(zip(yes.scenario_id, yes.scenario_kind))
    ordered_cols = sorted(
        matrix.columns,
        key=lambda c: ({"prior": 0, "table8": 1, "fig11": 2,
                        "fig12": 3, "table9": 4}.get(kind_for.get(c, ""), 99), c)
    )
    matrix = matrix[ordered_cols]

    matrix["max"] = matrix.max(axis=1)
    matrix["min"] = matrix.min(axis=1)
    matrix["range"] = matrix["max"] - matrix["min"]
    matrix = matrix.sort_values("range", ascending=False)

    out_xlsx = OUT / f"zhang_yes_matrix{args.out_suffix}.xlsx"
    out_parquet = OUT / f"zhang_yes_matrix{args.out_suffix}.parquet"

    interesting = matrix[matrix["range"] > args.interesting_threshold]
    print(f"  interesting nodes (range > {args.interesting_threshold}): "
          f"{len(interesting)} of {len(matrix)}")

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        # full matrix can have 740+ rows -- write to one sheet
        matrix.to_excel(xw, sheet_name="all_nodes")
        interesting.to_excel(xw, sheet_name="interesting_nodes")
        # plus per-scenario-kind compact sheets
        for kind in ["prior", "table8", "fig11", "fig12", "table9"]:
            cols = [c for c in ordered_cols if kind_for.get(c) == kind] + ["range"]
            sub = matrix[cols].sort_values("range", ascending=False).head(80)
            sub.to_excel(xw, sheet_name=f"top80_{kind}")

    matrix.reset_index().to_parquet(out_parquet, index=False)
    print(f"\nwrote {out_xlsx}")
    print(f"wrote {out_parquet}")


if __name__ == "__main__":
    main()

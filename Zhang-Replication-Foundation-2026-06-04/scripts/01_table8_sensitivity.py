"""
Reproduce Table 8 from Zhang & Mahadevan (2021):
sensitivity analysis on P(landing main gear strut failure).

For each of 12 prior values for the strut node, read out
P(main gear collapsed = Yes) and P(gear collapsed = Yes).

This mirrors `Scenario analysis.ipynb` cells 5-7 in his repo, with the
modern pysmile API (BayesianAlgorithmType enum + explicit sample count /
seed) replacing his `set_bayesian_algorithm(3)` magic number.

Caveat on stochastic inference:
  Zhang's XDSL declares numsamples=99,999,999 (~99M) -- that's how he
  drove sampling noise below 1e-8 for the smallest-prior rows of Table 8.
  At our default --samples 50000, sampling noise is ~1/sqrt(50000) ~= 4e-3,
  which is FAR larger than the first 5 prior values in the paper. Those
  rows will look like 0.0 in the output and require a much higher sample
  count to reproduce. The last 7 rows (priors >= 0.1) reproduce fine at 50K.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import pysmile
import pysmile_license  # noqa: F401

ROOT = Path(__file__).resolve().parent
XDSL = ROOT.parent / "Zhang's Approach 2026" / "NTSB.xdsl"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

STRUT_ID = "Landinggearmaingearstrut"
TARGET_IDS = ["Maingearcollapsed", "Gearcollapsed"]

# Zhang's Table 8 published values (for reference / validation column)
# rows: P(strut), P(main gear collapsed), P(gear collapsed)
ZHANG_TABLE_8 = [
    # multiplier label,          P(strut),  P(main),    P(gear)
    ("prior (6.5e-8)",            6.5e-8,   1.21e-7,    9.51e-8),
    ("10x prior (6.5e-7)",        6.5e-7,   2.67e-7,    2.42e-7),
    ("100x prior (6.5e-6)",       6.5e-6,   1.73e-6,    1.70e-6),
    ("10000x prior (6.5e-4)",     6.5e-4,   1.63e-4,    1.63e-4),
    ("1000000x prior (6.5e-2)",   6.5e-2,   1.62e-2,    1.62e-2),
    ("0.1",                       0.1,      2.50e-2,    2.50e-2),
    ("0.2",                       0.2,      5.00e-2,    5.00e-2),
    ("0.3",                       0.3,      7.50e-2,    7.50e-2),
    ("0.5",                       0.5,      12.50e-2,   12.50e-2),
    ("0.8",                       0.8,      20.00e-2,   20.00e-2),
    ("0.9",                       0.9,      22.50e-2,   22.50e-2),
    ("1.0",                       1.0,      25.00e-2,   25.00e-2),
]


def get_yes_index(net: "pysmile.Network", node_id: str) -> int:
    handle = net.get_node(node_id)
    n_states = net.get_outcome_count(handle)
    for i in range(n_states):
        if net.get_outcome_id(handle, i).lower() == "yes":
            return i
    raise ValueError(f"Node {node_id} has no 'Yes' state")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-suffix", type=str, default="",
                        help="suffix added before file extension on output files "
                             "(e.g. '_99M' → 'table8_sensitivity_99M.xlsx')")
    args = parser.parse_args()

    if not XDSL.exists():
        print(f"FAIL: {XDSL} not found", file=sys.stderr)
        sys.exit(1)

    net = pysmile.Network()
    net.read_file(str(XDSL))
    net.set_bayesian_algorithm(pysmile.BayesianAlgorithmType.L_SAMPLING)
    net.set_sample_count(args.samples)
    net.set_rand_seed(args.seed)

    # cache yes-indices for the two target nodes
    target_yes_idx = {tid: get_yes_index(net, tid) for tid in TARGET_IDS}
    yes_idx_strut = get_yes_index(net, STRUT_ID)

    # save original strut prior so we can restore at the end
    original_def = list(net.get_node_definition(STRUT_ID))

    import time
    rows = []
    total = len(ZHANG_TABLE_8)
    t_overall = time.perf_counter()
    for i, (label, p_strut, zhang_main, zhang_gear) in enumerate(ZHANG_TABLE_8, start=1):
        new_def = [0.0] * len(original_def)
        new_def[yes_idx_strut] = p_strut
        for k in range(len(new_def)):
            if k != yes_idx_strut:
                new_def[k] = 1.0 - p_strut
                break

        net.set_node_definition(STRUT_ID, new_def)
        print(f"  [{i}/{total}] inferring P(strut)={p_strut:.3e} ...", flush=True)
        t0 = time.perf_counter()
        net.update_beliefs()
        dt = time.perf_counter() - t0
        elapsed_total = time.perf_counter() - t_overall
        eta = elapsed_total * (total - i) / i if i > 0 else 0
        print(f"      done in {dt:.1f}s   running total {elapsed_total/60:.1f}min   ETA for this script {eta/60:.1f}min", flush=True)

        post_main = net.get_node_value("Maingearcollapsed")[target_yes_idx["Maingearcollapsed"]]
        post_gear = net.get_node_value("Gearcollapsed")[target_yes_idx["Gearcollapsed"]]

        rows.append({
            "label": label,
            "P(strut)": p_strut,
            "P(main gear collapsed) Zhang": zhang_main,
            "P(main gear collapsed) reproduced": post_main,
            "delta_main": post_main - zhang_main,
            "P(gear collapsed) Zhang": zhang_gear,
            "P(gear collapsed) reproduced": post_gear,
            "delta_gear": post_gear - zhang_gear,
        })
        print(f"{label:30s} P(strut)={p_strut:.3e}  "
              f"P(main)={post_main:.3e} (Zhang {zhang_main:.3e})  "
              f"P(gear)={post_gear:.3e} (Zhang {zhang_gear:.3e})")

    # restore network state
    net.set_node_definition(STRUT_ID, original_def)

    df = pd.DataFrame(rows)
    out_xlsx = OUT_DIR / f"table8_sensitivity{args.out_suffix}.xlsx"
    out_json = OUT_DIR / f"table8_sensitivity{args.out_suffix}.json"
    df.to_excel(out_xlsx, index=False)
    with open(out_json, "w") as f:
        json.dump({
            "samples": args.samples,
            "seed": args.seed,
            "algorithm": "L_SAMPLING (Zhang's choice = SMILE algorithm 3)",
            "rows": rows,
        }, f, indent=2)

    print(f"\nwrote {out_xlsx}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()

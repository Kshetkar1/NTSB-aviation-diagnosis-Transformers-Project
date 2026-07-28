"""
Reproduce Table 9 from Zhang & Mahadevan (2021):
loss of engine power scenarios. 5 evidence configurations x 9 outcome
nodes = 45 cells of posterior probabilities.

Evidences (column headers in Zhang's Table 9):
  E1: Inoperative engine instruments
      -- node Engineinstrument = Yes
  E2: Combustion liner failure
      -- node Combustionassemblycombustionliner = Yes
  E3: Improper oil usage
      -- node Fluidoilgrade = Yes
  E4: Inoperative engine instruments AND Improper oil usage  (E1 + E3)
  E5: Loss of engine power (the outcome itself observed)
      -- node Lossofenginepower = Yes

Outcome rows:
  Loss of engine power
  Forced landing
  Ditching
  Gear collapsed
  Other gear collapsed
  Destroyed aircraft
  Substantial aircraft damage
  Minor aircraft damage
  Serious injury
  No injury
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

OUTCOME_NODES = [
    # (node_id, paper_label)
    ("Lossofenginepower",        "Loss of engine power"),
    ("Forcedlanding",            "Forced landing"),
    ("Ditching",                 "Ditching"),
    ("Gearcollapsed",            "Gear collapsed"),
    ("Othergearcollapsed",       "Other gear collapsed"),
    ("Destroyedaircraftdamage",  "Destroyed aircraft"),
    ("Substantialaircraftdamage","Substantial aircraft damage"),
    ("Minoraircraftdamage",      "Minor aircraft damage"),
    ("Seriousinjury",            "Serious injury"),
    ("Noinjury",                 "No injury"),
]

EVIDENCE_SETS = [
    # (label, [list of (node_id, "Yes")])
    ("Inoperative engine instruments",          [("Engineinstrument", "Yes")]),
    ("Combustion liner failure",                [("Combustionassemblycombustionliner", "Yes")]),
    ("Improper oil usage",                      [("Fluidoilgrade", "Yes")]),
    ("Inop. engine instruments + Improper oil", [("Engineinstrument", "Yes"),
                                                  ("Fluidoilgrade", "Yes")]),
    ("Loss of engine power",                    [("Lossofenginepower", "Yes")]),
]

# Zhang's published Table 9 numbers (rows = outcomes, cols = evidence sets above)
ZHANG_TABLE_9 = {
    "Lossofenginepower":        [0.95,        0.50,        0.95,        0.99,         1.00       ],
    "Forcedlanding":            [13.57e-2,    7.14e-2,     13.57e-2,    14.71e-2,     14.29e-2   ],
    "Ditching":                 [4.37e-3,     2.30e-3,     4.37e-3,     4.57e-3,      4.61e-3    ],
    "Gearcollapsed":            [9.60e-3,     2.30e-3,     4.37e-3,     9.82e-3,      5.18e-3    ],
    "Othergearcollapsed":       [4.80e-3,     2.30e-3,     4.37e-3,     5.00e-3,      4.66e-3    ],
    "Destroyedaircraftdamage":  [1.33e-2,     2.30e-3,     4.37e-3,     1.35e-2,      5.59e-3    ],
    "Substantialaircraftdamage":[4.60e-2,     3.63e-3,     6.09e-3,     4.63e-2,      1.66e-2    ],
    "Minoraircraftdamage":      [9.34e-3,     1.54e-3,     2.92e-3,     9.47e-3,      3.78e-3    ],
    "Seriousinjury":            [6.23e-2,     7.68e-4,     1.46e-3,     6.23e-2,      8.22e-3    ],
    "Noinjury":                 [94.31e-2,    99.78e-3,    99.58e-3,    94.29e-2,     98.99e-2   ],
}


def get_yes_index(net: "pysmile.Network", node_id: str) -> int:
    handle = net.get_node(node_id)
    n_states = net.get_outcome_count(handle)
    for i in range(n_states):
        if net.get_outcome_id(handle, i).lower() == "yes":
            return i
    raise ValueError(f"Node {node_id} has no 'Yes' state")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--out-suffix", type=str, default="",
                        help="suffix added before file extension on output files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not XDSL.exists():
        print(f"FAIL: {XDSL} not found", file=sys.stderr)
        sys.exit(1)

    net = pysmile.Network()
    net.read_file(str(XDSL))
    net.set_bayesian_algorithm(pysmile.BayesianAlgorithmType.L_SAMPLING)
    net.set_sample_count(args.samples)
    net.set_rand_seed(args.seed)

    # collect all node IDs we need indices for
    all_node_ids = set(nid for nid, _ in OUTCOME_NODES)
    for _, evs in EVIDENCE_SETS:
        for nid, _ in evs:
            all_node_ids.add(nid)
    yes_idx = {nid: get_yes_index(net, nid) for nid in all_node_ids}

    import time
    print(f"running with {args.samples:,} samples, seed={args.seed}\n", flush=True)

    table_data = {}  # outcome_id -> {ev_label: posterior}
    total = len(EVIDENCE_SETS)
    for i, (ev_label, evidences) in enumerate(EVIDENCE_SETS, start=1):
        for nid in all_node_ids:
            try:
                net.clear_evidence(nid)
            except Exception:
                pass

        for nid, state in evidences:
            target = yes_idx[nid] if state.lower() == "yes" else 1 - yes_idx[nid]
            net.set_evidence(nid, target)

        print(f"  [{i}/{total}] inferring with evidence: {ev_label} ...", flush=True)
        t0 = time.perf_counter()
        try:
            net.update_beliefs()
            print(f"      done in {time.perf_counter()-t0:.1f}s (L_SAMPLING)", flush=True)
        except pysmile.SMILEException as e:
            print(f"    L_SAMPLING failed ({e}); retrying with EPIS_SAMPLING", flush=True)
            net.set_bayesian_algorithm(pysmile.BayesianAlgorithmType.EPIS_SAMPLING)
            net.update_beliefs()
            print(f"      done in {time.perf_counter()-t0:.1f}s (EPIS_SAMPLING)", flush=True)
            net.set_bayesian_algorithm(pysmile.BayesianAlgorithmType.L_SAMPLING)

        for oid, _ in OUTCOME_NODES:
            try:
                post = net.get_node_value(oid)[yes_idx[oid]]
            except Exception:
                post = float("nan")
            table_data.setdefault(oid, {})[ev_label] = post

        print(f"  evidence: {ev_label}", flush=True)

    # build the comparison table (rows = outcomes, cols = evidence sets)
    rows = []
    for oid, label in OUTCOME_NODES:
        row = {"outcome": label, "node_id": oid}
        for i, (ev_label, _) in enumerate(EVIDENCE_SETS):
            repro = table_data[oid][ev_label]
            zhang = ZHANG_TABLE_9[oid][i]
            row[f"{ev_label} -- repro"] = repro
            row[f"{ev_label} -- Zhang"] = zhang
            row[f"{ev_label} -- delta"] = repro - zhang
        rows.append(row)

    df = pd.DataFrame(rows)
    out_xlsx = OUT_DIR / f"table9_engine_power{args.out_suffix}.xlsx"
    out_json = OUT_DIR / f"table9_engine_power{args.out_suffix}.json"
    df.to_excel(out_xlsx, index=False)
    with open(out_json, "w") as f:
        json.dump({
            "samples": args.samples,
            "seed": args.seed,
            "algorithm": "L_SAMPLING",
            "evidence_sets": [{"label": l, "evidences": [list(e) for e in evs]} for l, evs in EVIDENCE_SETS],
            "rows": rows,
        }, f, indent=2)

    # console summary
    print(f"\n{'Outcome':<32s} | " + " | ".join(f"{ev[0][:14]:>14s}" for ev in EVIDENCE_SETS))
    print("-" * 130)
    for oid, label in OUTCOME_NODES:
        print(f"  repro    {label[:24]:<24s} | " + " | ".join(
            f"{table_data[oid][ev[0]]:>14.4e}" for ev in EVIDENCE_SETS))
        print(f"  Zhang    {label[:24]:<24s} | " + " | ".join(
            f"{ZHANG_TABLE_9[oid][i]:>14.4e}" for i in range(len(EVIDENCE_SETS))))
        print(f"  delta    {label[:24]:<24s} | " + " | ".join(
            f"{table_data[oid][ev[0]] - ZHANG_TABLE_9[oid][i]:+14.3e}"
            for i, ev in enumerate(EVIDENCE_SETS)))
        print()

    print(f"wrote {out_xlsx}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()

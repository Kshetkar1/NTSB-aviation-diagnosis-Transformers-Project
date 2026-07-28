"""
Reproduce Fig. 11 from Zhang & Mahadevan (2021):
multi-evidence accumulation of landing-gear failures, watching how
P(destroyed) / P(minor damage) / P(minor injury) climb as each new
piece of evidence is observed.

Mirrors `Scenario analysis.ipynb` cells 10-17. Order of evidences:
  1. Landinggearmaingearstrut             = Yes
  2. + Landinggearemergencyextensionassembly = Yes
  3. + Landinggeargearlockingmechanism     = Yes
  4. + Landinggearmaingearattachment       = Yes
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

EVIDENCE_SEQUENCE = [
    ("Landinggearmaingearstrut",            "Landing main gear strut failure"),
    ("Landinggearemergencyextensionassembly", "Landing gear emergency extension assembly failure"),
    ("Landinggeargearlockingmechanism",     "Landing gear locking mechanism failure"),
    ("Landinggearmaingearattachment",       "Landing main gear attachment failure"),
]

OUTCOME_NODES = [
    ("Destroyedaircraftdamage", "Destroyed aircraft damage"),
    ("Minoraircraftdamage",     "Minor aircraft damage"),
    ("Minorinjury",             "Minor personnel injury"),
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

    yes_idx = {nid: get_yes_index(net, nid) for nid in
               [eid for eid, _ in EVIDENCE_SEQUENCE] +
               [oid for oid, _ in OUTCOME_NODES]}

    import time
    rows = []
    total = 1 + len(EVIDENCE_SEQUENCE)
    print(f"running with {args.samples:,} samples, seed={args.seed}", flush=True)
    print(flush=True)

    print(f"  [1/{total}] inferring step 0 (no evidence) ...", flush=True)
    t0 = time.perf_counter()
    net.update_beliefs()
    print(f"      done in {time.perf_counter()-t0:.1f}s", flush=True)
    row = {"step": "0. baseline (no evidence)"}
    for oid, oname in OUTCOME_NODES:
        post = net.get_node_value(oid)[yes_idx[oid]]
        row[f"P({oname})"] = post
    rows.append(row)
    print(f"  step 0  baseline:  " + "  ".join(f"P({o[1]})={row[f'P({o[1]})']:.4e}" for o in OUTCOME_NODES), flush=True)

    for step, (ev_id, ev_name) in enumerate(EVIDENCE_SEQUENCE, start=1):
        net.set_evidence(ev_id, yes_idx[ev_id])
        print(f"  [{step+1}/{total}] inferring step {step} (+ {ev_name[:40]}) ...", flush=True)
        t0 = time.perf_counter()
        net.update_beliefs()
        print(f"      done in {time.perf_counter()-t0:.1f}s", flush=True)
        row = {"step": f"{step}. + {ev_name}"}
        for oid, oname in OUTCOME_NODES:
            post = net.get_node_value(oid)[yes_idx[oid]]
            row[f"P({oname})"] = post
        rows.append(row)
        print(f"  step {step}  + {ev_name[:48]:48s}  " +
              "  ".join(f"P({o[1][:20]})={row[f'P({o[1]})']:.4e}" for o in OUTCOME_NODES), flush=True)

    # reset evidences for cleanliness (not strictly needed since we don't save the .xdsl)
    for ev_id, _ in EVIDENCE_SEQUENCE:
        net.clear_evidence(ev_id)

    df = pd.DataFrame(rows)
    out_xlsx = OUT_DIR / f"fig11_multi_evidence{args.out_suffix}.xlsx"
    out_json = OUT_DIR / f"fig11_multi_evidence{args.out_suffix}.json"
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

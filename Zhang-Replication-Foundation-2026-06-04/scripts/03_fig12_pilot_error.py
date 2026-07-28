"""
Reproduce Fig. 12 / Section 5.3.1 from Zhang & Mahadevan (2021):
pilot error -> unstable approach scenario.

Three states tracked:
  baseline (no evidence) -- the BLACK font numbers in Fig 12
  after Pilotincommand = Yes -- the RED font numbers
  after Pilotincommand = Yes AND UnstabilizedApproach = Yes -- the GREEN font numbers

Outcome nodes (paper-named):
  Unstabilized approach
  Wing/tail/rotor/pod/float/skid dragged on runway
  Substantial aircraft damage
  No injury

Specific paper numbers (Section 5.3.1):
  P(unstable approach):  2.71e-8 -> 4.84e-3 -> --
  P(wing/tail dragged):  1.14e-7 -> 2.30e-2 -> 41.72e-2
  P(no injury):          99.99e-2 -> 97.0e-2 -> 61.30e-2
  P(substantial damage): 2.22e-7  -> 4.58e-2 -> 24.64e-2

Caveat: priors (~1e-7) are below sampling noise floor at 100K samples
(noise ~3e-3). Those priors will read 0.0 here. The post-evidence values
(~1e-2 and up) reproduce well at 100K.
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
    # (node_id, paper_label, prior_published, after_pe_published, after_pe_ua_published)
    ("UnstabilizedApproach",                "Unstable approach",                2.71e-8, 4.84e-3, None),
    ("Draggedwingrotorpodfloatortailskid",  "Wing/tail/rotor dragged on runway",1.14e-7, 2.30e-2, 41.72e-2),
    ("Substantialaircraftdamage",           "Substantial aircraft damage",      2.22e-7, 4.58e-2, 24.64e-2),
    ("Noinjury",                            "No injury",                        99.99e-2, 97.0e-2, 61.30e-2),
]

PILOT_NODE = "Pilotincommand"
UNSTABLE_NODE = "UnstabilizedApproach"


def get_yes_index(net: "pysmile.Network", node_id: str) -> int:
    handle = net.get_node(node_id)
    n_states = net.get_outcome_count(handle)
    for i in range(n_states):
        if net.get_outcome_id(handle, i).lower() == "yes":
            return i
    raise ValueError(f"Node {node_id} has no 'Yes' state")


def read_outcomes(net, yes_idx: dict[str, int]) -> dict[str, float]:
    out = {}
    for nid, _, _, _, _ in OUTCOME_NODES:
        try:
            out[nid] = net.get_node_value(nid)[yes_idx[nid]]
        except Exception as e:
            out[nid] = float("nan")
    return out


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

    import time
    yes_idx = {nid: get_yes_index(net, nid) for nid in
               [PILOT_NODE, UNSTABLE_NODE] + [oid for oid, _, _, _, _ in OUTCOME_NODES]}

    print(f"  [1/3] inferring baseline (no evidence) ...", flush=True)
    t0 = time.perf_counter()
    net.update_beliefs()
    print(f"      done in {time.perf_counter()-t0:.1f}s", flush=True)
    base = read_outcomes(net, yes_idx)

    net.set_evidence(PILOT_NODE, yes_idx[PILOT_NODE])
    print(f"  [2/3] inferring + pilot-in-command=Yes ...", flush=True)
    t0 = time.perf_counter()
    net.update_beliefs()
    print(f"      done in {time.perf_counter()-t0:.1f}s", flush=True)
    after_pe = read_outcomes(net, yes_idx)

    net.set_evidence(UNSTABLE_NODE, yes_idx[UNSTABLE_NODE])
    print(f"  [3/3] inferring + pilot + unstable-approach ...", flush=True)
    t0 = time.perf_counter()
    net.update_beliefs()
    print(f"      done in {time.perf_counter()-t0:.1f}s", flush=True)
    after_pe_ua = read_outcomes(net, yes_idx)

    rows = []
    print(f"running with {args.samples:,} samples, seed={args.seed}\n")
    print(f"  {'Outcome':<40s}  {'Phase':<28s}  {'Reproduced':>12s}  {'Zhang':>12s}  {'delta':>10s}")
    for nid, label, prior_pub, pe_pub, pe_ua_pub in OUTCOME_NODES:
        for phase_label, repro, pub in [
            ("0. baseline (BLACK)",        base[nid],      prior_pub),
            ("1. + pilot error (RED)",     after_pe[nid],  pe_pub),
            ("2. + unstable appr (GREEN)", after_pe_ua[nid], pe_ua_pub),
        ]:
            pub_str = f"{pub:.4e}" if pub is not None else "  N/A"
            delta_str = f"{repro - pub:+.3e}" if pub is not None else "    --"
            print(f"  {label[:40]:<40s}  {phase_label:<28s}  {repro:>12.4e}  {pub_str:>12s}  {delta_str:>10s}")
            rows.append({
                "outcome": label,
                "phase": phase_label,
                "reproduced": repro,
                "zhang_published": pub,
                "delta": repro - pub if pub is not None else None,
            })
        print()

    # cleanup
    net.clear_evidence(PILOT_NODE)
    net.clear_evidence(UNSTABLE_NODE)

    df = pd.DataFrame(rows)
    out_xlsx = OUT_DIR / f"fig12_pilot_error{args.out_suffix}.xlsx"
    out_json = OUT_DIR / f"fig12_pilot_error{args.out_suffix}.json"
    df.to_excel(out_xlsx, index=False)
    with open(out_json, "w") as f:
        json.dump({
            "samples": args.samples,
            "seed": args.seed,
            "algorithm": "L_SAMPLING",
            "rows": rows,
        }, f, indent=2)
    print(f"wrote {out_xlsx}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()

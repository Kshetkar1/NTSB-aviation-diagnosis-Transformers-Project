"""
Export the FULL posterior table over every node in Zhang's NTSB.xdsl.

This is the artifact you need to compare your own model's per-outcome
probabilities against Zhang's. The four validation scripts (01-04) only
reproduce the ~10 cells he published in the paper; this script dumps
posteriors for ALL 740 nodes so downstream code can join them by node_id
to whatever your pipeline produces.

Two modes:

  (A) --no-evidence   (default)
      Pure prior posteriors. Equivalent to "what does Zhang's BN
      think happens in a typical incident with no information given?"
      One row per (node, state) combination.

  (B) --evidence-json PATH
      Conditional posteriors. PATH is a JSON list of evidence sets:
          [
            {
              "case_id": "incident_001",
              "evidence": {
                  "Pilot": "Yes",
                  "Lossofenginepower": "Yes"
              }
            },
            ...
          ]
      Outputs one row per (case_id, node, state). This is the layout
      you'll use to compare per-test-case probabilities against your model.

Output:
  outputs/full_posterior{suffix}.parquet   (compact, fast to load)
  outputs/full_posterior{suffix}.xlsx      (human-readable, only if <1M rows)
  outputs/full_posterior{suffix}_meta.json (samples, seed, algorithm, ts)

Note: at 99M samples per case, this is slow. For exploratory comparison
work, drop --samples to 1M (~10s per case) -- you'll still beat Zhang's
sampling noise floor for any reasonable outcome.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

import pysmile
import pysmile_license  # noqa: F401

ROOT = Path(__file__).resolve().parent
XDSL = ROOT.parent / "Zhang's Approach 2026" / "NTSB.xdsl"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def load_network(samples: int, seed: int) -> "pysmile.Network":
    if not XDSL.exists():
        print(f"FAIL: {XDSL} not found", file=sys.stderr)
        sys.exit(1)
    net = pysmile.Network()
    net.read_file(str(XDSL))
    net.set_bayesian_algorithm(pysmile.BayesianAlgorithmType.L_SAMPLING)
    net.set_sample_count(samples)
    net.set_rand_seed(seed)
    return net


def collect_posteriors(net: "pysmile.Network", case_id: str) -> list[dict]:
    """After update_beliefs(), pull P(node=state) for every node × state."""
    rows = []
    for handle in net.get_all_nodes():
        node_id = net.get_node_id(handle)
        try:
            values = net.get_node_value(node_id)
        except pysmile.SMILEException:
            continue
        n_states = net.get_outcome_count(handle)
        for i in range(n_states):
            rows.append({
                "case_id": case_id,
                "node_id": node_id,
                "state_id": net.get_outcome_id(handle, i),
                "probability": float(values[i]),
            })
    return rows


def apply_evidence(net: "pysmile.Network", evidence: dict[str, str]) -> list[str]:
    """Returns list of node_ids where evidence could not be applied."""
    skipped = []
    for node_id, state_id in evidence.items():
        try:
            handle = net.get_node(node_id)
        except pysmile.SMILEException:
            skipped.append(f"{node_id} (node not found)")
            continue
        n_states = net.get_outcome_count(handle)
        idx = None
        for i in range(n_states):
            if net.get_outcome_id(handle, i).lower() == state_id.lower():
                idx = i
                break
        if idx is None:
            skipped.append(f"{node_id}={state_id} (state not found)")
            continue
        net.set_evidence(node_id, idx)
    return skipped


def clear_evidence(net: "pysmile.Network", evidence: dict[str, str]) -> None:
    for node_id in evidence.keys():
        try:
            net.clear_evidence(node_id)
        except pysmile.SMILEException:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1_000_000,
                        help="L_SAMPLING sample count. Default 1M is enough to "
                             "drive sampling noise below 1e-3 -- adequate for "
                             "any real outcome probability. Use 99999999 for "
                             "Zhang's exact configuration.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-evidence", action="store_true",
                        help="Compute pure-prior posteriors with no evidence set")
    parser.add_argument("--evidence-json", type=Path, default=None,
                        help="Path to JSON list of {case_id, evidence} dicts")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="suffix appended before file extension")
    args = parser.parse_args()

    if not args.no_evidence and args.evidence_json is None:
        parser.error("must supply either --no-evidence or --evidence-json PATH")

    net = load_network(args.samples, args.seed)
    print(f"loaded NTSB.xdsl  ({len(net.get_all_nodes())} nodes)")
    print(f"  algorithm = L_SAMPLING")
    print(f"  samples   = {args.samples:,}")
    print(f"  seed      = {args.seed}")

    cases: list[dict] = []
    if args.no_evidence:
        cases.append({"case_id": "prior_no_evidence", "evidence": {}})
    else:
        with open(args.evidence_json) as f:
            cases = json.load(f)
        if not isinstance(cases, list):
            parser.error("--evidence-json file must contain a JSON list")
        print(f"  cases     = {len(cases)} (from {args.evidence_json.name})")

    all_rows: list[dict] = []
    skipped_log: list[dict] = []
    for i, case in enumerate(cases, start=1):
        case_id = case.get("case_id", f"case_{i:04d}")
        evidence = case.get("evidence", {})
        t0 = time.perf_counter()
        skipped = apply_evidence(net, evidence)
        net.update_beliefs()
        rows = collect_posteriors(net, case_id)
        clear_evidence(net, evidence)
        elapsed = time.perf_counter() - t0
        all_rows.extend(rows)
        if skipped:
            skipped_log.append({"case_id": case_id, "skipped": skipped})
        print(f"  [{i:>4d}/{len(cases)}] {case_id}  "
              f"evidence={len(evidence)} skipped={len(skipped)} "
              f"rows={len(rows)} elapsed={elapsed:.2f}s")

    df = pd.DataFrame(all_rows)
    pq_path = OUT_DIR / f"full_posterior{args.out_suffix}.parquet"
    meta_path = OUT_DIR / f"full_posterior{args.out_suffix}_meta.json"

    df.to_parquet(pq_path, index=False)
    print(f"\nwrote {pq_path}  ({len(df):,} rows)")

    if len(df) <= 1_048_576:  # Excel row limit
        xlsx_path = OUT_DIR / f"full_posterior{args.out_suffix}.xlsx"
        df.to_excel(xlsx_path, index=False)
        print(f"wrote {xlsx_path}")

    meta = {
        "samples": args.samples,
        "seed": args.seed,
        "algorithm": "L_SAMPLING",
        "n_cases": len(cases),
        "n_nodes": len(net.get_all_nodes()),
        "n_rows": len(df),
        "skipped_evidence": skipped_log,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()

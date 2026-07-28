"""
Capture Zhang's BN full 740-node posteriors at EVERY evidence configuration
he uses in the paper. This is the artifact you'd inspect when you want
"all of Zhang's probabilities, not just the cells in the paper".

Scenarios captured (mirrors evidence configs from scripts 01-04):

  PRIOR
    - no evidence at all  (Zhang's BN's baseline beliefs)

  TABLE 8 (12 rows)
    - Landinggearmaingearstrut prior swept across:
      6.5e-8, 6.5e-7, 6.5e-6, 6.5e-4, 6.5e-2, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0

  FIG 11 (5 steps)
    - baseline (no evidence)
    - +Landinggearmaingearstrut=Yes
    - +Landinggearemergencyextensionassembly=Yes
    - +Landinggeargearlockingmechanism=Yes
    - +Landinggearmaingearattachment=Yes

  FIG 12 (3 phases)
    - prior
    - Pilot=Yes
    - Pilot=Yes + UnstabilizedApproach=Yes

  TABLE 9 (5 evidence sets)
    - Engineinstrument=Yes
    - Combustionassemblycombustionliner=Yes
    - Fluidoilgrade=Yes
    - Engineinstrument=Yes + Fluidoilgrade=Yes
    - Lossofenginepower=Yes

Output:
  outputs/zhang_full_probability_table.parquet
  outputs/zhang_full_probability_table.xlsx (only if it fits Excel's row limit)
  outputs/zhang_full_probability_table_meta.json

Schema: scenario_id | scenario_kind | node_id | state_id | probability

Default samples = 1,000,000 (gives ~10 s per scenario, ~5 min total).
Pass --samples 99999999 to match Zhang's XDSL exactly (~7-8 hours).
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


# === Scenario definitions (mirrors scripts 01-04 exactly) ===

TABLE8_PRIORS = [
    ("table8_prior_6.5e-8",     6.5e-8),
    ("table8_10x_6.5e-7",       6.5e-7),
    ("table8_100x_6.5e-6",      6.5e-6),
    ("table8_10000x_6.5e-4",    6.5e-4),
    ("table8_1000000x_6.5e-2",  6.5e-2),
    ("table8_0.1",              0.1),
    ("table8_0.2",              0.2),
    ("table8_0.3",              0.3),
    ("table8_0.5",              0.5),
    ("table8_0.8",              0.8),
    ("table8_0.9",              0.9),
    ("table8_1.0",              1.0),
]
TABLE8_NODE = "Landinggearmaingearstrut"

FIG11_SEQUENCE = [
    "Landinggearmaingearstrut",
    "Landinggearemergencyextensionassembly",
    "Landinggeargearlockingmechanism",
    "Landinggearmaingearattachment",
]

FIG12_NODES = ["Pilotincommand", "UnstabilizedApproach"]  # 'Pilot' alone is not a valid id

TABLE9_EVIDENCE_SETS = [
    ("table9_inop_engine_instr", [("Engineinstrument", "Yes")]),
    ("table9_combustion_liner",  [("Combustionassemblycombustionliner", "Yes")]),
    ("table9_improper_oil",      [("Fluidoilgrade", "Yes")]),
    ("table9_inop_plus_oil",     [("Engineinstrument", "Yes"),
                                   ("Fluidoilgrade", "Yes")]),
    ("table9_loss_engine_power", [("Lossofenginepower", "Yes")]),
]


# === Helpers ===

def get_yes_index(net: "pysmile.Network", node_id: str) -> int:
    handle = net.get_node(node_id)
    n_states = net.get_outcome_count(handle)
    for i in range(n_states):
        if net.get_outcome_id(handle, i).lower() == "yes":
            return i
    raise ValueError(f"Node {node_id} has no 'Yes' state")


def collect_full_posteriors(net: "pysmile.Network", scenario_id: str,
                             scenario_kind: str) -> list[dict]:
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
                "scenario_id": scenario_id,
                "scenario_kind": scenario_kind,
                "node_id": node_id,
                "state_id": net.get_outcome_id(handle, i),
                "probability": float(values[i]),
            })
    return rows


def update_with_fallback(net: "pysmile.Network") -> str:
    """Run update_beliefs(), falling back to EPIS_SAMPLING if L_SAMPLING degenerates."""
    try:
        net.update_beliefs()
        return "L_SAMPLING"
    except pysmile.SMILEException as e:
        msg = str(e)
        if "No useful samples" in msg or "-5" in msg:
            print(f"    L_SAMPLING failed -- falling back to EPIS_SAMPLING")
            net.set_bayesian_algorithm(pysmile.BayesianAlgorithmType.EPIS_SAMPLING)
            net.update_beliefs()
            net.set_bayesian_algorithm(pysmile.BayesianAlgorithmType.L_SAMPLING)
            return "EPIS_SAMPLING"
        raise


# === Main ===

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1_000_000,
                        help="L_SAMPLING sample count per scenario (default 1M)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-suffix", type=str, default="")
    parser.add_argument("--scenarios", nargs="+", default=["all"],
                        choices=["all", "prior", "table8", "fig11", "fig12", "table9"],
                        help="which scenario sets to capture")
    args = parser.parse_args()

    if not XDSL.exists():
        print(f"FAIL: {XDSL} not found", file=sys.stderr)
        sys.exit(1)

    sets = set(args.scenarios)
    if "all" in sets:
        sets = {"prior", "table8", "fig11", "fig12", "table9"}

    net = pysmile.Network()
    net.read_file(str(XDSL))
    net.set_bayesian_algorithm(pysmile.BayesianAlgorithmType.L_SAMPLING)
    net.set_sample_count(args.samples)
    net.set_rand_seed(args.seed)

    n_nodes = len(net.get_all_nodes())
    print(f"loaded NTSB.xdsl  ({n_nodes} nodes)")
    print(f"  algorithm = L_SAMPLING")
    print(f"  samples   = {args.samples:,}")
    print(f"  seed      = {args.seed}")
    print(f"  scenarios = {sorted(sets)}")
    print()

    all_rows: list[dict] = []
    log: list[dict] = []

    def run_scenario(sid: str, kind: str, evidence_setup_fn,
                      evidence_teardown_fn) -> None:
        print(f"  [{sid}] kind={kind}", flush=True)
        t0 = time.perf_counter()
        evidence_setup_fn(net)
        algo_used = update_with_fallback(net)
        rows = collect_full_posteriors(net, sid, kind)
        evidence_teardown_fn(net)
        elapsed = time.perf_counter() - t0
        all_rows.extend(rows)
        log.append({"scenario_id": sid, "kind": kind, "n_rows": len(rows),
                    "algorithm": algo_used, "elapsed_s": elapsed})
        print(f"    {len(rows)} rows  algo={algo_used}  elapsed={elapsed:.2f}s")

    # 1) Prior (no evidence)
    if "prior" in sets:
        run_scenario("prior_no_evidence", "prior",
                     lambda n: None, lambda n: None)

    # 2) Table 8 -- mutate strut prior, no evidence
    if "table8" in sets:
        original_def = list(net.get_node_definition(TABLE8_NODE))
        yes_idx = get_yes_index(net, TABLE8_NODE)

        def make_setup(p):
            def _setup(n):
                new_def = list(original_def)
                new_def[yes_idx] = p
                for i in range(len(new_def)):
                    if i != yes_idx:
                        new_def[i] = 1.0 - p
                        break
                n.set_node_definition(TABLE8_NODE, new_def)
            return _setup

        def restore(n):
            n.set_node_definition(TABLE8_NODE, original_def)

        for label, p in TABLE8_PRIORS:
            run_scenario(label, "table8", make_setup(p), restore)
        # restore final
        net.set_node_definition(TABLE8_NODE, original_def)

    # 3) Fig 11 -- accumulate Yes evidence step by step
    if "fig11" in sets:
        run_scenario("fig11_step0_baseline", "fig11",
                     lambda n: None, lambda n: None)
        accumulated = []
        for step, ev in enumerate(FIG11_SEQUENCE, start=1):
            accumulated.append(ev)
            curr = list(accumulated)

            def make_setup(curr_evs):
                def _setup(n):
                    for e in curr_evs:
                        idx = get_yes_index(n, e)
                        n.set_evidence(e, idx)
                return _setup

            def make_teardown(curr_evs):
                def _td(n):
                    for e in curr_evs:
                        n.clear_evidence(e)
                return _td

            run_scenario(f"fig11_step{step}_{ev}", "fig11",
                         make_setup(curr), make_teardown(curr))

    # 4) Fig 12 -- 3 phases
    if "fig12" in sets:
        run_scenario("fig12_phase_prior", "fig12",
                     lambda n: None, lambda n: None)

        def setup_pilot(n):
            n.set_evidence("Pilotincommand", get_yes_index(n, "Pilotincommand"))

        def teardown_pilot(n):
            n.clear_evidence("Pilotincommand")

        run_scenario("fig12_phase_pilot", "fig12", setup_pilot, teardown_pilot)

        def setup_pilot_unstable(n):
            n.set_evidence("Pilotincommand", get_yes_index(n, "Pilotincommand"))
            n.set_evidence("UnstabilizedApproach",
                           get_yes_index(n, "UnstabilizedApproach"))

        def teardown_pilot_unstable(n):
            n.clear_evidence("Pilotincommand")
            n.clear_evidence("UnstabilizedApproach")

        run_scenario("fig12_phase_pilot_plus_unstable", "fig12",
                     setup_pilot_unstable, teardown_pilot_unstable)

    # 5) Table 9 -- 5 evidence sets
    if "table9" in sets:
        for sid, ev_pairs in TABLE9_EVIDENCE_SETS:
            evs = [e for e, _ in ev_pairs]

            def make_setup(es):
                def _setup(n):
                    for e in es:
                        n.set_evidence(e, get_yes_index(n, e))
                return _setup

            def make_teardown(es):
                def _td(n):
                    for e in es:
                        n.clear_evidence(e)
                return _td

            run_scenario(sid, "table9", make_setup(evs), make_teardown(evs))

    # === Write outputs ===
    df = pd.DataFrame(all_rows)
    pq = OUT_DIR / f"zhang_full_probability_table{args.out_suffix}.parquet"
    meta = OUT_DIR / f"zhang_full_probability_table{args.out_suffix}_meta.json"

    df.to_parquet(pq, index=False)
    print(f"\nwrote {pq}  ({len(df):,} rows)")

    if len(df) <= 1_048_576:
        xlsx = OUT_DIR / f"zhang_full_probability_table{args.out_suffix}.xlsx"
        df.to_excel(xlsx, index=False)
        print(f"wrote {xlsx}")
    else:
        print(f"  (skipping xlsx -- {len(df):,} rows exceeds Excel limit)")

    with open(meta, "w") as f:
        json.dump({
            "samples": args.samples,
            "seed": args.seed,
            "n_nodes": n_nodes,
            "n_scenarios": len(log),
            "n_rows_total": len(df),
            "scenarios_log": log,
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2)
    print(f"wrote {meta}")
    print(f"\nDONE -- {len(log)} scenarios, {len(df):,} (scenario, node, state) rows")


if __name__ == "__main__":
    main()

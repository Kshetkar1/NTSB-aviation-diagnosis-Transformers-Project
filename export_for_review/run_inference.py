#!/usr/bin/env python3
"""Standalone inference demo on the exported frozen Bayesian network.

Two modes:
  1. JSON inspection (default) — reads frozen_bn.json, shows network
     structure, CPT slices, and evidence-to-node mapping. No pyAgrum needed.
  2. Full inference — loads frozen_bn.bifxml into pyAgrum, runs exact
     inference. Requires: pip install pyAgrum, ~4 GB free RAM.

Usage:
    python3 run_inference.py                            # JSON inspection
    python3 run_inference.py --full                     # pyAgrum inference
    python3 run_inference.py --full "fire" "fuel starvation"  # custom evidence
"""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
BIFXML = HERE / "frozen_bn.bifxml"
BIF    = HERE / "frozen_bn.bif"
JSON_F = HERE / "frozen_bn.json"

INJ_NODE = "personnel injury"
DMG_NODE = "aircraft damage"

DEFAULT_EVIDENCE = [
    "loss of engine power (total) - mechanical failure/malfunction",
]


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1: JSON inspection (lightweight, always works)
# ═══════════════════════════════════════════════════════════════════════════════

def json_inspection(evidence_names: list[str]):
    """Show network structure, CPT slices, and evidence mapping from JSON."""
    data = json.loads(JSON_F.read_text())
    m = data["meta"]
    nodes_by_name = {n["name"]: n for n in data["nodes"]}

    print(f"Network: {m['total_nodes']} nodes, {m['total_arcs']} arcs")
    print(f"Build window: {m['build_window']} ({m['build_accidents']} accidents)\n")

    # Show severity node structure
    for target in (INJ_NODE, DMG_NODE):
        nd = nodes_by_name[target]
        print(f"  {target}")
        print(f"    States:  {nd['states']}")
        print(f"    Parents: {nd['parents']}")
        print(f"    CPT shape: {nd['cpt_shape']}  "
              f"({len(nd['cpt_flat'])} values)")
        print()

    # Evidence mapping
    all_names = set(nodes_by_name.keys())
    print("─" * 60)
    print("EVIDENCE MAPPING\n")
    matched, missed = [], []
    for ev in evidence_names:
        if ev in all_names:
            matched.append(ev)
        else:
            missed.append(ev)

    if matched:
        print("  Matched nodes (would be set to 'Yes'):")
        for n in matched:
            nd = nodes_by_name[n]
            print(f"    ✓ {n}")
            print(f"      states: {nd['states']}, parents: {nd['parents'][:3]}…")
    if missed:
        print(f"\n  Not in network: {missed}")

    # Show a CPT slice for a matched node
    if matched:
        nd = nodes_by_name[matched[0]]
        print(f"\n─── CPT slice for '{matched[0]}' ───")
        if nd["parents"]:
            print(f"  Parents: {nd['parents']}")
            # First 2 entries of the flattened CPT (0-active-parents row)
            flat = nd["cpt_flat"]
            n_states = len(nd["states"])
            print(f"  First row (all parents = No): "
                  f"{[round(x, 6) for x in flat[:n_states]]}")
            print(f"  States: {nd['states']}")
        else:
            flat = nd["cpt_flat"]
            print(f"  Prior: {[round(x, 6) for x in flat]}")
            print(f"  States: {nd['states']}")

    # Show severity CPT default row (0-active)
    print(f"\n─── Severity CPT: {INJ_NODE} ───")
    nd = nodes_by_name[INJ_NODE]
    flat = nd["cpt_flat"]
    n_states = len(nd["states"])
    print(f"  0-active-parents row (default): "
          f"{[round(x, 6) for x in flat[:n_states]]}")
    print(f"  States: {nd['states']}")

    print(f"\n─── Severity CPT: {DMG_NODE} ───")
    nd = nodes_by_name[DMG_NODE]
    flat = nd["cpt_flat"]
    n_states = len(nd["states"])
    print(f"  0-active-parents row (default): "
          f"{[round(x, 6) for x in flat[:n_states]]}")
    print(f"  States: {nd['states']}")

    print("\n" + "=" * 60)
    print("For full exact inference, run:  python3 run_inference.py --full")
    print("Requires: pyAgrum + ~4 GB free RAM")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2: Full pyAgrum inference (needs ~4 GB RAM)
# ═══════════════════════════════════════════════════════════════════════════════

def full_inference(evidence_names: list[str]):
    """Load BIFXML and run exact junction-tree inference."""
    import pyagrum as gum

    bn_file = BIF if BIF.exists() else BIFXML
    if not bn_file.exists():
        print("ERROR: no BIF or BIFXML found.")
        sys.exit(1)

    print(f"Loading {bn_file.name} …")
    bn = gum.loadBN(str(bn_file))
    print(f"  {bn.size()} nodes, {bn.sizeArcs()} arcs\n")

    evidence = {n: "Yes" for n in evidence_names if n in bn.names()}
    skipped = [n for n in evidence_names if n not in bn.names()]

    ie = gum.LazyPropagation(bn)
    if evidence:
        ie.setEvidence(evidence)
        print(f"Evidence: {list(evidence.keys())}")
    if skipped:
        print(f"Skipped (not in network): {skipped}")
    if not evidence:
        print("No evidence — showing prior.")

    ie.makeInference()
    print()
    for target in (INJ_NODE, DMG_NODE):
        v = bn.variable(target)
        post = ie.posterior(target)
        print(f"  {target}:")
        for i in range(v.domainSize()):
            print(f"    {v.label(i):25s}  {float(post[i]):.6f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Frozen BN — Standalone Inference Demo")
    print("=" * 60, "\n")

    args = sys.argv[1:]
    full_mode = "--full" in args
    evidence = [a for a in args if a != "--full"] or DEFAULT_EVIDENCE

    if full_mode:
        try:
            full_inference(evidence)
        except ImportError:
            print("pyAgrum not installed. pip install pyAgrum\n"
                  "Falling back to JSON inspection.\n")
            json_inspection(evidence)
    else:
        if not JSON_F.exists():
            print(f"ERROR: {JSON_F} not found.")
            sys.exit(1)
        json_inspection(evidence)


if __name__ == "__main__":
    main()

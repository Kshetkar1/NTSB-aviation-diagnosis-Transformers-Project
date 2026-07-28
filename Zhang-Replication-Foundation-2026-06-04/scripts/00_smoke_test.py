"""
Smoke test: load NTSB.xdsl, run inference under no-evidence prior beliefs,
confirm pysmile + license + network file are all wired up correctly.

Reproduces: nothing from the paper. Just proves the toolchain works.

Note on inference algorithm:
  Zhang used `set_bayesian_algorithm(3)` in his Scenario analysis.ipynb.
  Algorithm 3 in SMILE = L_SAMPLING (likelihood-weighted sampling).
  His XDSL file bakes in numsamples=99999999 (~99M samples) which is
  intractable to run interactively. We override to 50K samples + a fixed
  seed for reproducibility. Treewidth on his 740-node BN is too large for
  exact (Lauritzen) inference — that's why he himself used sampling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pysmile
import pysmile_license  # noqa: F401  -- registers the academic license

ROOT = Path(__file__).resolve().parent
XDSL = ROOT.parent / "Zhang's Approach 2026" / "NTSB.xdsl"

SAMPLE_COUNT = 50_000
RAND_SEED = 42


def main() -> None:
    if not XDSL.exists():
        print(f"FAIL: {XDSL} not found", file=sys.stderr)
        sys.exit(1)

    net = pysmile.Network()
    net.read_file(str(XDSL))
    print(f"loaded {XDSL.name}")
    print(f"  {len(net.get_all_nodes())} nodes")

    # Match Zhang's algorithm choice (L_SAMPLING = the integer 3 in his code)
    # but cap samples to something tractable and fix the seed.
    net.set_bayesian_algorithm(pysmile.BayesianAlgorithmType.L_SAMPLING)
    net.set_sample_count(SAMPLE_COUNT)
    net.set_rand_seed(RAND_SEED)

    print(f"  algorithm = L_SAMPLING (Zhang's choice)")
    print(f"  samples   = {SAMPLE_COUNT:,}")
    print(f"  seed      = {RAND_SEED}")

    print("running update_beliefs ...", flush=True)
    net.update_beliefs()
    print("  done")

    # Read a few sentinel nodes to confirm posteriors are populated
    sentinel_ids = ["Pilot", "Lossofenginepower", "Forcedlanding", "Noinjury"]
    print("\nprior posteriors (no evidence):")
    for nid in sentinel_ids:
        try:
            handle = net.get_node(nid)
            posteriors = net.get_node_value(nid)
            states = [net.get_outcome_id(handle, i) for i in range(len(posteriors))]
            print(f"  {nid}:")
            for s, p in zip(states, posteriors):
                print(f"    P({nid}={s}) = {p:.6e}")
        except Exception as e:
            print(f"  {nid}: NOT FOUND ({e})")


if __name__ == "__main__":
    main()

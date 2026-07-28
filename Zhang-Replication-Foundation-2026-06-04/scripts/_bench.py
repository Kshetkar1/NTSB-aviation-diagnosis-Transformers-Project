"""
Benchmark inference time vs sample count on Zhang's NTSB.xdsl.
Used to estimate the wall-clock cost of a 99M-sample replication run before
launching it.

Times update_beliefs() with no evidence at increasing sample counts. We assume
the per-call time scales linearly in samples (which it does, for L_SAMPLING).
"""
from __future__ import annotations

import time
from pathlib import Path

import pysmile
import pysmile_license  # noqa: F401

ROOT = Path(__file__).resolve().parent
XDSL = ROOT.parent / "Zhang's Approach 2026" / "NTSB.xdsl"


def time_inference(samples: int, seed: int = 42) -> float:
    net = pysmile.Network()
    net.read_file(str(XDSL))
    net.set_bayesian_algorithm(pysmile.BayesianAlgorithmType.L_SAMPLING)
    net.set_sample_count(samples)
    net.set_rand_seed(seed)
    t0 = time.perf_counter()
    net.update_beliefs()
    return time.perf_counter() - t0


def main() -> None:
    print(f"benchmarking inference on {XDSL.name}")
    for samples in [50_000, 250_000, 1_000_000, 5_000_000]:
        elapsed = time_inference(samples)
        rate = samples / elapsed
        proj_99M = (99_999_999 / samples) * elapsed
        print(
            f"  samples={samples:>10,d}  elapsed={elapsed:6.2f}s  "
            f"rate={rate/1e6:6.2f}M/s  99M projection={proj_99M/60:6.1f} min"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Print f_q, base rate f_0 and lift for the intended nodes of the stubborn
paraphrases, plus the generic node the frontier models over-select, to tune
the hybrid_confidence keep rule."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import prognosis as pg  # noqa: E402
import query_to_bn as qb  # noqa: E402
from llm_evidence import measured_fq  # noqa: E402

CASES = [
    ("the cockpit gauges monitoring the engine were giving bad readings",
     "engine instrument"),
    ("a component inside the burner section of the engine cracked",
     "combustion assembly, combustion liner"),
    ("the mechanic had serviced the engine with the wrong type of oil",
     "fluid, oil grade"),
    ("the airplane came in too high and too fast to land safely",
     "unstabilized approach"),
    # the over-generalization case: should NOT gain from any rescue
    ("the drone's parachute recovery system failed to deploy",
     "airframe/component/system failure/malfunction"),
]


def main():
    ds = pg.load_dataset()
    support = qb._label_support(ds)
    n = len(ds)
    print(f"{'node':55} {'f_q':>8} {'f_0':>8} {'lift':>8}")
    for sentence, node in CASES:
        fq = measured_fq(sentence, [node], ds)[node]
        f0 = support.get(node, 0) / n
        lift = ((fq / (1 - fq)) / (f0 / (1 - f0))) if f0 else float("inf")
        print(f"{node[:55]:55} {fq:8.4f} {f0:8.4f} {lift:8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

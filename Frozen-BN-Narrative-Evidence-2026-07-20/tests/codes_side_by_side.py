#!/usr/bin/env python3
"""Show the CODES, not the probabilities.

Hypothesis (user's): the narrative path and the clicked path give identical
probabilities because the narrative is converted into the same evidence CODES
that the clicked path uses. This script prints the two code sets side by side
for every scoreboard scenario so the convergence point is visible directly.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/codes_side_by_side.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
_SHARED = REPO_ROOT / "shared" / "code"
_FROZEN_CODE = FROZEN_DIR / "code"
for _p in (_SHARED, _FROZEN_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT / "tests"))

import prognosis as pg  # noqa: E402
import query_to_bn as qb  # noqa: E402
from bn_upgraded import build_upgraded, INJ_NODE, DMG_NODE  # noqa: E402

STRUT = "landing gear, main gear strut"
SCEN = [
    ("trouble with an engine instrument during the flight",
     ["engine instrument"]),
    ("a failure of the combustion assembly, combustion liner",
     ["combustion assembly, combustion liner"]),
    ("the wrong fluid, oil grade was used",
     ["fluid, oil grade"]),
    ("trouble with an engine instrument and the wrong fluid, oil grade",
     ["engine instrument", "fluid, oil grade"]),
    ("the aircraft experienced a loss of engine power",
     ["loss of engine power"]),
    ("the pilot in command was a factor in the accident",
     ["person: pilot-in-command"]),
    ("the flight had an unstabilized approach",
     ["unstabilized approach"]),
    ("a failure of the landing gear, main gear strut",
     [STRUT]),
    ("a failure of the landing gear, main gear strut and the landing gear, "
     "emergency extension assembly",
     [STRUT, "landing gear, emergency extension assembly"]),
    ("a failure of the landing gear, main gear strut, the landing gear, "
     "emergency extension assembly and the landing gear, gear locking "
     "mechanism",
     [STRUT, "landing gear, emergency extension assembly",
      "landing gear, gear locking mechanism"]),
    ("a failure of the landing gear, main gear strut, the landing gear, "
     "emergency extension assembly, the landing gear, gear locking mechanism "
     "and the landing gear, main gear attachment",
     [STRUT, "landing gear, emergency extension assembly",
      "landing gear, gear locking mechanism",
      "landing gear, main gear attachment"]),
]


def main():
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]

    n_same = 0
    for sentence, clicked in SCEN:
        p = qb.parse_query_to_bn_evidence(sentence, names, dataset=ds,
                                          semantic=True)
        parsed = sorted(p["evidence"])
        same = parsed == sorted(clicked)
        confs = {k: round(v, 3) for k, v in p["confidence"].items()}
        n_same += same
        print(f'SENTENCE : "{sentence}"')
        print(f"  clicked codes : {sorted(clicked)}")
        print(f"  parsed codes  : {parsed}")
        print(f"  confidences   : {confs}")
        print(f"  SAME CODES?   : {'YES' if same else 'NO  <-- differs'}")
        print()
    print(f"{n_same}/{len(SCEN)} scenarios: parsed codes == clicked codes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Run structural prognosis evaluation with explicit presets (A0 / A1 / A2).

Forwards to Testing_Structural_Mapping/scripts/eval_prognosis_structural.py.

  python data/Testing_Data_Metrics/scripts/run_structural_prognosis_eval.py a0 --n 20
  python data/Testing_Data_Metrics/scripts/run_structural_prognosis_eval.py a1 --n 20 --alpha 0.5
  python data/Testing_Data_Metrics/scripts/run_structural_prognosis_eval.py a2 --n 20 --alpha 2.0

A1/A2 require --structural in the underlying script; this wrapper adds it for a1/a2.
See data/Testing_Data_Metrics/STRUCTURAL_MAPPING_EVAL.md.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_EVAL = _REPO / "Testing_Structural_Mapping" / "scripts" / "eval_prognosis_structural.py"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Preset wrapper for eval_prognosis_structural.py (a0/a1/a2).",
    )
    ap.add_argument("preset", choices=("a0", "a1", "a2"))
    ap.add_argument("forward_args", nargs=argparse.REMAINDER, help="Args for eval script")
    args = ap.parse_args()
    extra = list(args.forward_args)
    if extra and extra[0] == "--":
        extra = extra[1:]

    cmd = [sys.executable, str(_EVAL)]
    if args.preset == "a0":
        cmd += extra
    elif args.preset == "a1":
        cmd += ["--structural", "--struct-version", "v1"] + extra
    else:
        cmd += ["--structural", "--struct-version", "v2"] + extra

    os.chdir(_REPO)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run steps 1–5 (+ audit) in order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def run(name: str) -> None:
    path = SCRIPTS / name
    print(f"\n{'='*60}\nRunning {name}\n{'='*60}")
    subprocess.check_call([sys.executable, str(path)])


def main() -> None:
    for step in (
        "step01_label_incidents.py",
        "step02_table4_cpt.py",
        "step05_table4_cpt_restricted.py",
        "step06_keyword_vs_struct.py",
        "step03_table5_cpt.py",
        "step04_audit_sample.py",
        "step07_diagnosis_vs_zhang.py",
        "step08_doubt_register.py",
        "step09_recreate_zhang.py",
        "build_html_report.py",
    ):
        run(step)
    print("\nDone — self-contained run (all inputs vendored under data/).")
    print("  MASTER report   → outputs/maha_probability_comparison.html")
    print("  Diagnosis vs Zhang → outputs/step07_diagnosis_vs_zhang.md")
    print("  Doubt register  → outputs/step08_doubt_register.md")


if __name__ == "__main__":
    main()

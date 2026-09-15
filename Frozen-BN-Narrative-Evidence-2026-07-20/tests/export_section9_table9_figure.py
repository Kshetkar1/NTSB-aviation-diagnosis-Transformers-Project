#!/usr/bin/env python3
"""Paper figure for §9.2: BN posterior readout (Table 9 engine-instruments scenario).

This is NOT the Markov prognosis tree (trees.build_prognosis_tree). It visualizes
the same estimand as §9.2 / Table 8: hard evidence on engine instrument → frozen BN
→ downstream posteriors (verified in ALL_TABLES_EXACT_COMPARISON.md).
"""
from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

FROZEN_DIR = Path(__file__).resolve().parents[1]
OUT = FROZEN_DIR / "docs_FrozenBN" / "figures" / "tree_table9_engine_instrument_bn.png"

# Upgraded network, narrative-driven column (matches §9.2 Table 8).
ROWS = [
    ("Loss of engine power", 0.950164),
    ("Forced landing", 0.135738),
    ("Substantial aircraft damage", 0.051214),
    ("Serious injury", 0.076016),
    ("No injury", 0.889034),
]


def _wrap(s: str, w: int = 22) -> str:
    return "\\n".join(textwrap.wrap(s, width=w))


def build_dot() -> str:
    lines = [
        "digraph G {",
        '  rankdir=LR;',
        '  bgcolor="white";',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];',
        '  edge [fontname="Helvetica", fontsize=10, color="#8e44ad"];',
        '  query [label="Query\\nTrouble with an engine\\ninstrument during the flight.", '
        'fillcolor="#ecf0f1", fontcolor="#2c3e50"];',
        '  parse [label="Hard evidence\\nengine instrument = Yes\\n(c = 1.0)", '
        'fillcolor="#f39c12", fontcolor="white"];',
        '  bn [label="Frozen BN\\n(one propagation)", '
        'fillcolor="#8e44ad", fontcolor="white"];',
        "  query -> parse -> bn;",
    ]
    for i, (label, prob) in enumerate(ROWS):
        nid = f"t{i}"
        fill = "#e67e22" if i == 0 else "#2c3e50"
        fc = "white"
        lines.append(
            f'  {nid} [label="{_wrap(label)}\\nP = {prob:.3f}", '
            f'fillcolor="{fill}", fontcolor="{fc}"];'
        )
        lines.append(f'  bn -> {nid} [label="posterior"];')
    lines.append("}")
    return "\n".join(lines)


def main() -> int:
    dot = build_dot()
    dot_path = OUT.with_suffix(".dot")
    dot_path.write_text(dot, encoding="utf-8")
    for cmd in (
        ["dot", "-Tpng", str(dot_path), "-o", str(OUT)],
        ["/opt/homebrew/bin/dot", "-Tpng", str(dot_path), "-o", str(OUT)],
    ):
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Wrote {OUT}")
            return 0
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    print(f"Install graphviz (brew install graphviz). DOT saved: {dot_path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

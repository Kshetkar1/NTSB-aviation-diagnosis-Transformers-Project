#!/usr/bin/env python3
"""Export diagnosis tree PNG using the Streamlit app's Graphviz styling.

The old build_diagnosis_figure.py matplotlib export used drop_generic=True, which
removes Zhang Table 7's top cause (airframe/component failure). This script
matches frozenbn_streamlit_diagnosis_prognosis_demo.py defaults except
drop_generic=False so level-1 matches Table 7.

Usage:
  python3.11 tests/export_streamlit_diagnosis_tree.py
  python3.11 tests/export_streamlit_diagnosis_tree.py --query "..." --query-pool
  python3.11 tests/export_streamlit_diagnosis_tree.py --out docs_FrozenBN/figures/tree_diagnosis_fire.png
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "shared" / "code", FROZEN_DIR / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NTSB_USE_WINDOW_INDEX", "1")

import trees  # noqa: E402
import zhang_diagnosis as zd  # noqa: E402

DEFAULT_QUERY = "engine caught fire during takeoff"
DEFAULT_OUT = FROZEN_DIR / "docs_FrozenBN" / "figures" / "tree_diagnosis_streamlit.png"

_DIAG_PALETTE = {"root": "#c0392b", "node": "#2980b9", "edge": "#c0392b", "leaf": "#7f8c8d"}


def _esc(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def _wrap(label: str, width: int = 26) -> str:
    return "\\n".join(textwrap.wrap(str(label), width=width)) or str(label)


def tree_to_dot(result: dict) -> str:
    """Same Graphviz DOT as frozenbn_streamlit_diagnosis_prognosis_demo.tree_to_dot."""
    meta = result.get("meta", {})
    root = result.get("tree")
    kind = meta.get("kind", "diagnosis")
    pal = _DIAG_PALETTE
    lines = [
        "digraph G {",
        "  rankdir=LR;",
        '  bgcolor="white";',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", '
        'fontsize=11, color="#34495e", penwidth=1.2];',
        f'  edge [fontname="Helvetica", fontsize=10, color="{pal["edge"]}", '
        'fontcolor="#2c3e50", penwidth=1.4];',
    ]

    def emit(node):
        nid = node["id"]
        is_root = node["depth"] == 0
        fill = pal["root"] if is_root else pal["node"]
        if is_root:
            denom = node.get("denom")
            extra = f"\\n(OUTCOME · N={denom})" if denom is not None else "\\n(OUTCOME)"
            label = _wrap(node["label"]) + extra
        else:
            label = _wrap(node["label"]) + f"\\npath p={node['path_prob']:.3f}"
        lines.append(
            f'  {nid} [label="{_esc(label)}", fillcolor="{fill}", fontcolor="white"];'
        )
        for c in node["children"]:
            supp = (
                f"{c['n']}/{c['denom']}"
                if c.get("n") is not None and c.get("denom")
                else ""
            )
            elabel = f"p={c['edge_prob']:.2f}" + (f"\\n{supp}" if supp else "")
            attrs = [f'label="{_esc(elabel)}"']
            if c.get("source") == "global-backoff":
                attrs.append('style="dashed"')
            lines.append(f'  {nid} -> {c["id"]} [{", ".join(attrs)}];')
            emit(c)

    emit(root)
    lines.append("}")
    return "\n".join(lines)


def export_png(dot: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path = out_path.with_suffix(".dot")
    dot_path.write_text(dot, encoding="utf-8")
    for cmd in (
        ["dot", "-Tpng", str(dot_path), "-o", str(out_path)],
        ["/opt/homebrew/bin/dot", "-Tpng", str(dot_path), "-o", str(out_path)],
    ):
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Wrote {out_path}")
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    raise SystemExit(
        "Graphviz 'dot' not found. Install: brew install graphviz\n"
        f"DOT source saved to {dot_path}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument(
        "--query-pool",
        action="store_true",
        help="Count over retrieved pool (Streamlit Table 7 population OFF)",
    )
    ap.add_argument("--top-n-incidents", type=int, default=300)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    import main_app  # noqa: E402

    if not getattr(main_app, "DATA_LOADED", False):
        raise SystemExit("main_app dataset not loaded")

    # Streamlit defaults except drop_generic=False (keep airframe 32/102 at L1).
    result = trees.build_diagnosis_tree(
        args.query,
        top_n_incidents=args.top_n_incidents,
        branching=4,
        depth=2,
        min_prob=0.03,
        min_n=2,
        drop_generic=False,
        main_app=main_app,
        cause_factor_only=True,
        exclude_responses=True,
        full_population=not args.query_pool,
    )
    if result["meta"].get("error") or result["tree"] is None:
        raise SystemExit(result["meta"].get("error", "no tree"))

    meta = result["meta"]
    print(f"Query: {args.query!r}")
    print(f"Population: {'query pool' if args.query_pool else 'Table 7 (all 102 fires)'}")
    print(f"Denominator N={meta['outcome_count_in_pool']} nodes={meta['n_nodes']}")

    # Level-1 sanity for paper
    l1 = [
        (c["label"], c["edge_prob"], c.get("n"), c.get("denom"))
        for c in result["tree"].get("children") or []
    ]
    print("Level 1:")
    for row in l1:
        print(f"  {row[0][:50]!r} p={row[1]:.4f} n/d={row[2]}/{row[3]}")

    sidecar = args.out.with_suffix(".json")
    sidecar.write_text(json.dumps(result, indent=2), encoding="utf-8")
    export_png(tree_to_dot(result), args.out)
    print(f"Meta JSON: {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export prognosis tree PNG using the Streamlit app's Graphviz styling.

Matches frozenbn_streamlit_diagnosis_prognosis_demo.py Prognosis-mode defaults:
  branching=3, depth=3, min_prob=0.05, min_n=2,
  Zhang population ON, damage/injury leaves ON, drop_generic ON.

Usage:
  python3.11 tests/export_streamlit_prognosis_tree.py
  python3.11 tests/export_streamlit_prognosis_tree.py --query "loss of engine power during cruise"
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

DEFAULT_QUERY = "engine caught fire during takeoff"
DEFAULT_OUT = FROZEN_DIR / "docs_FrozenBN" / "figures" / "tree_prognosis_streamlit.png"

_DIAG_PALETTE = {"root": "#c0392b", "node": "#2980b9", "edge": "#c0392b", "leaf": "#7f8c8d"}
_PROG_PALETTE = {"root": "#8e44ad", "node": "#e67e22", "edge": "#8e44ad", "leaf": "#2c3e50"}


def _esc(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def _wrap(label: str, width: int = 26) -> str:
    return "\\n".join(textwrap.wrap(str(label), width=width)) or str(label)


def tree_to_dot(result: dict) -> str:
    """Same Graphviz DOT as frozenbn_streamlit_diagnosis_prognosis_demo.tree_to_dot."""
    meta = result.get("meta", {})
    root = result.get("tree")
    kind = meta.get("kind", "prognosis")
    pal = _DIAG_PALETTE if kind == "diagnosis" else _PROG_PALETTE
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
        is_leaf_outcome = (
            kind == "prognosis" and node.get("kind") == "outcome" and not is_root
        )
        fill = (
            pal["root"]
            if is_root
            else pal["leaf"]
            if is_leaf_outcome
            else pal["node"]
        )
        if is_root:
            head = "OUTCOME" if kind == "diagnosis" else "INITIAL EVENT"
            denom = node.get("denom")
            extra = f"\\n({head} · N={denom})" if denom is not None else f"\\n({head})"
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
            if kind == "prognosis" and c.get("kind") == "outcome":
                attrs.append(f'color="{pal["leaf"]}"')
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
    ap.add_argument("--top-n-incidents", type=int, default=300)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    import main_app  # noqa: E402

    if not getattr(main_app, "DATA_LOADED", False):
        raise SystemExit("main_app dataset not loaded")

    # Streamlit Prognosis-mode defaults (Zhang population ON).
    result = trees.build_prognosis_tree(
        args.query,
        top_n_incidents=args.top_n_incidents,
        branching=3,
        depth=3,
        min_prob=0.05,
        min_n=2,
        query_relevant=False,
        main_app=main_app,
        dataset=main_app.refined_dataset,
        add_outcome_leaves=True,
        drop_generic=True,
        deep_backoff=False,
    )
    if result["meta"].get("error") or result["tree"] is None:
        raise SystemExit(result["meta"].get("error", "no tree"))

    meta = result["meta"]
    print(f"Query: {args.query!r}")
    print(f"Seed: {meta.get('seed_event')}")
    print(f"Population: {meta.get('transition_population')} "
          f"(n={meta.get('incidents_in_population')}) nodes={meta.get('n_nodes')}")

    sidecar = args.out.with_suffix(".json")
    sidecar.write_text(json.dumps(result, indent=2), encoding="utf-8")
    export_png(tree_to_dot(result), args.out)
    print(f"Meta JSON: {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

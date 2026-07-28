#!/usr/bin/env python3
"""Regenerate docs/figures/tree_diagnosis_fire.png from the real diagnosis tree.

Uses faithful Table-7 counting (contributory Cause/Factor findings only) over all
102 fire accidents, with response labels (evacuation, emergency procedure, …)
excluded so level-1 branches are upstream causes, not downstream consequences.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 tests/build_diagnosis_figure.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
_SHARED = REPO_ROOT / "shared" / "code"
_FROZEN_CODE = FROZEN_DIR / "code"
for _p in (_SHARED, _FROZEN_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
ROOT = REPO_ROOT
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import trees
import zhang_diagnosis as zd

OUT = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "docs_FrozenBN" / "figures" / "tree_diagnosis_fire.png"
QUERY = "engine caught fire during takeoff"
DATASET = ROOT / "shared" / "data" / "processed" / "refined_dataset_1982_2006.json"

MPL_NAVY = "#0B2E59"
MPL_ACCENT = "#1F6FB2"
MPL_CARD = "#EEF3F8"
MPL_EDGE = "#BBC7D4"


def _short(label, n=34):
    return label if len(label) <= n else label[: n - 1] + "\u2026"


def _count_leaves(node):
    kids = node.get("children") or []
    if not kids:
        return 1
    return sum(_count_leaves(k) for k in kids)


def _tree_node_to_figure(node):
    """Convert internal tree node to the flat dict render_tree_figure expects."""
    return {
        "label": node["label"],
        "edge_prob": node["edge_prob"],
        "path_prob": node["path_prob"],
        "n": node.get("n"),
        "denom": node.get("denom"),
        "depth": node.get("depth", 0),
        "children": [_tree_node_to_figure(c) for c in node.get("children") or []],
    }


def render_tree_figure(root, out_path, title, root_caption):
    leaf_count = _count_leaves(root)
    fig_h = max(4.2, 0.62 * leaf_count)
    fig, ax = plt.subplots(figsize=(11.5, fig_h))
    ax.set_axis_off()

    x_gap = 3.55
    leaf_cursor = [0]
    positions = {}

    def assign(node, depth):
        kids = node.get("children") or []
        if not kids:
            y = leaf_cursor[0]
            leaf_cursor[0] += 1
        else:
            ys = [assign(k, depth + 1) for k in kids]
            y = sum(ys) / len(ys)
        positions[id(node)] = (depth * x_gap, y)
        return y

    assign(root, 0)

    def color_for(depth):
        if depth == 0:
            return MPL_NAVY, "white"
        if depth == 1:
            return MPL_ACCENT, "white"
        return MPL_CARD, "#1A1A1A"

    def draw(node, depth, parent_xy=None):
        x, y = positions[id(node)]
        if parent_xy is not None:
            px, py = parent_xy
            ax.plot([px + 1.45, x - 0.02], [py, y], color=MPL_EDGE, lw=1.3,
                    zorder=1, solid_capstyle="round")
        face, txt = color_for(depth)
        box_w, box_h = 2.95, 0.62
        box = FancyBboxPatch((x, y - box_h / 2), box_w, box_h,
                             boxstyle="round,pad=0.02,rounding_size=0.08",
                             linewidth=1.0, edgecolor=MPL_EDGE, facecolor=face,
                             zorder=2)
        ax.add_patch(box)
        if depth == 0:
            label = f"{node['label'].upper()}\n{root_caption}"
            ax.text(x + box_w / 2, y, label, ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=txt, zorder=3)
        else:
            p = node.get("edge_prob", 0.0)
            n = node.get("n")
            denom = node.get("denom")
            supp = f"{n}/{denom}" if n is not None and denom is not None else f"{denom}"
            head = f"\u2190 {_short(node['label'])}"
            sub = f"p={p:.3f}   N={supp}   path={node.get('path_prob', 0.0):.3f}"
            ax.text(x + box_w / 2, y + 0.10, head, ha="center", va="center",
                    fontsize=8.0, fontweight="bold", color=txt, zorder=3)
            ax.text(x + box_w / 2, y - 0.15, sub, ha="center", va="center",
                    fontsize=6.8, color=txt, zorder=3)
        for k in (node.get("children") or []):
            draw(k, depth + 1, (x, y))

    draw(root, 0)

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    ax.set_xlim(min(xs) - 0.3, max(xs) + 3.4)
    ax.set_ylim(min(ys) - 0.7, max(ys) + 0.9)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=12, fontweight="bold", color=MPL_NAVY, loc="left")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ds = json.loads(DATASET.read_text(encoding="utf-8"))
    result = trees.build_diagnosis_tree(
        QUERY, branching=4, depth=2, min_prob=0.02, drop_generic=True,
        dataset=ds, cause_factor_only=True, exclude_responses=True,
        full_population=True,
    )
    root = result["tree"]
    if root is None:
        raise SystemExit(result["meta"].get("error", "no tree"))

    # Sanity: no response labels at level 1
    for child in root.get("children") or []:
        if child["label"].lower() in zd.DIAGNOSIS_RESPONSE_LABELS:
            raise SystemExit(f"response label leaked to L1: {child['label']!r}")

    denom = result["meta"]["outcome_count_in_pool"]
    fig_root = _tree_node_to_figure(root)
    render_tree_figure(
        fig_root, str(OUT),
        title=f'Diagnosis tree  \u2014  query: "{QUERY}"',
        root_caption=f"N={denom} fire accidents  (upstream causes only)",
    )
    print(f"Wrote {OUT}")
    print(trees.render_tree(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

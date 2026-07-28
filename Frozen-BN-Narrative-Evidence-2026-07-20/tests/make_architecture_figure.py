#!/usr/bin/env python3
"""One-picture answer to "where do the narratives enter?".

BUILD TIME (once, frozen):
  row A  coded NTSB records -> Zhang Section 4 recipe -> the Bayesian network
  row B  accident narratives -> embedding index + clusters
QUERY TIME (every typed sentence), two paths:
  COUNTING path  similarity -> cluster weights -> law of total probability
  NETWORK path   two readers (deterministic parser, LLM) -> evidence with
                 measured strengths -> frozen network -> joint posterior

The narrative NEVER touches the network's construction; it only supplies
evidence at query time. Writes docs/figures/narrative_to_bn_architecture.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
_SHARED = REPO_ROOT / "shared" / "code"
_FROZEN_CODE = FROZEN_DIR / "code"
for _p in (_SHARED, _FROZEN_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
ROOT = REPO_ROOT
OUT = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "docs_FrozenBN" / "figures" / "narrative_to_bn_architecture.png"

BLUE = "#1f77b4"
DARKBLUE = "#0d3d63"
GREEN = "#2ca02c"
DARKGREEN = "#14571a"
ORANGE = "#ff7f0e"
GRAY = "#666666"


def box(ax, x, y, w, h, text, fc, fontsize=9.5, tc="white", bold=True):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.08",
        fc=fc, ec="none", zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=tc, zorder=3,
            fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, color=GRAY, ls="-", lw=2.2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
        color=color, lw=lw, linestyle=ls, zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(12.6, 8.6))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 9.4)
    ax.axis("off")

    # ---------------- punchline banner ---------------------------------------
    ax.text(6.3, 9.15,
            "The narrative never touches the network's construction -- "
            "it enters ONLY as evidence at query time.",
            fontsize=11, ha="center", color="#333333",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fff3cd", ec="#e0c060"))

    # ---------------- BUILD TIME ----------------------------------------------
    ax.text(0.15, 8.55, "BUILD TIME  (once - frozen 1982-2006 window)",
            fontsize=12, fontweight="bold", color=BLUE)

    # row A: coded records -> recipe -> network
    box(ax, 0.3, 7.3, 2.5, 1.0,
        "Coded NTSB records\noccurrences · findings\npersons · severity", BLUE)
    box(ax, 3.6, 7.3, 3.1, 1.0,
        "Zhang Section 4 recipe\npriors · edge ratios\nBeta-CDF CPTs · 12-parent cap",
        BLUE)
    box(ax, 8.6, 6.2, 3.6, 2.1,
        "BAYESIAN NETWORK\n(frozen)\n785 nodes · exact inference\n"
        "+ person findings\n+ multi-state severity", DARKBLUE, fontsize=10.5)
    arrow(ax, 2.8, 7.8, 3.6, 7.8, BLUE)
    arrow(ax, 6.7, 7.8, 8.6, 7.6, BLUE)
    ax.text(7.6, 8.05, "validated against\n93 published values",
            fontsize=8, color=BLUE, ha="center", style="italic")

    # row B: narratives -> embeddings + clusters
    box(ax, 0.3, 5.9, 2.5, 0.95,
        "Accident narratives\n(1,742, text)", BLUE)
    box(ax, 3.6, 5.9, 3.1, 0.95,
        "EMBEDDING INDEX + CLUSTERS\nevery narrative a meaning vector\n"
        "accidents grouped into families", BLUE, fontsize=8.5)
    arrow(ax, 2.8, 6.4, 3.6, 6.4, BLUE)

    # ---------------- separator -----------------------------------------------
    ax.axhline(5.45, color="#cccccc", lw=1, ls="--")
    ax.text(0.15, 5.1, "QUERY TIME  (every typed sentence)",
            fontsize=12, fontweight="bold", color=GREEN)

    # the typed query
    box(ax, 0.3, 2.1, 2.1, 1.9,
        "TYPED QUERY /\nNARRATIVE\n\"the windshield was\ncracking...\"",
        GREEN, fontsize=9.5)

    # ---------------- counting path -------------------------------------------
    ax.text(2.9, 4.75, "COUNTING PATH  (one conditional at a time)",
            fontsize=9.5, fontweight="bold", color=DARKGREEN)
    box(ax, 2.9, 3.7, 2.7, 0.85,
        "similarity\nembedding scores\nevery accident", GREEN, fontsize=8.5)
    box(ax, 6.1, 3.7, 2.6, 0.85,
        "cluster weights\nhow loud each\nfamily votes", GREEN, fontsize=8.5)
    box(ax, 9.2, 3.7, 3.1, 0.85,
        "LAW OF TOTAL PROBABILITY\nP(cause | query) =\n"
        "sum  P(cause | cluster) x weight", DARKGREEN, fontsize=8.5)
    arrow(ax, 2.4, 3.7, 2.9, 4.0, GREEN)
    arrow(ax, 5.6, 4.1, 6.1, 4.1, GREEN)
    arrow(ax, 8.7, 4.1, 9.2, 4.1, GREEN)
    # embeddings reused
    arrow(ax, 5.15, 5.9, 4.3, 4.55, BLUE, ls=":", lw=1.8)
    ax.text(4.05, 5.2, "reused", fontsize=8, color=BLUE, rotation=55)

    # ---------------- network path --------------------------------------------
    ax.text(2.9, 2.95, "NETWORK PATH  (several facts at once, what-ifs)",
            fontsize=9.5, fontweight="bold", color=ORANGE)
    box(ax, 2.9, 1.35, 2.9, 1.35,
        "TWO READERS\n1. deterministic parser\n(facts NAMED)\n"
        "2. LLM (facts only\nDESCRIBED)", ORANGE, fontsize=8.5)
    box(ax, 6.3, 1.35, 3.3, 1.35,
        "EVIDENCE, strengths from DATA\nHARD: named -> 100%\n"
        "SOFT: measured in 100 most\nsimilar accidents\n"
        "STATED severity: calibrated", ORANGE, fontsize=8.5)
    box(ax, 10.1, 1.5, 2.2, 1.1,
        "JOINT POSTERIOR\nP(cause | evidence)\nP(severity | evidence)",
        DARKGREEN, fontsize=8.8)
    arrow(ax, 2.4, 2.6, 2.9, 2.2, GREEN)
    arrow(ax, 5.8, 2.0, 6.3, 2.0, ORANGE)
    # evidence up to the frozen network, posterior back down
    arrow(ax, 8.6, 2.7, 9.6, 6.2, GREEN, ls="--")
    ax.text(8.5, 4.6, "evidence in", fontsize=8.5, color=GREEN, rotation=74)
    arrow(ax, 11.2, 6.2, 11.2, 2.6, DARKGREEN, ls="--")
    ax.text(11.35, 4.4, "propagation", fontsize=8.5, color=DARKGREEN,
            rotation=-90)
    # embeddings reused for soft strengths
    arrow(ax, 5.7, 5.9, 7.3, 2.7, BLUE, ls=":", lw=1.8)
    ax.text(6.75, 4.35, "reused", fontsize=8, color=BLUE, rotation=-62)

    ax.set_ylim(0.9, 9.55)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

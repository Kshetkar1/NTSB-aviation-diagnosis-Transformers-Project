"""One-slide figure: why Beta-CDF (not narrative) for sparse forward cells.

Output: docs/figures/sparse_decision_summary.png (+ .svg)
Uses PIL only (no matplotlib) for fast generation.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
_SHARED = REPO_ROOT / "shared" / "code"
_FROZEN_CODE = FROZEN_DIR / "code"
for _p in (_SHARED, _FROZEN_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
ROOT = REPO_ROOT
RESULTS = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "docs_FrozenBN" / "sparse_robustness_results.json"
OUT_PNG = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "docs_FrozenBN" / "figures" / "sparse_decision_summary.png"
OUT_SVG = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "docs_FrozenBN" / "figures" / "sparse_decision_summary.svg"

COLORS = {
    "raw": "#888888",
    "cap": "#d62728",
    "beta_cdf": "#1f77b4",
    "semantic": "#2ca02c",
}
SHORT = {"raw": "Raw", "cap": "Cap", "beta_cdf": "Beta-CDF", "semantic": "Narrative"}
METHODS = ["raw", "cap", "beta_cdf", "semantic"]

LOO_BRIER = {"raw": 0.148, "cap": 0.148, "beta_cdf": 0.171, "semantic": 0.167}
LOO_LL = {"raw": 0.471, "cap": 0.471, "beta_cdf": 0.530, "semantic": 0.600}


def load_mae():
    mae = json.loads(RESULTS.read_text())["summary_error"]["mae"]
    return {m: (mae[m]["1"]["mean"], mae[m]["10"]["mean"]) for m in METHODS}


def bar_chart(draw, x0, y0, w, h, values, labels, colors, title, subtitle, ymax, note=None):
    draw.text((x0, y0), title, fill="#111", font=_font(14, bold=True))
    draw.text((x0, y0 + 20), subtitle, fill="#444", font=_font(11))
    base = y0 + h + 50
    left, bw = x0 + 50, (w - 100) / len(values) * 0.55
    gap = (w - 100) / len(values)
    for i, (v, lab, col) in enumerate(zip(values, labels, colors)):
        cx = left + i * gap + gap * 0.15
        bh = int((v / ymax) * (h - 20))
        draw.rectangle([cx, base - bh, cx + bw, base], fill=col, outline="#333")
        draw.text((cx, base + 4), lab, fill="#222", font=_font(10))
        draw.text((cx, base - bh - 16), f"{v:.2f}", fill="#222", font=_font(9))
    draw.line([x0 + 40, base, x0 + w - 20, base], fill="#333", width=2)
    draw.text((x0, base + 28), "lower = better", fill="#666", font=_font(10))
    if note:
        draw.text((x0, y0 + h + 62), note, fill=COLORS["semantic"], font=_font(10))


def grouped_bars(draw, x0, y0, w, h, n1, n10, ymax, title, subtitle, note):
    draw.text((x0, y0), title, fill="#111", font=_font(14, bold=True))
    draw.text((x0, y0 + 20), subtitle, fill="#444", font=_font(11))
    base = y0 + h + 50
    gap = (w - 100) / len(METHODS)
    bw = gap * 0.22
    for i, m in enumerate(METHODS):
        cx = x0 + 50 + i * gap + gap * 0.1
        for j, val in enumerate((n1[m], n10[m])):
            bx = cx + j * (bw + 4)
            bh = int((val / ymax) * (h - 20))
            alpha = 180 if j == 0 else 255
            col = COLORS[m]
            draw.rectangle([bx, base - bh, bx + bw, base], fill=col, outline="#333")
        draw.text((cx, base + 4), SHORT[m], fill="#222", font=_font(10))
    draw.line([x0 + 40, base, x0 + w - 20, base], fill="#333", width=2)
    draw.text((x0 + 50, base + 28), "light = n=1   dark = n=10", fill="#666", font=_font(10))
    draw.text((x0, y0 + h + 62), note, fill=COLORS["semantic"], font=_font(10, bold=True))


def _font(size, bold=False):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default()


def main():
    mae = load_mae()
    W, H = 1200, 520
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    draw.text((20, 8), "Sparse cells: tested narrative → use Zhang Beta-CDF (forward CPT only)",
              fill="#111", font=_font(16, bold=True))
    draw.text((20, 32), "Table 7 diagnosis = plain counting (no Beta-CDF)",
              fill="#555", font=_font(11))

    n1 = {m: mae[m][0] for m in METHODS}
    n10 = {m: mae[m][1] for m in METHODS}
    grouped_bars(
        draw, 20, 55, 560, 220, n1, n10, ymax=0.32,
        title="A. Optimistic test (subsample)",
        subtitle="semantic borrows from whole database",
        note="Green looks best here — unfair vs other methods",
    )

    brier = [LOO_BRIER[m] for m in METHODS]
    bar_chart(
        draw, 610, 55, 560, 220, brier, [SHORT[m] for m in METHODS],
        [COLORS[m] for m in METHODS],
        title="B. Honest test (held-out)",
        subtitle="each incident hidden once — no peeking",
        ymax=0.22,
        note=f"Narrative log-loss worst ({LOO_LL['semantic']:.2f}) — poor calibration",
    )
    # highlight Beta-CDF bar
    gap = (560 - 100) / 4
    bx = 610 + 50 + 2 * gap + gap * 0.15
    draw.rectangle([bx - 2, 55 + 50, bx + gap * 0.55 + 2, 55 + 50 + 220], outline="#003366", width=3)
    draw.text((610 + 50 + 2 * gap, 55 + 50 + 230), "← We use Beta-CDF (Zhang parity)",
              fill="#003366", font=_font(11, bold=True))

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT_PNG, "PNG")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()

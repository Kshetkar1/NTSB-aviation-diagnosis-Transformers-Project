"""
Optimal K Selection: Silhouette + Elbow Sweep
=============================================
Addresses Dr. Mahadevan's question: "Why did you choose K=50 clusters?"

Sweeps K from 10 to 100 and computes three metrics for each value:
  - Silhouette Score     (higher = better)
  - Davies-Bouldin Index (lower = better)
  - Inertia / Elbow     (look for the "elbow" in the curve)

Marks K=50 (production value) on every chart so you can see exactly
where it sits relative to the data-driven optimum.

Output:
  tests/04_optimal_k_sweep.html

Run from project root:
    python tests/04_optimal_k_sweep.py
"""

import sys
import os

os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"]   = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from config import EMBEDDINGS_PATH, EMBEDDINGS_MAP_PATH

OUT_HTML  = Path("tests/04_optimal_k_sweep.html")
K_VALUES  = list(range(10, 105, 5))   # 10, 15, 20, ... 100
PROD_K    = 50
PCA_DIMS  = 100
SEED      = 42

# ---------------------------------------------------------------------------
# Load embeddings
# ---------------------------------------------------------------------------
print("Loading embeddings ...", flush=True)
embeddings = np.load(EMBEDDINGS_PATH)
with open(EMBEDDINGS_MAP_PATH) as f:
    emb_map = json.load(f)

seen, indices = set(), []
for i, m in enumerate(emb_map):
    ev_id = m.get("ev_id")
    if m.get("source") == "incident" and ev_id and ev_id not in seen:
        seen.add(ev_id)
        indices.append(i)

X_full = embeddings[indices]
print(f"  {len(X_full):,} unique incident embeddings", flush=True)

print(f"\nReducing to {PCA_DIMS} dims via PCA ...", flush=True)
pca   = PCA(n_components=PCA_DIMS, random_state=SEED)
X_pca = pca.fit_transform(X_full)
var_explained = pca.explained_variance_ratio_.sum()
print(f"  Variance explained: {var_explained:.1%}", flush=True)

# ---------------------------------------------------------------------------
# Sweep K
# ---------------------------------------------------------------------------
print(f"\nSweeping K = {K_VALUES[0]} to {K_VALUES[-1]} ...", flush=True)

results = []
for k in K_VALUES:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10, max_iter=300)
    labels = km.fit_predict(X_pca)

    sil = silhouette_score(X_pca, labels, sample_size=min(1000, len(X_pca)))
    db  = davies_bouldin_score(X_pca, labels)
    inertia = km.inertia_

    results.append({"k": k, "sil": sil, "db": db, "inertia": inertia})
    marker = " ← PRODUCTION" if k == PROD_K else ""
    print(f"  K={k:3d}  Silhouette={sil:.4f}  DB={db:.4f}  Inertia={inertia:.1f}{marker}", flush=True)

# Find optima
best_sil_k  = max(results, key=lambda r: r["sil"])["k"]
best_db_k   = min(results, key=lambda r: r["db"])["k"]

# Elbow: find point of maximum curvature in inertia curve
inertias = [r["inertia"] for r in results]
ks       = [r["k"] for r in results]

# Second derivative approximation
def elbow_k(ks, vals):
    if len(ks) < 3:
        return ks[0]
    diffs1 = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
    diffs2 = [diffs1[i+1] - diffs1[i] for i in range(len(diffs1)-1)]
    elbow_idx = int(np.argmax(np.abs(diffs2))) + 1
    return ks[elbow_idx]

elbow = elbow_k(ks, inertias)

print(f"\nBest Silhouette at K={best_sil_k}", flush=True)
print(f"Best Davies-Bouldin at K={best_db_k}", flush=True)
print(f"Elbow (inertia) at K={elbow}", flush=True)
print(f"Production K={PROD_K}", flush=True)

# ---------------------------------------------------------------------------
# Build HTML with inline SVG charts
# ---------------------------------------------------------------------------

def svg_line_chart(data_points, label, color, prod_k, width=700, height=220,
                   good="high", extra_markers=None):
    """
    data_points: list of (k, value)
    good: "high" or "low" — which direction is better
    extra_markers: list of (k, label, color) to annotate
    """
    pad_l, pad_r, pad_t, pad_b = 60, 20, 20, 40
    W = width - pad_l - pad_r
    H = height - pad_t - pad_b

    xs = [p[0] for p in data_points]
    ys = [p[1] for p in data_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    y_pad = (y_max - y_min) * 0.1 or 0.01
    y_min -= y_pad
    y_max += y_pad

    def px(k): return pad_l + (k - x_min) / (x_max - x_min) * W
    def py(v): return pad_t + H - (v - y_min) / (y_max - y_min) * H

    # Build polyline points
    pts = " ".join(f"{px(k):.1f},{py(v):.1f}" for k, v in data_points)

    # Y-axis ticks
    y_ticks = np.linspace(y_min, y_max, 5)
    y_tick_els = "".join(
        f'<text x="{pad_l-6}" y="{py(v)+4:.1f}" text-anchor="end" '
        f'font-size="10" fill="#8b949e">{v:.3f}</text>'
        f'<line x1="{pad_l}" y1="{py(v):.1f}" x2="{pad_l+W}" y2="{py(v):.1f}" '
        f'stroke="#21262d" stroke-width="1"/>'
        for v in y_ticks
    )

    # X-axis ticks (every 10)
    x_tick_els = "".join(
        f'<text x="{px(k):.1f}" y="{pad_t+H+18}" text-anchor="middle" '
        f'font-size="10" fill="#8b949e">{k}</text>'
        for k in xs if k % 10 == 0
    )

    # Production K vertical line
    pk_x = px(prod_k)
    prod_line = (
        f'<line x1="{pk_x:.1f}" y1="{pad_t}" x2="{pk_x:.1f}" y2="{pad_t+H}" '
        f'stroke="#f39c12" stroke-width="1.5" stroke-dasharray="4,3"/>'
        f'<text x="{pk_x+4:.1f}" y="{pad_t+14}" font-size="10" fill="#f39c12">K=50</text>'
    )

    # Extra marker dots + labels
    marker_els = ""
    if extra_markers:
        for mk, ml, mc in extra_markers:
            mv  = dict(data_points).get(mk)
            if mv is None:
                continue
            mx = px(mk)
            my = py(mv)
            marker_els += (
                f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="5" fill="{mc}" stroke="#0d1117" stroke-width="1.5"/>'
                f'<text x="{mx+7:.1f}" y="{my+4:.1f}" font-size="10" fill="{mc}">{ml}</text>'
            )

    # Best value dot
    best_v = max(ys) if good == "high" else min(ys)
    best_k = xs[ys.index(best_v)]
    bx, by = px(best_k), py(best_v)
    best_dot = (
        f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="5" fill="{color}" stroke="#0d1117" stroke-width="1.5"/>'
        f'<text x="{bx+7:.1f}" y="{by+4:.1f}" font-size="10" fill="{color}">best K={best_k}</text>'
    )

    return f"""
<svg width="{width}" height="{height}" style="overflow:visible">
  <text x="{pad_l}" y="14" font-size="12" fill="#e6edf3" font-weight="bold">{label}</text>
  {y_tick_els}
  {x_tick_els}
  <text x="{pad_l + W//2}" y="{pad_t+H+34}" text-anchor="middle"
        font-size="11" fill="#8b949e">Number of clusters (K)</text>
  <polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>
  {prod_line}
  {best_dot}
  {marker_els}
</svg>"""

sil_chart = svg_line_chart(
    [(r["k"], r["sil"]) for r in results],
    "Silhouette Score (higher = better)",
    "#2ecc71", PROD_K, good="high"
)
db_chart = svg_line_chart(
    [(r["k"], r["db"]) for r in results],
    "Davies-Bouldin Index (lower = better)",
    "#e74c3c", PROD_K, good="low"
)
inertia_chart = svg_line_chart(
    [(r["k"], r["inertia"]) for r in results],
    "Inertia / Within-Cluster Sum of Squares (Elbow Method)",
    "#58a6ff", PROD_K, good="low",
    extra_markers=[(elbow, f"elbow K={elbow}", "#9b59b6")]
)

# Table rows
table_rows = ""
for r in results:
    is_prod = r["k"] == PROD_K
    row_style = 'style="background:#1c2128"' if is_prod else ""
    prod_tag  = ' <span style="color:#f39c12;font-size:10px">★ production</span>' if is_prod else ""
    sil_best  = r["k"] == best_sil_k
    db_best   = r["k"] == best_db_k
    sil_col   = "#2ecc71" if sil_best else "#c9d1d9"
    db_col    = "#2ecc71" if db_best else "#c9d1d9"
    table_rows += f"""
    <tr {row_style}>
      <td style="text-align:center;font-weight:bold">K={r['k']}{prod_tag}</td>
      <td style="text-align:center;color:{sil_col}">{'★ ' if sil_best else ''}{r['sil']:.4f}</td>
      <td style="text-align:center;color:{db_col}">{'★ ' if db_best else ''}{r['db']:.4f}</td>
      <td style="text-align:center;color:#8b949e">{r['inertia']:.0f}</td>
    </tr>"""

# Verdict
prod_sil = next(r["sil"] for r in results if r["k"] == PROD_K)
best_sil = max(r["sil"] for r in results)
pct_diff = (best_sil - prod_sil) / best_sil * 100

if best_sil_k == PROD_K:
    verdict = f"K=50 <b>is the optimal choice</b> — it achieves the highest silhouette score ({prod_sil:.4f}) of any K tested."
    verdict_color = "#2ecc71"
elif pct_diff < 5:
    verdict = (
        f"K=50 is within <b>{pct_diff:.1f}%</b> of the peak silhouette score "
        f"(best at K={best_sil_k}, score={best_sil:.4f} vs K=50 score={prod_sil:.4f}). "
        f"The difference is negligible — K=50 is a well-justified choice."
    )
    verdict_color = "#f39c12"
else:
    verdict = (
        f"The data suggests K={best_sil_k} may produce better-separated clusters "
        f"(silhouette={best_sil:.4f} vs K=50={prod_sil:.4f}, a {pct_diff:.1f}% improvement). "
        f"Re-running the pipeline with K={best_sil_k} is recommended."
    )
    verdict_color = "#e74c3c"

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Optimal K Sweep — K-Means Cluster Selection</title>
  <style>
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ font-family:'Segoe UI',Arial,sans-serif; background:#0d1117;
            color:#c9d1d9; padding:32px; max-width:900px; margin:0 auto; }}
    h1   {{ color:#58a6ff; font-size:20px; margin-bottom:6px; }}
    .sub {{ color:#8b949e; font-size:13px; margin-bottom:28px; line-height:1.7; }}
    h2   {{ color:#e6edf3; font-size:15px; margin:32px 0 14px; }}
    .chart-box {{ background:#161b22; border-radius:8px; padding:20px 24px;
                  margin-bottom:24px; overflow-x:auto; }}
    table {{ width:100%; border-collapse:collapse; }}
    th {{ padding:10px 14px; background:#161b22; color:#58a6ff;
          font-size:12px; text-align:left; border-bottom:2px solid #30363d; }}
    td {{ padding:9px 14px; border-bottom:1px solid #21262d; font-size:13px; }}
    tr:hover td {{ background:#161b22; }}
    .verdict {{
      padding:18px 22px; border-radius:6px;
      border-left:4px solid {verdict_color};
      background:#161b22; margin-top:8px; margin-bottom:28px;
    }}
    .verdict-body {{ font-size:14px; color:#c9d1d9; line-height:1.7; }}
  </style>
</head>
<body>
  <h1>Optimal K Selection: Silhouette &amp; Elbow Sweep</h1>
  <p class="sub">
    K-Means was run for every K from 10 to 100 (step 5) on
    <b>{len(X_full):,} NTSB incident embeddings</b> (PCA-{PCA_DIMS}, {var_explained:.1%} variance retained).<br>
    The orange dashed line marks <b>K=50</b> — the value used in production.<br>
    Green dots mark the data-driven best K for each metric.
  </p>

  <h2>Verdict</h2>
  <div class="verdict">
    <div class="verdict-body">{verdict}</div>
  </div>

  <h2>Silhouette Score vs K</h2>
  <div class="chart-box">{sil_chart}</div>

  <h2>Davies-Bouldin Index vs K</h2>
  <div class="chart-box">{db_chart}</div>

  <h2>Elbow Curve (Inertia) vs K</h2>
  <div class="chart-box">{inertia_chart}</div>

  <h2>Full Results Table</h2>
  <table>
    <thead>
      <tr>
        <th>K</th>
        <th style="text-align:center">Silhouette ↑</th>
        <th style="text-align:center">Davies-Bouldin ↓</th>
        <th style="text-align:center">Inertia ↓</th>
      </tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>
</body>
</html>"""

OUT_HTML.write_text(html, encoding="utf-8")
print(f"\nSaved: {OUT_HTML}", flush=True)

import subprocess
try:
    subprocess.Popen(["open", str(OUT_HTML)])
except Exception:
    import webbrowser
    webbrowser.open(OUT_HTML.resolve().as_uri())

print("Done.", flush=True)

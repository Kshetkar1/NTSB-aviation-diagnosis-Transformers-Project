"""
Clustering Method Comparison: K-Means vs GMM
=============================================
Addresses Dr. Mahadevan's question: "Is K-means the best clustering approach?"

Compares K-Means (used in production) against Gaussian Mixture Models (GMM)
using three standard cluster quality metrics:
  - Silhouette Score     (higher = better, -1 to 1)
  - Davies-Bouldin Index (lower = better, >= 0)
  - Calinski-Harabasz   (higher = better)

Both methods use K=50 clusters.
Embeddings are PCA-reduced to 100 dims for a fair comparison
(GMM is numerically unstable in 1536-dim space with only ~1800 points).

Output:
  tests/03_clustering_comparison.html

Run from project root:
    python tests/03_clustering_comparison.py
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
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from config import REFINED_DATA_PATH, EMBEDDINGS_PATH, EMBEDDINGS_MAP_PATH

OUT_HTML  = Path("tests/03_clustering_comparison.html")
K         = 50
PCA_DIMS  = 100   # reduce for GMM numerical stability
SEED      = 42

# ---------------------------------------------------------------------------
# Load embeddings (incident-only, deduplicated)
# ---------------------------------------------------------------------------
print("Loading embeddings ...", flush=True)
embeddings = np.load(EMBEDDINGS_PATH)
with open(EMBEDDINGS_MAP_PATH) as f:
    emb_map = json.load(f)
with open(REFINED_DATA_PATH) as f:
    dataset = json.load(f)

# Keep only unique incident-level embeddings
seen, indices = set(), []
for i, m in enumerate(emb_map):
    ev_id = m.get("ev_id")
    if m.get("source") == "incident" and ev_id and ev_id not in seen:
        seen.add(ev_id)
        indices.append(i)

X_full = embeddings[indices]
print(f"  {len(X_full):,} unique incident embeddings  (dim={X_full.shape[1]})", flush=True)

# ---------------------------------------------------------------------------
# PCA reduction (fair comparison space for both methods)
# ---------------------------------------------------------------------------
print(f"\nReducing to {PCA_DIMS} dims via PCA ...", flush=True)
pca   = PCA(n_components=PCA_DIMS, random_state=SEED)
X_pca = pca.fit_transform(X_full)
var_explained = pca.explained_variance_ratio_.sum()
print(f"  Variance explained: {var_explained:.1%}", flush=True)

# ---------------------------------------------------------------------------
# K-Means  (production method)
# ---------------------------------------------------------------------------
print(f"\nRunning K-Means (K={K}) ...", flush=True)
km = KMeans(n_clusters=K, random_state=SEED, n_init=10, max_iter=300)
km_labels = km.fit_predict(X_pca)

km_sil = silhouette_score(X_pca, km_labels, sample_size=min(1000, len(X_pca)))
km_db  = davies_bouldin_score(X_pca, km_labels)
km_ch  = calinski_harabasz_score(X_pca, km_labels)

sizes_km = np.bincount(km_labels)
print(f"  Silhouette:         {km_sil:.4f}", flush=True)
print(f"  Davies-Bouldin:     {km_db:.4f}", flush=True)
print(f"  Calinski-Harabasz:  {km_ch:.1f}", flush=True)
print(f"  Cluster sizes:  min={sizes_km.min()}  max={sizes_km.max()}  mean={sizes_km.mean():.1f}", flush=True)

# ---------------------------------------------------------------------------
# GMM  (alternative)
# ---------------------------------------------------------------------------
print(f"\nRunning GMM (K={K}, covariance=diag) ...", flush=True)
gmm = GaussianMixture(
    n_components=K,
    covariance_type="diag",   # diagonal: stable in high dims
    random_state=SEED,
    max_iter=200,
    n_init=3,
)
gmm.fit(X_pca)
gmm_labels = gmm.predict(X_pca)
gmm_probs  = gmm.predict_proba(X_pca)   # soft memberships

gmm_sil = silhouette_score(X_pca, gmm_labels, sample_size=min(1000, len(X_pca)))
gmm_db  = davies_bouldin_score(X_pca, gmm_labels)
gmm_ch  = calinski_harabasz_score(X_pca, gmm_labels)

sizes_gmm = np.bincount(gmm_labels)

# GMM-specific: average entropy of soft assignments (lower = more confident)
entropy = -np.sum(gmm_probs * np.log(gmm_probs + 1e-12), axis=1)
avg_entropy = entropy.mean()
max_entropy = np.log(K)   # theoretical max when uniform

print(f"  Silhouette:         {gmm_sil:.4f}", flush=True)
print(f"  Davies-Bouldin:     {gmm_db:.4f}", flush=True)
print(f"  Calinski-Harabasz:  {gmm_ch:.1f}", flush=True)
print(f"  Cluster sizes:  min={sizes_gmm.min()}  max={sizes_gmm.max()}  mean={sizes_gmm.mean():.1f}", flush=True)
print(f"  Avg assignment entropy: {avg_entropy:.3f} / {max_entropy:.3f} (lower = more confident)", flush=True)

# ---------------------------------------------------------------------------
# Build HTML
# ---------------------------------------------------------------------------
def bar(val, min_val, max_val, good="high", width=160):
    """Render a simple CSS progress bar."""
    pct = (val - min_val) / (max_val - min_val + 1e-9)
    if good == "low":
        pct = 1 - pct
    fill = int(pct * width)
    color = "#2ecc71" if pct >= 0.5 else "#f39c12" if pct >= 0.25 else "#e74c3c"
    return (
        f'<div style="background:#21262d;border-radius:4px;width:{width}px;height:14px;display:inline-block">'
        f'<div style="background:{color};width:{fill}px;height:14px;border-radius:4px"></div></div>'
    )

metrics = [
    {
        "name": "Silhouette Score",
        "desc": "Measures how similar each point is to its own cluster vs. other clusters. Range: −1 to 1. Higher is better.",
        "good": "high",
        "km":   km_sil,
        "gmm":  gmm_sil,
        "fmt":  ".4f",
    },
    {
        "name": "Davies-Bouldin Index",
        "desc": "Average similarity between each cluster and its most similar neighbour. Lower is better.",
        "good": "low",
        "km":   km_db,
        "gmm":  gmm_db,
        "fmt":  ".4f",
    },
    {
        "name": "Calinski-Harabasz Score",
        "desc": "Ratio of between-cluster dispersion to within-cluster dispersion. Higher is better.",
        "good": "high",
        "km":   km_ch,
        "gmm":  gmm_ch,
        "fmt":  ".1f",
    },
]

def winner_badge(km_val, gmm_val, good):
    if good == "high":
        km_wins = km_val >= gmm_val
    else:
        km_wins = km_val <= gmm_val
    if abs(km_val - gmm_val) / (abs(gmm_val) + 1e-9) < 0.01:
        return '<span style="color:#8b949e">Tie</span>'
    if km_wins:
        return '<span style="color:#2ecc71;font-weight:bold">K-Means ✓</span>'
    return '<span style="color:#58a6ff;font-weight:bold">GMM ✓</span>'

rows = []
for m in metrics:
    min_v = min(m["km"], m["gmm"])
    max_v = max(m["km"], m["gmm"])
    rows.append(f"""
    <tr>
      <td>
        <b>{m['name']}</b><br>
        <span style="font-size:11px;color:#8b949e">{m['desc']}</span>
      </td>
      <td style="text-align:center">
        <code style="color:#f39c12">{m['km']:{m['fmt']}}</code><br>
        {bar(m['km'], min_v, max_v, m['good'])}
      </td>
      <td style="text-align:center">
        <code style="color:#58a6ff">{m['gmm']:{m['fmt']}}</code><br>
        {bar(m['gmm'], min_v, max_v, m['good'])}
      </td>
      <td style="text-align:center">{winner_badge(m['km'], m['gmm'], m['good'])}</td>
    </tr>""")

rows_html = "\n".join(rows)

# Cluster size distribution rows
size_rows = "".join(
    f'<tr><td style="color:#8b949e;font-size:12px">{label}</td>'
    f'<td style="text-align:center"><code style="color:#f39c12">{km_v}</code></td>'
    f'<td style="text-align:center"><code style="color:#58a6ff">{gmm_v}</code></td></tr>'
    for label, km_v, gmm_v in [
        ("Min cluster size", sizes_km.min(), sizes_gmm.min()),
        ("Max cluster size", sizes_km.max(), sizes_gmm.max()),
        ("Mean cluster size", f"{sizes_km.mean():.1f}", f"{sizes_gmm.mean():.1f}"),
        ("Empty clusters", int((sizes_km == 0).sum()), int((sizes_gmm == 0).sum())),
    ]
)

# Determine overall winner
km_wins_count  = sum(
    1 for m in metrics
    if (m["km"] >= m["gmm"] if m["good"] == "high" else m["km"] <= m["gmm"])
)
gmm_wins_count = len(metrics) - km_wins_count

if km_wins_count > gmm_wins_count:
    verdict_color = "#f39c12"
    verdict = f"K-Means wins {km_wins_count}–{gmm_wins_count} on standard cluster quality metrics."
    verdict_detail = (
        "K-Means is the stronger choice for this dataset. It also has the practical advantage "
        "of hard cluster assignments, which plug directly into the Law of Total Probability "
        "used in the Bayesian diagnosis framework."
    )
elif gmm_wins_count > km_wins_count:
    verdict_color = "#58a6ff"
    verdict = f"GMM wins {gmm_wins_count}–{km_wins_count} on standard cluster quality metrics."
    verdict_detail = (
        "GMM produces slightly better-separated clusters on this dataset. Its soft probability "
        "assignments could further improve the Bayesian framework by providing richer "
        "P(cluster | incident) weights instead of hard 0/1 membership. "
        "This is a natural future extension of the pipeline."
    )
else:
    verdict_color = "#8b949e"
    verdict = "Methods tied on cluster quality metrics."
    verdict_detail = "Both methods produce comparable cluster quality. K-Means is preferred for simplicity."

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Clustering Comparison: K-Means vs GMM</title>
  <style>
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ font-family:'Segoe UI',Arial,sans-serif; background:#0d1117;
            color:#c9d1d9; padding:32px; max-width:1000px; margin:0 auto; }}
    h1   {{ color:#58a6ff; font-size:20px; margin-bottom:6px; }}
    .sub {{ color:#8b949e; font-size:13px; margin-bottom:28px; line-height:1.7; }}
    h2   {{ color:#e6edf3; font-size:15px; margin:28px 0 12px; }}
    table {{ width:100%; border-collapse:collapse; margin-bottom:32px; }}
    th {{ padding:10px 14px; background:#161b22; color:#58a6ff;
          font-size:12px; text-align:left; border-bottom:2px solid #30363d; }}
    td {{ padding:12px 14px; border-bottom:1px solid #21262d;
          vertical-align:middle; font-size:13px; }}
    tr:hover td {{ background:#161b22; }}
    .verdict {{
      padding:18px 22px; border-radius:6px; border-left:4px solid {verdict_color};
      background:#161b22; margin-top:8px;
    }}
    .verdict-title {{ font-size:16px; font-weight:bold; color:{verdict_color};
                      margin-bottom:6px; }}
    .verdict-body  {{ font-size:13px; color:#c9d1d9; line-height:1.6; }}
    .note {{ background:#161b22; border-radius:6px; padding:14px 18px;
             font-size:12px; color:#8b949e; line-height:1.7; margin-top:24px; }}
  </style>
</head>
<body>
  <h1>Clustering Method Comparison: K-Means vs GMM</h1>
  <p class="sub">
    Both methods use <b>K = {K} clusters</b> on the same dataset of
    <b>{len(X_full):,} NTSB incident embeddings</b>.<br>
    Embeddings are PCA-reduced to <b>{PCA_DIMS} dimensions</b> (explains
    <b>{var_explained:.1%}</b> of variance) for a fair, numerically stable comparison.<br>
    All metrics are computed in the same PCA-{PCA_DIMS} space.
  </p>

  <h2>Cluster Quality Metrics</h2>
  <table>
    <thead>
      <tr>
        <th style="width:40%">Metric</th>
        <th style="width:20%;text-align:center">
          <span style="color:#f39c12">K-Means</span>
          <span style="font-size:10px;color:#8b949e"> (production)</span>
        </th>
        <th style="width:20%;text-align:center">
          <span style="color:#58a6ff">GMM</span>
          <span style="font-size:10px;color:#8b949e"> (alternative)</span>
        </th>
        <th style="width:20%;text-align:center">Winner</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>

  <h2>Cluster Size Distribution</h2>
  <table>
    <thead>
      <tr>
        <th>Stat</th>
        <th style="text-align:center;color:#f39c12">K-Means</th>
        <th style="text-align:center;color:#58a6ff">GMM</th>
      </tr>
    </thead>
    <tbody>
      {size_rows}
    </tbody>
  </table>

  <h2>GMM Soft-Assignment Entropy</h2>
  <p style="font-size:13px;margin-bottom:12px;color:#c9d1d9">
    GMM assigns each incident a probability distribution over all {K} clusters.
    Entropy measures how uncertain these assignments are.
    Low entropy = the model is confident. High entropy = the incident could belong to many clusters.
  </p>
  <table>
    <thead>
      <tr>
        <th>Metric</th>
        <th style="text-align:center">Value</th>
        <th>Interpretation</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Average assignment entropy</td>
        <td style="text-align:center"><code style="color:#58a6ff">{avg_entropy:.3f}</code></td>
        <td style="font-size:12px;color:#8b949e">
          out of max possible {max_entropy:.3f} (uniform over {K} clusters) —
          {avg_entropy/max_entropy:.0%} of maximum uncertainty
        </td>
      </tr>
    </tbody>
  </table>

  <h2>Verdict</h2>
  <div class="verdict">
    <div class="verdict-title">{verdict}</div>
    <div class="verdict-body">{verdict_detail}</div>
  </div>

  <div class="note">
    <b>Note on methodology:</b> K-Means in production runs on full 1,536-dim unit-normalized
    embeddings (equivalent to spherical K-Means optimizing cosine similarity).
    GMM requires PCA reduction for numerical stability in high dimensions.
    Both are evaluated here in the same PCA-{PCA_DIMS} space for a fair metric comparison.
    The slight disadvantage this introduces for K-Means means the production K-Means
    performance is likely <i>at least as good</i> as shown here.
  </div>
</body>
</html>"""

OUT_HTML.write_text(html, encoding="utf-8")
print(f"\nSaved: {OUT_HTML}", flush=True)

import webbrowser, subprocess
try:
    subprocess.Popen(["open", str(OUT_HTML)])
except Exception:
    webbrowser.open(OUT_HTML.resolve().as_uri())

print("Done.", flush=True)

"""
Cluster Validation
==================
Addresses Dr. Mahadevan's request: "Show every incident and which cluster
it ended up in — validate that the clustering makes sense."

Outputs:
  1. Console: cluster summary table (name, size, sample narratives)
  2. tests/01_cluster_validation.html — full interactive breakdown
     with every cluster and its incidents, searchable and filterable.

No API calls needed — uses pre-computed cluster_label in refined_dataset.json.

Run from project root:
    python tests/01_validate_clusters.py
"""

import sys
import os

os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"]   = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
from collections import defaultdict, Counter
from config import REFINED_DATA_PATH


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_clusters() -> dict:
    """Returns {cluster_label: [incident_dict, ...]} sorted by cluster size."""
    print("📂  Loading refined dataset …")
    with open(REFINED_DATA_PATH, "r") as f:
        dataset = json.load(f)
    print(f"    {len(dataset):,} incidents loaded.\n")

    clusters: dict = defaultdict(list)
    no_label = 0

    for ev_id, inc in dataset.items():
        label = inc.get("cluster_label")
        if not label:
            no_label += 1
            label = "⚠️  No cluster assigned"

        narr = (inc.get("narr_cause") or inc.get("narr_accp")
                or inc.get("narr_accf") or "No narrative")

        clusters[label].append({
            "ev_id":    ev_id,
            "narr":     narr,
            "date":     inc.get("ev_date", "")[:10],
            "state":    inc.get("ev_state", ""),
            "aircraft": f"{inc.get('acft_make','')} {inc.get('acft_model','')}".strip(),
            "damage":   inc.get("damage", ""),
            "injury":   inc.get("ev_highest_injury", ""),
        })

    if no_label:
        print(f"⚠️   {no_label} incidents had no cluster_label "
              f"(run 3_precompute_clusters.py to fix).\n")

    # Sort clusters by descending size
    return dict(sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True))


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(clusters: dict) -> None:
    total = sum(len(v) for v in clusters.values())
    W = 42

    print("=" * 70)
    print(f"  CLUSTER VALIDATION SUMMARY  —  {len(clusters)} clusters, {total} incidents")
    print("=" * 70)
    print(f"{'Cluster Name':<{W}} {'Count':>6}  {'%':>5}  Sample narrative (first incident)")
    print("-" * 70)

    for label, incs in clusters.items():
        pct  = len(incs) / total * 100
        sample = incs[0]["narr"][:55].replace("\n", " ")
        if len(incs[0]["narr"]) > 55:
            sample += "…"
        label_disp = (label[:W - 2] + "…") if len(label) > W else label
        print(f"{label_disp:<{W}} {len(incs):>6}  {pct:>4.1f}%  {sample}")

    print("=" * 70)
    print(f"  Total: {total} incidents across {len(clusters)} clusters.\n")


# ---------------------------------------------------------------------------
# HTML report — full breakdown, every incident
# ---------------------------------------------------------------------------

def build_html(clusters: dict, output_path: Path) -> None:
    total = sum(len(v) for v in clusters.values())

    # Colour palette (cycles if > 20 clusters)
    palette = [
        "#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
        "#1abc9c","#e67e22","#d35400","#e91e63","#00bcd4",
        "#ff5722","#607d8b","#8bc34a","#ff9800","#795548",
        "#c0392b","#2980b9","#27ae60","#f1c40f","#8e44ad",
        "#16a085","#e74c3c","#2c3e50","#7f8c8d","#bdc3c7",
    ]
    colour_map = {
        label: palette[i % len(palette)]
        for i, label in enumerate(clusters.keys())
    }

    # ── Cluster cards HTML ──────────────────────────────────────────────────
    cards_html = ""
    for label, incs in clusters.items():
        col  = colour_map[label]
        pct  = len(incs) / total * 100

        rows = ""
        for inc in incs:
            narr_short = inc["narr"][:120].replace("<", "&lt;").replace(">", "&gt;")
            if len(inc["narr"]) > 120:
                narr_short += "…"
            rows += (
                f'<tr>'
                f'<td class="ev">{inc["ev_id"]}</td>'
                f'<td>{inc["date"]}</td>'
                f'<td>{inc["state"]}</td>'
                f'<td>{inc["aircraft"]}</td>'
                f'<td>{inc["damage"]}</td>'
                f'<td>{inc["injury"]}</td>'
                f'<td class="narr">{narr_short}</td>'
                f'</tr>'
            )

        cards_html += f"""
<div class="cluster-card" data-label="{label.lower()}" style="border-left:5px solid {col}">
  <div class="cluster-header" onclick="toggle(this)">
    <span class="badge" style="background:{col}">{len(incs)}</span>
    <strong>{label}</strong>
    <span class="pct">{pct:.1f}% of all incidents</span>
    <span class="chevron">▼</span>
  </div>
  <div class="cluster-body" style="display:none">
    <table>
      <thead>
        <tr>
          <th>Event ID</th><th>Date</th><th>State</th>
          <th>Aircraft</th><th>Damage</th><th>Injury</th><th>Cause narrative</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Cluster Validation — NTSB</title>
  <style>
    *    {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ font-family:'Segoe UI',Arial,sans-serif; background:#0d1117;
            color:#c9d1d9; padding:24px; }}
    h1   {{ color:#58a6ff; font-size:20px; margin-bottom:4px; }}
    .sub {{ color:#8b949e; font-size:13px; margin-bottom:16px; }}
    /* search */
    #search {{ width:100%; padding:10px 14px; font-size:14px; border-radius:6px;
               border:1px solid #30363d; background:#161b22; color:#c9d1d9;
               margin-bottom:14px; }}
    /* cards */
    .cluster-card  {{ background:#161b22; border-radius:6px; margin-bottom:10px;
                      overflow:hidden; }}
    .cluster-header{{ display:flex; align-items:center; gap:10px; padding:10px 14px;
                      cursor:pointer; user-select:none; }}
    .cluster-header:hover {{ background:#21262d; }}
    .badge  {{ border-radius:12px; padding:2px 9px; font-size:12px;
               font-weight:bold; color:#fff; flex-shrink:0; }}
    .pct    {{ color:#8b949e; font-size:12px; margin-left:auto; }}
    .chevron{{ color:#8b949e; font-size:12px; margin-left:8px; }}
    /* table */
    .cluster-body {{ padding:0 14px 14px; overflow-x:auto; }}
    table   {{ width:100%; border-collapse:collapse; font-size:12px; }}
    th      {{ background:#21262d; color:#8b949e; padding:6px 8px;
               text-align:left; position:sticky; top:0; }}
    td      {{ padding:5px 8px; border-bottom:1px solid #21262d;
               vertical-align:top; }}
    td.ev   {{ font-family:monospace; white-space:nowrap; }}
    td.narr {{ max-width:380px; }}
    tr:hover td {{ background:#1c2128; }}
    /* expand all btn */
    .btn {{ padding:6px 14px; background:#21262d; border:1px solid #30363d;
            color:#c9d1d9; border-radius:6px; cursor:pointer; font-size:13px;
            margin-bottom:12px; margin-right:8px; }}
    .btn:hover {{ background:#30363d; }}
    footer {{ margin-top:16px; font-size:11px; color:#6e7681; }}
  </style>
</head>
<body>
  <h1>✈️ Cluster Validation — NTSB Incident Dataset</h1>
  <p class="sub">
    {len(clusters)} clusters &nbsp;·&nbsp; {total:,} incidents &nbsp;·&nbsp;
    Pre-computed cluster labels from K-means (k=50) + GPT-4o-mini naming.
  </p>

  <input id="search" type="text" placeholder="Filter clusters by name … (e.g. engine, bird, fuel)">
  <button class="btn" onclick="expandAll()">Expand All</button>
  <button class="btn" onclick="collapseAll()">Collapse All</button>

  <div id="cards">{cards_html}</div>

  <footer>Source: {REFINED_DATA_PATH.name} &nbsp;·&nbsp; cluster_label field</footer>

  <script>
    function toggle(header) {{
      const body = header.nextElementSibling;
      const chev = header.querySelector('.chevron');
      if (body.style.display === 'none') {{
        body.style.display = 'block'; chev.textContent = '▲';
      }} else {{
        body.style.display = 'none';  chev.textContent = '▼';
      }}
    }}
    function expandAll()  {{ document.querySelectorAll('.cluster-body').forEach(b => {{ b.style.display='block';  b.previousElementSibling.querySelector('.chevron').textContent='▲'; }}); }}
    function collapseAll(){{ document.querySelectorAll('.cluster-body').forEach(b => {{ b.style.display='none';   b.previousElementSibling.querySelector('.chevron').textContent='▼'; }}); }}

    document.getElementById('search').addEventListener('input', function() {{
      const q = this.value.toLowerCase().trim();
      document.querySelectorAll('.cluster-card').forEach(card => {{
        card.style.display = (!q || card.dataset.label.includes(q)) ? '' : 'none';
      }});
    }});
  </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    clusters = load_clusters()
    print_summary(clusters)

    out = Path(__file__).parent / "01_cluster_validation.html"
    print("🌐  Building HTML report …")
    build_html(clusters, out)
    print(f"✅  Saved → {out}")
    print("    Open in any browser — no internet needed.\n")

    try:
        import webbrowser
        webbrowser.open(f"file://{out.absolute()}")
        print("    Opening in browser automatically …")
    except Exception:
        pass


if __name__ == "__main__":
    main()

"""
Incident-to-Incident Similarity Network
=========================================
Addresses Dr. Mahadevan's question: "Are the embeddings actually grouping
similar incidents together, or is the similarity arbitrary?"

No query is used. We load all 2,243 incident embeddings directly and compute
pairwise cosine similarity — pure incident-to-incident comparison.

Outputs:
  tests/02_incident_network.html  — interactive vis.js network
      Nodes  = sampled NTSB incidents, coloured by cluster
      Edges  = pairs with cosine similarity >= EDGE_THRESHOLD
      Hover  = tooltip with ev_id, cluster, similarity score, and narrative

  tests/02_incident_table.html    — searchable table
      Each row = one incident
      Columns = ev_id, cluster, and its top-5 most similar incidents with scores

Run from project root:
    python tests/02_incident_similarity_network.py
"""

import sys
import os

# Prevent NumPy/OpenBLAS from spawning threads that deadlock in restricted envs
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"]   = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from config import REFINED_DATA_PATH, EMBEDDINGS_PATH, EMBEDDINGS_MAP_PATH, EMBEDDING_MODEL

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MAX_PER_CLUSTER  = 8     # incidents sampled per cluster for the network
EDGE_THRESHOLD   = 0.55  # minimum cosine similarity to draw an edge
TOP_K_TABLE      = 5     # how many similar incidents to show per row in the table

PALETTE = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#d35400", "#e91e63", "#00bcd4",
    "#ff5722", "#607d8b", "#8bc34a", "#ff9800", "#795548",
    "#c0392b", "#2980b9", "#27ae60", "#f1c40f", "#8e44ad",
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#d35400", "#e91e63", "#00bcd4",
    "#ff5722", "#607d8b", "#8bc34a", "#ff9800", "#795548",
    "#c0392b", "#2980b9", "#27ae60", "#f1c40f", "#8e44ad",
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#d35400", "#e91e63", "#00bcd4",
]


# ---------------------------------------------------------------------------
# Step 1 — Load data
# ---------------------------------------------------------------------------

def load_data():
    print("Loading embeddings and dataset ...", flush=True)
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(EMBEDDINGS_MAP_PATH, "r") as f:
        emb_map = json.load(f)
    with open(REFINED_DATA_PATH, "r") as f:
        dataset = json.load(f)
    print(f"  {len(embeddings):,} embeddings  |  {len(dataset):,} incidents", flush=True)
    return embeddings, emb_map, dataset


# ---------------------------------------------------------------------------
# Step 2 — Build deduplicated incident list (one embedding per ev_id)
# ---------------------------------------------------------------------------

def build_incident_list(embeddings, emb_map, dataset):
    """
    Returns list of dicts:
        ev_id, cluster_label, narrative, embedding (np.ndarray)
    One entry per unique incident (first embedding wins if multiple exist).
    """
    seen = set()
    incidents = []

    for idx, meta in enumerate(emb_map):
        if meta.get("source") != "incident":
            continue
        ev_id = meta.get("ev_id")
        if not ev_id or ev_id in seen:
            continue

        inc     = dataset.get(ev_id, {})
        cluster = inc.get("cluster_label", "Unknown")
        narr    = (inc.get("narr_cause") or inc.get("narr_accp")
                   or inc.get("narr_accf") or "No narrative available")

        incidents.append({
            "ev_id":   ev_id,
            "cluster": cluster,
            "narr":    narr,
            "emb":     embeddings[idx],
        })
        seen.add(ev_id)

    print(f"  {len(incidents):,} unique incidents extracted", flush=True)
    return incidents


# ---------------------------------------------------------------------------
# Step 3 — Sample incidents (up to MAX_PER_CLUSTER per cluster)
# ---------------------------------------------------------------------------

def sample_incidents(incidents):
    by_cluster = defaultdict(list)
    for inc in incidents:
        by_cluster[inc["cluster"]].append(inc)

    sampled = []
    for cluster, group in sorted(by_cluster.items()):
        sampled.extend(group[:MAX_PER_CLUSTER])

    print(f"  Sampled {len(sampled)} incidents across {len(by_cluster)} clusters "
          f"(up to {MAX_PER_CLUSTER} per cluster)", flush=True)
    return sampled, by_cluster


# ---------------------------------------------------------------------------
# Step 4 — Compute pairwise cosine similarity for sampled set
# ---------------------------------------------------------------------------

def compute_similarity_matrix(incidents):
    emb_matrix = np.array([inc["emb"] for inc in incidents], dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = emb_matrix / norms
    sim_matrix = normed @ normed.T  # shape: (N, N)
    np.fill_diagonal(sim_matrix, 0.0)  # exclude self-similarity
    return sim_matrix


# ---------------------------------------------------------------------------
# Step 5 — Build and save the network HTML
# ---------------------------------------------------------------------------

def build_network_html(sampled, sim_matrix, output_path: Path):
    clusters_ordered = list(dict.fromkeys(inc["cluster"] for inc in sampled))
    colour_map = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(clusters_ordered)}

    # --- Nodes ---
    nodes = []
    for i, inc in enumerate(sampled):
        col   = colour_map[inc["cluster"]]
        short = inc["narr"][:55] + "…" if len(inc["narr"]) > 55 else inc["narr"]
        top_sims = sorted(
            [(j, float(sim_matrix[i, j])) for j in range(len(sampled)) if j != i],
            key=lambda x: x[1], reverse=True
        )[:3]
        top_str = "<br>".join(
            f"  #{r+1}: {sampled[j]['ev_id']} ({score:.3f})"
            for r, (j, score) in enumerate(top_sims)
        )
        tooltip = (
            f"<b>ID:</b> {inc['ev_id']}<br>"
            f"<b>Cluster:</b> {inc['cluster']}<br><br>"
            f"<b>Top-3 most similar:</b><br>{top_str}<br><br>"
            f"<b>Narrative:</b> {inc['narr'][:300]}"
        )
        nodes.append({
            "id":    i,
            "label": short,
            "title": tooltip,
            "color": {"background": col, "border": col,
                      "highlight": {"background": "#ffffff", "border": col}},
            "font":  {"color": "#ffffff", "size": 10},
            "shape": "box",
        })

    # --- Edges ---
    edges   = []
    edge_id = 0
    n = len(sampled)
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i, j])
            if score < EDGE_THRESHOLD:
                continue
            ci = sampled[i]["cluster"]
            cj = sampled[j]["cluster"]
            col = colour_map[ci] if ci == cj else "#888888"
            edges.append({
                "id":    edge_id,
                "from":  i,
                "to":    j,
                "value": round(score, 3),
                "title": (f"Similarity: {score:.3f}<br>"
                          f"{sampled[i]['ev_id']} ↔ {sampled[j]['ev_id']}<br>"
                          f"Cluster A: {ci}<br>Cluster B: {cj}"),
                "color": {"color": col, "opacity": 0.75},
                "width": round((score - EDGE_THRESHOLD) * 20, 1),
            })
            edge_id += 1

    # --- Legend ---
    legend = "".join(
        f'<div class="leg-item">'
        f'<div class="swatch" style="background:{colour_map[c]}"></div>'
        f'<span>{c}</span>'
        f'</div>'
        for c in clusters_ordered
    )

    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>NTSB Incident-to-Incident Similarity Network</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ font-family:'Segoe UI',Arial,sans-serif; background:#0d1117; color:#c9d1d9; padding:20px; }}
    h1   {{ color:#58a6ff; font-size:19px; margin-bottom:4px; }}
    .sub {{ color:#8b949e; font-size:13px; margin-bottom:10px; }}
    #net {{ width:100%; height:760px; border:1px solid #30363d;
            background:#161b22; border-radius:8px; }}
    .legend   {{ display:flex; flex-wrap:wrap; gap:10px; margin:8px 0 10px; max-height:80px; overflow-y:auto; }}
    .leg-item {{ display:flex; align-items:center; gap:5px; font-size:11px; }}
    .swatch   {{ width:12px; height:12px; border-radius:2px; flex-shrink:0; }}
    footer    {{ margin-top:10px; font-size:11px; color:#6e7681; }}
  </style>
</head>
<body>
  <h1>NTSB Incident-to-Incident Similarity Network</h1>
  <p class="sub">
    {len(sampled)} incidents sampled (up to {MAX_PER_CLUSTER} per cluster), coloured by cluster type.
    Edges appear only when two incidents share &ge; {int(EDGE_THRESHOLD*100)}% cosine similarity.
    No query used &mdash; pure incident-to-incident comparison.
    Hover any node or edge for details. Drag to rearrange.
  </p>
  <div class="legend">{legend}</div>
  <button id="togglePhysics" style="margin-bottom:8px;padding:6px 14px;background:#21262d;
    border:1px solid #30363d;color:#c9d1d9;border-radius:4px;cursor:pointer;font-size:12px;">
    Freeze Layout
  </button>
  <div id="net"></div>
  <footer>
    Model: {EMBEDDING_MODEL} &nbsp;&middot;&nbsp; Dimensions: 1,536
    &nbsp;&middot;&nbsp; Similarity metric: cosine
    &nbsp;&middot;&nbsp; Edge threshold: {int(EDGE_THRESHOLD*100)}%
    &nbsp;&middot;&nbsp; Edges drawn: {len(edges):,}
  </footer>
  <script>
    const nodes = new vis.DataSet({nodes_json});
    const edges = new vis.DataSet({edges_json});
    const opt = {{
      nodes: {{
        widthConstraint: {{ maximum: 200 }},
        margin: {{ top:5, right:8, bottom:5, left:8 }},
        shadow: {{ enabled:true, color:'rgba(0,0,0,.6)', size:8 }}
      }},
      edges: {{
        smooth: {{ type:'continuous' }},
        scaling:{{ min:1, max:10 }}
      }},
      physics: {{
        stabilization: {{ iterations:800 }},
        barnesHut: {{
          gravitationalConstant: -20000,
          springLength:          260,
          springConstant:        0.02,
          damping:               0.15,
          avoidOverlap:          0.7
        }}
      }},
      interaction: {{
        hover:             true,
        tooltipDelay:      100,
        navigationButtons: true,
        keyboard:          true
      }}
    }};
    const network = new vis.Network(document.getElementById('net'), {{nodes, edges}}, opt);

    // Freeze layout as soon as stabilization finishes — stops the vibrating
    network.once('stabilizationIterationsDone', function() {{
      network.setOptions({{ physics: {{ enabled: false }} }});
    }});

    // "Unfreeze / Re-freeze" toggle button
    document.getElementById('togglePhysics').addEventListener('click', function() {{
      const enabled = network.physics.options.enabled;
      network.setOptions({{ physics: {{ enabled: !enabled }} }});
      this.textContent = enabled ? 'Unfreeze Layout' : 'Freeze Layout';
    }});
  </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Network: {output_path}  ({len(nodes)} nodes, {len(edges)} edges)", flush=True)


# ---------------------------------------------------------------------------
# Step 6 — Build and save the similarity table HTML
# ---------------------------------------------------------------------------

def build_table_html(sampled, sim_matrix, output_path: Path):
    clusters_ordered = list(dict.fromkeys(inc["cluster"] for inc in sampled))
    colour_map = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(clusters_ordered)}

    rows = []
    for i, inc in enumerate(sampled):
        top_sims = sorted(
            [(j, float(sim_matrix[i, j])) for j in range(len(sampled)) if j != i],
            key=lambda x: x[1], reverse=True
        )[:TOP_K_TABLE]

        sim_cells = "".join(
            f'<td style="font-size:11px">'
            f'<b>{sampled[j]["ev_id"]}</b><br>'
            f'<span style="color:#8b949e">{sampled[j]["cluster"]}</span><br>'
            f'<span style="color:#2ecc71">{score:.3f}</span>'
            f'</td>'
            for j, score in top_sims
        )

        col = colour_map[inc["cluster"]]
        rows.append(
            f'<tr>'
            f'<td><code>{inc["ev_id"]}</code></td>'
            f'<td><span style="background:{col};padding:2px 6px;border-radius:3px;'
            f'font-size:11px;color:#fff">{inc["cluster"]}</span></td>'
            f'<td style="font-size:11px;max-width:300px">{inc["narr"][:150]}…</td>'
            f'{sim_cells}'
            f'</tr>'
        )

    header_extra = "".join(
        f'<th>Top-{k+1} Similar</th>' for k in range(TOP_K_TABLE)
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>NTSB Incident Similarity Table</title>
  <style>
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ font-family:'Segoe UI',Arial,sans-serif; background:#0d1117; color:#c9d1d9;
            padding:20px; }}
    h1   {{ color:#58a6ff; font-size:18px; margin-bottom:6px; }}
    .sub {{ color:#8b949e; font-size:12px; margin-bottom:14px; }}
    input {{ width:340px; padding:6px 10px; background:#161b22; border:1px solid #30363d;
             color:#c9d1d9; border-radius:4px; font-size:13px; margin-bottom:12px; }}
    table {{ border-collapse:collapse; width:100%; font-size:12px; }}
    th {{ background:#161b22; color:#58a6ff; padding:8px 10px; text-align:left;
          border-bottom:2px solid #30363d; position:sticky; top:0; }}
    td {{ padding:7px 10px; border-bottom:1px solid #21262d; vertical-align:top; }}
    tr:hover td {{ background:#161b22; }}
    code {{ font-size:11px; color:#79c0ff; }}
  </style>
</head>
<body>
  <h1>NTSB Incident Similarity Table</h1>
  <p class="sub">
    {len(sampled)} incidents. For each incident, the top {TOP_K_TABLE} most similar incidents
    are shown with their cosine similarity score. No query used.
  </p>
  <input type="text" id="search" placeholder="Filter by ev_id or cluster..." oninput="filterTable()">
  <table id="tbl">
    <thead>
      <tr>
        <th>Event ID</th><th>Cluster</th><th>Narrative (excerpt)</th>
        {header_extra}
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
  <script>
    function filterTable() {{
      const q = document.getElementById('search').value.toLowerCase();
      document.querySelectorAll('#tbl tbody tr').forEach(row => {{
        row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none';
      }});
    }}
  </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Table:   {output_path}  ({len(sampled)} rows)", flush=True)


# ---------------------------------------------------------------------------
# Step 7 — Print console summary
# ---------------------------------------------------------------------------

def print_summary(sampled, sim_matrix, by_cluster):
    n = len(sampled)
    all_scores = [
        float(sim_matrix[i, j])
        for i in range(n) for j in range(i + 1, n)
    ]
    above = sum(1 for s in all_scores if s >= EDGE_THRESHOLD)

    print(f"\n{'='*70}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Total incidents in dataset   : {sum(len(v) for v in by_cluster.values()):,}", flush=True)
    print(f"  Incidents sampled for network: {n}", flush=True)
    print(f"  Total clusters               : {len(by_cluster)}", flush=True)
    print(f"  Pairwise pairs evaluated     : {len(all_scores):,}", flush=True)
    print(f"  Pairs above threshold ({int(EDGE_THRESHOLD*100)}%)  : {above:,} "
          f"({100*above/len(all_scores):.1f}%)", flush=True)
    print(f"  Mean similarity (all pairs)  : {np.mean(all_scores):.3f}", flush=True)
    print(f"  Max similarity               : {np.max(all_scores):.3f}", flush=True)
    print(f"  Min similarity               : {np.min(all_scores):.3f}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Per-cluster intra-cluster mean similarity
    print("  Intra-cluster avg similarity (do same-cluster incidents score higher?):", flush=True)
    print(f"  {'Cluster':<45} {'Size':>5}  {'Avg Intra-Sim':>14}", flush=True)
    print(f"  {'-'*65}", flush=True)
    idx_map = {inc["ev_id"]: i for i, inc in enumerate(sampled)}
    for cluster, group in sorted(by_cluster.items()):
        sampled_group = [inc for inc in group[:MAX_PER_CLUSTER]]
        idxs = [idx_map[inc["ev_id"]] for inc in sampled_group if inc["ev_id"] in idx_map]
        if len(idxs) < 2:
            continue
        scores = [float(sim_matrix[a, b]) for x, a in enumerate(idxs)
                  for b in idxs[x+1:]]
        avg = np.mean(scores) if scores else 0.0
        print(f"  {cluster:<45} {len(idxs):>5}  {avg:>14.3f}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*70, flush=True)
    print("  02 — INCIDENT-TO-INCIDENT SIMILARITY NETWORK", flush=True)
    print("  Proves embeddings group similar incidents together (no query used)", flush=True)
    print("="*70 + "\n", flush=True)

    here = Path(__file__).parent

    embeddings, emb_map, dataset = load_data()
    all_incidents = build_incident_list(embeddings, emb_map, dataset)
    sampled, by_cluster = sample_incidents(all_incidents)

    print("\nComputing pairwise cosine similarity matrix ...", flush=True)
    sim_matrix = compute_similarity_matrix(sampled)
    print(f"  Matrix shape: {sim_matrix.shape}", flush=True)

    print("\nBuilding outputs ...", flush=True)
    build_network_html(sampled, sim_matrix, here / "02_incident_network.html")
    build_table_html(sampled, sim_matrix, here / "02_incident_table.html")

    print_summary(sampled, sim_matrix, by_cluster)

    # Auto-open network in browser
    try:
        import webbrowser
        net_path = (here / "02_incident_network.html").absolute()
        webbrowser.open(f"file://{net_path}")
        print(f"Opening network in browser ...", flush=True)
    except Exception:
        pass

    print("Done. Open tests/02_incident_network.html and tests/02_incident_table.html.\n",
          flush=True)


if __name__ == "__main__":
    main()

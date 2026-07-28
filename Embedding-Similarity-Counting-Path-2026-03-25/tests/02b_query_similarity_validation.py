"""
Query-to-Incident Similarity Validation
=========================================
Addresses Dr. Mahadevan's question: "Is cosine similarity actually pointing
to the right incidents, or is it arbitrary?"

Runs 6 different queries across different incident types, including
one deliberately out-of-domain query to show the system degrades
gracefully (low scores, wrong clusters) rather than confidently
hallucinating a match.

For each query, shows the top 10 most similar real NTSB incidents
with their full narrative text so relevance can be judged directly.

All queries are embedded in a single batch API call.

Output:
  tests/02b_query_validation.html — one section per query,
      ranked table of top-10 results with similarity score,
      cluster, and full narrative text.

Run from project root:
    python tests/02b_query_similarity_validation.py
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
from openai import OpenAI
from config import (
    get_openai_api_key, EMBEDDING_MODEL,
    REFINED_DATA_PATH, EMBEDDINGS_PATH, EMBEDDINGS_MAP_PATH
)

# ---------------------------------------------------------------------------
# Test queries — 5 different incident types
# ---------------------------------------------------------------------------
QUERIES = [
    {
        "text":     "engine lost power during cruise flight",
        "expected": "Engine Failure / Engine Fire",
        "color":    "#e74c3c",
    },
    {
        "text":     "bird ingested into engine on approach to landing",
        "expected": "Bird Strike",
        "color":    "#2ecc71",
    },
    {
        "text":     "pilot failed to maintain adequate airspeed on final approach",
        "expected": "Pilot Error / Landing Error",
        "color":    "#3498db",
    },
    {
        "text":     "aircraft encountered severe turbulence during cruise",
        "expected": "Turbulence / Severe Turbulence",
        "color":    "#f39c12",
    },
    {
        "text":     "landing gear failed to extend for landing",
        "expected": "Landing Gear Failure",
        "color":    "#9b59b6",
    },
    {
        "text":     "spontaneous combustion of alien spacecraft component",
        "expected": "No match expected — topic outside aviation dataset",
        "color":    "#95a5a6",
    },
]

TOP_N = 10   # results per query


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    print("Loading embeddings and dataset ...", flush=True)
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(EMBEDDINGS_MAP_PATH, "r") as f:
        emb_map = json.load(f)
    with open(REFINED_DATA_PATH, "r") as f:
        dataset = json.load(f)
    print(f"  {len(embeddings):,} embeddings  |  {len(dataset):,} incidents\n", flush=True)
    return embeddings, emb_map, dataset


# ---------------------------------------------------------------------------
# Embed all queries in one batch API call
# ---------------------------------------------------------------------------

def embed_queries(client):
    texts = [q["text"] for q in QUERIES]
    print(f"Embedding {len(texts)} queries in one batch API call ...", flush=True)
    resp = client.embeddings.create(
        input=[t.replace("\n", " ") for t in texts],
        model=EMBEDDING_MODEL
    )
    embs = {texts[i]: np.array(item.embedding) for i, item in enumerate(resp.data)}
    print("  Done.\n", flush=True)
    return embs


# ---------------------------------------------------------------------------
# Find top N results for one query
# ---------------------------------------------------------------------------

def top_results(query_emb, embeddings, emb_map, dataset, n=TOP_N):
    sims = np.dot(embeddings, query_emb)   # OpenAI embeddings are unit-normalised
    sorted_idx = np.argsort(sims)[::-1]

    seen = set()
    results = []
    for idx in sorted_idx:
        if len(results) >= n:
            break
        meta  = emb_map[idx]
        ev_id = meta.get("ev_id")
        if not ev_id or ev_id in seen or meta.get("source") != "incident":
            continue
        inc     = dataset.get(ev_id, {})
        cluster = inc.get("cluster_label", "Unknown")
        narr    = (inc.get("narr_cause") or inc.get("narr_accp")
                   or inc.get("narr_accf") or "No narrative available.")
        results.append({
            "rank":    len(results) + 1,
            "ev_id":   ev_id,
            "score":   float(sims[idx]),
            "cluster": cluster,
            "narr":    narr,
        })
        seen.add(ev_id)
    return results


# ---------------------------------------------------------------------------
# Build HTML
# ---------------------------------------------------------------------------

def build_html(all_results, output_path: Path):

    sections = []
    for q, results in zip(QUERIES, all_results):
        color = q["color"]

        rows = []
        for r in results:
            score_color = (
                "#2ecc71" if r["score"] >= 0.70 else
                "#f39c12" if r["score"] >= 0.60 else
                "#e74c3c"
            )
            rows.append(f"""
              <tr>
                <td style="text-align:center;font-weight:bold">{r['rank']}</td>
                <td><code style="color:#79c0ff">{r['ev_id']}</code></td>
                <td style="text-align:center;font-weight:bold;color:{score_color}">
                  {r['score']:.3f}
                </td>
                <td>
                  <span style="background:{color};padding:2px 7px;border-radius:3px;
                    font-size:11px;color:#fff;white-space:nowrap">
                    {r['cluster']}
                  </span>
                </td>
                <td style="font-size:12px;color:#c9d1d9">{r['narr']}</td>
              </tr>""")

        rows_html = "\n".join(rows)
        sections.append(f"""
        <div class="section">
          <div class="query-header" style="border-left:4px solid {color}">
            <div class="query-label">Query {QUERIES.index(q)+1} of {len(QUERIES)}</div>
            <div class="query-text">"{q['text']}"</div>
            <div class="query-expect">Expected cluster type: <b>{q['expected']}</b></div>
          </div>
          <table>
            <thead>
              <tr>
                <th style="width:40px">#</th>
                <th style="width:150px">Event ID</th>
                <th style="width:70px">Similarity</th>
                <th style="width:180px">Cluster Assigned</th>
                <th>Full Narrative</th>
              </tr>
            </thead>
            <tbody>
              {rows_html}
            </tbody>
          </table>
        </div>""")

    sections_html = "\n".join(sections)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Query-to-Incident Similarity Validation</title>
  <style>
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ font-family:'Segoe UI',Arial,sans-serif; background:#0d1117;
            color:#c9d1d9; padding:28px; max-width:1400px; margin:0 auto; }}
    h1   {{ color:#58a6ff; font-size:20px; margin-bottom:6px; }}
    .sub {{ color:#8b949e; font-size:13px; margin-bottom:28px; line-height:1.6; }}

    .section {{ margin-bottom:44px; }}

    .query-header {{
      padding:14px 18px; background:#161b22;
      border-radius:6px 6px 0 0; margin-bottom:0;
    }}
    .query-label  {{ font-size:11px; color:#8b949e; text-transform:uppercase;
                     letter-spacing:.06em; margin-bottom:4px; }}
    .query-text   {{ font-size:17px; color:#e6edf3; font-weight:600;
                     margin-bottom:4px; }}
    .query-expect {{ font-size:12px; color:#8b949e; }}

    table {{ width:100%; border-collapse:collapse; }}
    thead tr {{ background:#161b22; }}
    th {{ padding:10px 12px; text-align:left; font-size:12px;
          color:#58a6ff; border-bottom:2px solid #30363d;
          position:sticky; top:0; background:#161b22; }}
    td {{ padding:10px 12px; border-bottom:1px solid #21262d;
          vertical-align:top; font-size:13px; }}
    tr:hover td {{ background:#161b22; }}
    code {{ font-size:12px; }}

    .score-high   {{ color:#2ecc71; font-weight:bold; }}
    .score-medium {{ color:#f39c12; font-weight:bold; }}
    .score-low    {{ color:#e74c3c; font-weight:bold; }}

    .legend {{
      display:flex; gap:20px; margin-bottom:24px; flex-wrap:wrap;
      background:#161b22; padding:12px 16px; border-radius:6px;
      font-size:12px;
    }}
    .leg {{ display:flex; align-items:center; gap:6px; }}
    .dot {{ width:12px; height:12px; border-radius:50%; flex-shrink:0; }}
  </style>
</head>
<body>
  <h1>Query-to-Incident Similarity Validation</h1>
  <p class="sub">
    6 test queries across different incident types (including 1 out-of-domain control).
    For each query, the top {TOP_N} most similar real NTSB incidents are shown with their full narrative text.<br>
    <b>Purpose:</b> verify that cosine similarity retrieves relevant incidents,
    not random ones. Read the narratives and judge relevance directly.<br>
    Query 6 is an <b>out-of-domain control</b> — a nonsense query with no match in
    the aviation dataset. Low scores and unrelated clusters in that section confirm
    the system does not hallucinate confident matches for topics outside its knowledge base.<br>
    Model: <code>{EMBEDDING_MODEL}</code> &nbsp;·&nbsp; Dimensions: 1,536
    &nbsp;·&nbsp; Metric: cosine similarity
  </p>
  <div class="legend">
    <span style="color:#8b949e;font-weight:600">Similarity score colours:</span>
    <div class="leg"><div class="dot" style="background:#2ecc71"></div>
      &ge; 0.70 &mdash; very high</div>
    <div class="leg"><div class="dot" style="background:#f39c12"></div>
      0.60 &ndash; 0.69 &mdash; high</div>
    <div class="leg"><div class="dot" style="background:#e74c3c"></div>
      &lt; 0.60 &mdash; moderate</div>
  </div>

  {sections_html}

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {output_path}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*70, flush=True)
    print("  02b — QUERY-TO-INCIDENT SIMILARITY VALIDATION", flush=True)
    print("  5 queries x top-10 results with full narratives", flush=True)
    print("="*70 + "\n", flush=True)

    client = OpenAI(api_key=get_openai_api_key())
    embeddings, emb_map, dataset = load_data()
    query_embs = embed_queries(client)

    all_results = []
    for q in QUERIES:
        print(f'Query: "{q["text"]}"', flush=True)
        results = top_results(query_embs[q["text"]], embeddings, emb_map, dataset)
        all_results.append(results)
        for r in results:
            print(f'  #{r["rank"]:2d}  {r["score"]:.3f}  [{r["cluster"]:<35}]  '
                  f'{r["narr"][:80]}...', flush=True)
        print(flush=True)

    out = Path(__file__).parent / "02b_query_validation.html"
    build_html(all_results, out)

    try:
        import webbrowser
        webbrowser.open(f"file://{out.absolute()}")
        print("Opening in browser ...", flush=True)
    except Exception:
        pass

    print("\nDone. Open tests/02b_query_validation.html\n", flush=True)


if __name__ == "__main__":
    main()

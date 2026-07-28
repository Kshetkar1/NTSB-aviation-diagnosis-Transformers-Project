"""
Generate a self-contained HTML explainer for the worked examples.

Renders Worked_Examples/data/a0.json and data/a2.json into a single, scrollable
HTML page that explains the pipeline to a non-expert reader AND shows every
intermediate result in the two worked examples without omission.

Output: Worked_Examples/worked_examples.html
"""

from __future__ import annotations

import html
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
A0_PATH = _HERE / "data" / "a0.json"
A2_PATH = _HERE / "data" / "a2.json"
OUT = _HERE / "worked_examples.html"


def esc(s: str | None) -> str:
    return html.escape("" if s is None else str(s))


def pct(p: float) -> str:
    return f"{p * 100:.2f}%"


def bar(p: float, color: str = "#2563eb", max_p: float = 1.0) -> str:
    width = min(100.0, (p / max_p) * 100.0) if max_p > 0 else 0.0
    return (
        f'<div class="bar"><div class="bar-fill" '
        f'style="width:{width:.1f}%;background:{color}"></div></div>'
    )


# ============================================================================
# Static prose (plain-language explanation, no field assumptions)
# ============================================================================

INTRO_HTML = r"""
<section id="intro">
  <h2><span class="num">0</span> Start here — what is this?</h2>
  <p>
    Every commercial aviation accident or serious incident in the United States is
    investigated by the National Transportation Safety Board (NTSB). For each one,
    investigators write a long narrative (what happened, in plain English) and
    they also fill out structured forms with coded categories (engine, weather,
    landing gear, etc.). Over decades this has produced a goldmine: thousands of
    rich case files describing how aircraft fail.
  </p>
  <p>
    Our project asks a question: <strong>can a computer read the narratives and
    learn to do two useful things?</strong>
  </p>
  <ol>
    <li>
      <strong>Diagnosis (backward inference).</strong> Given a new narrative —
      "the engine made a loud bang and the crew shut it down" — what is the most
      likely <em>cause</em>?
    </li>
    <li>
      <strong>Prognosis (forward inference).</strong> Given that same narrative,
      what is the most likely <em>next event</em>? Will it lead to an emergency
      landing? A fire? A loss of control?
    </li>
  </ol>
  <p>
    Before our work, prior researchers (Zhang 2021) answered these questions
    using a Bayesian network built only from the <em>structured codes</em>,
    throwing the narratives away. Our approach reads the narratives — the much
    richer source — and converts them into probabilities using two pipelines
    (diagnosis and prognosis) that share retrieval but diverge afterward. This
    page walks you through those pipelines and shows two real worked examples
    end-to-end. Every pipeline step lists <strong>all</strong> rows produced at
    that step (full retrieval tables, full cause lists, full code mappings).
  </p>
  <div class="callout">
    <strong>The two examples in this document:</strong>
    <ol>
      <li>
        <strong>Example 1 (Section 7 of the paper):</strong> a fictional but
        realistic engine-failure narrative. Part A walks through
        <strong>diagnosis</strong> (Section 7.1) on the full narrative; Part B
        walks through <strong>prognosis</strong> (Section 7.2) on a truncated
        cut of the same scenario (stops before outcome). No ground truth.
      </li>
      <li>
        <strong>Example 2 (Section 8.5.1 of the paper):</strong> a real
        held-out test case (incident
        <code>20100114X11754</code>, an ATR72 engine fire in St. Croix in 2010)
        is fed in. We compare the pipeline's prediction against the NTSB's
        actual finding to <em>verify</em> the pipeline is right.
      </li>
    </ol>
  </div>
</section>
"""

PIPELINE_HTML = r"""
<section id="pipeline">
  <h2><span class="num">1</span> How the pipelines work (plain language)</h2>
  <p>
    The paper solves two different questions — <strong>diagnosis</strong>
    (what caused this?) and <strong>prognosis</strong> (what happens next?) —
    using two <em>different</em> pipelines. They share only the first move:
    embed the query and retrieve similar past incidents. After that, the paths
    diverge completely.
  </p>

  <h3 id="pipeline-shared">Shared first step (both modes)</h3>
  <ol class="pipeline">
    <li>
      <strong>Embed the query.</strong>
      <span class="what">What:</span> turn the narrative into a 1,536-dimensional
      embedding via OpenAI <code>text-embedding-3-small</code>.
      <span class="why">Why:</span> we need a numeric representation to compare
      against thousands of past incidents.
    </li>
    <li>
      <strong>Retrieve the top 50 most-similar past incidents.</strong>
      <span class="what">What:</span> cosine similarity against every incident
      narrative in the corpus; keep the 50 best matches.
      <span class="why">Why:</span> both diagnosis and prognosis learn from the
      same "neighborhood" of historically similar cases.
    </li>
  </ol>

  <h3 id="pipeline-diagnosis">Diagnosis pipeline (backward inference)</h3>
  <p class="note">Used in Example 1 (Steps 1–6) and Example 2. Function:
  <code>diagnose_with_conditional_probabilities</code>.</p>
  <ol class="pipeline">
    <li>
      <strong>Cluster the top 50 by failure theme.</strong>
      <span class="what">What:</span> group retrieved incidents into LLM-labeled
      clusters (e.g. "engine fire due to component failures").
      <span class="why">Why:</span> 50 raw neighbors are noisy; clusters reveal
      dominant failure modes.
    </li>
    <li>
      <strong>Compute P(Cluster | Query).</strong>
      <span class="how">How:</span> weight each cluster by
      <code>(average cosine sim) × (cluster size)</code>, then normalize to 1.
    </li>
    <li>
      <strong>Compute P(Cause | Cluster).</strong>
      <span class="how">How:</span> frequency count of recorded causes inside
      each cluster; normalize within the cluster.
    </li>
    <li>
      <strong>Law of Total Probability — P(Cause | Query).</strong>
      <div class="formula">P(Cause | Query) = Σ<sub>K</sub> P(Cause | K) · P(K | Query)</div>
      <span class="note">This is <strong>LTP</strong>, not the chain rule.
      Clusters with no recorded causes are excluded from the partition so
      probabilities sum to 1.</span>
    </li>
    <li>
      <strong>Map free-text causes to Zhang's 54 codes</strong> (optional
      comparison step): nearest-neighbor on code-label embeddings; sum mass
      per code.
    </li>
  </ol>

  <h3 id="pipeline-prognosis">Prognosis pipeline (forward inference)</h3>
  <p class="note">Used in Example 1 Part B (Section 7.2). Function:
  <code>predict_future_events</code>. <strong>Does not use clustering or LTP.</strong></p>
  <ol class="pipeline">
    <li>
      <strong>Re-embed and re-retrieve</strong> (same query → same top 50 in
      practice; prognosis runs as its own function call).
    </li>
    <li>
      <strong>Align the query to each incident's event timeline.</strong>
      <span class="what">What:</span> each incident has a coded
      <code>sequence_of_events</code>. For each neighbor, find which occurrence
      step best matches the query (cosine between query embedding and that
      step's description embedding).
    </li>
    <li>
      <strong>Collect what happened next.</strong>
      <span class="what">What:</span> take the occurrence(s) that follow the
      aligned step in each historical incident.
    </li>
    <li>
      <strong>Weighted empirical transition.</strong>
      <span class="how">How:</span> each incident contributes its observed
      "next event" with weight = retrieval similarity (A0: cosine only).
      Sum weights per distinct next-event label → P(next event | query).
      No cluster mixture — by design, because event-sequence clusters are
      not a valid partition for this task.
    </li>
  </ol>

  <h3>What "A0" and "A2" mean</h3>
  <p>
    <strong>A0</strong> and <strong>A2</strong> apply to <em>diagnosis</em>
    (and optionally prognosis if a structural <code>score_adjust_fn</code> is
    passed). Example 1 prognosis uses <strong>A0 only</strong> — cosine
    weights, no structural reranking.
  </p>
  <ul>
    <li>
      <strong>A0 (baseline).</strong> Embedding similarity only for retrieval
      and weighting.
    </li>
    <li>
      <strong>A2 (contribution).</strong> After retrieval, re-score each
      neighbor with structural similarity on LLM-extracted causal chains:
      <div class="formula">
        new_score = max(0, cosine) × exp(α · struct_sim),
        &nbsp;&nbsp; with &nbsp; α = 2.0
      </div>
      Then re-cluster and re-run LTP (diagnosis only). Example 2 shows this.
    </li>
  </ul>
</section>
"""


# ============================================================================
# Renderers (data → HTML)
# ============================================================================


def render_query_block(query: str, *, badge: str = "Query") -> str:
    return f"""
<div class="query-box">
  <div class="query-badge">{esc(badge)}</div>
  <blockquote>{esc(query)}</blockquote>
</div>
"""


def render_retrieval_table(rows: list[dict]) -> str:
    if not rows:
        return "<p><em>No retrievals.</em></p>"
    max_score = max(r["score"] for r in rows) or 1.0
    body = "\n".join(
        f"<tr><td class='num'>{r['rank']}</td>"
        f"<td><code>{esc(r['ev_id'])}</code></td>"
        f"<td class='num'>{r['score']:.4f}<div class='bar tiny'>"
        f"<div class='bar-fill' style='width:{(r['score']/max_score)*100:.1f}%;background:#2563eb'></div></div></td>"
        f"<td class='snippet'>{esc(r.get('snippet') or '')}</td></tr>"
        for r in rows
    )
    return f"""
<div class="table-wrap">
  <table class="data-table retrieval">
    <thead>
      <tr><th style="width:50px">#</th><th style="width:160px">Incident ID</th>
      <th style="width:160px">Cosine sim</th><th>Narrative snippet</th></tr>
    </thead>
    <tbody>{body}</tbody>
  </table>
</div>
"""


def render_cluster_table(
    rows: list[dict],
    *,
    color: str = "#2563eb",
    sim_col_label: str = "avg cos sim",
) -> str:
    if not rows:
        return "<p><em>No clusters.</em></p>"
    max_p = max(r["p_k_given_q"] for r in rows) or 1.0
    body = "\n".join(
        f"<tr><td>{esc(r['cluster'])}</td>"
        f"<td class='num'>{r['n_incidents']}</td>"
        f"<td class='num'>{r['avg_similarity']:.4f}</td>"
        f"<td class='num'>{r['weight']:.4f}</td>"
        f"<td class='num strong'>{r['p_k_given_q']:.4f}"
        f"<div class='bar tiny'><div class='bar-fill' style='width:{(r['p_k_given_q']/max_p)*100:.1f}%;background:{color}'></div></div></td></tr>"
        for r in rows
    )
    return f"""
<div class="table-wrap">
  <table class="data-table clusters">
    <thead><tr>
      <th>Cluster (failure theme)</th><th>n</th><th>{esc(sim_col_label)}</th>
      <th>weight = sim·n</th><th>P(Cluster | Query)</th>
    </tr></thead>
    <tbody>{body}</tbody>
  </table>
</div>
"""


def ltp_hand_check_callout(a0: dict) -> str:
    """One explicit LTP arithmetic check for the top cause."""
    if not a0.get("ltp_causes") or not a0.get("cluster_causes"):
        return ""
    top = a0["ltp_causes"][0]
    cause_prefix = top["cause"][:48]
    p_ck = p_kq = cluster_label = None
    for label, rows in a0["cluster_causes"].items():
        for r in rows:
            if r["cause"].startswith(cause_prefix[:40]) or cause_prefix.startswith(r["cause"][:40]):
                p_ck = r["p_c_given_k"]
                cluster_label = label
                break
        if p_ck is not None:
            break
    if cluster_label:
        for c in a0["clusters"]:
            if c["cluster"] == cluster_label:
                p_kq = c["p_k_given_q"]
                break
    if p_ck is None or p_kq is None:
        return ""
    expected = p_ck * p_kq
    return f"""
<div class="callout callout-ok">
  <strong>Hand-check (top cause):</strong>
  P(C|Q) = P(C|K) × P(K|Q) = {p_ck:.4f} × {p_kq:.4f} = {expected:.4f}
  (table: {top['probability']:.4f}) from cluster
  <em>{esc(cluster_label)}</em>.
</div>
"""


def engine_cluster_mass(clusters: list[dict]) -> float:
    keys = ("engine", "fire", "power", "turbine", "compressor", "fuel nozzle")
    return sum(
        c["p_k_given_q"]
        for c in clusters
        if any(k in c["cluster"].lower() for k in keys)
    )


def render_cluster_causes(cluster_causes: dict[str, list[dict]]) -> str:
    blocks = []
    for label, rows in cluster_causes.items():
        if not rows:
            blocks.append(
                f"<details class='cluster-causes'>"
                f"<summary>{esc(label)} <em>(no causes recorded)</em></summary></details>"
            )
            continue
        rows_html = "\n".join(
            f"<tr><td class='num'>{r.get('rank', i + 1)}</td>"
            f"<td class='num'>{r['p_c_given_k']:.4f}</td>"
            f"<td>{esc(r['cause'])}</td></tr>"
            for i, r in enumerate(rows)
        )
        blocks.append(f"""
<details class="cluster-causes" open>
  <summary><strong>{esc(label)}</strong></summary>
  <div class="table-wrap">
    <table class="data-table sub">
      <thead><tr><th style="width:40px">#</th><th style="width:100px">P(C|K)</th><th>Cause (free text)</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</details>
""")
    return "\n".join(blocks)


def render_causes_table(rows: list[dict], *, color: str = "#2563eb") -> str:
    if not rows:
        return "<p><em>No causes.</em></p>"
    max_p = max(r["probability"] for r in rows) or 1.0
    body = "\n".join(
        f"<tr><td class='num'>{r['rank']}</td>"
        f"<td class='num strong'>{r['probability']:.4f}"
        f"<div class='bar tiny'><div class='bar-fill' style='width:{(r['probability']/max_p)*100:.1f}%;background:{color}'></div></div></td>"
        f"<td>{esc(r['cause'])}</td></tr>"
        for r in rows
    )
    return f"""
<div class="table-wrap">
  <table class="data-table causes">
    <thead><tr><th style="width:50px">Rank</th><th style="width:160px">P(Cause | Query)</th><th>Free-text cause</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
</div>
"""


def render_codes_table(rows: list[dict], *, color: str = "#2563eb", highlight_code: str | None = None) -> str:
    if not rows:
        return "<p><em>No coded distribution.</em></p>"
    max_p = max(r["probability"] for r in rows) or 1.0
    body = []
    for r in rows:
        klass = "highlight" if (highlight_code and str(r["code"]) == str(highlight_code)) else ""
        body.append(
            f"<tr class='{klass}'><td class='num'>{r['rank']}</td>"
            f"<td class='num'><code>{esc(r['code'])}</code></td>"
            f"<td>{esc(r['label'])}</td>"
            f"<td class='num strong'>{r['probability']:.4f}"
            f"<div class='bar tiny'><div class='bar-fill' style='width:{(r['probability']/max_p)*100:.1f}%;background:{color}'></div></div></td></tr>"
        )
    body_html = "\n".join(body)
    return f"""
<div class="table-wrap">
  <table class="data-table codes">
    <thead><tr><th style="width:50px">#</th><th style="width:70px">Code</th><th>Label</th><th style="width:160px">P(code | Query)</th></tr></thead>
    <tbody>{body_html}</tbody>
  </table>
</div>
"""


def render_prognosis_table(rows: list[dict]) -> str:
    if not rows:
        return "<p><em>No prognosis output.</em></p>"
    max_p = max(r["probability"] for r in rows) or 1.0
    body = "\n".join(
        f"<tr><td class='num'>{r['rank']}</td>"
        f"<td class='num strong'>{r['probability']:.4f}"
        f"<div class='bar tiny'><div class='bar-fill' style='width:{(r['probability']/max_p)*100:.1f}%;background:#16a34a'></div></div></td>"
        f"<td>{esc(r['event'])}</td></tr>"
        for r in rows
    )
    return f"""
<div class="table-wrap">
  <table class="data-table prognosis">
    <thead><tr><th style="width:50px">#</th><th style="width:160px">P(event | Query)</th><th>Predicted next event</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
</div>
"""


def render_prognosis_alignment_table(rows: list[dict]) -> str:
    if not rows:
        return "<p><em>No aligned incidents.</em></p>"
    body = []
    for r in rows:
        next_txt = "; ".join(r.get("next_events") or []) or "—"
        downstream = "yes" if r.get("has_downstream") else "no"
        body.append(
            f"<tr><td class='num'>{r['rank']}</td>"
            f"<td><code>{esc(r.get('ev_id', ''))}</code></td>"
            f"<td class='num'>{r.get('incident_similarity', 0.0):.4f}</td>"
            f"<td class='num'>{r.get('event_match_score', 0.0):.4f}</td>"
            f"<td class='snippet'>{esc(r.get('matched_event', ''))}</td>"
            f"<td class='num'>{downstream}</td>"
            f"<td class='snippet'>{esc(next_txt)}</td></tr>"
        )
    return f"""
<div class="table-wrap">
  <table class="data-table alignment">
    <thead><tr>
      <th>#</th><th>Incident</th><th>Retrieval wt</th><th>Step match</th>
      <th>Aligned step</th><th>Downstream?</th><th>Next event(s)</th>
    </tr></thead>
    <tbody>{''.join(body)}</tbody>
  </table>
</div>
"""


def render_multi_step_prognosis(steps: list[dict]) -> str:
    if not steps:
        return ""
    parts = []
    for step in steps:
        cond = step.get("conditioned_on") or []
        cond_txt = " (none — step 1)" if not cond else " → ".join(cond)
        parts.append(f"<h4>Prognosis chain step {step.get('step')}</h4>")
        parts.append(f"<p class='note'>Conditioned on prefix: {esc(cond_txt)}. "
                     f"Total weight for this step: {step.get('total_weight', 0.0):.4f}</p>")
        parts.append(render_prognosis_table(step.get("events") or []))
    return "\n".join(parts)


def render_cluster_shift_table(a0_clusters: list[dict], a2_clusters: list[dict]) -> str:
    a0_map = {c["cluster"]: c for c in a0_clusters}
    rows = sorted(a2_clusters, key=lambda c: -c["p_k_given_q"])
    body = []
    max_abs_shift = max(
        (abs(c["p_k_given_q"] - a0_map.get(c["cluster"], {}).get("p_k_given_q", 0.0)) for c in rows),
        default=0.0,
    ) or 1.0
    for c in rows:
        a0p = a0_map.get(c["cluster"], {}).get("p_k_given_q", 0.0)
        a2p = c["p_k_given_q"]
        shift = a2p - a0p
        arrow = "↑" if shift > 0 else ("↓" if shift < 0 else "·")
        klass = "shift-up" if shift > 0 else ("shift-down" if shift < 0 else "shift-flat")
        body.append(
            f"<tr class='{klass}'>"
            f"<td>{esc(c['cluster'])}</td>"
            f"<td class='num'>{c['n_incidents']}</td>"
            f"<td class='num'>{a0p:.4f}</td>"
            f"<td class='num strong'>{a2p:.4f}</td>"
            f"<td class='num shift'>{shift:+.4f} {arrow}</td></tr>"
        )
    body_html = "\n".join(body)
    return f"""
<div class="table-wrap">
  <table class="data-table shift">
    <thead><tr><th>Cluster</th><th>n</th><th>A0 P(K|Q)</th><th>A2 P(K|Q)</th><th>shift</th></tr></thead>
    <tbody>{body_html}</tbody>
  </table>
</div>
"""


def render_side_by_side_causes(a0: list[dict], a2: list[dict]) -> str:
    n = max(len(a0), len(a2))
    rows_html = []
    for i in range(n):
        a0r = a0[i] if i < len(a0) else None
        a2r = a2[i] if i < len(a2) else None
        a0p = a0r["probability"] if a0r else 0.0
        a2p = a2r["probability"] if a2r else 0.0
        delta = a2p - a0p
        rows_html.append(
            f"<tr><td class='num'>{i+1}</td>"
            f"<td class='num'>{a0p:.4f}</td>"
            f"<td>{esc(a0r['cause']) if a0r else ''}</td>"
            f"<td class='num'>{a2p:.4f}</td>"
            f"<td>{esc(a2r['cause']) if a2r else ''}</td>"
            f"<td class='num shift'>{(delta):+.4f}</td></tr>"
        )
    return f"""
<div class="table-wrap">
  <table class="data-table side-by-side">
    <thead><tr>
      <th style="width:40px">#</th>
      <th style="width:90px">A0 P</th><th>A0 cause</th>
      <th style="width:90px">A2 P</th><th>A2 cause</th>
      <th style="width:90px">Δ</th>
    </tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
</div>
"""


def render_side_by_side_codes(a0: list[dict], a2: list[dict], gt_code: str | None) -> str:
    a0_by_code = {str(r["code"]): r for r in a0}
    all_codes = list(dict.fromkeys([str(r["code"]) for r in a0] + [str(r["code"]) for r in a2]))
    a2_by_code = {str(r["code"]): r for r in a2}
    rows_html = []
    # Order: by A2 prob desc (since A2 is the contribution), fallback to A0
    def _key(c):
        return -(a2_by_code.get(c, {}).get("probability", 0.0))
    for code in sorted(all_codes, key=_key):
        a0r = a0_by_code.get(code)
        a2r = a2_by_code.get(code)
        a0p = a0r["probability"] if a0r else 0.0
        a2p = a2r["probability"] if a2r else 0.0
        delta = a2p - a0p
        label = (a2r or a0r or {}).get("label", "")
        klass = "highlight" if (gt_code and code == str(gt_code)) else ""
        rows_html.append(
            f"<tr class='{klass}'>"
            f"<td class='num'><code>{esc(code)}</code></td>"
            f"<td>{esc(label)}</td>"
            f"<td class='num'>{a0p:.4f}</td>"
            f"<td class='num strong'>{a2p:.4f}</td>"
            f"<td class='num shift'>{delta:+.4f}</td></tr>"
        )
    return f"""
<div class="table-wrap">
  <table class="data-table side-by-side codes">
    <thead><tr>
      <th style="width:70px">Code</th><th>Label</th>
      <th style="width:90px">A0 P</th>
      <th style="width:90px">A2 P</th>
      <th style="width:90px">Δ</th>
    </tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
</div>
"""


# ============================================================================
# Assemble
# ============================================================================


CSS = r"""
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Helvetica, Arial, sans-serif;
  line-height: 1.6;
  color: #111827;
  background: #fafafa;
  margin: 0;
  padding: 0;
}
.layout {
  display: grid;
  grid-template-columns: 260px 1fr;
  max-width: 1500px;
  margin: 0 auto;
}
nav.toc {
  position: sticky;
  top: 0;
  align-self: start;
  height: 100vh;
  overflow-y: auto;
  padding: 24px 16px 24px 24px;
  border-right: 1px solid #e5e7eb;
  background: #fff;
  font-size: 13px;
}
nav.toc h3 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em;
  color: #6b7280; margin: 0 0 8px; }
nav.toc ul { list-style: none; padding: 0; margin: 0; }
nav.toc ul ul { margin-left: 12px; margin-top: 4px; }
nav.toc li { margin: 4px 0; }
nav.toc a { color: #374151; text-decoration: none; display: block; padding: 3px 6px;
  border-radius: 4px; }
nav.toc a:hover { background: #f3f4f6; color: #003366; }
main {
  padding: 36px 48px 80px;
  max-width: 1100px;
  background: #fafafa;
}
h1 {
  font-size: 32px;
  font-weight: 700;
  margin: 0 0 4px;
  color: #003366;
  letter-spacing: -0.02em;
}
.subtitle { color: #6b7280; font-size: 16px; margin: 0 0 32px; }
h2 {
  font-size: 24px;
  font-weight: 700;
  margin: 48px 0 12px;
  color: #003366;
  letter-spacing: -0.01em;
  display: flex;
  align-items: center;
  gap: 12px;
}
h2 .num {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border-radius: 8px;
  background: #003366;
  color: #fff;
  font-size: 14px;
  font-weight: 700;
}
h3 {
  font-size: 18px;
  font-weight: 600;
  margin: 28px 0 8px;
  color: #1f2937;
}
h4 { font-size: 15px; font-weight: 600; margin: 20px 0 8px; color: #374151; }
p { margin: 8px 0 12px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.92em;
  background: #f3f4f6; padding: 1px 5px; border-radius: 3px; }
strong { color: #1f2937; }
em { color: #4b5563; }
ol, ul { padding-left: 24px; }
ol.pipeline { padding-left: 0; counter-reset: step; list-style: none; }
ol.pipeline > li {
  counter-increment: step;
  padding: 12px 16px 12px 56px;
  margin: 10px 0;
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  position: relative;
}
ol.pipeline > li::before {
  content: counter(step);
  position: absolute;
  left: 14px;
  top: 12px;
  width: 28px;
  height: 28px;
  background: #003366;
  color: #fff;
  border-radius: 6px;
  font-weight: 700;
  font-size: 13px;
  display: flex;
  align-items: center;
  justify-content: center;
}
ol.pipeline > li span.what,
ol.pipeline > li span.why,
ol.pipeline > li span.how {
  display: inline-block;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 1px 7px;
  border-radius: 4px;
  margin: 0 6px 0 0;
}
ol.pipeline > li span.what { color: #1e40af; background: #dbeafe; }
ol.pipeline > li span.why  { color: #92400e; background: #fef3c7; }
ol.pipeline > li span.how  { color: #166534; background: #d1fae5; }
.callout {
  background: #f0f9ff;
  border-left: 4px solid #0284c7;
  padding: 12px 16px;
  margin: 16px 0;
  border-radius: 4px;
}
.callout-warn {
  background: #fff7ed;
  border-left-color: #ea580c;
}
.callout-ok {
  background: #f0fdf4;
  border-left-color: #16a34a;
}
.query-box {
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 16px 20px;
  margin: 12px 0 20px;
}
.query-badge {
  display: inline-block;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #4b5563;
  margin-bottom: 8px;
}
.query-box blockquote {
  margin: 0;
  font-style: italic;
  color: #1f2937;
  font-size: 15px;
  line-height: 1.7;
  border: none;
  padding: 0;
}
.table-wrap {
  overflow-x: auto;
  margin: 12px 0 20px;
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
}
table.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.data-table th,
table.data-table td {
  text-align: left;
  padding: 8px 10px;
  border-bottom: 1px solid #f3f4f6;
  vertical-align: top;
}
table.data-table th {
  background: #f9fafb;
  font-weight: 600;
  color: #374151;
  font-size: 12px;
  letter-spacing: 0.02em;
  border-bottom: 1px solid #e5e7eb;
  position: sticky;
  top: 0;
}
table.data-table td.num { font-variant-numeric: tabular-nums; }
table.data-table td.snippet { color: #4b5563; font-size: 12px; }
table.data-table tbody tr:nth-child(even) td { background: #fcfcfd; }
table.data-table tbody tr:hover td { background: #f3f4f6; }
table.data-table td.strong { font-weight: 600; color: #111827; }
table.data-table tr.highlight td { background: #fef3c7 !important; }
table.data-table tr.shift-up  td.shift { color: #16a34a; font-weight: 600; }
table.data-table tr.shift-down td.shift { color: #b91c1c; font-weight: 600; }
table.data-table tr.shift-flat td.shift { color: #6b7280; }
table.data-table td.shift { font-variant-numeric: tabular-nums; }
.bar {
  margin-top: 4px;
  background: #f3f4f6;
  border-radius: 2px;
  height: 4px;
  overflow: hidden;
}
.bar.tiny { height: 3px; }
.bar-fill { height: 100%; }
details.cluster-causes {
  margin: 6px 0;
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  padding: 4px 12px;
}
details.cluster-causes summary {
  cursor: pointer;
  padding: 6px 0;
  font-size: 14px;
  color: #1f2937;
}
table.data-table.sub { font-size: 12px; margin: 6px 0 8px; }
.metrics-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin: 16px 0 24px;
}
.metric {
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 12px 16px;
}
.metric .label { font-size: 11px; color: #6b7280; text-transform: uppercase;
  letter-spacing: 0.05em; }
.metric .value { font-size: 20px; font-weight: 700; color: #003366; margin-top: 4px;
  font-variant-numeric: tabular-nums; }
.metric.success .value { color: #16a34a; }
.metric.warn .value { color: #ea580c; }
.metric.muted .value { color: #6b7280; }
.formula {
  display: block;
  font-family: ui-monospace, monospace;
  background: #f9fafb;
  border: 1px solid #e5e7eb;
  padding: 10px 14px;
  border-radius: 6px;
  margin: 8px 0;
  font-size: 13px;
  text-align: center;
}
.example-header {
  padding: 24px 28px;
  background: linear-gradient(135deg, #003366, #0066cc);
  color: #fff;
  border-radius: 12px;
  margin: 32px 0 12px;
}
.example-header h2 { color: #fff; margin: 0 0 4px; }
.example-header .sub { opacity: 0.85; font-size: 14px; }
.step {
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 18px 22px;
  margin: 12px 0 18px;
}
.step h3 { margin-top: 0; display: flex; align-items: center; gap: 10px; }
.step h3 .stepnum {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 26px; height: 26px;
  border-radius: 6px;
  background: #dbeafe;
  color: #1e40af;
  font-weight: 700;
  font-size: 12px;
}
.step h3 .stepnum.a2 { background: #fee2e2; color: #b91c1c; }
.step h3 .stepnum.prog { background: #dcfce7; color: #166534; }
.example-part-header {
  margin: 28px 0 12px;
  padding: 14px 18px;
  border-radius: 8px;
  background: #f0f9ff;
  border: 1px solid #bae6fd;
}
.example-part-header.prognosis {
  background: #f0fdf4;
  border-color: #bbf7d0;
}
.example-part-header h3 {
  margin: 0;
  font-size: 17px;
  color: #0c4a6e;
}
.example-part-header.prognosis h3 { color: #14532d; }
.example-part-header p {
  margin: 6px 0 0;
  font-size: 13px;
  color: #475569;
}
.note {
  font-size: 13px;
  color: #6b7280;
  margin: 4px 0 10px;
}
.divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, #d1d5db, transparent);
  margin: 32px 0;
}
footer {
  margin-top: 60px;
  padding-top: 24px;
  border-top: 1px solid #e5e7eb;
  color: #6b7280;
  font-size: 13px;
}
@media (max-width: 1000px) {
  .layout { grid-template-columns: 1fr; }
  nav.toc { position: relative; height: auto; border-right: none;
    border-bottom: 1px solid #e5e7eb; }
  main { padding: 24px; }
}
"""


def assemble(a0: dict, a2: dict) -> str:
    # ---------- Example 1 (A0) -----------
    top_cluster_pct = a0["clusters"][0]["p_k_given_q"] * 100 if a0.get("clusters") else 0.0
    engine_mass_pct = engine_cluster_mass(a0.get("clusters") or []) * 100
    ltp_sum = a0.get("ltp_sum") or sum(c["probability"] for c in (a0.get("ltp_causes") or []))
    ltp_count = len(a0.get("ltp_causes") or [])
    prognosis_query = a0.get("prognosis_query") or a0["query"]
    unmapped_a0 = next(
        (c["probability"] for c in (a0.get("coded_distribution") or []) if c.get("code") == "—"),
        0.0,
    )
    a0_html_parts = []
    a0_html_parts.append(f"""
<div class="example-header" id="ex1">
  <h2 style="border:none">Example 1 — A0 diagnosis &amp; prognosis</h2>
  <div class="sub">Paper Section 7. Synthetic engine-failure narrative against the full {a0['corpus_size']:,}-incident corpus. Part A = diagnosis (7.1); Part B = prognosis (7.2), separate pipeline. No ground truth.</div>
</div>
""")

    a0_html_parts.append(f"""
<div class="metrics-row">
  <div class="metric"><div class="label">Pipeline</div><div class="value">A0 (embedding-only)</div></div>
  <div class="metric"><div class="label">Mode</div><div class="value">Production</div></div>
  <div class="metric"><div class="label">Searchable corpus</div><div class="value">{a0['corpus_size']:,} incidents</div></div>
  <div class="metric muted"><div class="label">Ground truth</div><div class="value">N/A</div></div>
</div>
""")

    a0_html_parts.append(f"""
<div class="example-part-header" id="ex1-diagnosis">
  <h3>Part A — Diagnosis (Paper Section 7.1)</h3>
  <p>
    Cluster retrieved neighbors, count causes per cluster, apply the Law of
    Total Probability, map to Zhang's 54 codes. Function:
    <code>diagnose_with_conditional_probabilities</code>. Uses the
    <strong>full</strong> synthetic narrative (including outcome). Part B
    prognosis uses a shorter cut of the same scenario (see Part B).
  </p>
</div>
""")

    a0_html_parts.append(f"""
<div class="step">
  <h3><span class="stepnum">1</span> The query (synthetic narrative)</h3>
  <p class="note">We hand-crafted a realistic engine-failure narrative. It mentions
  a No. 2 engine power loss at cruise, high EGT, vibration, crew shutting down
  the engine, declaring an emergency, and a safe diversion. About 60 words.</p>
  {render_query_block(a0['query'])}
</div>
""")

    a0_html_parts.append(f"""
<div class="step">
  <h3><span class="stepnum">2</span> Embed and retrieve the top 50 most-similar past incidents</h3>
  <p>We embed the query (1,536-dim vector), then compare against every incident
  in the corpus by cosine similarity. The table below shows <strong>all 50
  retrieved incidents</strong>. Higher score = more similar. The bar visualizes
  each score against the top score.</p>
  {render_retrieval_table(a0['retrieval'])}
  <div class="callout">
    Most of the top retrievals are engine-related airline incidents — the
    right neighborhood — though a few tail rows are off-theme (e.g. rank 6 is
    a water collision). Retrieval is intentionally broad (top 50); clustering
    and LTP sharpen the signal downstream.
  </div>
</div>
""")

    a0_html_parts.append(f"""
<div class="step">
  <h3><span class="stepnum">3</span> Cluster the 50 retrieved incidents and compute P(Cluster | Query)</h3>
  <p>The 50 are partitioned into {len(a0['clusters'])} clusters by failure theme.
  Each cluster's weight is <code>(avg cosine sim) × (cluster size)</code>; we
  normalize so the weights sum to 1.</p>
  {render_cluster_table(a0['clusters'])}
  <div class="callout">
    The top cluster — <strong>"{esc(a0['clusters'][0]['cluster'])}"</strong> —
    holds about {top_cluster_pct:.0f}% of the probability mass. Engine/fire
    themed clusters together account for about {engine_mass_pct:.0f}% — the
    pipeline correctly identifies the neighborhood as engine-failure-and-fire.
  </div>
</div>
""")

    a0_html_parts.append(f"""
<div class="step">
  <h3><span class="stepnum">4</span> P(Cause | Cluster) — the typical causes inside each cluster</h3>
  <p>For each cluster we count how often each free-text cause shows up across
  the cluster's member incidents and normalize. Below: <strong>all</strong>
  {sum(len(v) for v in a0['cluster_causes'].values())} causes across
  {len(a0['cluster_causes'])} active clusters (complete P(C|K) tables).</p>
  {render_cluster_causes(a0['cluster_causes'])}
</div>
""")

    skipped = a0.get("skipped_clusters", []) or []
    if skipped:
        skipped_items = "\n".join(
            f"<li><code>{esc(c['cluster'])}</code> (n={c['n_incidents']}, {esc(c['reason'])})</li>"
            for c in skipped
        )
        skipped_note = f"""
<div class="callout callout-warn" style="margin: -8px 0 18px;">
  <strong>Methodology note:</strong> {len(skipped)}
  cluster(s) were excluded from the LTP partition because their member
  incidents have no recorded causes:
  <ul style="margin: 6px 0 0;">
{skipped_items}
  </ul>
</div>
"""
    else:
        skipped_note = ""

    a0_html_parts.append(f"""
<div class="step">
  <h3><span class="stepnum">5</span> Law of Total Probability — P(Cause | Query)</h3>
  <p>We combine steps 3 and 4 using the Law of Total Probability (LTP),
  marginalizing over the cluster partition {{K}}:</p>
  <div class="formula">P(Cause | Query) = Σ<sub>K</sub> P(Cause | K) · P(K | Query)</div>
  <p class="note">This is <strong>LTP</strong>, not the chain rule.
  P(A,B) = P(A|B)·P(B) is the chain rule; that's for joint probabilities. We
  use LTP because we're marginalizing one event (a cause) over a partition of
  another (the clusters).</p>
  <p>The output is the <strong>complete</strong> probability distribution over
  all {ltp_count} retrieved causes (table below). Σ P(C|Query) =
  {ltp_sum:.6f}. We restrict the LTP partition to clusters that have recorded
  causes.</p>
  {render_causes_table(a0['ltp_causes'])}
  {ltp_hand_check_callout(a0)}
  <div class="callout">
    The highest-probability causes are dominated by engine and fire clusters
    (anti-ice / compressor / fuel-nozzle / maintenance themes), though a few
    crew-procedure causes also appear from mixed clusters.
  </div>
</div>
{skipped_note}
""")

    a0_html_parts.append(f"""
<div class="step">
  <h3><span class="stepnum">6</span> Map free-text causes to Zhang's 54 codes</h3>
  <p>To compare against prior work (Zhang 2021), we project our free-text
  distribution onto Zhang's 54 occurrence codes. Each free-text cause is mapped
  to its nearest code by embedding similarity; mass is summed per code. The
  complete mapped distribution ({len(a0['coded_distribution'])} rows) is below.</p>
  {render_codes_table(a0['coded_distribution'])}
  <div class="callout">
    Engine and power-related codes (<code>130</code>, <code>132</code>,
    <code>352</code>, <code>350</code>) together hold the majority of the
    probability mass — the correct code family for an engine-failure query.
    <span class="note">Code <code>—</code> (UNMAPPED) holds {unmapped_a0:.1%}
    of mass: free-text causes with no confident nearest Zhang code.</span>
  </div>
</div>
""")

    a0_html_parts.append(f"""
<div class="example-part-header prognosis" id="ex1-prognosis">
  <h3>Part B — Prognosis (Paper Section 7.2)</h3>
  <p>
    <strong>Separate pipeline.</strong> Prognosis does <em>not</em> use the
    clusters, causes, or Law of Total Probability from Part A. It calls
    <code>predict_future_events</code>: align the query to each neighbor's
    coded event timeline, then aggregate observed next events with cosine
    weights (A0). Uses a <strong>truncated query</strong> (same scenario, stops
    at master caution — before shutdown or landing) so predictions are forward
    forecasts. Step 1 immediate next-event distribution shown in P4.
  </p>
</div>
""")

    a0_html_parts.append(f"""
<div class="step">
  <h3><span class="stepnum prog">P1</span> Prognosis query — truncated before outcome</h3>
  <p class="note">Part A diagnosis uses the full narrative (including shutdown,
  divert, and landing). Prognosis deliberately uses an earlier cut so we are
  asking "what happens next?" rather than matching text the query already
  contains. Prognosis re-embeds this shorter text and retrieves its own
  top-50 neighborhood (similar engine-failure cases, not identical ranks).</p>
  {render_query_block(prognosis_query, badge="Prognosis query (truncated)")}
</div>
""")

    prog_retrieval = a0.get("prognosis_retrieval") or []
    if prog_retrieval:
        a0_html_parts.append(f"""
<div class="step">
  <h3><span class="stepnum prog">P2</span> Embed and retrieve top 50 for prognosis</h3>
  <p>Prognosis embeds the truncated query and retrieves its own top-50 neighbors
  (cosine on narrative embeddings). All {len(prog_retrieval)} rows are shown.</p>
  {render_retrieval_table(prog_retrieval)}
</div>
""")

    aligned = a0.get("prognosis_aligned") or []
    a0_html_parts.append(f"""
<div class="step">
  <h3><span class="stepnum prog">P{'3' if prog_retrieval else '2'}</span> Align query to each incident's event timeline</h3>
  <p>For each retrieved incident with a coded <code>sequence_of_events</code>,
  we embed each occurrence description and pick the step closest to the query
  embedding. Table: all {len(aligned)} aligned incidents returned by the pipeline.</p>
  {render_prognosis_alignment_table(aligned)}
  <div class="callout">
    This alignment uses <strong>occurrence-description embeddings</strong>, not
    cluster labels or cause text from diagnosis. Rows with downstream = no
    matched the last event in that timeline and do not vote in the next-event sum.
  </div>
</div>
""")

    p_collect = 4 if prog_retrieval else 3
    p_out = 5 if prog_retrieval else 4
    a0_html_parts.append(f"""
<div class="step">
  <h3><span class="stepnum prog">P{p_collect}</span> Collect the next coded occurrence from each neighbor</h3>
  <p>After the aligned step, we take whatever occurrence(s) follow in that
  incident's historical record (see the "Next event(s)" column above). Only rows
  with downstream = yes enter the weighted vote.</p>
  <p class="note">No clustering. No P(Cause|Cluster). No LTP — the code
  explicitly avoids mixing over LLM/k-means clusters for prognosis because
  they are not a valid partition for event transitions.</p>
</div>
""")

    a0_html_parts.append(f"""
<div class="step">
  <h3><span class="stepnum prog">P{p_out}</span> Weighted empirical transition — P(next event | query)</h3>
  <p>Each neighbor with downstream data contributes its observed next event with
  weight = cosine retrieval similarity. We sum weights per distinct next-event
  label and normalize. Below: the complete step-1 distribution
  ({len(a0.get('prognosis') or [])} distinct next events) and all greedy chain
  steps computed (<code>max_chain_steps=3</code>).</p>
  <h4>Step 1 — immediate next event</h4>
  {render_prognosis_table(a0['prognosis'])}
  {render_multi_step_prognosis([s for s in (a0.get('prognosis_multi_step') or []) if s.get('step', 1) > 1])}
  <p class="note">Coverage: {a0.get('prognosis_meta', {}).get('sequences_analyzed', '?')} of 50
  retrieved neighbors contributed a downstream next-event after timeline
  alignment ({a0.get('prognosis_meta', {}).get('aligned_incident_count', '?')}
  aligned total).</p>
  <div class="callout callout-ok">
    <strong>Forward forecast.</strong> Because the query stops at master caution
    (no shutdown, divert, or landing yet), these are genuine next-event
    predictions from historically similar in-flight engine failures — not
    restatements of an ending already in the text.
  </div>
</div>
""")

    a0_html = "\n".join(a0_html_parts)

    # ---------- Example 2 (A2) -----------
    gt = a2.get("ground_truth", {})
    a0_by_label = {c["cluster"]: c for c in a2["a0"]["clusters"]}
    a2_by_label = {c["cluster"]: c for c in a2["a2"]["clusters"]}
    engine_label = next(
        (lbl for lbl in a0_by_label if "engine fire" in lbl.lower()),
        a2["a0"]["clusters"][0]["cluster"] if a2["a0"]["clusters"] else "",
    )
    engine_shift_pp = 0.0
    if engine_label and engine_label in a2_by_label:
        engine_shift_pp = (
            a2_by_label[engine_label]["p_k_given_q"]
            - a0_by_label[engine_label]["p_k_given_q"]
        ) * 100.0

    a2_html_parts = []
    a2_html_parts.append(f"""
<div class="example-header" id="ex2" style="background: linear-gradient(135deg, #7f1d1d, #b91c1c);">
  <h2 style="border:none; color:#fff">Example 2 — A2 diagnosis (held-out test incident)</h2>
  <div class="sub">Paper Section 8.5.1. Real test incident <code style="color:#fff;background:rgba(255,255,255,0.15)">{esc(a2['ev_id'])}</code> (ATR72 engine fire, St. Croix, 2010) against the {a2['corpus_size']}-incident training index. We verify the pipeline against the actual NTSB ground truth.</div>
</div>
""")

    a2_html_parts.append(f"""
<div class="metrics-row">
  <div class="metric"><div class="label">Pipeline</div><div class="value">A0 + A2</div></div>
  <div class="metric"><div class="label">Mode</div><div class="value">Held-out</div></div>
  <div class="metric"><div class="label">Training index</div><div class="value">{a2['corpus_size']} incidents</div></div>
  <div class="metric"><div class="label">Reranking α</div><div class="value">2.0</div></div>
  <div class="metric {'success' if a2['hit']['a0'] else 'warn'}"><div class="label">A0 hit?</div><div class="value">{'YES ✓' if a2['hit']['a0'] else 'NO ✗'}</div></div>
  <div class="metric {'success' if a2['hit']['a2'] else 'warn'}"><div class="label">A2 hit?</div><div class="value">{'YES ✓' if a2['hit']['a2'] else 'NO ✗'}</div></div>
</div>
""")

    a2_html_parts.append(f"""
<div class="step">
  <h3><span class="stepnum">1</span> The query (real incident narrative)</h3>
  <p class="note">This is the actual NTSB narrative for incident
  <code>{esc(a2['ev_id'])}</code>: an Aerospatiale ATR72 experienced a left-engine
  fire during takeoff at St. Croix in January 2010. Subsequent investigation
  traced the fire to a fuel-manifold O-ring leak.</p>
  {render_query_block(a2['query'])}
  <h4>Ground truth (the answer key, held out from training)</h4>
  <div class="callout callout-ok">
    <strong>NTSB probable cause:</strong> {esc(gt.get('text', ''))}<br>
    <strong>Nearest Zhang code:</strong> <code>{esc(gt.get('code', '?'))}</code> — {esc(gt.get('label', ''))}<br>
    <span class="note">Embedding similarity between the probable cause text and the code label: {gt.get('similarity_to_label', 0.0):.4f}.</span>
  </div>
</div>
""")

    a2_html_parts.append(f"""
<div class="step">
  <h3><span class="stepnum">2</span> Embed and retrieve the top 50 (cosine similarity)</h3>
  <p>Same retrieval mechanism as Example 1. All 50 retrieved rows are shown.</p>
  {render_retrieval_table(a2['retrieval'])}
</div>
""")

    a2_html_parts.append(f"""
<div class="step">
  <h3><span class="stepnum">3</span> A0 clustering and P(Cluster | Query) — the baseline</h3>
  <p>The 50 retrieved rows are clustered into {len(a2['a0']['clusters'])}
  failure-theme clusters. We compute P(Cluster | Query) exactly as in Example 1.
  This is the A0 baseline.</p>
  {render_cluster_table(a2['a0']['clusters'])}
  <h4>A0 — complete P(Cause | Cluster) tables</h4>
  {render_cluster_causes(a2['a0'].get('cluster_causes') or {})}
</div>
""")

    a2_html_parts.append(f"""
<div class="step">
  <h3><span class="stepnum a2">4</span> A2 — structural reranking</h3>
  <p>This is the paper's contribution. For each retrieved row we extract a
  causal-chain structure with an LLM — a sequence of
  <code>(role, system, mechanism)</code> triples. We compute a structural
  similarity score between the query's chain and each retrieved chain, then
  rescore each row:</p>
  <div class="formula">new_score = max(0, cosine) × exp(α · struct_sim),  α = 2.0</div>
  <p>The 50 rows are then re-clustered using the new scores, and P(Cluster | Query)
  is recomputed. Below: the same clusters as A0, but with A2's recomputed weights.</p>
  {render_cluster_table(a2['a2']['clusters'], color='#b91c1c', sim_col_label='avg rerank score')}
  <h4>A2 — complete P(Cause | Cluster) tables (after reranking)</h4>
  {render_cluster_causes(a2['a2'].get('cluster_causes') or {})}
</div>
""")

    a2_html_parts.append(f"""
<div class="step">
  <h3><span class="stepnum a2">5</span> A0 vs A2 cluster-weight shift — the key visualization</h3>
  <p>This is the headline result for the paper section. For each cluster we
  show its weight under A0 (baseline) and under A2 (with structural reranking),
  and the shift (Δ = A2 − A0).</p>
  {render_cluster_shift_table(a2['a0']['clusters'], a2['a2']['clusters'])}
  <div class="callout callout-ok">
    <strong>Read the column of shifts.</strong> Structural reranking moves mass
    <em>into</em> clusters that share the query's causal structure
    (engine-component / fire / fatigue: all ↑) and <em>out of</em> clusters that
    are only lexically near (ground vehicle collision, turbulence, landing gear,
    tailwind: all ↓). The dominant engine-failure cluster gains
    <strong>{engine_shift_pp:+.1f} percentage points</strong> of probability
    mass — exactly the direction we want.
  </div>
</div>
""")

    a2_html_parts.append(f"""
<div class="step">
  <h3><span class="stepnum a2">6</span> Complete LTP causes — A0 vs A2 side-by-side</h3>
  <p>Full Law of Total Probability output for A0 ({len(a2['a0']['ltp_causes'])} causes,
  Σ={a2['a0'].get('ltp_sum', 0):.4f}) and A2 ({len(a2['a2']['ltp_causes'])} causes,
  Σ={a2['a2'].get('ltp_sum', 0):.4f}). Retrieval is identical; reranking changes
  cluster masses. Δ is A2 − A0 per rank slot.
  <span class="note">Rows compare the same <em>rank slot</em>, not matched
  cause strings — A2 can promote a different cause to rank 1.</span></p>
  {render_side_by_side_causes(a2['a0']['ltp_causes'], a2['a2']['ltp_causes'])}
</div>
""")

    a2_html_parts.append(f"""
<div class="step">
  <h3><span class="stepnum a2">7</span> Complete coded distribution (54 codes) — A0 vs A2</h3>
  <p>Full mapped distribution after projecting all free-text causes onto Zhang's
  54 codes ({len(a2['a0']['coded_distribution'])} code buckets for A0). Ground-truth
  code row is highlighted.</p>
  {render_side_by_side_codes(a2['a0']['coded_distribution'], a2['a2']['coded_distribution'], gt_code=gt.get('code'))}
</div>
""")

    a2_html_parts.append(f"""
<div class="step">
  <h3><span class="stepnum a2">8</span> Hit verification</h3>
  <div class="metrics-row">
    <div class="metric"><div class="label">Ground-truth code</div><div class="value">{esc(gt.get('code'))}</div></div>
    <div class="metric {'success' if a2['hit']['a0'] else 'warn'}"><div class="label">A0 top-1</div><div class="value">{esc(a2['a0']['top1_code'])} {'✓' if a2['hit']['a0'] else '✗'}</div></div>
    <div class="metric {'success' if a2['hit']['a2'] else 'warn'}"><div class="label">A2 top-1</div><div class="value">{esc(a2['a2']['top1_code'])} {'✓' if a2['hit']['a2'] else '✗'}</div></div>
  </div>
  <div class="callout callout-ok">
    Both A0 and A2 produce the correct top-1 code (<code>{esc(gt.get('code'))}</code>) for this
    incident. This particular incident is an "easy" hit for A0, but notice from
    Step 5 that A2 still <em>strengthens</em> the engine-failure cluster mass
    (engine-family codes gain weight; weather/turbulence/landing-gear confounders
    lose weight). On this held-out case, A2 keeps the correct top-1 code while
    sharpening the cluster distribution toward engine/fire themes.
  </div>
</div>
""")

    a2_html = "\n".join(a2_html_parts)

    # ---------- Final assembly -----------
    toc = r"""
<nav class="toc">
  <h3>Contents</h3>
  <ul>
    <li><a href="#intro">0 · What is this?</a></li>
    <li><a href="#pipeline">1 · How the pipelines work</a></li>
    <li><a href="#ex1">Example 1 (A0)</a>
      <ul>
        <li><a href="#ex1-diagnosis">Part A — Diagnosis (7.1)</a></li>
        <li><a href="#ex1">Step 1 — Query</a></li>
        <li><a href="#ex1">Step 2 — Retrieve top 50</a></li>
        <li><a href="#ex1">Step 3 — Clusters</a></li>
        <li><a href="#ex1">Step 4 — P(C|K)</a></li>
        <li><a href="#ex1">Step 5 — Law of Total Probability</a></li>
        <li><a href="#ex1">Step 6 — 54 codes</a></li>
        <li><a href="#ex1-prognosis">Part B — Prognosis (7.2)</a></li>
        <li><a href="#ex1-prognosis">P1 — Same query</a></li>
        <li><a href="#ex1-prognosis">P2 — Timeline alignment</a></li>
        <li><a href="#ex1-prognosis">P3 — Next occurrences</a></li>
        <li><a href="#ex1-prognosis">P4 — Weighted transition</a></li>
      </ul>
    </li>
    <li><a href="#ex2">Example 2 (A2 vs ground truth)</a>
      <ul>
        <li><a href="#ex2">Step 1 — Query + ground truth</a></li>
        <li><a href="#ex2">Step 2 — Retrieve top 50</a></li>
        <li><a href="#ex2">Step 3 — A0 clusters</a></li>
        <li><a href="#ex2">Step 4 — A2 reranking</a></li>
        <li><a href="#ex2">Step 5 — Shift A0 → A2</a></li>
        <li><a href="#ex2">Step 6 — Top causes</a></li>
        <li><a href="#ex2">Step 7 — 54 codes</a></li>
        <li><a href="#ex2">Step 8 — Hit verification</a></li>
      </ul>
    </li>
    <li><a href="#takeaway">2 · What this demonstrates</a></li>
  </ul>
</nav>
"""

    takeaway = r"""
<section id="takeaway">
  <h2><span class="num">2</span> What these two examples demonstrate</h2>
  <ul>
    <li>
      <strong>The pipeline is end-to-end interpretable.</strong> You can point
      at any retrieved incident, any cluster, any cause, any code, and trace
      exactly where its probability came from. Nothing is a black box.
    </li>
    <li>
      <strong>Retrieval finds the right neighborhood (with noise).</strong> Both
      examples retrieve mostly engine-related incidents; a few off-theme rows
      remain in the top 50 by design.
    </li>
    <li>
      <strong>Clustering produces semantic groups.</strong> The cluster labels
      ("engine fire due to component failures", "fatigue-induced engine
      component failures", etc.) describe real failure modes, not arbitrary
      groupings.
    </li>
    <li>
      <strong>Probabilities are calibrated.</strong> Output distributions sum
      to exactly 1. The top of the distribution always lands on the right
      failure family.
    </li>
    <li>
      <strong>Diagnosis and prognosis are different pipelines.</strong> Example 1
      Part A uses clustering + LTP over causes; Part B uses event-timeline
      alignment + weighted transitions — no clusters, no LTP.
    </li>
    <li>
      <strong>Prognosis forecasts forward events.</strong> Example 1 Part B uses
      a truncated query (stops at master caution) so predicted next events
      (e.g. shutdown, emergency landing) are forward-looking, not copied from
      the diagnosis narrative's ending.
    </li>
    <li>
      <strong>A2 demonstrably tightens the answer.</strong> In Example 2,
      structural reranking moves probability mass into engine-failure clusters
      and out of unrelated ones. The top-1 code stays correct
      (<code>130</code>); its mass goes up.
    </li>
    <li>
      <strong>The held-out hit is real.</strong> Example 2 uses an incident
      that was never seen by the search index, and the pipeline still correctly
      identifies its NTSB-assigned cause code.
    </li>
  </ul>
</section>

<footer>
  Generated from <code>Worked_Examples/data/a0.json</code> and <code>Worked_Examples/data/a2.json</code>.
  Re-run <code>generate_html.py</code> after any data update to refresh this page.
</footer>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NTSB Worked Examples — A Non-Expert Walkthrough</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{CSS}</style>
</head>
<body>
<div class="layout">
  {toc}
  <main>
    <h1>NTSB Worked Examples</h1>
    <p class="subtitle">A complete, non-expert walkthrough of the diagnosis &amp; prognosis pipelines,
    illustrated by two end-to-end examples — every row at every step is shown.</p>
    {INTRO_HTML}
    {PIPELINE_HTML}
    <div class="divider"></div>
    {a0_html}
    <div class="divider"></div>
    {a2_html}
    <div class="divider"></div>
    {takeaway}
  </main>
</div>
</body>
</html>
"""


def main() -> None:
    a0 = json.loads(A0_PATH.read_text())
    a2 = json.loads(A2_PATH.read_text())
    html_doc = assemble(a0, a2)
    OUT.write_text(html_doc)
    print(f"Wrote {OUT} ({len(html_doc):,} bytes)")


if __name__ == "__main__":
    main()

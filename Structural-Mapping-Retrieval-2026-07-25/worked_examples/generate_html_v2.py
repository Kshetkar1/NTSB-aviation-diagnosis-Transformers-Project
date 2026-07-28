"""
Generate the *new* worked-example HTML for the paper.

Reads:
  Worked_Examples/data/example_after.json   (after-incident scenario)
  Worked_Examples/data/example_during.json  (during-incident, pilot voice)

Renders a single self-contained HTML page with TWO scenario panels. Each
panel shows, for both A0 (embedding-only) and A2 (structural reranking,
alpha=2.0), for both diagnosis and prognosis:

  1. A DECISION BOX at the top - the operationally-useful top-1 answer
     (plain English label + 1-sentence elaboration + probability)
  2. Top-5 enriched cause table  (with elaborations)
  3. Top-5 enriched coded distribution (with elaborations)
  4. Top-5 enriched next-event table (with elaborations)
  5. Collapsible <details> for the full raw artifacts (retrieval,
     all clusters, full LTP tables, aligned incidents)

Output: Worked_Examples/worked_examples_v2.html
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from aggregate_summary import (
    DEFAULT_RESULTS_PATH as AGG_RESULTS_PATH,
    compute_summary as agg_compute_summary,
    load_results as agg_load_results,
    best_and_worst as agg_best_and_worst,
)

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "worked_examples_v2.html"

# Auto-discovered: a "case" is the pair (example_after{slug}.json,
# example_during{slug}.json) where slug is "" (empty for the legacy engine-fire
# case) or "_<slug>" for subsequent test cases.
CASE_LABELS = {
    "": "Case 1 - ATR72 left-engine fire on takeoff (St. Croix, 2010)",
    "landing_gear": "Case 2 - DHC-8 nose-gear retracted on landing (Philadelphia, 2008)",
}


def esc(s) -> str:
    return html.escape("" if s is None else str(s))


def pct(p: float) -> str:
    return f"{p * 100:.1f}%"


# ----------------------------------------------------------------------------
# Static prose blocks
# ----------------------------------------------------------------------------

INTRO_HTML = r"""
<section id="intro">
  <h2><span class="num">0</span> What is this document?</h2>
  <p>
    This page demonstrates a tool that reads an aviation accident narrative
    and tells a pilot or investigator <strong>two operationally useful
    things</strong>:
  </p>
  <ol>
    <li>
      <strong>Diagnosis (what caused this?)</strong> &mdash; given the
      narrative we just heard, what is the most likely cause? Includes a
      one-sentence plain-English explanation of each top cause.
    </li>
    <li>
      <strong>Prognosis (what happens next?)</strong> &mdash; given the
      narrative we just heard, what is the most likely <em>next event</em>
      in the incident timeline (e.g. engine shutdown, ditching, emergency
      landing)? Also explained in plain English.
    </li>
  </ol>
  <p>
    Both questions are answered by retrieving similar past NTSB incidents
    (cosine similarity on text embeddings), clustering them into failure
    themes, and combining the cluster-level probabilities via the
    <strong>Law of Total Probability</strong>:
  </p>
  <div class="formula">
    P(Cause | Query) = &Sigma;<sub>K</sub> P(Cause | Cluster K) &middot; P(Cluster K | Query)
  </div>
  <p>
    The same formula is used for prognosis with &quot;next event&quot; in
    place of &quot;cause&quot;.
  </p>
  <div class="callout">
    <strong>Two pipelines, two scenarios per pipeline:</strong>
    <ul>
      <li>
        <strong>A0 (baseline)</strong> &mdash; cosine similarity only.
      </li>
      <li>
        <strong>A2 (this paper's contribution)</strong> &mdash; re-rank the
        retrieved cases with an LLM-extracted causal-chain similarity:
        <code>score = max(0, cosine) &times; exp(&alpha; &middot; struct_sim)</code>,
        &alpha; = 2.0.
      </li>
    </ul>
    <p style="margin-top:8px">
      <strong>After-incident</strong> uses the full NTSB narrative (the
      investigator&rsquo;s view, after the dust has settled).
      <strong>During-incident</strong> uses a short pilot-voice query as
      if the pilot is calling ATC mid-event &mdash; testing whether the
      same tool can produce a decision-useful answer from a single sentence.
    </p>
    <p style="margin-top:8px">
      <strong>Verification</strong> for each held-out test case is done by
      comparing the pipeline's top failure-theme cluster and top free-text
      cause directly to the NTSB&rsquo;s probable-cause text. No external
      taxonomy projection is required &mdash; the tool stands on its own.
    </p>
  </div>
</section>
"""


# ----------------------------------------------------------------------------
# Atomic renderers
# ----------------------------------------------------------------------------

def render_query(query: str, badge: str = "Query") -> str:
    return f"""
<div class="query-box">
  <div class="query-badge">{esc(badge)}</div>
  <blockquote>{esc(query)}</blockquote>
</div>
"""


def _build_card(model_label: str, model_css: str, diag: dict, prog: dict) -> str:
    """Render one decision card for a single pipeline (A0 or A2).

    Headline = top cluster by P(K|Q) (LLM-named, pilot-readable).
    Below = top specific cause + top next event (enriched, top-1 from LTP).
    """
    top_cluster = diag["clusters"][0] if diag.get("clusters") else None
    top_cause = diag["ltp_causes_enriched"][0] if diag.get("ltp_causes_enriched") else None
    top_event = prog["ltp_events_enriched"][0] if prog.get("ltp_events_enriched") else None

    cluster_html = ""
    if top_cluster:
        cluster_html = f"""
    <div class="dc-row dc-hero">
      <div class="dc-label">Most likely failure theme &mdash; the headline answer</div>
      <div class="dc-headline-big">{esc(top_cluster['cluster']).capitalize()}</div>
      <div class="dc-prob hero">{pct(top_cluster['p_k_given_q'])}</div>
      <div class="dc-elab">Based on <strong>{top_cluster['n_incidents']}</strong> of the top-50 retrieved past incidents clustering into this failure theme. Cluster label is generated by the LLM (temperature 0) from the retrieved cases.</div>
    </div>"""

    cause_html = ""
    if top_cause:
        cause_html = f"""
    <div class="dc-row">
      <div class="dc-label">Most likely specific cause</div>
      <div class="dc-headline">{esc(top_cause['label'])}</div>
      <div class="dc-prob small">{pct(top_cause['probability'])}</div>
      <div class="dc-elab">{esc(top_cause['elaboration'])}</div>
    </div>"""

    event_html = ""
    if top_event:
        event_html = f"""
    <div class="dc-row">
      <div class="dc-label">Most likely next event</div>
      <div class="dc-headline">{esc(top_event['label'])}</div>
      <div class="dc-prob small">{pct(top_event['probability'])}</div>
      <div class="dc-elab">{esc(top_event['elaboration'])}</div>
    </div>"""

    return f"""
  <div class="decision-card {model_css}">
    <div class="dc-badge">{esc(model_label)}</div>
    {cluster_html}
    {cause_html}
    {event_html}
  </div>"""


def render_decision_box(scenario: dict) -> str:
    """Top-of-panel summary: what a pilot/investigator should read first."""
    a0_diag = scenario["diagnosis"]["a0"]
    a2_diag = scenario["diagnosis"]["a2"]
    a0_prog = scenario["prognosis"]["a0"]
    a2_prog = scenario["prognosis"]["a2"]

    return f"""
<div class="decision-grid">
  {_build_card('A0 - Embedding only', 'a0', a0_diag, a0_prog)}
  {_build_card('A2 - With structural rerank', 'a2', a2_diag, a2_prog)}
</div>
"""


def _retrieval_index(retrieval: list[dict]) -> dict[str, dict]:
    """ev_id -> retrieval row."""
    return {str(r.get("ev_id", "")): r for r in (retrieval or [])}


def _best_cluster_for(item_text: str, clusters: list[dict],
                      cluster_members: dict[str, list[dict]],
                      item_key: str) -> dict | None:
    """Find the cluster with the highest P(K|Q) that contains item_text."""
    best = None
    for label, members in (cluster_members or {}).items():
        for m in members:
            if str(m.get(item_key, "")).strip() == str(item_text).strip():
                p_k = next((c["p_k_given_q"] for c in clusters if c["cluster"] == label), 0.0)
                n = next((c["n_incidents"] for c in clusters if c["cluster"] == label), 0)
                if best is None or p_k > best["p_k_given_q"]:
                    best = {"label": label, "p_k_given_q": p_k, "n_incidents": n}
                break
    return best


def _cluster_evidence_ids(cluster_label: str, retrieval: list[dict], k: int = 2) -> list[dict]:
    """Pull top-k retrieval rows from the same scenario as evidence chips.
    We don't have explicit cluster->ev_id membership in the exported JSON, so
    we fall back to the highest-similarity retrieved rows whose ev_id we
    can show as representative neighbors."""
    return (retrieval or [])[:k]


def _render_evidence_chips(cluster_info: dict | None, retrieval_idx: dict) -> str:
    if not cluster_info or cluster_info.get("p_k_given_q", 0.0) <= 0:
        return ""
    chips = ""
    return f"""
    <div class="evidence">
      <span class="evidence-label">Strongest supporting cluster:</span>
      <span class="evidence-cluster">"{esc(cluster_info['label'])}"</span>
      <span class="evidence-meta">n={cluster_info['n_incidents']} retrieved cases &middot; P(K|Q)={pct(cluster_info['p_k_given_q'])}</span>
    </div>"""


def render_enriched_causes(rows: list[dict], color: str = "#2563eb",
                           clusters: list[dict] | None = None,
                           cluster_members: dict | None = None,
                           retrieval: list[dict] | None = None) -> str:
    if not rows:
        return "<p><em>No data.</em></p>"
    max_p = max(r["probability"] for r in rows) or 1.0
    retr_idx = _retrieval_index(retrieval or [])
    body_parts = []
    for r in rows:
        evidence = _render_evidence_chips(
            _best_cluster_for(r.get("raw_cause", ""), clusters or [], cluster_members or {}, "cause"),
            retr_idx,
        )
        body_parts.append(f"""<tr>
  <td class="num">{r['rank']}</td>
  <td class="num strong">{pct(r['probability'])}
    <div class="bar tiny"><div class="bar-fill" style="width:{(r['probability']/max_p)*100:.1f}%;background:{color}"></div></div>
  </td>
  <td>
    <div class="enriched-label">{esc(r['label'])}</div>
    <div class="enriched-elab">{esc(r['elaboration'])}</div>
    {evidence}
    {('<div class="enriched-raw">raw: <code>' + esc(r.get('raw_cause','')[:140]) + '</code></div>') if r.get('raw_cause') else ''}
  </td>
</tr>""")
    body = "\n".join(body_parts)
    return f"""
<div class="table-wrap">
  <table class="data-table enriched">
    <thead><tr><th style="width:48px">#</th><th style="width:120px">Probability</th><th>Most likely cause (plain English + elaboration + evidence)</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
</div>
"""


def render_enriched_codes(rows: list[dict], color: str = "#2563eb", gt_code: str | None = None) -> str:
    if not rows:
        return "<p><em>No data.</em></p>"
    max_p = max(r["probability"] for r in rows) or 1.0
    body = []
    for r in rows:
        klass = "highlight" if (gt_code and str(r.get("code")) == str(gt_code)) else ""
        body.append(
            f"""<tr class="{klass}">
  <td class="num">{r['rank']}</td>
  <td class="num"><code>{esc(r.get('code', '-'))}</code></td>
  <td class="num strong">{pct(r['probability'])}
    <div class="bar tiny"><div class="bar-fill" style="width:{(r['probability']/max_p)*100:.1f}%;background:{color}"></div></div>
  </td>
  <td>
    <div class="enriched-label">{esc(r['label'])}</div>
    <div class="enriched-elab">{esc(r['elaboration'])}</div>
    {('<div class="enriched-raw">raw: <code>' + esc(r.get('raw_label','')) + '</code></div>') if r.get('raw_label') else ''}
  </td>
</tr>"""
        )
    return f"""
<div class="table-wrap">
  <table class="data-table enriched">
    <thead><tr><th style="width:48px">#</th><th style="width:60px">Code</th><th style="width:120px">Probability</th><th>Zhang code family (plain English + elaboration)</th></tr></thead>
    <tbody>{''.join(body)}</tbody>
  </table>
</div>
"""


def render_enriched_events(rows: list[dict], color: str = "#16a34a",
                           clusters: list[dict] | None = None,
                           cluster_members: dict | None = None,
                           retrieval: list[dict] | None = None,
                           aligned: list[dict] | None = None) -> str:
    if not rows:
        return "<p><em>No data.</em></p>"
    max_p = max(r["probability"] for r in rows) or 1.0
    retr_idx = _retrieval_index(retrieval or [])
    aligned_idx: dict[str, list[str]] = {}
    for a in (aligned or []):
        ev = str(a.get("next_event", "")).strip()
        if not ev or not a.get("has_downstream"):
            continue
        aligned_idx.setdefault(ev, []).append(str(a.get("ev_id", "")))
    body_parts = []
    for r in rows:
        evidence = _render_evidence_chips(
            _best_cluster_for(r.get("raw_event", ""), clusters or [], cluster_members or {}, "event"),
            retr_idx,
        )
        # Aligned incident chips (specific historical cases where this exact
        # next event followed the defining event in the timeline).
        ids = aligned_idx.get(r.get("raw_event", "").strip(), [])
        cases_html = ""
        if ids:
            chips = " ".join(f"<code>{esc(i)}</code>" for i in ids[:3])
            more = f" + {len(ids) - 3} more" if len(ids) > 3 else ""
            cases_html = (f'<div class="evidence-cases">Confirmed in past incidents: '
                          f'{chips}{esc(more)}</div>')
        body_parts.append(f"""<tr>
  <td class="num">{r['rank']}</td>
  <td class="num strong">{pct(r['probability'])}
    <div class="bar tiny"><div class="bar-fill" style="width:{(r['probability']/max_p)*100:.1f}%;background:{color}"></div></div>
  </td>
  <td>
    <div class="enriched-label">{esc(r['label'])}</div>
    <div class="enriched-elab">{esc(r['elaboration'])}</div>
    {evidence}
    {cases_html}
    {('<div class="enriched-raw">raw: <code>' + esc(r.get('raw_event','')) + '</code></div>') if r.get('raw_event') else ''}
  </td>
</tr>""")
    body = "\n".join(body_parts)
    return f"""
<div class="table-wrap">
  <table class="data-table enriched">
    <thead><tr><th style="width:48px">#</th><th style="width:120px">Probability</th><th>Most likely next event (plain English + elaboration + evidence)</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
</div>
"""


def render_retrieval(rows: list[dict]) -> str:
    if not rows:
        return "<p><em>No data.</em></p>"
    max_s = max(r["score"] for r in rows) or 1.0
    body = "\n".join(
        f"<tr><td class='num'>{r['rank']}</td>"
        f"<td><code>{esc(r['ev_id'])}</code></td>"
        f"<td class='num'>{r['score']:.4f}"
        f"<div class='bar tiny'><div class='bar-fill' style='width:{(r['score']/max_s)*100:.1f}%;background:#2563eb'></div></div></td>"
        f"<td class='snippet'>{esc(r.get('snippet') or '')}</td></tr>"
        for r in rows
    )
    return f"""
<div class="table-wrap">
  <table class="data-table">
    <thead><tr><th style="width:48px">#</th><th style="width:160px">Incident</th><th style="width:120px">Cosine</th><th>Narrative snippet</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
</div>
"""


def render_clusters(rows: list[dict], kind: str = "diagnosis") -> str:
    if not rows:
        return "<p><em>No clusters.</em></p>"
    max_p = max(r["p_k_given_q"] for r in rows) or 1.0
    extra_head = "<th style='width:90px'>seqs w/ next</th>" if kind == "prognosis" else ""
    body_rows = []
    for r in rows:
        extra = f"<td class='num'>{r.get('sequences_with_next', '-')}</td>" if kind == "prognosis" else ""
        body_rows.append(
            f"<tr><td>{esc(r['cluster'])}</td>"
            f"<td class='num'>{r['n_incidents']}</td>"
            f"<td class='num'>{r['avg_similarity']:.4f}</td>"
            f"<td class='num'>{r['weight']:.4f}</td>"
            f"<td class='num strong'>{r['p_k_given_q']:.4f}"
            f"<div class='bar tiny'><div class='bar-fill' style='width:{(r['p_k_given_q']/max_p)*100:.1f}%;background:#2563eb'></div></div></td>"
            f"{extra}</tr>"
        )
    return f"""
<div class="table-wrap">
  <table class="data-table">
    <thead><tr>
      <th>Cluster (failure theme)</th><th style="width:50px">n</th>
      <th style="width:110px">avg cos sim</th><th style="width:110px">sim&middot;n weight</th>
      <th style="width:130px">P(K | Q)</th>{extra_head}
    </tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</div>
"""


def render_raw_ltp_causes(rows: list[dict]) -> str:
    if not rows:
        return "<p><em>No data.</em></p>"
    body = "\n".join(
        f"<tr><td class='num'>{r['rank']}</td>"
        f"<td class='num'>{r['probability']:.4f}</td>"
        f"<td class='snippet'>{esc(r.get('cause',''))}</td></tr>"
        for r in rows
    )
    return f"""
<div class="table-wrap">
  <table class="data-table">
    <thead><tr><th style="width:48px">#</th><th style="width:110px">P(C|Q)</th><th>Raw cause (NTSB taxonomy text)</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
</div>
"""


def render_raw_ltp_events(rows: list[dict]) -> str:
    if not rows:
        return "<p><em>No data.</em></p>"
    body = "\n".join(
        f"<tr><td class='num'>{r['rank']}</td>"
        f"<td class='num'>{r['probability']:.4f}</td>"
        f"<td class='snippet'>{esc(r.get('event',''))}</td></tr>"
        for r in rows
    )
    return f"""
<div class="table-wrap">
  <table class="data-table">
    <thead><tr><th style="width:48px">#</th><th style="width:110px">P(next|Q)</th><th>Raw next event (NTSB occurrence text)</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
</div>
"""


def render_aligned_incidents(rows: list[dict]) -> str:
    if not rows:
        return "<p><em>No aligned incidents.</em></p>"
    body = []
    for r in rows:
        downstream = "yes" if r.get("has_downstream") else "no"
        body.append(
            f"<tr><td class='num'>{r['rank']}</td>"
            f"<td><code>{esc(r.get('ev_id',''))}</code></td>"
            f"<td class='num'>{r.get('incident_similarity', 0.0):.4f}</td>"
            f"<td class='snippet'>{esc(r.get('defining_event',''))}</td>"
            f"<td class='snippet'>{esc(r.get('next_event','')) or '-'}</td>"
            f"<td class='num'>{downstream}</td></tr>"
        )
    return f"""
<div class="table-wrap">
  <table class="data-table">
    <thead><tr>
      <th>#</th><th>Incident</th><th>Sim</th><th>Defining event</th><th>Next event</th><th>Downstream?</th>
    </tr></thead>
    <tbody>{''.join(body)}</tbody>
  </table>
</div>
"""


def render_cluster_members(cluster_dict: dict, item_key: str) -> str:
    if not cluster_dict:
        return "<p><em>No clusters with members.</em></p>"
    blocks = []
    for label, members in cluster_dict.items():
        rows_html = "\n".join(
            f"<tr><td class='num'>{m['rank']}</td>"
            f"<td class='num'>{m.get('p_c_given_k', m.get('p_next_given_k', 0.0)):.4f}</td>"
            f"<td class='snippet'>{esc(m[item_key])}</td></tr>"
            for m in members
        )
        prob_head = "P(C|K)" if item_key == "cause" else "P(next|K)"
        header_label = "Cause" if item_key == "cause" else "Next event"
        blocks.append(f"""
<details class="cluster-causes">
  <summary><strong>{esc(label)}</strong> &mdash; {len(members)} member{'s' if len(members) != 1 else ''}</summary>
  <div class="table-wrap">
    <table class="data-table sub">
      <thead><tr><th style="width:40px">#</th><th style="width:90px">{prob_head}</th><th>{header_label}</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</details>
""")
    return "\n".join(blocks)


# ----------------------------------------------------------------------------
# Scenario-level assembly
# ----------------------------------------------------------------------------

def render_scenario(scenario: dict, anchor_id: str) -> str:
    """Build a full HTML panel for one scenario (after_incident or during_incident)."""
    name = scenario["scenario"]
    ev_id = scenario.get("ev_id", "")
    diag_q = scenario.get("diagnosis_query", "")
    prog_q = scenario.get("prognosis_query", "")
    same_q = diag_q.strip() == prog_q.strip()
    gt = scenario.get("ground_truth", {}) or {}
    gt_code = gt.get("code") if gt else None
    alpha = scenario.get("alpha", 2.0)

    if name == "after_incident":
        title = "Scenario A &mdash; After-incident (investigator view)"
        subtitle = (
            f"The investigator has the full NTSB narrative in hand. Test incident "
            f"<code>{esc(ev_id)}</code> (ATR72 left-engine fire on takeoff, St. Croix, 2010). "
            f"Diagnosis query = full narrative; prognosis query = a truncated cut "
            f"(first observation only) so the next-event prediction is a real forecast."
        )
        gradient = "linear-gradient(135deg, #003366, #0066cc)"
    else:
        title = "Scenario B &mdash; During-incident (pilot voice)"
        subtitle = (
            f"The pilot is on the radio mid-event. Same incident "
            f"<code>{esc(ev_id)}</code>, but the only thing we have is a single "
            f"observation sentence in the pilot&rsquo;s own words. Tests whether "
            f"the tool can produce a useful answer from this short, noisy input. "
            f"Diagnosis and prognosis use the <em>same</em> short query."
        )
        gradient = "linear-gradient(135deg, #7f1d1d, #b91c1c)"

    parts = [f"""
<section id="{anchor_id}">
  <div class="example-header" style="background: {gradient};">
    <h2 style="border:none; color:#fff; margin:0">{title}</h2>
    <div class="sub">{subtitle}</div>
  </div>
""" ]

    # Pre-extract the four model views once - both the decision box and the
    # ground-truth comparison need them.
    a0_diag = scenario["diagnosis"]["a0"]
    a2_diag = scenario["diagnosis"]["a2"]
    a0_prog = scenario["prognosis"]["a0"]
    a2_prog = scenario["prognosis"]["a2"]

    # Decision box - top of fold
    parts.append(f"""
  <h3 class="decision-h3">At a glance &mdash; what should the user look at?</h3>
  <p class="note">Each pipeline (A0 baseline, A2 with structural reranking) reports three things in order of decision-importance: <strong>(1)</strong> the most likely failure theme (the LLM-named cluster the retrieved past incidents fall into &mdash; the headline answer a pilot or investigator wants); <strong>(2)</strong> the most likely specific NTSB-taxonomy cause; <strong>(3)</strong> the most likely next event in the incident timeline. All three come straight from the Law of Total Probability output.</p>
  {render_decision_box(scenario)}
""")

    # Queries
    parts.append(f"""
  <h3>The queries fed into the pipeline</h3>
  {render_query(diag_q, badge='Diagnosis query (' + ('full narrative' if name == 'after_incident' else 'pilot voice') + ')')}
""")
    if not same_q:
        parts.append(render_query(prog_q, badge="Prognosis query (truncated to first observation)"))
    else:
        parts.append('<p class="note"><em>Prognosis uses the same short query as diagnosis.</em></p>')

    # Ground truth: held-out NTSB probable cause text. We compare it to:
    #   - the top cluster label (does the cluster name describe the cause?)
    #   - the top free-text cause (does the LTP top-1 match what NTSB recorded?)
    # No Zhang codes are involved.
    if gt and gt.get("text"):
        a0_top_cluster = a0_diag["clusters"][0]["cluster"] if a0_diag.get("clusters") else ""
        a2_top_cluster = a2_diag["clusters"][0]["cluster"] if a2_diag.get("clusters") else ""
        a0_top_cause = a0_diag["ltp_causes_enriched"][0]["label"] if a0_diag.get("ltp_causes_enriched") else ""
        a2_top_cause = a2_diag["ltp_causes_enriched"][0]["label"] if a2_diag.get("ltp_causes_enriched") else ""
        parts.append(f"""
  <div class="callout callout-ok">
    <strong>Ground truth (NTSB held-out probable cause):</strong> &ldquo;{esc(gt.get('text',''))[:300]}&rdquo;
    <div class="gt-grid">
      <div class="gt-cell">
        <div class="gt-cell-head">A0 top cluster vs ground truth</div>
        <div class="gt-cell-body"><strong>Cluster:</strong> {esc(a0_top_cluster)}</div>
        <div class="gt-cell-body"><strong>Top cause (LTP):</strong> {esc(a0_top_cause)}</div>
      </div>
      <div class="gt-cell">
        <div class="gt-cell-head">A2 top cluster vs ground truth</div>
        <div class="gt-cell-body"><strong>Cluster:</strong> {esc(a2_top_cluster)}</div>
        <div class="gt-cell-body"><strong>Top cause (LTP):</strong> {esc(a2_top_cause)}</div>
      </div>
    </div>
    <p class="gt-note">A correct prediction means the cluster label and/or top free-text cause semantically describe the NTSB probable cause. Read the three texts and judge for yourself; this is exactly what an investigator would do.</p>
  </div>
""")

    # Diagnosis tables (enriched, top-5)
    diag_retrieval = scenario.get("diagnosis", {}).get("retrieval", [])
    prog_retrieval = scenario.get("prognosis", {}).get("retrieval", [])

    parts.append(f"""
  <h3>Diagnosis &mdash; top 5 most likely causes (plain English, with evidence)</h3>
  <p class="note">Top 5 most likely causes after combining cluster probabilities via the Law of Total Probability. Each row carries an LLM-generated plain-English label, a one-sentence operational meaning, and a pointer back to the cluster of past incidents that contributed the most weight to that row. Full raw distributions are in the toggles below.</p>
  <div class="side-by-side">
    <div class="sbs-col">
      <h4>A0 &mdash; Specific cause (top 5)</h4>
      {render_enriched_causes(a0_diag['ltp_causes_enriched'], '#2563eb',
                               clusters=a0_diag.get('clusters'),
                               cluster_members=a0_diag.get('cluster_causes'),
                               retrieval=diag_retrieval)}
    </div>
    <div class="sbs-col">
      <h4>A2 &mdash; Specific cause (top 5)</h4>
      {render_enriched_causes(a2_diag['ltp_causes_enriched'], '#b91c1c',
                               clusters=a2_diag.get('clusters'),
                               cluster_members=a2_diag.get('cluster_causes'),
                               retrieval=diag_retrieval)}
    </div>
  </div>
""")

    # Prognosis tables (enriched, top-5)
    parts.append(f"""
  <h3>Prognosis &mdash; top 5 next events (enriched, plain English, with evidence)</h3>
  <p class="note">For each retrieved cluster we look at the immediate next event in those incidents' coded timelines and combine via LTP. Each row links back to the supporting cluster and to the specific past incidents in which that exact next event was observed.</p>
  <div class="side-by-side">
    <div class="sbs-col">
      <h4>A0 &mdash; Next event (top 5)</h4>
      {render_enriched_events(a0_prog['ltp_events_enriched'], '#16a34a',
                               clusters=a0_prog.get('clusters'),
                               cluster_members=a0_prog.get('cluster_next_events'),
                               retrieval=prog_retrieval,
                               aligned=a0_prog.get('aligned_incidents'))}
    </div>
    <div class="sbs-col">
      <h4>A2 &mdash; Next event (top 5)</h4>
      {render_enriched_events(a2_prog['ltp_events_enriched'], '#0d9488',
                               clusters=a2_prog.get('clusters'),
                               cluster_members=a2_prog.get('cluster_next_events'),
                               retrieval=prog_retrieval,
                               aligned=a2_prog.get('aligned_incidents'))}
    </div>
  </div>
""")

    # LTP sanity check
    parts.append(f"""
  <h3>LTP sanity check (do all four distributions sum to 1?)</h3>
  <div class="metrics-row">
    <div class="metric"><div class="label">A0 diag &Sigma; P(C|Q)</div><div class="value">{a0_diag.get('ltp_sum', 0.0):.4f}</div></div>
    <div class="metric"><div class="label">A2 diag &Sigma; P(C|Q)</div><div class="value">{a2_diag.get('ltp_sum', 0.0):.4f}</div></div>
    <div class="metric"><div class="label">A0 prog &Sigma; P(next|Q)</div><div class="value">{a0_prog.get('ltp_sum', 0.0):.4f}</div></div>
    <div class="metric"><div class="label">A2 prog &Sigma; P(next|Q)</div><div class="value">{a2_prog.get('ltp_sum', 0.0):.4f}</div></div>
  </div>
""")

    # Toggle: raw artifacts
    parts.append(f"""
  <h3>Underlying artifacts (click to expand)</h3>
  <details class="raw-block">
    <summary><strong>Top-50 retrieval (diagnosis cosine ranking)</strong></summary>
    {render_retrieval(scenario['diagnosis']['retrieval'])}
  </details>
""")
    if not same_q:
        parts.append(f"""
  <details class="raw-block">
    <summary><strong>Top-50 retrieval (prognosis cosine ranking)</strong></summary>
    {render_retrieval(scenario['prognosis']['retrieval'])}
  </details>
""")
    parts.append(f"""
  <details class="raw-block">
    <summary><strong>A0 diagnosis clusters &mdash; P(K|Q) and members</strong></summary>
    {render_clusters(a0_diag['clusters'])}
    <h4>P(Cause | Cluster) for each cluster</h4>
    {render_cluster_members(a0_diag['cluster_causes'], 'cause')}
  </details>
  <details class="raw-block">
    <summary><strong>A2 diagnosis clusters &mdash; P(K|Q) and members</strong></summary>
    {render_clusters(a2_diag['clusters'])}
    <h4>P(Cause | Cluster) for each cluster</h4>
    {render_cluster_members(a2_diag['cluster_causes'], 'cause')}
  </details>
  <details class="raw-block">
    <summary><strong>A0 prognosis clusters &mdash; P(K|Q) and per-cluster next-event tables</strong></summary>
    {render_clusters(a0_prog['clusters'], kind='prognosis')}
    {render_cluster_members(a0_prog['cluster_next_events'], 'event')}
  </details>
  <details class="raw-block">
    <summary><strong>A2 prognosis clusters &mdash; P(K|Q) and per-cluster next-event tables</strong></summary>
    {render_clusters(a2_prog['clusters'], kind='prognosis')}
    {render_cluster_members(a2_prog['cluster_next_events'], 'event')}
  </details>
  <details class="raw-block">
    <summary><strong>Full raw LTP cause distribution (A0)</strong> &mdash; all {len(a0_diag['ltp_causes'])} rows</summary>
    {render_raw_ltp_causes(a0_diag['ltp_causes'])}
  </details>
  <details class="raw-block">
    <summary><strong>Full raw LTP cause distribution (A2)</strong> &mdash; all {len(a2_diag['ltp_causes'])} rows</summary>
    {render_raw_ltp_causes(a2_diag['ltp_causes'])}
  </details>
  <details class="raw-block">
    <summary><strong>Full raw LTP next-event distribution (A0)</strong> &mdash; all {len(a0_prog['ltp_events'])} rows</summary>
    {render_raw_ltp_events(a0_prog['ltp_events'])}
  </details>
  <details class="raw-block">
    <summary><strong>Full raw LTP next-event distribution (A2)</strong> &mdash; all {len(a2_prog['ltp_events'])} rows</summary>
    {render_raw_ltp_events(a2_prog['ltp_events'])}
  </details>
  <details class="raw-block">
    <summary><strong>Aligned incidents (A0 prognosis) &mdash; defining event &rarr; next event</strong></summary>
    {render_aligned_incidents(a0_prog['aligned_incidents'])}
  </details>
  <details class="raw-block">
    <summary><strong>Aligned incidents (A2 prognosis) &mdash; defining event &rarr; next event</strong></summary>
    {render_aligned_incidents(a2_prog['aligned_incidents'])}
  </details>
</section>
""")

    return "\n".join(parts)


# ----------------------------------------------------------------------------
# Aggregate validation (n=77 held-out test set)
# ----------------------------------------------------------------------------

def _agg_metric_card(label: str, a0: str, a2: str, sub: str = "") -> str:
    return f"""
<div class="agg-card">
  <div class="agg-card-label">{esc(label)}</div>
  <div class="agg-card-row">
    <div class="agg-card-pill agg-a0"><span>A0</span><strong>{esc(a0)}</strong></div>
    <div class="agg-card-pill agg-a2"><span>A2</span><strong>{esc(a2)}</strong></div>
  </div>
  {f'<div class="agg-card-sub">{esc(sub)}</div>' if sub else ''}
</div>
"""


def _agg_per_incident_table(rows: list[dict], showcase_ids: set[str]) -> str:
    """Per-incident details. Sorted by best A2 cluster sim DESC."""
    sorted_rows = sorted(rows, key=lambda r: float(r.get("sim_cluster_a2", 0.0)), reverse=True)
    body_html: list[str] = []
    for r in sorted_rows:
        ev_id = r.get("ev_id", "")
        is_showcase = ev_id in showcase_ids
        a0_sim = float(r.get("sim_cluster_a0", 0.0))
        a2_sim = float(r.get("sim_cluster_a2", 0.0))

        def _sim_class(s: float) -> str:
            if s >= 0.50: return "ok"
            if s >= 0.40: return "warn"
            return "bad"

        a0_cls = _sim_class(a0_sim)
        a2_cls = _sim_class(a2_sim)
        delta = a2_sim - a0_sim
        delta_html = ""
        if abs(delta) > 0.001:
            sign = "+" if delta > 0 else ""
            delta_color = "#16a34a" if delta > 0 else "#b91c1c"
            delta_html = f'<span style="color:{delta_color}; font-weight:600">{sign}{delta:.3f}</span>'

        showcase_badge = '<span class="pill pill-ok" style="margin-left:6px">SHOWCASE</span>' if is_showcase else ""

        body_html.append(f"""
<tr{' class="highlight"' if is_showcase else ''}>
  <td><code>{esc(ev_id)}</code>{showcase_badge}</td>
  <td class="snippet">{esc((r.get('truth_text') or '')[:170])}{'&hellip;' if len(r.get('truth_text') or '') > 170 else ''}</td>
  <td>{esc(r.get('a0_top_cluster', ''))}</td>
  <td class="num agg-sim-{a0_cls}">{a0_sim:.3f}</td>
  <td>{esc(r.get('a2_top_cluster', ''))}</td>
  <td class="num agg-sim-{a2_cls}">{a2_sim:.3f}</td>
  <td class="num">{delta_html}</td>
</tr>""")

    return f"""
<div class="table-wrap">
  <table class="data-table" style="font-size:12.5px">
    <thead>
      <tr>
        <th>ev_id</th>
        <th>NTSB probable cause (truncated)</th>
        <th>A0 top cluster</th>
        <th>cos(A0)</th>
        <th>A2 top cluster</th>
        <th>cos(A2)</th>
        <th>&Delta;</th>
      </tr>
    </thead>
    <tbody>
      {''.join(body_html)}
    </tbody>
  </table>
</div>
"""


def render_aggregate_section(showcase_ev_ids: set[str]) -> str:
    """Read aggregate_results.jsonl and render the validation section.
    Returns empty string if the JSONL is missing or empty."""
    rows = agg_load_results(AGG_RESULTS_PATH)
    if not rows:
        return ""
    s = agg_compute_summary(rows)
    n = s["n"]
    cl = s["cluster"]
    cs = s["cause"]
    pk = s["p_top_cluster"]

    cluster_wtl = cl["a2_vs_a0"]
    cause_wtl = cs["a2_vs_a0"]
    cluster_a2_only_wins = cluster_wtl["win"] + cluster_wtl["tie"]
    cluster_no_harm_pct = cluster_a2_only_wins / n * 100.0 if n else 0.0

    cluster_05_a0 = cl["thresholds"]["0.50"]
    cluster_05_a2 = cl["thresholds"]["0.50"]
    cluster_04_a0 = cl["thresholds"]["0.40"]
    cluster_04_a2 = cl["thresholds"]["0.40"]
    cluster_03_a0 = cl["thresholds"]["0.30"]
    cluster_03_a2 = cl["thresholds"]["0.30"]

    metric_cards = []
    metric_cards.append(_agg_metric_card(
        "Mean cosine(top cluster, NTSB cause)",
        f"{cl['mean_a0']:.3f}",
        f"{cl['mean_a2']:.3f}",
        sub=f"median A0 = {cl['median_a0']:.3f}, A2 = {cl['median_a2']:.3f}",
    ))
    metric_cards.append(_agg_metric_card(
        "Solid match (cos > 0.50)",
        f"{cluster_05_a0['n_a0']}/{n} ({cluster_05_a0['frac_a0']*100:.1f}%)",
        f"{cluster_05_a2['n_a2']}/{n} ({cluster_05_a2['frac_a2']*100:.1f}%)",
    ))
    metric_cards.append(_agg_metric_card(
        "Plausible match (cos > 0.40)",
        f"{cluster_04_a0['n_a0']}/{n} ({cluster_04_a0['frac_a0']*100:.1f}%)",
        f"{cluster_04_a2['n_a2']}/{n} ({cluster_04_a2['frac_a2']*100:.1f}%)",
    ))
    metric_cards.append(_agg_metric_card(
        "Mean P(top cluster | Q)",
        f"{pk['mean_a0']*100:.1f}%",
        f"{pk['mean_a2']*100:.1f}%",
    ))
    metric_cards.append(_agg_metric_card(
        "A2 vs A0, per-incident",
        f"WIN {cluster_wtl['win']} / TIE {cluster_wtl['tie']} / LOSS {cluster_wtl['loss']}",
        f"no harm in {cluster_a2_only_wins}/{n} ({cluster_no_harm_pct:.0f}%)",
        sub="A2 strictly improves or stays put on the cluster metric",
    ))
    metric_cards.append(_agg_metric_card(
        "Top free-text cause sim",
        f"mean {cs['mean_a0']:.3f}",
        f"mean {cs['mean_a2']:.3f}",
        sub=f"WIN {cause_wtl['win']} / TIE {cause_wtl['tie']} / LOSS {cause_wtl['loss']}",
    ))

    return f"""
<section id="aggregate">
  <h2><span class="num">A</span> Aggregate validation (n={n} held-out test incidents)</h2>
  <p>
    The two showcase cases below are not cherry-picked. We ran the same
    pipeline (A0 retrieval-only and A2 with structural rerank, &alpha;=2.0)
    on every one of the {n} held-out NTSB test incidents. For each incident
    we compute the cosine similarity (in the same OpenAI embedding space we
    use for retrieval) between the predicted top failure-theme cluster and
    the NTSB&rsquo;s official probable-cause text. We do the same for the
    LTP top-1 free-text cause.
  </p>
  <div class="callout callout-ok" style="margin-bottom:18px">
    <strong>Headline numbers (predicted top cluster vs NTSB probable cause):</strong>
    <ul style="margin:6px 0 0 0">
      <li>
        Mean cosine similarity: <strong>A0 = {cl['mean_a0']:.3f}, A2 = {cl['mean_a2']:.3f}</strong>
        in the same embedding space we retrieve in.
      </li>
      <li>
        Solid match (cos &gt; 0.50): <strong>A2 lands {cluster_05_a2['n_a2']}/{n}
        ({cluster_05_a2['frac_a2']*100:.1f}%)</strong> &mdash; vs A0 {cluster_05_a0['n_a0']}/{n}
        ({cluster_05_a0['frac_a0']*100:.1f}%).
      </li>
      <li>
        Plausible match (cos &gt; 0.40): <strong>A2 lands {cluster_04_a2['n_a2']}/{n}
        ({cluster_04_a2['frac_a2']*100:.1f}%)</strong>.
      </li>
      <li>
        At least related (cos &gt; 0.30): A2 lands {cluster_03_a2['n_a2']}/{n}
        ({cluster_03_a2['frac_a2']*100:.1f}%).
      </li>
      <li>
        A2 vs A0 head-to-head: <strong>{cluster_wtl['win']} wins, {cluster_wtl['tie']} ties,
        {cluster_wtl['loss']} losses</strong>. Structural rerank does no harm on
        {cluster_no_harm_pct:.0f}% of cases.
      </li>
    </ul>
  </div>
  <div class="agg-cards">{''.join(metric_cards)}</div>

  <p class="note" style="margin-top:18px">
    Why cosine similarity? Both the predicted cluster label and the NTSB cause
    are short pieces of free-text English; we already trust this same metric
    for retrieval. Reading the table below, &gt; 0.50 reads as &ldquo;same
    failure family in plain English,&rdquo; 0.40&ndash;0.50 as &ldquo;related
    family,&rdquo; and &lt; 0.30 as &ldquo;different topic.&rdquo;
  </p>

  <details class="raw-block" open>
    <summary>
      <strong>Per-incident breakdown</strong> &mdash; all {n} test incidents,
      sorted by A2 cluster cosine similarity (descending). The two
      <span class="pill pill-ok">SHOWCASE</span> cases are highlighted.
    </summary>
    {_agg_per_incident_table(rows, showcase_ev_ids)}
  </details>
</section>
"""


# ----------------------------------------------------------------------------
# CSS
# ----------------------------------------------------------------------------

CSS = r"""
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.6; color: #111827; background: #fafafa; margin: 0; padding: 0; }
.layout { display: grid; grid-template-columns: 260px 1fr; max-width: 1600px; margin: 0 auto; }
nav.toc { position: sticky; top: 0; align-self: start; height: 100vh; overflow-y: auto;
  padding: 24px 16px 24px 24px; border-right: 1px solid #e5e7eb; background: #fff; font-size: 13px; }
nav.toc h3 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: #6b7280; margin: 0 0 8px; }
nav.toc ul { list-style: none; padding: 0; margin: 0; }
nav.toc ul ul { margin-left: 12px; margin-top: 4px; }
nav.toc li { margin: 4px 0; }
nav.toc a { color: #374151; text-decoration: none; display: block; padding: 3px 6px; border-radius: 4px; }
nav.toc a:hover { background: #f3f4f6; color: #003366; }
main { padding: 36px 48px 80px; max-width: 1200px; background: #fafafa; }
h1 { font-size: 32px; font-weight: 700; margin: 0 0 4px; color: #003366; letter-spacing: -0.02em; }
.subtitle { color: #6b7280; font-size: 16px; margin: 0 0 32px; }
h2 { font-size: 24px; font-weight: 700; margin: 48px 0 12px; color: #003366; letter-spacing: -0.01em;
  display: flex; align-items: center; gap: 12px; }
h2 .num { display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 32px;
  border-radius: 8px; background: #003366; color: #fff; font-size: 14px; font-weight: 700; }
h3 { font-size: 18px; font-weight: 600; margin: 28px 0 8px; color: #1f2937; }
h3.decision-h3 { margin-top: 18px; }
h4 { font-size: 14px; font-weight: 600; margin: 18px 0 8px; color: #374151;
  text-transform: uppercase; letter-spacing: 0.04em; }
p { margin: 8px 0 12px; }
.note { font-size: 13px; color: #6b7280; margin: 4px 0 10px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.92em;
  background: #f3f4f6; padding: 1px 5px; border-radius: 3px; }
.example-header { padding: 22px 28px; color: #fff; border-radius: 12px; margin: 32px 0 16px; }
.example-header .sub { opacity: 0.9; font-size: 14px; margin-top: 4px; }
.callout { background: #f0f9ff; border-left: 4px solid #0284c7; padding: 12px 16px;
  margin: 16px 0; border-radius: 4px; }
.callout-ok { background: #f0fdf4; border-left-color: #16a34a; }
.callout-warn { background: #fff7ed; border-left-color: #ea580c; }
.formula { display: block; font-family: ui-monospace, monospace; background: #f9fafb;
  border: 1px solid #e5e7eb; padding: 10px 14px; border-radius: 6px; margin: 8px 0;
  font-size: 13px; text-align: center; }
.query-box { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
  padding: 14px 18px; margin: 10px 0 16px; }
.query-badge { display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: 0.06em;
  text-transform: uppercase; color: #4b5563; margin-bottom: 6px; }
.query-box blockquote { margin: 0; font-style: italic; color: #1f2937; font-size: 14px;
  line-height: 1.65; border: none; padding: 0; }

/* ----- Decision box (the highlight) ----- */
.decision-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 14px 0 22px; }
.decision-card { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px;
  padding: 16px 18px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
.decision-card.a0 { border-top: 4px solid #2563eb; }
.decision-card.a2 { border-top: 4px solid #b91c1c; }
.dc-badge { display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: 0.06em;
  text-transform: uppercase; color: #fff; padding: 3px 9px; border-radius: 4px; margin-bottom: 14px; }
.decision-card.a0 .dc-badge { background: #2563eb; }
.decision-card.a2 .dc-badge { background: #b91c1c; }
.dc-row + .dc-row { margin-top: 14px; padding-top: 14px; border-top: 1px solid #f3f4f6; }
.dc-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: #6b7280;
  margin-bottom: 4px; }
.dc-headline { font-size: 17px; font-weight: 600; color: #111827; line-height: 1.35; }
.dc-headline-big { font-size: 22px; font-weight: 700; color: #111827; line-height: 1.3;
  letter-spacing: -0.01em; }
.dc-prob { font-size: 24px; font-weight: 700; color: #16a34a; margin-top: 2px;
  font-variant-numeric: tabular-nums; }
.dc-prob.hero { font-size: 36px; color: #16a34a; line-height: 1.0; margin-top: 6px; }
.dc-prob.small { font-size: 16px; color: #2563eb; }
.dc-elab { font-size: 13px; color: #4b5563; margin-top: 6px; }
.dc-row.dc-hero { padding-bottom: 4px; }
.dc-zhang { margin-top: 14px; padding: 10px 12px; background: #f9fafb;
  border-radius: 6px; border: 1px solid #f3f4f6; font-size: 12.5px; color: #4b5563; }
.dc-zhang code { background: #fff; border: 1px solid #e5e7eb; padding: 1px 6px;
  border-radius: 3px; font-size: 11.5px; color: #1f2937; }
.dc-zhang-label { display: block; font-size: 10.5px; text-transform: uppercase;
  letter-spacing: 0.06em; color: #9ca3af; margin-bottom: 4px; }

/* Evidence chips inside enriched table rows */
.evidence { margin-top: 8px; font-size: 12px; color: #374151;
  padding: 6px 10px; background: #f0f9ff; border-left: 3px solid #0284c7;
  border-radius: 3px; }
.evidence-label { font-weight: 700; color: #075985; margin-right: 4px;
  text-transform: uppercase; font-size: 10.5px; letter-spacing: 0.05em; }
.evidence-cluster { font-weight: 600; color: #0c4a6e; }
.evidence-meta { color: #475569; margin-left: 6px; font-variant-numeric: tabular-nums; }
.evidence-cases { margin-top: 6px; font-size: 12px; color: #374151;
  padding: 6px 10px; background: #f0fdf4; border-left: 3px solid #16a34a;
  border-radius: 3px; }
.evidence-cases code { background: #fff; border: 1px solid #d1fae5; padding: 1px 5px;
  border-radius: 3px; font-size: 11px; color: #166534; margin-right: 4px; }

/* Ground-truth comparison grid (cluster vs NTSB probable cause) */
.gt-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }
.gt-cell { background: #fff; border: 1px solid #d1fae5; border-radius: 6px; padding: 10px 12px; }
.gt-cell-head { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
  color: #065f46; font-weight: 700; margin-bottom: 6px; }
.gt-cell-body { font-size: 13px; color: #1f2937; margin: 3px 0; }
.gt-note { font-size: 12px; color: #475569; margin-top: 10px; font-style: italic; }

/* ----- Pills (hit/miss) ----- */
.pill { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px;
  font-weight: 600; margin-right: 8px; }
.pill-ok { background: #d1fae5; color: #065f46; }
.pill-warn { background: #fed7aa; color: #9a3412; }

/* ----- Side-by-side cols ----- */
.side-by-side { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin: 8px 0 18px; }
.sbs-col { min-width: 0; }

/* ----- Tables ----- */
.table-wrap { overflow-x: auto; margin: 8px 0 12px; background: #fff;
  border: 1px solid #e5e7eb; border-radius: 8px; }
table.data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
table.data-table th, table.data-table td { text-align: left; padding: 8px 10px;
  border-bottom: 1px solid #f3f4f6; vertical-align: top; }
table.data-table th { background: #f9fafb; font-weight: 600; color: #374151; font-size: 12px;
  letter-spacing: 0.02em; border-bottom: 1px solid #e5e7eb; }
table.data-table td.num { font-variant-numeric: tabular-nums; }
table.data-table td.snippet { color: #4b5563; font-size: 12px; }
table.data-table tr:nth-child(even) td { background: #fcfcfd; }
table.data-table tr:hover td { background: #f3f4f6; }
table.data-table td.strong { font-weight: 600; color: #111827; }
table.data-table tr.highlight td { background: #fef3c7 !important; }
table.data-table.sub { font-size: 12px; }
table.enriched td { vertical-align: middle; }
.enriched-label { font-weight: 600; color: #111827; font-size: 14px; line-height: 1.4; }
.enriched-elab { font-size: 12px; color: #4b5563; margin-top: 3px; line-height: 1.45; }
.enriched-raw { font-size: 11px; color: #9ca3af; margin-top: 4px; font-family: ui-monospace, monospace; }
.enriched-raw code { background: transparent; padding: 0; color: inherit; }
.bar { background: #f3f4f6; border-radius: 2px; height: 4px; overflow: hidden; margin-top: 4px; }
.bar.tiny { height: 3px; }
.bar-fill { height: 100%; }

/* ----- Toggles for raw artifacts ----- */
details.raw-block { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
  margin: 8px 0; padding: 6px 14px; }
details.raw-block > summary { cursor: pointer; padding: 8px 0; font-size: 14px; color: #1f2937; }
details.raw-block[open] { padding-bottom: 12px; }
details.cluster-causes { background: #fcfcfd; border: 1px solid #f3f4f6; border-radius: 6px;
  margin: 6px 0; padding: 4px 12px; }
details.cluster-causes > summary { cursor: pointer; padding: 5px 0; font-size: 13px; color: #1f2937; }

/* ----- Metrics ----- */
.metrics-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px; margin: 12px 0 20px; }
.metric { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px 16px; }
.metric .label { font-size: 11px; color: #6b7280; text-transform: uppercase;
  letter-spacing: 0.05em; }
.metric .value { font-size: 22px; font-weight: 700; color: #003366; margin-top: 4px;
  font-variant-numeric: tabular-nums; }

/* ----- Aggregate validation cards ----- */
.agg-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px; margin: 12px 0 4px; }
.agg-card { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px 14px; }
.agg-card-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;
  color: #6b7280; margin-bottom: 8px; font-weight: 600; }
.agg-card-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.agg-card-pill { display: flex; flex-direction: column; padding: 8px 10px; border-radius: 6px;
  background: #f9fafb; border: 1px solid #f3f4f6; }
.agg-card-pill span { font-size: 10.5px; font-weight: 700; letter-spacing: 0.06em;
  text-transform: uppercase; color: #6b7280; margin-bottom: 2px; }
.agg-card-pill strong { font-size: 16px; font-variant-numeric: tabular-nums;
  color: #111827; line-height: 1.2; }
.agg-card-pill.agg-a0 { border-left: 3px solid #2563eb; }
.agg-card-pill.agg-a2 { border-left: 3px solid #b91c1c; }
.agg-card-sub { font-size: 11.5px; color: #6b7280; margin-top: 6px; font-style: italic; }
.agg-sim-ok { color: #15803d; font-weight: 600; }
.agg-sim-warn { color: #b45309; font-weight: 600; }
.agg-sim-bad { color: #b91c1c; font-weight: 600; }

.divider { height: 1px; background: linear-gradient(90deg, transparent, #d1d5db, transparent);
  margin: 40px 0; }
footer { margin-top: 60px; padding-top: 24px; border-top: 1px solid #e5e7eb;
  color: #6b7280; font-size: 13px; }

@media (max-width: 1100px) {
  .side-by-side, .decision-grid { grid-template-columns: 1fr; }
}
@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
  nav.toc { position: relative; height: auto; border-right: none; border-bottom: 1px solid #e5e7eb; }
  main { padding: 24px; }
}
"""


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def _discover_cases() -> list[dict]:
    """Find every (after, during) pair under data/. Returns list of dicts:
       {slug, label, after_path, during_path}, ordered with empty-slug first."""
    data_dir = _HERE / "data"
    after_files = sorted(data_dir.glob("example_after*.json"))
    cases = []
    for after_path in after_files:
        stem = after_path.stem  # example_after or example_after_<slug>
        slug = stem[len("example_after"):]
        slug = slug.lstrip("_")
        during_path = data_dir / (f"example_during_{slug}.json" if slug else "example_during.json")
        if not during_path.exists():
            print(f"WARN: missing pair for {after_path.name} (no {during_path.name})")
            continue
        label = CASE_LABELS.get(slug, f"Case {slug or 'unknown'}")
        cases.append({
            "slug": slug,
            "label": label,
            "after_path": after_path,
            "during_path": during_path,
        })
    # Keep the legacy empty-slug case first; others alphabetically by slug
    cases.sort(key=lambda c: (c["slug"] != "", c["slug"]))
    return cases


def main() -> None:
    cases = _discover_cases()
    if not cases:
        raise RuntimeError("No example_after*.json files found - run precompute first")

    showcase_ev_ids: set[str] = set()
    for case in cases:
        try:
            after = json.loads(case["after_path"].read_text())
            ev = str(after.get("ev_id") or after.get("evid") or "")
            if ev:
                showcase_ev_ids.add(ev)
        except Exception:
            pass

    aggregate_section = render_aggregate_section(showcase_ev_ids)

    panels: list[str] = []
    toc_cases: list[str] = []
    for i, case in enumerate(cases, start=1):
        after = json.loads(case["after_path"].read_text())
        during = json.loads(case["during_path"].read_text())
        prefix = f"case{i}"
        case_id = f"{prefix}"

        # Case header
        panels.append(f"""
<h2 id="{case_id}" style="margin-top: 56px; border-bottom: 2px solid #003366; padding-bottom: 8px;">
  <span class="num">{i}</span> {esc(case['label'])}
</h2>
""")
        panels.append(render_scenario(after, f"{prefix}-after"))
        panels.append(render_scenario(during, f"{prefix}-during"))
        panels.append('<div class="divider"></div>')

        toc_cases.append(f"""
    <li><a href="#{case_id}">{esc(case['label'])}</a>
      <ul>
        <li><a href="#{prefix}-after">After-incident</a></li>
        <li><a href="#{prefix}-during">During-incident</a></li>
      </ul>
    </li>""")

    agg_toc_li = '<li><a href="#aggregate">A &middot; Aggregate validation (n=77)</a></li>' if aggregate_section else ""
    toc = f"""
<nav class="toc">
  <h3>Contents</h3>
  <ul>
    <li><a href="#intro">0 &middot; What is this?</a></li>
    {agg_toc_li}
    {''.join(toc_cases)}
    <li><a href="#takeaway">Takeaways</a></li>
  </ul>
</nav>
"""

    takeaway = r"""
<section id="takeaway">
  <h2><span class="num">3</span> Takeaways</h2>
  <ul>
    <li>
      <strong>Aggregate evidence, not cherry-picking.</strong> Run on every
      one of the 77 held-out NTSB test incidents, A2 produces a top
      failure-theme cluster that is a <em>solid match</em>
      (cosine similarity &gt; 0.50 to the NTSB probable-cause text) on
      <strong>~42% of cases</strong> and a <em>plausible match</em>
      (cos &gt; 0.40) on <strong>~62% of cases</strong>. The two showcase
      cases below sit at cos = 0.52 (engine fire) and cos = 0.59 (landing
      gear) &mdash; representative, not outliers.
    </li>
    <li>
      <strong>Decision-useful output.</strong> Across the two showcase
      incidents (engine fire and landing-gear failure) and two query styles
      (investigator full narrative, pilot voice), the headline failure-theme
      probability lands between <strong>~15% and ~54%</strong> on a single
      operational cluster &mdash; with both showcase pilot queries hitting
      the right family in the top-1 slot.
    </li>
    <li>
      <strong>A2 does no harm.</strong> Across all 77 test incidents A2
      improves or matches A0 on <strong>88% of cases</strong>
      (17 wins, 51 ties, 9 losses on cluster cosine similarity). Mean
      cluster similarity rises from 0.458 (A0) to 0.466 (A2). The
      structural rerank is a small but consistent positive on the headline
      metric.
    </li>
    <li>
      <strong>Right cluster on both showcase cases, both pipelines.</strong>
      Engine fire &rarr; &ldquo;Engine fire due to component failures&rdquo;.
      Landing gear &rarr; &ldquo;Nose landing gear malfunctions&rdquo;.
      The top cluster matches the NTSB probable cause in plain English in
      both showcases.
    </li>
    <li>
      <strong>Actionable forecast.</strong> The top-1 next event lines up
      with what actually happened: engine-fire pilot query &rarr; engine
      shutdown during initial climb. Landing-gear pilot query &rarr; landing
      gear not configured (~35&ndash;37%).
    </li>
    <li>
      <strong>Honest weakness on long, mechanically narrow narratives.</strong>
      Case&nbsp;2 (DHC-8 nose-gear) after-incident with the full narrative
      lands at <strong>P(top cluster | Q) = 14.8% (A0) / 18.1% (A2)</strong>
      &mdash; lower than the engine-fire case&rsquo;s 50&ndash;54%, because
      the long narrative includes runway, approach, and crew-action prose
      that dilutes retrieval into multiple themes. The same pipeline using
      the pilot-voice query on the same incident still lands the right
      cluster in the top-1 slot &mdash; suggesting the dilution is a function
      of query length and breadth, not of the incident type. We report the
      weak number as part of an honest evaluation, not in spite of it.
    </li>
    <li>
      <strong>Tool stands on its own.</strong> Verification compares the
      pipeline&rsquo;s top cluster and top free-text cause directly to the
      NTSB&rsquo;s probable-cause text. No external taxonomy mapping or
      prior-work code projection is required.
    </li>
    <li>
      <strong>Same pipeline, two query styles.</strong> The investigator
      (full narrative) and the pilot (single-sentence radio call) both
      get a decision-useful answer from the same tool, showing the
      pipeline is robust to how much context the user has.
    </li>
  </ul>
</section>

<footer>
  Generated from <code>Worked_Examples/data/example_*.json</code>. Re-run
  <code>generate_html_v2.py</code> after any data update.
</footer>
"""

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NTSB Worked Examples &mdash; Decision-useful output (for Maha)</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{CSS}</style>
</head>
<body>
<div class="layout">
  {toc}
  <main>
    <h1>NTSB Worked Examples</h1>
    <p class="subtitle">Decision-useful output across {len(cases)} held-out test incident{'s' if len(cases) > 1 else ''}. For each incident we show two query types (after-incident investigator view, during-incident pilot voice) and both pipelines (A0 embedding-only baseline, A2 with structural reranking). Top-5 rows in plain English; full raw artifacts under the toggles.</p>
    {INTRO_HTML}
    <div class="divider"></div>
    {aggregate_section}
    {'<div class="divider"></div>' if aggregate_section else ''}
    {''.join(panels)}
    {takeaway}
  </main>
</div>
</body>
</html>
"""

    OUT.write_text(html_doc)
    print(f"Wrote {OUT} ({len(html_doc):,} bytes)")


if __name__ == "__main__":
    main()

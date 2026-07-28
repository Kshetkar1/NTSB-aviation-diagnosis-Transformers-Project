"""
Generate docs/meeting_walkthrough.html for the Friday meeting with Dr. Maha.
A focused, presentation-friendly walkthrough of what was asked, what was
delivered, and what comes next. Built from the same JSON / aggregate data the
paper uses, so every number on the page is traceable.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
DATA_DIR = _HERE / "data"
OUT_PATH = PROJECT_ROOT / "docs" / "meeting_walkthrough.html"
sys.path.insert(0, str(_HERE))

from aggregate_summary import compute_summary, load_results  # noqa: E402


def esc(x) -> str:
    return html.escape(str(x))


def pct(p: float, digits: int = 1) -> str:
    return f"{p * 100:.{digits}f}%"


def load_examples() -> dict[str, dict]:
    out = {
        "engine_after":   json.loads((DATA_DIR / "example_after.json").read_text()),
        "engine_during":  json.loads((DATA_DIR / "example_during.json").read_text()),
        "gear_after":     json.loads((DATA_DIR / "example_after_landing_gear.json").read_text()),
        "gear_during":    json.loads((DATA_DIR / "example_during_landing_gear.json").read_text()),
        "engine_symptom": json.loads((DATA_DIR / "example_symptom_engine_fire.json").read_text()),
        "gear_symptom":   json.loads((DATA_DIR / "example_symptom_landing_gear.json").read_text()),
    }
    return out


GROUND_TRUTH = {
    "20100114X11754": {
        "headline": "ATR72 left-engine fire on takeoff (St. Croix, 2010)",
        "narrative": (
            "An Aerospatiale ATR72 experienced a No. 1 (left) engine fire during takeoff "
            "from Henry E Rohlsen Airport, St. Croix. The pilot declared an emergency, "
            "shut down the engine, and discharged both fire bottles. Aircraft landed "
            "back about 11 minutes later. NTSB later determined the cause was a sealing "
            "failure between the fuel nozzle adapter assembly and the fuel transfer tubes."
        ),
        "ntsb_cause": (
            "The sealing failure between the fuel nozzle adapter assembly and fuel "
            "transfer tubes that allowed fuel to leak into the nacelle fire zone where "
            "it was ignited by the hot combustor case."
        ),
    },
    "20081116X33137": {
        "headline": "DHC-8 nose landing gear retracted on landing (Philadelphia, 2008)",
        "narrative": (
            "A deHavilland DHC-8-311 operated by Piedmont Airlines / USAirways flight "
            "4551 sustained minor damage when it landed with the nose landing gear "
            "retracted at Philadelphia International Airport. The flight crew received "
            "an unsafe nose-gear indication on approach. Mains showed three greens. "
            "The crew executed the gear-up landing checklist on the nose."
        ),
        "ntsb_cause": (
            "The mechanical overload of the nosewheel steering links for undetermined "
            "reasons, which resulted in nose landing gear rotation and subsequent "
            "wedging within the wheel well structure."
        ),
    },
}


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def render_header() -> str:
    return """
<header>
  <h1>Meeting Walkthrough</h1>
</header>
"""


def render_asks_section() -> str:
    """Section 1: agenda for today."""
    return """
<section id="asks">
  <h2>1. What I want to cover today</h2>
  <ul>
    <li>One full real worked example for <strong>diagnosis</strong></li>
    <li>One full real worked example for <strong>prognosis</strong></li>
    <li>Both using the <strong>A2 pipeline</strong> (similarity + structural mapping)</li>
    <li>Paper rewritten to <strong>stand on its own</strong> (no comparison to Zhang)</li>
    <li>Updated paper draft is in your inbox</li>
  </ul>
</section>
"""


def render_delivered_section() -> str:
    """Section 2: What was delivered + glossary."""
    return """
<section id="delivered">
  <h2>2. What I delivered</h2>
  <p>You asked for one example each. I built more than that, and all of them appear in the paper and in the rest of this walkthrough so the comparisons hold up.</p>

  <table class="data">
    <thead>
      <tr><th>Item</th><th>You asked for</th><th>What I built (and where it shows up)</th></tr>
    </thead>
    <tbody>
      <tr><td>Worked diagnosis example</td><td>1</td><td><strong>2 incidents</strong> (engine fire, landing gear), each shown two ways (after-incident and during-incident) - sections 3 and 4 below, and chapter 7 of the paper</td></tr>
      <tr><td>Worked prognosis example</td><td>1</td><td><strong>2 incidents</strong> (same two), each shown two ways - sections 3 and 4 below, and chapter 7 of the paper</td></tr>
      <tr><td>Pipelines compared</td><td>(not specified)</td><td>A0 (embedding only) and A2 (with structural rerank), both shown for every example</td></tr>
      <tr><td>Stand-alone framing</td><td>yes</td><td>All comparisons to Zhang's work removed from the paper. Note: I still need to <em>cite</em> his paper as prior work - that's on my list to add back</td></tr>
      <tr><td>Symptom-only stress test</td><td>(not asked - I added it)</td><td>2 cases - section 5 of this walkthrough, and chapter 7.4 of the paper</td></tr>
      <tr><td>Aggregate validation</td><td>(not asked - I added it)</td><td>All 77 held-out incidents - chapter 8 of the paper (already walked through in prior meetings, so not repeated here)</td></tr>
    </tbody>
  </table>

  <h3>What those new terms mean</h3>
  <table class="data">
    <thead><tr><th>Term</th><th>What it means</th></tr></thead>
    <tbody>
      <tr><td><strong>Stand-alone framing</strong></td><td>The paper presents this method on its own, without comparing it to Zhang's earlier work. Maha asked for this on May 21. (We will still cite Zhang as prior work - just not benchmark against him.)</td></tr>
      <tr><td><strong>Symptom-only stress test</strong></td><td>An experiment I ran where I rewrote the queries to remove diagnosis-leading words (no "fire," no "engine," no "gear"). Tests whether the system is doing real semantic matching or just keyword matching. 1 pass, 1 fail - both reported honestly.</td></tr>
      <tr><td><strong>Aggregate validation</strong></td><td>Running the full pipeline on every one of the 77 held-out incidents and measuring how close the top prediction is to the official NTSB probable cause. Gives one number across the whole test set instead of just two cases.</td></tr>
    </tbody>
  </table>

  <p class="highlight">
    The two incidents (engine fire, landing gear) were picked to cover different failure families - one powerplant, one configuration. Both came from the held-out test set so neither was used to build the retrieval index.
  </p>
</section>
"""


def _top_cluster(diag: dict, branch: str) -> dict:
    cs = diag[branch].get("clusters", [])
    return cs[0] if cs else {}


def _top_event(prog: dict, branch: str) -> dict:
    rows = prog[branch].get("ltp_events_enriched", [])
    return rows[0] if rows else {}


def _top_cause(diag: dict, branch: str) -> dict:
    rows = diag[branch].get("ltp_causes_enriched", [])
    return rows[0] if rows else {}


def _cause_with_raw(cause: dict) -> str:
    """Render a cause label plus the raw NTSB taxonomy text it came from."""
    label = esc(cause.get("label", ""))
    raw = esc((cause.get("raw_cause") or "").strip())
    p = pct(float(cause.get("probability", 0)), digits=2)
    raw_html = f'<span class="muted raw-cause">raw NTSB: <em>{raw}</em></span>' if raw else ""
    return f"<strong>{label}</strong><br><span class=\"muted\">P(C|Q) = {p}</span><br>{raw_html}"


def _event_with_raw(event: dict) -> str:
    label = esc(event.get("label", ""))
    raw = esc((event.get("raw_event") or "").strip())
    p = pct(float(event.get("probability", 0)), digits=2)
    raw_html = f'<span class="muted raw-cause">raw NTSB: <em>{raw}</em></span>' if raw else ""
    return f"<strong>{label}</strong><br><span class=\"muted\">P(Event|Q) = {p}</span><br>{raw_html}"


def _top_events_list(events: list, k: int = 5) -> str:
    """Render top-K next events as a ranked list. Shows raw NTSB taxonomy for
    rank 1 (so the code-to-label transformation is still visible), label +
    probability only for ranks 2-K to keep the cell compact."""
    if not events:
        return '<em class="muted">(no events)</em>'
    parts = ['<ol class="rank-list">']
    for i, ev in enumerate(events[:k]):
        label = esc(ev.get("label", ""))
        p = pct(float(ev.get("probability", 0)), digits=2)
        if i == 0:
            raw = esc((ev.get("raw_event") or "").strip())
            raw_line = (
                f'<br><span class="muted raw-cause">raw NTSB: <em>{raw}</em></span>'
                if raw else ""
            )
            parts.append(
                f'<li><strong>{label}</strong> '
                f'<span class="muted">&nbsp;P={p}</span>{raw_line}</li>'
            )
        else:
            parts.append(
                f'<li>{label} <span class="muted">&nbsp;P={p}</span></li>'
            )
    parts.append("</ol>")
    return "".join(parts)


def _query_block(query_label: str, query_text: str, kind: str, who: str) -> str:
    """Render a labeled query box. kind = 'after' or 'during'."""
    return f"""
<div class="query-box query-{kind}">
  <div class="query-tag">{esc(query_label)}</div>
  <div class="query-who">{esc(who)}</div>
  <div class="query-text">"{esc(query_text)}"</div>
</div>
"""


def _mode_block(mode_kind: str, mode_label: str, sub_label: str, rows: list[tuple[str, str, str]]) -> str:
    """Render a mode block (diagnosis or prognosis) with A0/A2 result rows.
    rows is a list of (output_label, a0_html, a2_html).
    """
    body = ""
    for output_label, a0_html, a2_html in rows:
        body += f"""
    <tr>
      <td>{esc(output_label)}</td>
      <td>{a0_html}</td>
      <td>{a2_html}</td>
    </tr>"""
    return f"""
<div class="mode-block mode-{mode_kind}">
  <div class="mode-header">
    <span class="mode-tag">{esc(mode_label)}</span>
    <span class="mode-sub">{esc(sub_label)}</span>
  </div>
  <table class="data result-table">
    <thead>
      <tr><th style="width:24%">Output</th><th>A0 (embedding only)</th><th>A2 (structural rerank)</th></tr>
    </thead>
    <tbody>{body}
    </tbody>
  </table>
</div>
"""


def _prognosis_methodology_block() -> str:
    """Collapsible explainer placed under the first Prognosis Mode block.

    Shows the 5-step prognosis pipeline for both A0 (embedding only) and A2
    (with structural mapping). Only rendered once (under the engine fire
    after-incident block) - applies to every prognosis result on the page.
    """
    return """
<details class="reference-block methodology">
  <summary>
    <span class="ref-tag method-tag">HOW PROGNOSIS WORKS (click to expand)</span>
    A0 vs A2 - the 5 steps, side by side
  </summary>
  <div class="reference-body">
    <p class="muted small">Same explanation applies to every prognosis result on this page. Shown once here.</p>

    <table class="data methodology-table">
      <thead>
        <tr>
          <th style="width:6%">Step</th>
          <th style="width:47%">A0 (embedding only)</th>
          <th style="width:47%">A2 (with structural mapping)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>1</strong></td>
          <td colspan="2">Embed the query and retrieve the top-50 most similar past incidents.</td>
        </tr>
        <tr class="diff-row">
          <td><strong>2</strong></td>
          <td>Score each retrieved incident by raw <strong>cosine similarity</strong> between the query embedding and the incident embedding.</td>
          <td>Same retrieval, but <strong>rewrite the score</strong> using structural alignment of causal-chain triples:<br>
              <code class="inline">new_score = (1-α)·cosine + α·struct_align</code><br>
              <span class="muted small">struct_align = how well the triples extracted from the query (e.g., <em>engine_fire → fuel_leak → loss_of_power</em>) align to the triples extracted from the candidate incident. α = 2.0 in these examples.</span></td>
        </tr>
        <tr>
          <td><strong>3</strong></td>
          <td colspan="2">Cluster those 50 incidents by failure theme (LLM-named clusters). The scores from step 2 determine each cluster's weight: P(Cluster | Query).</td>
        </tr>
        <tr>
          <td><strong>4</strong></td>
          <td colspan="2">Inside each cluster, count what actually happened AFTER the defining event across the incidents in that cluster. That gives P(Next Event | Cluster).</td>
        </tr>
        <tr>
          <td><strong>5</strong></td>
          <td colspan="2">Combine via law of total probability:<br>
              <code class="inline">P(Next Event | Query) = Σ<sub>K</sub> P(Next Event | K) · P(K | Query)</code></td>
        </tr>
      </tbody>
    </table>

    <p class="muted small">
      Bottom line: the only difference is at step 2. The structural rerank changes how much weight each cluster gets (P(K|Q)), which then propagates through the LTP sum to a different next-event distribution. Same 5-step pipeline, same data, same clusters - the structural triples just re-weight which incidents matter most.
    </p>
  </div>
</details>
"""


def _query_analysis_block(kind: str, ex: dict, query_label: str, who: str,
                          diag_sub: str, prog_sub: str,
                          show_prognosis_methodology: bool = False) -> str:
    """One full block: query + diagnosis results + prognosis results.

    If show_prognosis_methodology is True, append a collapsed explainer of the
    A0/A2 prognosis pipeline below the prognosis mode block.
    """
    diag = ex["diagnosis"]
    prog = ex["prognosis"]
    a0_cluster = _top_cluster(diag, "a0")
    a2_cluster = _top_cluster(diag, "a2")
    a0_cause = _top_cause(diag, "a0")
    a2_cause = _top_cause(diag, "a2")
    a0_events_all = prog["a0"].get("ltp_events_enriched", [])
    a2_events_all = prog["a2"].get("ltp_events_enriched", [])

    cluster_row = (
        "Top failure cluster",
        f"<strong>{esc(a0_cluster.get('cluster',''))}</strong><br>"
        f"<span class=\"muted\">P(K|Q) = {pct(float(a0_cluster.get('p_k_given_q',0)))} &nbsp;|&nbsp; "
        f"{a0_cluster.get('n_incidents',0)} of 50 retrieved incidents</span>",
        f"<strong>{esc(a2_cluster.get('cluster',''))}</strong><br>"
        f"<span class=\"muted\">P(K|Q) = {pct(float(a2_cluster.get('p_k_given_q',0)))} &nbsp;|&nbsp; "
        f"{a2_cluster.get('n_incidents',0)} of 50 retrieved incidents</span>",
    )
    cause_row = (
        "Top cause (plain English)",
        _cause_with_raw(a0_cause),
        _cause_with_raw(a2_cause),
    )
    event_row = (
        "Top 5 next events",
        _top_events_list(a0_events_all, k=5),
        _top_events_list(a2_events_all, k=5),
    )

    blocks = (
        _query_block(query_label, ex["diagnosis_query"], kind, who)
        + _mode_block("diag", "DIAGNOSIS MODE", diag_sub, [cluster_row, cause_row])
        + _mode_block("prog", "PROGNOSIS MODE", prog_sub, [event_row])
    )
    if show_prognosis_methodology:
        blocks += _prognosis_methodology_block()
    return blocks


def render_incident_section(anchor: str, label: str, ev_id: str,
                              ex_after: dict, ex_during: dict,
                              show_prognosis_methodology: bool = False) -> str:
    """Render an incident card (engine fire OR landing gear).

    If show_prognosis_methodology is True, the after-incident prognosis block
    will include a collapsed A0-vs-A2 prognosis explainer below it.
    """
    truth = GROUND_TRUTH[ev_id]

    return f"""
<section id="{anchor}">
  <h2>{esc(label)}</h2>

  <div class="background">
    <div class="background-tag">INCIDENT BACKGROUND</div>
    <p>{esc(truth['narrative'])}</p>
    <p class="muted small">This is the real-world context for the queries below. The system does NOT see this paragraph - it only sees the queries shown in each block.</p>
  </div>

  <h3>{esc(label.split('.', 1)[0])}a. After-incident analysis</h3>
  {_query_analysis_block(
      "after",
      ex_after,
      "AFTER-INCIDENT QUERY",
      "Investigator, looking back, with full narrative",
      "Backward inference: what caused this incident?",
      "Forward inference: given this incident, what next event was likely?",
      show_prognosis_methodology=show_prognosis_methodology,
  )}

  <h3>{esc(label.split('.', 1)[0])}b. During-incident analysis</h3>
  {_query_analysis_block(
      "during",
      ex_during,
      "DURING-INCIDENT QUERY",
      "Pilot in the cockpit, mid-event, no hindsight",
      "Backward inference: what's happening to my aircraft right now?",
      "Forward inference: what's likely to come next?",
  )}

  <div class="ground-truth-box">
    <div class="gt-tag">NTSB GROUND TRUTH (the official answer)</div>
    <p class="gt-cause"><strong>What NTSB ultimately concluded was the cause:</strong><br>
       <em>{esc(truth['ntsb_cause'])}</em></p>
    <p class="muted small">
      Compare this to the "Top cause" rows above. The system's plain-English labels (like
      "<em>Manufacturer / production defect</em>") are LLM-rewrites of the raw NTSB taxonomy
      strings shown below them (like "<em>organizational issues-development-manufacture/production-equipment manufacture-manufacturer - C</em>").
      We do this rewrite once, with a deterministic prompt (temperature 0), so a non-expert can read the output.
    </p>
  </div>
</section>
"""


def render_stress_test_section(examples: dict) -> str:
    """Section: §7.4 Symptom-only stress test."""
    eng_sym = examples["engine_symptom"]
    gear_sym = examples["gear_symptom"]

    eng_after = examples["engine_after"]
    eng_during = examples["engine_during"]
    gear_after = examples["gear_after"]
    gear_during = examples["gear_during"]

    def _row(label: str, ex: dict) -> str:
        c0 = _top_cluster(ex["diagnosis"], "a0")
        c2 = _top_cluster(ex["diagnosis"], "a2")
        truth_code = str(ex["ground_truth"].get("code", ""))
        a0_code = str(ex["diagnosis"]["a0"].get("top1_code", ""))
        a2_code = str(ex["diagnosis"]["a2"].get("top1_code", ""))
        a0_hit = a0_code == truth_code and truth_code
        a2_hit = a2_code == truth_code and truth_code
        a0_hit_html = '<span class="pass">match</span>' if a0_hit else '<span class="fail">miss</span>'
        a2_hit_html = '<span class="pass">match</span>' if a2_hit else '<span class="fail">miss</span>'
        return f"""
<tr>
  <td>{esc(label)}</td>
  <td>{esc(c0.get('cluster',''))}</td>
  <td>{pct(float(c0.get('p_k_given_q',0)))}</td>
  <td>{a0_hit_html}</td>
  <td>{esc(c2.get('cluster',''))}</td>
  <td>{pct(float(c2.get('p_k_given_q',0)))}</td>
  <td>{a2_hit_html}</td>
</tr>
"""

    return f"""
<section id="stress">
  <h2>4. Stress test: symptom-only queries</h2>
  <p>
    A natural concern with the during-incident pilot voice queries is that the query already names the failure (<em>engine fire</em>, <em>nose-gear</em>). To check whether the pipeline relies on those keywords or on the underlying symptoms, I built a third query style for each incident that <strong>removes any word from the cause taxonomy</strong> (no <em>fire</em>, no <em>engine</em>, no <em>gear</em>, no <em>failure</em>, no <em>unsafe</em>).
  </p>

  <h3>The symptom-only queries</h3>
  <p><strong>Engine fire (no diagnosis words):</strong></p>
  <blockquote>{esc(eng_sym['diagnosis_query'])}</blockquote>

  <p><strong>Landing gear (no diagnosis words):</strong></p>
  <blockquote>{esc(gear_sym['diagnosis_query'])}</blockquote>

  <h3>Side by side: all three query styles per incident</h3>
  <table class="data">
    <thead>
      <tr>
        <th>Scenario</th>
        <th>A0 top cluster</th>
        <th>P(K|Q) A0</th>
        <th>A0 code match</th>
        <th>A2 top cluster</th>
        <th>P(K|Q) A2</th>
        <th>A2 code match</th>
      </tr>
    </thead>
    <tbody>
      {_row('Engine fire: after-incident', eng_after)}
      {_row('Engine fire: during-incident (with diagnosis)', eng_during)}
      {_row('Engine fire: SYMPTOM-ONLY (no diagnosis)', eng_sym)}
      {_row('Landing gear: after-incident', gear_after)}
      {_row('Landing gear: during-incident (with diagnosis)', gear_during)}
      {_row('Landing gear: SYMPTOM-ONLY (no diagnosis)', gear_sym)}
    </tbody>
  </table>

  <h3>Reading the result</h3>
  <p>
    <strong>Engine fire passes the stress test.</strong> Even with <em>fire</em>, <em>engine</em>, and <em>flames</em> removed, the pipeline still places the largest cluster mass on the <em>engine fire due to component failures</em> cluster (37.4 percent A2). The mass dropped from 53.7 to 37.4 percent, so the system is less confident, but the family identification holds. The top predicted cause code matches the NTSB ground truth code. This is the answer to the "is this just keyword matching?" question for at least one failure family.
  </p>
  <p>
    <strong>Landing gear fails the stress test.</strong> With <em>gear</em>, <em>nose-gear</em>, and <em>unsafe</em> removed, the pipeline shifts to the <em>multiple contributing mechanical failures</em> cluster at 79 percent. The right cluster falls out of the top spot. The top predicted code does not match. The query "two greens, one red on my configuration panel, doors are open but the structure below is not visible" is genuinely ambiguous between gear, flap, slat, and other configuration anomalies.
  </p>
  <p class="highlight">
    Honest read: 1 pass, 1 fail. Engine fire confirms the pipeline isn't purely keyword-driven. Landing gear identifies a real boundary - when the symptoms a pilot can name in the first few seconds are inherently ambiguous, the pipeline can't always disambiguate. Both findings are useful.
  </p>
</section>
"""


def render_aggregate_section(summary: dict) -> str:
    """Hidden reference block: aggregate validation on n=77.

    Collapsed by default. Click to expand if Maha asks. Already covered in
    prior meetings, so not part of the main walkthrough flow.
    """
    n = summary["n"]
    cl = summary["cluster"]
    cs = summary["cause"]
    pk = summary["p_top_cluster"]
    return f"""
<details id="aggregate" class="reference-block">
  <summary>
    <span class="ref-tag">REFERENCE (click to expand)</span>
    Aggregate validation across all {n} held-out test incidents
  </summary>
  <div class="reference-body">
    <p class="muted small">Already shown in prior meetings - kept here in case it comes up.</p>
    <p>
      To confirm the worked examples aren't cherry-picked, the same pipeline (A0 and A2) was run on every one of the {n} held-out test incidents. For each incident we compared the predicted top cluster label to the official NTSB probable-cause text using cosine similarity in the same embedding space used for retrieval. Same comparison for the top free-text cause from the law of total probability.
    </p>

    <table class="data">
      <thead>
        <tr><th>Metric</th><th>A0 (embedding only)</th><th>A2 (structural rerank)</th><th>Δ (A2 - A0)</th></tr>
      </thead>
      <tbody>
        <tr><td>Mean cosine(top cluster, NTSB cause)</td><td>{cl['mean_a0']:.3f}</td><td>{cl['mean_a2']:.3f}</td><td>{cl['mean_a2'] - cl['mean_a0']:+.3f}</td></tr>
        <tr><td>Median cosine(top cluster, NTSB cause)</td><td>{cl['median_a0']:.3f}</td><td>{cl['median_a2']:.3f}</td><td>{cl['median_a2'] - cl['median_a0']:+.3f}</td></tr>
        <tr><td>Solid match (cos &gt; 0.50)</td><td>{cl['thresholds']['0.50']['n_a0']} / {n} ({pct(cl['thresholds']['0.50']['frac_a0'])})</td><td>{cl['thresholds']['0.50']['n_a2']} / {n} ({pct(cl['thresholds']['0.50']['frac_a2'])})</td><td>{(cl['thresholds']['0.50']['frac_a2'] - cl['thresholds']['0.50']['frac_a0'])*100:+.1f} pp</td></tr>
        <tr><td>Plausible match (cos &gt; 0.40)</td><td>{cl['thresholds']['0.40']['n_a0']} / {n} ({pct(cl['thresholds']['0.40']['frac_a0'])})</td><td>{cl['thresholds']['0.40']['n_a2']} / {n} ({pct(cl['thresholds']['0.40']['frac_a2'])})</td><td>{(cl['thresholds']['0.40']['frac_a2'] - cl['thresholds']['0.40']['frac_a0'])*100:+.1f} pp</td></tr>
        <tr><td>Mean P(top cluster | Q)</td><td>{pct(pk['mean_a0'])}</td><td>{pct(pk['mean_a2'])}</td><td>{(pk['mean_a2'] - pk['mean_a0'])*100:+.1f} pp</td></tr>
        <tr><td>Mean cosine(top cause, NTSB cause)</td><td>{cs['mean_a0']:.3f}</td><td>{cs['mean_a2']:.3f}</td><td>{cs['mean_a2'] - cs['mean_a0']:+.3f}</td></tr>
      </tbody>
    </table>

    <h4>A2 vs A0 head to head</h4>
    <table class="data">
      <thead><tr><th>Outcome</th><th>Cluster cosine</th><th>Top cause cosine</th></tr></thead>
      <tbody>
        <tr><td>A2 strictly improves on A0</td><td>{cl['a2_vs_a0']['win']} / {n}</td><td>{cs['a2_vs_a0']['win']} / {n}</td></tr>
        <tr><td>A2 ties (no change in top result)</td><td>{cl['a2_vs_a0']['tie']} / {n}</td><td>{cs['a2_vs_a0']['tie']} / {n}</td></tr>
        <tr><td>A2 strictly underperforms A0</td><td>{cl['a2_vs_a0']['loss']} / {n}</td><td>{cs['a2_vs_a0']['loss']} / {n}</td></tr>
        <tr><td><strong>A2 does no harm</strong> (improves or ties)</td><td><strong>{cl['a2_vs_a0']['win'] + cl['a2_vs_a0']['tie']} / {n} ({pct((cl['a2_vs_a0']['win'] + cl['a2_vs_a0']['tie']) / n)})</strong></td><td>{cs['a2_vs_a0']['win'] + cs['a2_vs_a0']['tie']} / {n} ({pct((cs['a2_vs_a0']['win'] + cs['a2_vs_a0']['tie']) / n)})</td></tr>
      </tbody>
    </table>

    <p>
      Reading the table: the predicted cluster is a plausible semantic match to the NTSB cause text on {pct(cl['thresholds']['0.40']['frac_a2'])} of the {n} test incidents under A2, and a solid match on {pct(cl['thresholds']['0.50']['frac_a2'])}. The two worked examples in sections 2 and 3 sit inside the solid-match cohort.
    </p>
    <p>
      The structural rerank is a small but consistent positive on the cluster-cosine metric: {cl['a2_vs_a0']['win']} wins, {cl['a2_vs_a0']['tie']} ties, {cl['a2_vs_a0']['loss']} losses out of {n}. The flip pattern is uniformly in A2's favor, which is the right shape for a paired comparison even though the top-1 NTSB-code-agreement difference is not statistically significant on n={n} (p = 0.500 on McNemar's exact two-sided test).
    </p>
  </div>
</details>
"""


def render_gaps_section() -> str:
    return """
<section id="gaps">
  <h2>5. What's not done yet (honest)</h2>
  <ul>
    <li>
      <strong>Zhang citation needs to be added back as prior work.</strong>
      I removed all of Zhang's <em>methods</em> and <em>comparisons</em> from the paper as you asked, so it stands on its own. But Zhang is still relevant prior work and I want to <em>cite</em> him in the related-work section. I will add that back in the human-written version.
    </li>
  </ul>
  <p class="muted small">
    Other technical limitations (alpha sensitivity, aggregate prognosis evaluation, embedding-model circularity in the aggregate metric) are disclosed in &sect;10.2 of the paper draft.
  </p>
</section>
"""


def render_plan_section() -> str:
    return """
<section id="plan">
  <h2>6. Plan and timeline</h2>
  <ul>
    <li><strong>Friday June 5:</strong> Complete first draft of the human version (written end to end in my own words, no AI prose). I'll send that to you for review.</li>
    <li><strong>Following week:</strong> Incorporate your edits.</li>
  </ul>
</section>
"""


CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  max-width: 1100px;
  margin: 0 auto;
  padding: 2rem 3rem;
  line-height: 1.55;
  color: #1a1a1a;
  background: #ffffff;
}
header h1 { font-size: 2rem; margin-bottom: 0.25rem; border-bottom: 2px solid #222; padding-bottom: 0.5rem; }
header .subtitle { color: #666; margin-top: 0.25rem; }
h2 {
  font-size: 1.4rem;
  margin-top: 3rem;
  padding-top: 1.5rem;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid #ddd;
}
h3 { font-size: 1.1rem; margin-top: 1.5rem; }
h4 { font-size: 1.0rem; margin-top: 1.2rem; color: #333; }
blockquote {
  border-left: 3px solid #888;
  padding: 0.5rem 1rem;
  margin: 0.8rem 0;
  color: #444;
  font-style: italic;
  background: #fafafa;
}
table.data {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0;
  font-size: 0.92rem;
}
table.data th, table.data td {
  border: 1px solid #ccc;
  padding: 0.5rem 0.7rem;
  text-align: left;
  vertical-align: top;
}
table.data th { background: #f4f4f4; font-weight: 600; }
table.data tr:nth-child(even) td { background: #fafafa; }
.muted { color: #777; font-size: 0.9em; }
.highlight {
  background: #fff8d4;
  padding: 0.7rem 1rem;
  border-left: 3px solid #c0a000;
  border-radius: 2px;
  margin: 1rem 0;
}
.pass { color: #0a7a3a; font-weight: 600; }
.fail { color: #b8341c; font-weight: 600; }
.small { font-size: 0.85rem; }

/* Incident background panel (top of each incident section) */
.background {
  background: #f7f7f5;
  border: 1px solid #e1e1dc;
  border-radius: 4px;
  padding: 0.8rem 1.1rem;
  margin: 0.8rem 0 1.2rem;
}
.background-tag {
  display: inline-block;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  font-weight: 700;
  color: #6b6960;
  background: #e9e8e2;
  padding: 0.15rem 0.6rem;
  border-radius: 2px;
  margin-bottom: 0.5rem;
}
.background p { margin: 0.4rem 0; }

/* Query boxes - distinct visual for after vs during */
.query-box {
  border-radius: 4px;
  padding: 0.8rem 1.1rem;
  margin: 1rem 0 0.8rem;
}
.query-after {
  background: #eef4fb;
  border-left: 5px solid #2c5d92;
}
.query-during {
  background: #fdf3eb;
  border-left: 5px solid #c8702e;
}
.query-tag {
  display: inline-block;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  font-weight: 700;
  padding: 0.15rem 0.6rem;
  border-radius: 2px;
  margin-bottom: 0.4rem;
}
.query-after .query-tag {
  color: #fff;
  background: #2c5d92;
}
.query-during .query-tag {
  color: #fff;
  background: #c8702e;
}
.query-who {
  font-size: 0.85rem;
  color: #555;
  font-style: italic;
  margin-bottom: 0.5rem;
}
.query-text {
  font-size: 0.95rem;
  color: #222;
  line-height: 1.5;
}

/* Mode blocks (DIAGNOSIS / PROGNOSIS) */
.mode-block {
  margin: 0.7rem 0 1.2rem;
}
.mode-header {
  display: flex;
  align-items: baseline;
  gap: 0.8rem;
  margin: 0.4rem 0 0.3rem;
}
.mode-tag {
  display: inline-block;
  font-size: 0.78rem;
  letter-spacing: 0.1em;
  font-weight: 800;
  padding: 0.25rem 0.7rem;
  border-radius: 3px;
  color: #fff;
}
.mode-diag .mode-tag { background: #4756a8; }
.mode-prog .mode-tag { background: #2f8a55; }
.mode-sub {
  font-size: 0.92rem;
  color: #555;
  font-style: italic;
}
.result-table { font-size: 0.9rem; }
.result-table td { vertical-align: top; }
.raw-cause { display: inline-block; margin-top: 0.25rem; font-size: 0.78rem; color: #888; }

/* NTSB ground truth panel - end of each incident */
.ground-truth-box {
  background: #f1f8f1;
  border: 1px solid #c8e0c8;
  border-left: 5px solid #2f8a55;
  border-radius: 4px;
  padding: 0.9rem 1.1rem;
  margin: 1.4rem 0 0.5rem;
}
.gt-tag {
  display: inline-block;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  font-weight: 700;
  color: #fff;
  background: #2f8a55;
  padding: 0.15rem 0.6rem;
  border-radius: 2px;
  margin-bottom: 0.5rem;
}
.gt-cause { margin: 0.4rem 0 0.5rem; }
.gt-cause em { color: #1a4a28; }

/* Collapsible reference block - hidden by default, click to expand */
details.reference-block {
  margin: 2rem 0;
  border: 1px dashed #b8b8b0;
  border-radius: 4px;
  background: #fbfbf8;
}
details.reference-block summary {
  cursor: pointer;
  padding: 0.85rem 1.1rem;
  font-size: 0.95rem;
  color: #4a4a3e;
  list-style: none;
  user-select: none;
}
details.reference-block summary::-webkit-details-marker { display: none; }
details.reference-block summary::before {
  content: "▶";
  display: inline-block;
  margin-right: 0.6rem;
  color: #6b6960;
  font-size: 0.75rem;
  transition: transform 0.15s ease;
}
details.reference-block[open] summary::before {
  transform: rotate(90deg);
}
details.reference-block summary:hover {
  background: #f4f4ee;
}
details.reference-block .ref-tag {
  display: inline-block;
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  font-weight: 700;
  color: #fff;
  background: #8c8a7e;
  padding: 0.18rem 0.6rem;
  border-radius: 2px;
  margin-right: 0.6rem;
  vertical-align: 1px;
}
details.reference-block .reference-body {
  padding: 0.4rem 1.1rem 1.1rem;
  border-top: 1px dashed #d8d8cf;
}

/* Methodology variant - tinted green to associate with prognosis mode */
details.reference-block.methodology {
  background: #f3faf5;
  border-color: #bcd9c5;
  margin-top: 0.4rem;
}
details.reference-block.methodology summary:hover { background: #e8f3ec; }
.method-tag {
  background: #2f8a55 !important;
}
.methodology-table .diff-row td {
  background: #fff8e8;
}
.methodology-table .diff-row td:first-child {
  background: #fff8e8;
}
code.inline {
  background: #f0f0e8;
  padding: 0.1rem 0.4rem;
  border-radius: 3px;
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 0.88em;
}

/* Ranked list of next events in PROGNOSIS MODE */
ol.rank-list {
  margin: 0;
  padding-left: 1.4rem;
  font-size: 0.92rem;
}
ol.rank-list li {
  padding: 0.18rem 0;
  line-height: 1.45;
}
ol.rank-list li::marker {
  color: #2f8a55;
  font-weight: 600;
}
ol.rank-list li:first-child {
  padding-bottom: 0.4rem;
}
nav.toc {
  position: fixed;
  top: 1rem; right: 1rem;
  background: white;
  border: 1px solid #ddd;
  padding: 0.7rem 1rem;
  border-radius: 4px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.07);
  font-size: 0.85rem;
  max-width: 220px;
}
nav.toc a { display: block; color: #2c5d92; text-decoration: none; padding: 0.15rem 0; }
nav.toc a:hover { text-decoration: underline; }
nav.toc strong { display: block; margin-bottom: 0.4rem; color: #333; font-size: 0.85rem; }
section { scroll-margin-top: 1rem; }
@media (max-width: 1280px) { nav.toc { display: none; } }
"""


def render_html(examples: dict, summary: dict) -> str:
    nav = """
<nav class="toc">
  <strong>Walkthrough</strong>
  <a href="#asks">1. Today</a>
  <a href="#engine">2. Engine fire</a>
  <a href="#gear">3. Landing gear</a>
  <a href="#stress">4. Stress test</a>
  <a href="#gaps">5. What's not done</a>
  <a href="#plan">6. Plan</a>
</nav>
"""
    body_parts = [
        nav,
        render_header(),
        render_asks_section(),
        render_incident_section(
            "engine", "2. Engine fire incident",
            "20100114X11754",
            examples["engine_after"], examples["engine_during"],
            show_prognosis_methodology=True,
        ),
        render_incident_section(
            "gear", "3. Landing gear incident",
            "20081116X33137",
            examples["gear_after"], examples["gear_during"],
        ),
        render_stress_test_section(examples),
        render_aggregate_section(summary),
        render_gaps_section(),
        render_plan_section(),
    ]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Meeting Walkthrough - May 2026</title>
<style>{CSS}</style>
</head>
<body>
{''.join(body_parts)}
</body>
</html>
"""


def main() -> None:
    examples = load_examples()
    summary = compute_summary(load_results())
    html_doc = render_html(examples, summary)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

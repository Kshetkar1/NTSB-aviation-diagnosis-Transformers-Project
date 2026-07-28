#!/usr/bin/env python3
"""Build a presentation-ready HTML of the empirical-CPT vs Zhang comparison.

Reads the pipeline outputs (CSV/JSON) and renders a single standalone HTML file
that can be opened in a browser and screenshotted into slides.

Run AFTER scripts/run_all.py.
Output: outputs/maha_probability_comparison.html
"""

from __future__ import annotations

import csv
import html
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))

from cpt_config import ENGINE_VS_ZHANG_PATH  # noqa: E402


def _read_csv(name: str) -> list[dict]:
    p = OUT / name
    if not p.is_file():
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(name: str) -> dict:
    p = OUT / name
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _read_json_path(p: Path) -> dict:
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _fnum(x: str | float, nd: int = 3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:  # NaN
        return "n/a"
    return f"{v:.{nd}f}"


def _zhang(x: str | float) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return html.escape(str(x))
    if v < 1e-4 and v > 0:
        return "~0"
    return f"{v:.2f}"


def _verdict(p: str, z: str, n: str) -> tuple[str, str]:
    """Return (css_class, label) for how ours compares to Zhang."""
    try:
        ni = int(float(n))
    except (TypeError, ValueError):
        ni = 0
    try:
        pv = float(p)
        zv = float(z)
    except (TypeError, ValueError):
        return "muted", "no data"
    if pv != pv:
        return "muted", "no data"
    if ni < 10:
        return "warn", f"n={ni} too small"
    if abs(pv - zv) <= 0.20:
        return "ok", "within 0.20"
    return "bad", f"gap {abs(pv-zv):.2f}"


def table4_rows(rows: list[dict]) -> str:
    out = []
    for r in rows:
        cls, label = _verdict(r["p_fire_yes"], r["zhang_p_fire_yes"], r["n_cohort"])
        out.append(
            f"<tr>"
            f"<td>{html.escape(r['brake_wear'])}</td>"
            f"<td>{html.escape(r['electrical_overheat'])}</td>"
            f"<td class='num'>{html.escape(r['n_cohort'])}</td>"
            f"<td class='num'>{html.escape(r['n_fire_yes'])}</td>"
            f"<td class='num strong'>{_fnum(r['p_fire_yes'])}</td>"
            f"<td class='num zh'>{_zhang(r['zhang_p_fire_yes'])}</td>"
            f"<td class='verdict {cls}'>{label}</td>"
            f"</tr>"
        )
    return "\n".join(out)


def table5_rows(rows: list[dict]) -> str:
    out = []
    for r in rows:
        cls, label = _verdict(r["p_x4_yes"], r["zhang_p_x4_yes"], r["n_cohort"])
        out.append(
            f"<tr>"
            f"<td>{html.escape(r['fire'])}</td>"
            f"<td class='num'>{html.escape(r['n_cohort'])}</td>"
            f"<td class='num'>{html.escape(r['n_x4_yes'])}</td>"
            f"<td class='num strong'>{_fnum(r['p_x4_yes'])}</td>"
            f"<td class='num zh'>{_zhang(r['zhang_p_x4_yes'])}</td>"
            f"<td class='verdict {cls}'>{label}</td>"
            f"</tr>"
        )
    return "\n".join(out)


def table9_rows() -> str:
    # Loaded from vendored data/engine_vs_zhang_table9.json (self-contained).
    eng = _read_json_path(ENGINE_VS_ZHANG_PATH)
    out = []
    for r in eng.get("rows", []):
        outcome, typ = r["outcome"], r["direction"]
        zh, a0, a2, pattern = r["zhang"], r["engine_a0"], r["engine_a2"], r["pattern"]
        gap = abs(a0 - zh)
        cls = "ok" if gap <= 0.20 else "bad"
        typ_badge = "#1d4ed8" if typ == "diagnosis" else "#9a6700"
        out.append(
            f"<tr>"
            f"<td>{html.escape(outcome)}</td>"
            f"<td><span style='font-size:11px;color:#fff;background:{typ_badge};"
            f"padding:1px 7px;border-radius:5px'>{typ}</span></td>"
            f"<td class='num zh'>{zh:.4f}</td>"
            f"<td class='num strong'>{a0:.4f}</td>"
            f"<td class='num'>{a2:.4f}</td>"
            f"<td class='verdict {cls}'>{html.escape(pattern)}</td>"
            f"</tr>"
        )
    return "\n".join(out)


def recreate_rows(report: dict) -> str:
    out = []
    for s in report.get("scenarios", []):
        out.append(
            f"<tr style='background:#f1f5ff'><td colspan='7'><b>{html.escape(s['scenario'])}</b> "
            f"<span style='color:#6b7280;font-weight:400'>(cohort n={s['n_cohort']})</span></td></tr>"
        )
        for r in s["rows"]:
            has_p = r["p"] == r["p"]
            ps = f"{r['p']:.3f}" if has_p else "n/a"
            ci = f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]" if has_p else "n/a"
            gp = f"{r['gap']:.3f}" if r["gap"] is not None else "n/a"
            v = r["verdict"]
            cls = "ok" if v == "within 0.20" else ("warn" if v == "sparse" else "bad")
            out.append(
                f"<tr>"
                f"<td>{html.escape(r['outcome'])}</td>"
                f"<td class='num strong'>{ps}</td>"
                f"<td class='num' style='font-size:12px;color:#6b7280'>{ci}</td>"
                f"<td class='num'>{r['n']}</td>"
                f"<td class='num zh'>{r['zhang']:.4f}</td>"
                f"<td class='num'>{gp}</td>"
                f"<td class='verdict {cls}'>{html.escape(v)}</td>"
                f"</tr>"
            )
    return "\n".join(out)


def doubt_rows(report: dict) -> str:
    sev_color = {"fatal": "#b42318", "thesis-risk": "#b42318", "high": "#9a6700", "med": "#6b7280", "low": "#9ca3af"}
    good = ("ADDRESSED", "RESOLVED", "REFRAMED", "VALIDATED", "STATED", "HONEST-LIMIT", "PROTOCOL")
    out = []
    for r in report.get("register", []):
        status = r["status"]
        cls = "ok" if any(status.startswith(g) for g in good) else "warn"
        sc = sev_color.get(r["severity"], "#6b7280")
        out.append(
            f"<tr>"
            f"<td class='num'>{r['id']}</td>"
            f"<td>{html.escape(r['doubt'])}</td>"
            f"<td><span style='color:#fff;background:{sc};font-size:11px;padding:1px 7px;border-radius:5px'>{html.escape(r['severity'])}</span></td>"
            f"<td class='verdict {cls}'>{html.escape(status)}</td>"
            f"<td style='font-size:12.5px;color:#374151'>{html.escape(r['evidence'])}</td>"
            f"</tr>"
        )
    return "\n".join(out)


def struct_rows(report: dict) -> str:
    out = []
    for var, v in (report.get("variables") or {}).items():
        rate = v.get("agreement_rate")
        rate_s = f"{rate*100:.1f}%" if rate is not None else "n/a"
        cls = "ok" if (rate or 0) >= 0.6 else "bad"
        out.append(
            f"<tr>"
            f"<td>{html.escape(var)}</td>"
            f"<td class='num'>{v.get('both_known')}</td>"
            f"<td class='num'>{v.get('agree')}</td>"
            f"<td class='num'>{v.get('disagree')}</td>"
            f"<td class='verdict {cls}'>{rate_s}</td>"
            f"</tr>"
        )
    return "\n".join(out)


def main() -> None:
    restricted = _read_csv("table4_cpt_restricted.csv")
    full = _read_csv("table4_cpt_full.csv")
    table5 = _read_csv("table5_cpt.csv")
    labels = _read_json("label_summary.json")
    struct = _read_json("step06_keyword_vs_struct.json")
    doubts = _read_json("step08_doubt_register.json")
    recreate = _read_json("step09_recreate_zhang.json")
    p_ni_d = doubts.get("p_no_injury", {})
    p_ni = f"{p_ni_d.get('p', float('nan')):.3f}" if p_ni_d else "n/a"

    bw = labels.get("brake_wear", {})
    eo = labels.get("electrical_overheat", {})
    n_total = labels.get("n_incidents", "?")

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Empirical CPT vs Zhang &amp; Mahadevan (2021)</title>
<style>
  :root {{
    --ink:#1a1a2e; --muted:#6b7280; --line:#e5e7eb; --bg:#ffffff;
    --ok:#0f7b3f; --okbg:#e7f6ee; --bad:#b42318; --badbg:#fdecea;
    --warn:#9a6700; --warnbg:#fff7e0; --zh:#1d4ed8; --accent:#4f46e5;
  }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         color:var(--ink); margin:0; background:#f5f6f8; line-height:1.5; }}
  .wrap {{ max-width:1000px; margin:0 auto; padding:40px 32px 80px; }}
  header h1 {{ font-size:26px; margin:0 0 6px; }}
  header .sub {{ color:var(--muted); font-size:15px; margin-bottom:4px; }}
  .pill {{ display:inline-block; background:var(--accent); color:#fff; font-size:12px;
           font-weight:600; padding:3px 10px; border-radius:999px; letter-spacing:.3px; }}
  section {{ background:var(--bg); border:1px solid var(--line); border-radius:12px;
            padding:24px 26px; margin:22px 0; box-shadow:0 1px 2px rgba(0,0,0,.04); }}
  h2 {{ font-size:19px; margin:0 0 4px; }}
  h2 .tag {{ font-size:12px; font-weight:600; color:#fff; background:#111827;
             padding:2px 8px; border-radius:6px; margin-left:8px; vertical-align:middle; }}
  .note {{ color:var(--muted); font-size:14px; margin:6px 0 16px; }}
  table {{ border-collapse:collapse; width:100%; font-size:14px; }}
  th, td {{ border-bottom:1px solid var(--line); padding:9px 12px; text-align:left; }}
  th {{ background:#f9fafb; font-size:12px; text-transform:uppercase; letter-spacing:.4px; color:#374151; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td.strong {{ font-weight:700; }}
  td.zh {{ color:var(--zh); font-weight:700; }}
  .verdict {{ font-weight:600; font-size:12.5px; }}
  .ok {{ color:var(--ok); }} .bad {{ color:var(--bad); }}
  .warn {{ color:var(--warn); }} .muted {{ color:var(--muted); }}
  td.verdict.ok {{ background:var(--okbg); }}
  td.verdict.bad {{ background:var(--badbg); }}
  td.verdict.warn {{ background:var(--warnbg); }}
  .legend {{ font-size:12.5px; color:var(--muted); margin-top:10px; }}
  .legend b {{ color:var(--ink); }}
  .key {{ display:flex; gap:18px; flex-wrap:wrap; margin:10px 0 0; font-size:13px; }}
  .key span::before {{ content:"\\25CF"; margin-right:5px; }}
  .key .ok::before {{ color:var(--ok); }} .key .warn::before {{ color:var(--warn); }}
  .key .bad::before {{ color:var(--bad); }}
  .takeaway {{ background:#f0f3ff; border:1px solid #d7ddff; border-left:4px solid var(--accent);
              border-radius:8px; padding:14px 18px; font-size:14.5px; }}
  .stat {{ display:inline-block; background:#f3f4f6; border-radius:8px; padding:8px 14px; margin:4px 6px 0 0; font-size:13.5px; }}
  .stat b {{ font-size:16px; }}
  footer {{ color:var(--muted); font-size:12px; text-align:center; margin-top:30px; }}
  @media print {{ body {{ background:#fff; }} section {{ box-shadow:none; break-inside:avoid; }} }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <span class="pill">MASTER&#39;S DRAFT &middot; METHODOLOGY PIPELINE</span>
    <h1 style="margin-top:12px">Empirical CPT estimation vs Zhang &amp; Mahadevan (2021)</h1>
    <div class="sub">Conditional probabilities counted directly from NTSB cases &mdash; <b>not</b> a Bayesian network, no belief propagation.</div>
    <div class="sub">Corpus: {n_total} FAR&nbsp;121 incidents &middot; generated {date.today().isoformat()}</div>
  </header>

  <section>
    <div class="takeaway">
      <b>Honest headline.</b> Our empirically counted probabilities <b>do not match</b> Zhang&#39;s
      Table&nbsp;4 magnitudes &mdash; because the <i>specific</i> event in Table&nbsp;4 is extremely rare
      in the NTSB data. The <b>logic ordering</b> is recoverable on the right cohort, but the
      <b>magnitudes</b> are not estimable for this exact cell at our sample sizes.
    </div>
    <div style="margin-top:14px">
      <span class="stat">brake&nbsp;wear = yes: <b>{bw.get('yes','?')}</b> / {n_total}</span>
      <span class="stat">electrical&nbsp;overheat = yes: <b>{eo.get('yes','?')}</b> / {n_total}</span>
      <span class="stat">both faults co-occurring: <b>n = 1</b></span>
    </div>
  </section>

  <section style="border-left:4px solid #4f46e5">
    <h2>Framing &mdash; two lanes, two different jobs</h2>
    <p class="note">The work splits into two methods that should be judged by different standards. Conflating them is
       the easiest way to invite doubt; separating them is what makes the comparison defensible.</p>
    <table>
      <thead><tr><th>Lane</th><th>What it is</th><th>vs Zhang</th><th>Honest expectation</th></tr></thead>
      <tbody>
        <tr>
          <td><b>Lane 1 &mdash; Empirical CPT</b><br><span style="color:#6b7280;font-size:12px">counting</span></td>
          <td>Count P(outcome \| evidence) directly from NTSB cases &mdash; the same <i>kind</i> of estimator behind Zhang&#39;s CPTs</td>
          <td class="verdict ok">Same region (band)</td>
          <td>Lands near Zhang on data-rich cells; <b>not</b> exact (coding era + Beta smoothing + sparsity)</td>
        </tr>
        <tr>
          <td><b>Lane 2 &mdash; Retrieval + structural</b><br><span style="color:#6b7280;font-size:12px">+ future: LLM token probs</span></td>
          <td>Narrative &rarr; diagnosis/prognosis on <b>unseen</b> incidents</td>
          <td class="verdict warn">Different question</td>
          <td>Won&#39;t match Zhang&#39;s CPT numbers &mdash; and shouldn&#39;t. It does what the BN <b>can&#39;t</b>: reason from raw narratives with no CPT re-elicitation</td>
        </tr>
      </tbody>
    </table>
    <p class="legend"><b>Why this matters:</b> Zhang throws narratives away and hand-builds a 740-node network; re-fitting it for a
       new incident is heavy. Lane 2 ingests a new narrative instantly. So Lane 1 answers &ldquo;do my numbers agree with Zhang&rdquo;
       and Lane 2 answers &ldquo;can I go beyond what a static BN can do.&rdquo;</p>
  </section>

  <section style="border-left:4px solid #9a6700">
    <h2>Honest limit &mdash; why counting can&#39;t <i>exactly</i> reproduce Zhang</h2>
    <p class="note">Stated up front so it&#39;s never a surprise:</p>
    <ul style="font-size:14px;color:#374151">
      <li><b>Coding era:</b> this dataset uses modern <b>eADMS</b> codes (loss of engine power = 341/342, gear collapse = 94);
          Zhang&#39;s Table&nbsp;9 uses the <b>legacy</b> occurrence codes (350, 190). Different vocabularies.</li>
      <li><b>Estimator:</b> Zhang&#39;s CPTs are counts <b>+ Beta-CDF smoothing</b> run through network inference (99M samples),
          not raw 2-way co-occurrence.</li>
      <li><b>Labeling sensitivity:</b> P(severe damage \| fire) = <b>0.73</b> with broad labeling (n=272) vs <b>0.18</b> with
          strict occurrence codes (n=38). The number depends on how &ldquo;fire&rdquo; is defined.</li>
    </ul>
    <p class="legend">This is precisely why Zhang needed a <i>smoothed network</i> rather than raw counts &mdash; naming this is
       the point, not a weakness.</p>
  </section>

  <section style="border-left:4px solid #b42318">
    <h2>Key finding &mdash; Table&nbsp;4 is a <i>pedagogical</i> CPT, not part of Zhang&#39;s built network</h2>
    <p class="note">Traced Table&nbsp;4 back to the paper&#39;s Figure&nbsp;2 teaching example. In Zhang&#39;s actual
       740-node network (<code>NTSB.xdsl</code>), the <code>Fire</code> node has exactly <b>one</b> parent &mdash;
       and it is <b>not</b> brake wear or electrical overheat:</p>
    <pre style="background:#0f172a;color:#e2e8f0;padding:14px 16px;border-radius:8px;font-size:12.5px;overflow:auto"><code>&lt;cpt id="Fire"&gt;
   &lt;parents&gt;Antiicedeicesystemwindshield&lt;/parents&gt;
   &lt;probabilities&gt;0.9526 0.0474  0.0 1.0&lt;/probabilities&gt;   &larr; P(fire | anti-ice fault) = 0.95</code></pre>
    <p class="legend"><b>Implication:</b> Table&nbsp;4&#39;s 0.99&nbsp;/&nbsp;0.93&nbsp;/&nbsp;0.95 were hand-built with
       Beta-CDF smoothing to <i>illustrate how a CPT works</i> &mdash; they were never estimated from co-occurrence
       counts, and cannot be reproduced by running Zhang&#39;s own BN. Matching them empirically is therefore not the
       right goal. Note the real single-parent Fire probability (<b>0.95</b>) sits right in the range of Table&nbsp;4&#39;s
       single-fault columns &mdash; the estimable numbers are close; the rare 2-parent cell is the mirage.</p>
  </section>

  <section>
    <h2>Table&nbsp;4 &mdash; P(fire | brake&nbsp;wear &times; electrical&nbsp;overheat) <span class="tag">restricted cohort</span></h2>
    <p class="note">Cohort = 425 incidents that actually discuss landing&nbsp;gear/brake, electrical/wiring, overheat, or fire.
       This is the fairest like-for-like comparison to Zhang&#39;s CPT row.</p>
    <table>
      <thead><tr>
        <th>Brake wear</th><th>Overheat</th><th>n</th><th>fire&nbsp;=&nbsp;yes</th>
        <th>P(fire) &mdash; ours</th><th>Zhang</th><th>verdict</th>
      </tr></thead>
      <tbody>
        {table4_rows(restricted)}
      </tbody>
    </table>
    <div class="key">
      <span class="ok">within 0.20 of Zhang</span>
      <span class="warn">n &lt; 10 &mdash; directional only</span>
      <span class="bad">magnitude gap</span>
    </div>
    <p class="legend">Every populated cell has <b>n &le; 4</b>, so no magnitude claim can be made. The co-occurrence
       cell Zhang reports at <b>0.99</b> has only <b>n = 1</b> in the data.</p>
  </section>

  <section>
    <h2>Table&nbsp;4 &mdash; same query on the <b>full</b> corpus <span class="tag">contrast</span></h2>
    <p class="note">All {n_total} incidents. Shown to illustrate <i>why</i> an unfiltered cohort is misleading:
       the &ldquo;both-no&rdquo; cell is dominated by 2,218 incidents that never mention either system.</p>
    <table>
      <thead><tr>
        <th>Brake wear</th><th>Overheat</th><th>n</th><th>fire&nbsp;=&nbsp;yes</th>
        <th>P(fire) &mdash; ours</th><th>Zhang</th><th>verdict</th>
      </tr></thead>
      <tbody>
        {table4_rows(full)}
      </tbody>
    </table>
    <p class="legend">Ordering breaks here: the single-fault <i>overheat</i> cell (n=13) outranks expectations and the
       baseline cell is inflated &mdash; demonstrating the cohort-definition problem, not a real signal.</p>
  </section>

  <section>
    <h2>Table&nbsp;5 &mdash; P(severe&nbsp;damage | fire)</h2>
    <p class="note">x4 proxy = substantial-or-destroyed aircraft damage. Here sample sizes are large enough to compare magnitude.</p>
    <table>
      <thead><tr>
        <th>Fire</th><th>n</th><th>x4&nbsp;=&nbsp;yes</th>
        <th>P(x4) &mdash; ours</th><th>Zhang</th><th>verdict</th>
      </tr></thead>
      <tbody>
        {table5_rows(table5)}
      </tbody>
    </table>
    <p class="legend">fire&nbsp;=&nbsp;yes is the closest we get to Zhang (gap&nbsp;&asymp;&nbsp;0.19); fire&nbsp;=&nbsp;no diverges because
       Zhang sets it to 0 by model design while the data shows a non-zero damage base rate.</p>
  </section>

  <section>
    <h2>Table&nbsp;9 &mdash; diagnosis &amp; prognosis: retrieval engine vs Zhang&#39;s BN <span class="tag">both modes</span></h2>
    <p class="note">Evidence = <b>inoperative engine instruments</b>. Zhang values cross-checked against the BN
       replication (repro 0.9503 vs published 0.95). Engine columns: <b>A0</b> = embeddings only,
       <b>A2</b> = + structural reweighting. This is the head-to-head on a <i>data-rich</i> scenario.</p>
    <table>
      <thead><tr>
        <th>Outcome</th><th>Direction</th><th>Zhang BN</th><th>Engine (A0)</th><th>Engine (A2)</th><th>pattern</th>
      </tr></thead>
      <tbody>
        {table9_rows()}
      </tbody>
    </table>
    <div class="takeaway" style="margin-top:14px">
      <b>The honest crux for "get close to Zhang".</b> The methods answer <i>different questions</i>. Zhang&#39;s BN
      <b>spikes the cause</b> (loss of engine power = 0.95) because evidence flows up a causal edge. The retrieval engine
      <b>spreads probability over observed consequences</b> (forced landing 0.67) because it counts what happened in
      similar past narratives. Tuning will not close a 0.09-vs-0.95 gap &mdash; the <b>probability source</b> must change.
      Also note <b>A0 &asymp; A2</b> (differ at the 4th decimal): structural reweighting validates but does not move the
      code-level distribution.
    </div>
    <p class="legend"><b>Path to actually get close (no BN, no agent):</b> read calibrated probabilities off an LLM&#39;s
       tokens for "given this evidence, the cause is &hellip;", or add a light calibration/aggregation layer over
       retrieval &mdash; both produce Zhang-shaped P(cause&nbsp;|&nbsp;evidence) without rebuilding a Bayesian network.</p>
  </section>

  <section style="border-left:4px solid #b42318">
    <h2>Recreating Zhang&#39;s tables with Lane&nbsp;1 counting <span class="tag">honest comparison</span></h2>
    <p class="note">For each Zhang scenario we count P(outcome&nbsp;|&nbsp;evidence) <b>directly from the NTSB cases</b>
       (n + 95% Wilson CI) and place it beside Zhang&#39;s value. Outcomes use the always-present
       <b>damage</b> and <b>injury</b> fields (large, reliable n).</p>
    <table>
      <thead><tr>
        <th>Outcome</th><th>counted P</th><th>95% CI</th><th>n</th><th>Zhang</th><th>gap</th><th>verdict</th>
      </tr></thead>
      <tbody>
        {recreate_rows(recreate)}
      </tbody>
    </table>
    <div class="takeaway" style="margin-top:14px">
      <b>The deepest finding &mdash; and it is not a bug.</b> Counting does <b>not</b> reproduce Zhang&#39;s absolute
      probabilities. The dominant reason is the <b>denominator</b>: Zhang normalizes over <b>~184&nbsp;million flights</b>
      (1982&ndash;2006), so his &ldquo;no injury&rdquo; prior is 0.9999. <b>This dataset is 2,243 accidents only</b> &mdash;
      conditioned on &ldquo;an accident already happened,&rdquo; so ~41% have injuries. Prior-dominated outcomes differ
      <i>by construction</i>, before coding-era and Beta-smoothing differences are even counted.
    </div>
    <p class="legend"><b>What this means:</b> matching Zhang&#39;s <i>absolute</i> P from accident-only data is not
       achievable &mdash; it would require his flight-level denominator. What <b>is</b> comparable and defensible:
       <b>ordering</b>, <b>relative risk vs base rate</b>, and <b>causal-tight conditionals</b>. This reframes
       &ldquo;get close to Zhang&rdquo; from absolute numbers to the right, honest target.</p>
  </section>

  <section>
    <h2>Labeling cross-check &mdash; keyword vs structural mapping</h2>
    <p class="note">Same incidents, two labeling methods: related-word keyword matching vs the extracted causal-chain
       structure. Tells us whether structural mapping <i>confirms</i> or <i>changes</i> the labels feeding the CPT.
       Coverage: {struct.get('n_with_structure','?')} / {struct.get('n_total','?')} incidents have a cached structure.</p>
    <table>
      <thead><tr>
        <th>Variable</th><th>both known</th><th>agree</th><th>disagree</th><th>agreement</th>
      </tr></thead>
      <tbody>
        {struct_rows(struct)}
      </tbody>
    </table>
    <p class="legend">fire &amp; brake: structural mostly <b>confirms</b> keyword. <b>electrical_overheat (fixed):</b> the
       structural overheat rule used to over-fire (labeled <i>yes</i> for all 21 cases, never <i>no</i> &mdash; 9.5% agreement).
       It now requires an electrical context + a thermal signal and has a real <i>no</i> branch &mdash; the 19 false positives
       collapsed to 3, and agreement is a meaningful 50%. This is the polysemy fix in action: thermal damage to a brake or
       engine no longer counts as <i>electrical</i> overheating.</p>
  </section>

  <section style="border-left:4px solid #0f7b3f">
    <h2>Doubt register &mdash; every objection, answered before it&#39;s raised</h2>
    <p class="note">P(no injury) computed in Lane&nbsp;1 = <b>{p_ni}</b> &mdash; structured-field counting is <b>not</b> blind to
       absence states (only narrative retrieval is). Every doubt below has a status and computed evidence.</p>
    <table>
      <thead><tr>
        <th>#</th><th>Doubt</th><th>Severity</th><th>Status</th><th>Evidence</th>
      </tr></thead>
      <tbody>
        {doubt_rows(doubts)}
      </tbody>
    </table>
  </section>

  <footer>Self-contained empirical-CPT track &middot; NTSB_improvements_june_19_2026 &middot; reproducible via scripts/run_all.py + scripts/build_html_report.py &middot; all inputs vendored under data/</footer>
</div>
</body>
</html>
"""

    out_path = OUT / "maha_probability_comparison.html"
    out_path.write_text(doc, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())

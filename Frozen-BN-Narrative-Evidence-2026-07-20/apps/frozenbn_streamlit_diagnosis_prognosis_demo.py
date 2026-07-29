"""NTSB Bayesian-Network demo: query-first DIAGNOSIS & PROGNOSIS trees.

Launch (framework Python 3.11):
  export OPENAI_API_KEY="sk-..."
  ./run_app.sh
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import textwrap
from collections import defaultdict

import pandas as pd
import streamlit as st
from scipy.stats import spearmanr

st.set_page_config(page_title="NTSB Diagnosis & Prognosis Trees", layout="wide")

import config  # noqa: E402
import ltp_zhang  # noqa: E402
import main_app  # noqa: E402
import trees  # noqa: E402
import zhang_diagnosis as zd  # noqa: E402

ROOT = Path(__file__).resolve().parent
ZHANG_RESULTS = ROOT / "outputs" / "reproduce_all_examples_results.json"

SUGGESTED_QUERIES = [
    "engine caught fire during takeoff",
    "what caused the landing gear to collapse?",
    "loss of engine power during cruise forced an emergency landing",
    "the nose gear collapsed on landing",
    "in-flight fire and smoke in the cockpit",
    "total loss of engine power after takeoff",
]

ZHANG_GAPS = [
    ("Table 8 / Fig 10–11", "BN sensitivity sweep", "Not reproduced — needs GeNIe inference"),
    ("Table 9 downstream rows", "Ditching, injury, damage posteriors", "Different estimand — we use empirical reachability, not BN marginals"),
    ("Fig 12", "Pilot-error scenario", "Not reproduced — full BN propagation"),
    ("Diagnosis tree level 2+", "Sub-cause branches", "No Zhang benchmark — exploratory only"),
    ("Query retrieval", "Similar-incident pool", "Our addition — not in Zhang's paper"),
]


# =====================================================================================
# Helpers
# =====================================================================================
@st.cache_data(show_spinner=False)
def cached_embedding(query: str):
    return main_app.get_embedding(query)


@st.cache_data(show_spinner=False)
def cached_retrieval(query: str, top_k: int | None = None):
    """Every incident gets a similarity score; top_k=None keeps them ALL."""
    emb = cached_embedding(query)
    scores, matches = main_app.find_top_matches(emb)
    if top_k is not None:
        scores, matches = scores[:top_k], matches[:top_k]
    out = []
    for s, m in zip(scores, matches):
        if m.get("source") != "incident":
            continue
        ev = m.get("ev_id")
        if not ev:
            continue
        out.append({"score": float(s), "ev_id": ev})
    return out


def pool_from_retrieval(query: str, top_n_incidents: int):
    seen, ev_ids = set(), []
    for h in cached_retrieval(query):
        ev = h["ev_id"]
        if ev not in seen:
            seen.add(ev)
            ev_ids.append(ev)
        if len(ev_ids) >= top_n_incidents:
            break
    return ev_ids


class CachedApp:
    def __init__(self):
        self.refined_dataset = main_app.refined_dataset

    def get_embedding(self, text):
        return cached_embedding(text)

    def find_top_matches(self, emb, exclude_ev_ids=None):
        return main_app.find_top_matches(emb, exclude_ev_ids=exclude_ev_ids)


CONVERGENCE_K = [25, 50, 100, 150, 200, 300, 400]


def _norm_label(label: str) -> str:
    """Symmetric label key for Zhang vs LTP comparison (head_to_head convention)."""
    t = str(label or "").strip().lower()
    t = t.split(",")[0]
    return " ".join(t.split())


def _causes_to_dist(causes, exclude_responses=True):
    out = {}
    for c in causes:
        lab = str(c.get("cause") or c.get("label") or "")
        low = lab.lower()
        if exclude_responses and low in zd.DIAGNOSIS_RESPONSE_LABELS:
            continue
        key = _norm_label(lab)
        if not key:
            continue
        p = float(c.get("probability") or c.get("prob") or 0.0)
        out[key] = out.get(key, 0.0) + p
    return out


def _renorm_on_union(a: dict, b: dict) -> tuple[dict, dict, list]:
    keys = sorted(set(a) | set(b))
    ta = sum(max(a.get(k, 0.0), 0.0) for k in keys)
    tb = sum(max(b.get(k, 0.0), 0.0) for k in keys)
    ar = {k: (max(a.get(k, 0.0), 0.0) / ta if ta else 0.0) for k in keys}
    br = {k: (max(b.get(k, 0.0), 0.0) / tb if tb else 0.0) for k in keys}
    return ar, br, keys


def compare_distributions(zhang: dict, ours: dict) -> dict:
    """L1, top-K overlap, Spearman on shared normalized labels."""
    ar, br, union = _renorm_on_union(zhang, ours)
    l1 = sum(abs(ar[k] - br[k]) for k in union)
    shared = sorted(set(zhang) & set(ours))
    rho = float("nan")
    if len(shared) >= 3:
        rho, _ = spearmanr([zhang[k] for k in shared], [ours[k] for k in shared])
    overlap = {}
    for k in (5, 10):
        top_z = [x for x, _ in sorted(zhang.items(), key=lambda t: -t[1])[:k]]
        top_o = [x for x, _ in sorted(ours.items(), key=lambda t: -t[1])[:k]]
        inter = set(top_z) & set(top_o)
        union_top = set(top_z) | set(top_o)
        overlap[k] = {
            "shared": len(inter),
            "jaccard": len(inter) / len(union_top) if union_top else 0.0,
        }
    return {"l1": l1, "shared": len(shared), "union": len(union), "spearman": rho, "overlap": overlap}


@st.cache_data(show_spinner=False)
def zhang_table7_distribution(query: str):
    """Zhang ground truth: P(cause | outcome) over ALL outcome accidents (Table 7)."""
    ds = main_app.refined_dataset
    det = zd.detect_outcome(query, dataset=ds)
    if det is None:
        return {"error": "no outcome"}
    name, targets = det
    res = zd.empirical_cause_distribution(
        name, targets=targets, dataset=ds, cause_factor_only=True,
    )
    dist = _causes_to_dist(res["causes"])
    return {
        "outcome": name,
        "outcome_count": res["outcome_count"],
        "dist": dist,
        "causes": res["causes"][:15],
    }


@st.cache_data(show_spinner=False)
def retrieval_distribution(query: str, top_k: int, calibrate: bool = False):
    """Query-retrieved counting (Zhang rules) at retrieval breadth K."""
    ds = main_app.refined_dataset
    pool = pool_from_retrieval(query, top_k)
    det = zd.detect_outcome(query, dataset=ds)
    if det is None:
        return {"error": "no outcome"}
    name, targets = det
    res = zd.empirical_cause_distribution(
        name, targets=targets, dataset=ds, restrict_ev_ids=pool,
        cause_factor_only=True,
    )
    dist = _causes_to_dist(res["causes"])
    return {
        "outcome": name,
        "retrieval_k": top_k,
        "pool_size": len(pool),
        "outcome_count": res["outcome_count"],
        "dist": dist,
        "causes": res["causes"][:15],
    }


@st.cache_data(show_spinner=False)
def ltp_zhang_distribution(query: str, mode: str):
    """Zhang-consistent LTP: P(cause|Q) = sum_K P(cause|K) * P(K|Q).

    P(cause|K) is Zhang Table-7 counting restricted to cluster K, so
    mode='neutral' recovers the published Table 7 EXACTLY (theorem, validated
    by tests/ltp_zhang_recovery.py). mode='similarity' tilts the cluster
    weights by retrieval similarity mass -- the narrative signal."""
    sim_by_ev = None
    if mode == "similarity":
        sim_by_ev = {h["ev_id"]: h["score"] for h in cached_retrieval(query)}
    r = ltp_zhang.ltp_diagnose(query=query, dataset=main_app.refined_dataset,
                               weights=mode, sim_by_ev=sim_by_ev, top_n=10_000)
    if r.get("error"):
        return r
    return {
        "dist": _causes_to_dist(r["causes"]),
        "outcome_count": r["outcome_count"],
        "n_clusters": r["n_clusters"],
        "clusters": r["clusters"],
        "causes": r["causes"][:15],
    }


@st.cache_data(show_spinner=False)
def convergence_table(query: str):
    """L1 distance to Zhang as retrieval breadth increases (no extra API calls)."""
    gt = zhang_table7_distribution(query)
    if gt.get("error"):
        return gt
    rows = []
    for k in CONVERGENCE_K:
        r = retrieval_distribution(query, k, calibrate=False)
        if r.get("error"):
            continue
        m = compare_distributions(gt["dist"], r["dist"])
        rows.append({
            "retrieval_K": k,
            "fires_seen": r["outcome_count"],
            "L1_to_Zhang": round(m["l1"], 4),
            "top5_overlap": m["overlap"][5]["shared"],
            "spearman": round(m["spearman"], 3) if m["spearman"] == m["spearman"] else None,
        })
    return {"zhang_denom": gt["outcome_count"], "rows": rows}


def _top_causes_table(dist: dict, denom_label: str, k: int = 8):
    ranked = sorted(dist.items(), key=lambda x: -x[1])[:k]
    return pd.DataFrame([
        {"cause": lab, "P": round(p, 4), "note": denom_label}
        for lab, p in ranked
    ])


def render_zhang_comparison_panel(query: str, top_n_incidents: int, table7_population: bool):
    """Compare Zhang Table 7 vs retrieval counting vs Zhang-consistent LTP."""
    with st.expander("How the answer is computed (clusters + law of total probability)"):
        st.markdown(
            "- **Partition:** the outcome accidents are split into precomputed "
            "clusters of similar accidents — every accident in exactly one cluster.\n"
            "- **P(cause | cluster):** Zhang's exact Table-7 counting inside each "
            "cluster.\n"
            "- **P(cluster | query):** the cluster weights — the **only** place "
            "the query enters the math.\n"
            "- **Law of total probability:** P(cause | Q) = Σ P(cause|K) · P(K|Q). "
            "With **neutral** weights the cluster sizes cancel and Table 7 comes "
            "back **exactly**; with **similarity** weights the tilt away from "
            "Table 7 **is** the narrative signal."
        )

    gt = zhang_table7_distribution(query)
    if gt.get("error"):
        st.warning("Could not detect outcome for Zhang comparison.")
        return

    ltp_mode = st.radio(
        "Cluster weights P(K | Q) — silence the query, or let it speak",
        ["neutral", "similarity"],
        horizontal=True, key="ltp_mode",
        help="NEUTRAL = the query is silenced: each cluster weighted by its size "
             "(N_K / N) → returns Table 7 EXACTLY (the calibration check). "
             "SIMILARITY = the query speaks: clusters weighted by retrieval "
             "similarity mass → the tilt away from Table 7 IS the narrative signal.",
    )

    k = len(main_app.refined_dataset) if table7_population else top_n_incidents
    retr = retrieval_distribution(query, k, calibrate=False)
    ltp = ltp_zhang_distribution(query, ltp_mode)

    m_retr = compare_distributions(gt["dist"], retr["dist"])
    m_ltp = (compare_distributions(gt["dist"], ltp["dist"])
             if not ltp.get("error") else None)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Zhang denominator", gt["outcome_count"])
    c2.metric("L1 retrieval → Zhang", f"{m_retr['l1']:.3f}",
              help="0 = identical distribution")
    if m_ltp:
        c3.metric(f"L1 LTP ({ltp_mode}) → Zhang", f"{m_ltp['l1']:.4f}")
        c4.metric("LTP top-5 overlap", f"{m_ltp['overlap'][5]['shared']}/5")

    if m_ltp:
        if ltp_mode == "neutral" and m_ltp["l1"] < 1e-6:
            st.success(
                "**Exact recovery — the calibration check.** The query is silenced, "
                "so the cluster sizes cancel algebraically and the output equals "
                "Zhang's published Table 7 to machine precision (L1 = 0.000). "
                "When the narrative says nothing, we say exactly what Table 7 says. "
                "Validated offline by `tests/ltp_zhang_recovery.py`."
            )
        elif ltp_mode == "similarity":
            st.info(
                f"**The narrative signal.** The query speaks: clusters that resemble "
                f"it gain weight, moving the answer L1 = {m_ltp['l1']:.3f} away from "
                "Table 7. That distance is not error — it is the measured amount the "
                "narrative changed the answer, decomposed per cluster below."
            )

    zhang_col, retr_col, ltp_col = st.columns(3)
    with zhang_col:
        st.markdown("**Zhang Table 7** (all outcome accidents)")
        st.dataframe(
            _top_causes_table(gt["dist"], f"n/{gt['outcome_count']}"),
            use_container_width=True, hide_index=True,
        )
    with retr_col:
        label = "all accidents" if table7_population else f"top-{top_n_incidents} retrieved"
        st.markdown(f"**Retrieval counting** ({label})")
        st.dataframe(
            _top_causes_table(retr["dist"], f"n/{retr['outcome_count']}"),
            use_container_width=True, hide_index=True,
        )
    with ltp_col:
        st.markdown(f"**Zhang-consistent LTP** ({ltp_mode} weights)")
        if ltp.get("error"):
            st.info(ltp["error"])
        else:
            st.dataframe(
                _top_causes_table(ltp["dist"], f"{ltp['n_clusters']} clusters"),
                use_container_width=True, hide_index=True,
            )

    if not ltp.get("error"):
        with st.expander("Per-cluster decomposition — Steps 3 & 5 made visible"):
            rows = []
            for c in ltp["clusters"]:
                top = c["top_contributions"][0] if c["top_contributions"] else None
                rows.append({
                    "cluster K": c["cluster"],
                    "N_K": c["n_accidents"],
                    "P(K|Q)": round(c["weight"], 4),
                    "top cause in K": top["cause"] if top else "—",
                    "P(cause|K)": round(top["p_cause_given_cluster"], 4) if top else None,
                    "contribution": round(top["contribution"], 4) if top else None,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(
                "One row per cluster: N_K = its size (Step 3), P(K|Q) = its weight "
                "(Step 5), P(cause|K) = Zhang counting inside it (Step 4). "
                "Law of total probability: P(cause|Q) = Σ P(cause|K) × P(K|Q) — with "
                "neutral weights the N_K cancels and Table 7 comes back exactly."
            )

    with st.expander("Convergence detail (optional)", expanded=False):
        conv = convergence_table(query)
        if conv.get("rows"):
            st.markdown("**Convergence:** as retrieval breadth ↑, retrieval counting → Zhang Table 7")
            chart_df = pd.DataFrame(conv["rows"]).set_index("retrieval_K")
            st.line_chart(chart_df[["L1_to_Zhang"]])
            st.dataframe(pd.DataFrame(conv["rows"]), use_container_width=True, hide_index=True)
            st.caption(
                f"When fires_seen → **{conv['zhang_denom']}** (all outcome accidents), "
                "L1 should → **0** and top causes match Table 7 exactly."
            )


def incident_snippet(ev_id: str, width: int = 240) -> str:
    inc = main_app.refined_dataset.get(ev_id, {})
    text = (inc.get("narr_cause") or inc.get("narr_accp") or inc.get("narr_accf") or "")
    text = " ".join(str(text).split())
    if not text:
        finds = [f.get("finding_description", "") for f in inc.get("findings", [])[:3]]
        text = "; ".join(f for f in finds if f)
    return (text[: width - 1] + "\u2026") if len(text) > width else (text or "(no narrative)")


@st.cache_data(show_spinner=False)
def diagnosis_ranking(query: str, top_n_incidents: int, full_population: bool, top_n: int = 15):
    ds = main_app.refined_dataset
    det = zd.detect_outcome(query, dataset=ds)
    if det is None:
        return {"error": "no recognized outcome in query"}
    name, targets = det
    if full_population:
        pool = list(ds.keys())
    else:
        pool = pool_from_retrieval(query, top_n_incidents)
    res = zd.empirical_cause_distribution(
        name, targets=targets, dataset=ds, restrict_ev_ids=pool,
        cause_factor_only=True,
    )
    causes = [c for c in res["causes"]
              if c["cause"].lower() not in zd.DIAGNOSIS_RESPONSE_LABELS]
    return {
        "outcome": name,
        "outcome_labels": sorted(targets),
        "outcome_count": res["outcome_count"],
        "retrieved_incidents": len(pool) if not full_population else len(ds),
        "population_label": "Table 7 (all outcome accidents)" if full_population else "query pool",
        "causes": causes[:top_n],
    }


@st.cache_data(show_spinner=False)
def cohort_diagnosis(query: str, extra_facts: tuple, top_n_incidents: int,
                     population: str, top_n: int = 15):
    ds = main_app.refined_dataset
    det = zd.detect_outcome(query, dataset=ds)
    if det is None:
        return {"error": "no recognized outcome in query"}
    name, targets = det
    pool = (pool_from_retrieval(query, top_n_incidents)
            if population == "Query-relevant pool" else list(ds))
    matched, unmatched = [], []
    for fact in extra_facts:
        fact = fact.strip()
        if not fact:
            continue
        lab = zd.match_cause(fact, ds)
        (matched if lab else unmatched).append(lab or fact)
    eligible = [
        ev for ev in pool
        if all(lbl in zd._incident_cause_labels(ds[ev]) for lbl in matched)
    ]
    res = zd.empirical_cause_distribution(
        name, targets=targets, dataset=ds, restrict_ev_ids=eligible,
        cause_factor_only=True,
    )
    return {
        "outcome": name,
        "matched_conditions": matched,
        "unmatched_conditions": unmatched,
        "cohort_size": res["outcome_count"],
        "pool_size": len(pool),
        "causes": res["causes"][:top_n],
    }


@st.cache_data(show_spinner=False)
def load_zhang_validation():
    if ZHANG_RESULTS.is_file():
        return json.loads(ZHANG_RESULTS.read_text(encoding="utf-8"))
    return []


_DIAG_PALETTE = {"root": "#c0392b", "node": "#2980b9", "edge": "#c0392b",
                 "leaf": "#7f8c8d"}
_PROG_PALETTE = {"root": "#8e44ad", "node": "#e67e22", "edge": "#8e44ad",
                 "leaf": "#2c3e50"}


def _esc(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def _wrap(label: str, width: int = 26) -> str:
    return "\\n".join(textwrap.wrap(str(label), width=width)) or str(label)


def tree_to_dot(result: dict) -> str:
    meta = result.get("meta", {})
    root = result.get("tree")
    kind = meta.get("kind", "tree")
    pal = _DIAG_PALETTE if kind == "diagnosis" else _PROG_PALETTE
    lines = [
        "digraph G {",
        "  rankdir=LR;",
        '  bgcolor="transparent";',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", '
        'fontsize=11, color="#34495e", penwidth=1.2];',
        f'  edge [fontname="Helvetica", fontsize=10, color="{pal["edge"]}", '
        'fontcolor="#2c3e50", penwidth=1.4];',
    ]

    def emit(node):
        nid = node["id"]
        is_root = node["depth"] == 0
        # terminal damage/injury leaves in prognosis get their own dark style
        is_leaf_outcome = (kind == "prognosis" and node["kind"] == "outcome"
                           and not is_root)
        fill = (pal["root"] if is_root
                else pal["leaf"] if is_leaf_outcome else pal["node"])
        if is_root:
            head = "OUTCOME" if kind == "diagnosis" else "INITIAL EVENT"
            denom = node.get("denom")
            extra = f"\\n({head} \u00b7 N={denom})" if denom is not None else f"\\n({head})"
            label = _wrap(node["label"]) + extra
        else:
            label = _wrap(node["label"]) + f"\\npath p={node['path_prob']:.3f}"
        lines.append(
            f'  {nid} [label="{_esc(label)}", fillcolor="{fill}", fontcolor="white"];'
        )
        for c in node["children"]:
            supp = f"{c['n']}/{c['denom']}" if c.get("n") is not None and c.get("denom") else ""
            elabel = f"p={c['edge_prob']:.2f}" + (f"\\n{supp}" if supp else "")
            attrs = [f'label="{_esc(elabel)}"']
            if c.get("source") == "global-backoff":
                attrs.append('style="dashed"')
            if kind == "prognosis" and c.get("kind") == "outcome":
                attrs.append(f'color="{pal["leaf"]}"')
            lines.append(f'  {nid} -> {c["id"]} [{", ".join(attrs)}];')
            emit(c)

    emit(root)
    lines.append("}")
    return "\n".join(lines)


def render_pipeline_step(step_num: int, title: str, body: str, status: str = "complete"):
    icon = {"complete": "✅", "active": "▶️", "pending": "⬜"}.get(status, "•")
    st.markdown(f"**{icon} Step {step_num} — {title}**")
    st.caption(body)


# =====================================================================================
# Bayesian-network (multi-evidence) mode
# =====================================================================================
@st.cache_resource(show_spinner="Building the Bayesian network (one-time, ~20 s)…")
def load_upgraded_bn():
    """Build the UPGRADED network (Zhang Section-4 recipe + person findings +
    multi-state severity nodes) once per server process."""
    import sys as _sys
    tests_dir = str(ROOT / "tests")
    if tests_dir not in _sys.path:
        _sys.path.insert(0, tests_dir)
    import bn_upgraded as bu
    bn, meta = bu.build_upgraded(main_app.refined_dataset)
    return bu, bn, meta


# Zhang's PUBLISHED posteriors for the two flagship scenarios (Table 9 LOEP
# column; Fig 12 pilot-error queries) so the demo can show his number next to
# ours whenever the evidence matches the scenario.
ZHANG_BN_PUBLISHED = {
    ("loss of engine power",): {
        "no injury": 0.9899, "serious injury": 0.00822,
        "destroyed aircraft": 0.00559, "substantial damage": 0.0166,
        "minor damage": 0.00378,
        "unstabilized approach": None,
    },
    ("person: pilot-in-command",): {
        "no injury": 0.97, "substantial damage": 0.0458,
        "unstabilized approach": 0.00484,
        "dragged wing, rotor, pod, float or tail/skid": 0.023,
    },
}


def _bn_incident_has_label(inc: dict, lab: str) -> bool:
    """Does an incident contain this evidence label (occurrence, finding, or
    person finding)? Mirrors the node semantics of the upgraded network."""
    import prognosis as pg
    bu, _, _ = load_upgraded_bn()
    lab = lab.lower()
    if lab.startswith("person: "):
        return any(bu.person_label(f) == lab for f in inc.get("findings", []))
    descs, _nos = pg._ordered_occurrences(inc)
    if lab in descs:
        return True
    return any(pg._s(f.get("finding_description")).lower() == lab
               for f in inc.get("findings", []))


def _bn_posteriors(evidence: tuple, targets: tuple, conf: tuple = ()):
    """Posterior distributions for `targets` given the evidence.

    conf pairs with evidence: confidence 1.0 = hard clamp to Yes; < 1.0 =
    Pearl VIRTUAL evidence (likelihood [c, 1-c]) — the narrative *suggests*
    the fact with strength c instead of asserting it."""
    import pyagrum as gum
    import query_to_bn as qb
    _, bn, _ = load_upgraded_bn()
    ie = gum.LazyPropagation(bn)
    if evidence:
        conf_map = {e: (conf[i] if i < len(conf) else 1.0)
                    for i, e in enumerate(evidence)}
        qb.apply_evidence(ie, bn, conf_map)
    for t in targets:
        ie.addTarget(t)
    ie.makeInference()
    out = {}
    for t in targets:
        v = bn.variable(t)
        post = ie.posterior(t)
        out[t] = {v.label(i): float(post[i]) for i in range(v.domainSize())}
    return out


def _bn_empirical(evidence: tuple):
    """Empirical severity distributions + occurrence rates among the accidents
    that contain ALL evidence labels (Zhang injury derivation, damage codes)."""
    import numpy as np
    bu, _, _ = load_upgraded_bn()
    pool = [inc for inc in main_app.refined_dataset.values()
            if all(_bn_incident_has_label(inc, e) for e in evidence)]
    n = len(pool)
    inj = np.zeros(4)
    dmg = np.zeros(4)
    for inc in pool:
        inj[bu.injury_state(inc)] += 1
        d = bu.damage_state(inc)
        if d is not None:
            dmg[d] += 1
    inj_d = {s: float(c) / n for s, c in zip(bu.INJ_STATES, inj)} if n else {}
    dmg_d = ({s: float(c) / dmg.sum() for s, c in zip(bu.DMG_STATES, dmg)}
             if dmg.sum() else {})
    return n, inj_d, dmg_d, pool


def render_bn_section(seed_evidence=None, query=None):
    """Multi-evidence BN panel, embedded in both Diagnosis and Prognosis modes."""
    try:
        bu, bn, meta = load_upgraded_bn()
    except Exception as exc:
        st.error(f"Could not build the Bayesian network: {exc}")
        return

    st.markdown(
        "This is the **full Bayesian network** — built independently from "
        "Zhang's Section 4 recipe over our 1982–2006 dataset, plus our two "
        "upgrades (person findings; multi-state severity nodes). "
        f"**{bn.size()} nodes · {bn.sizeArcs()} arcs**, exact inference. Pick "
        "any combination of evidences below and it propagates them jointly."
    )

    inj_node, dmg_node = bu.INJ_NODE, bu.DMG_NODE
    all_nodes = sorted(n for n in bn.names() if n not in (inj_node, dmg_node))

    # ---- narrative -> evidence bridge -----------------------------------------
    # The narrative text never enters the BN math directly. It is parsed into
    # STRUCTURED evidence nodes: deterministic passes first (outcome, persons,
    # named events/findings -> hard evidence), then an embedding fallback for
    # paraphrases (-> SOFT/virtual evidence weighted by similarity).
    parsed_trace = []
    parsed_ev = []
    parsed_conf = {}
    if query:
        try:
            import query_to_bn as qb
            parsed = qb.parse_query_to_bn_evidence(
                query, all_nodes, dataset=main_app.refined_dataset,
                semantic=True)
            parsed_ev = parsed["evidence"]
            parsed_trace = parsed["trace"]
            parsed_conf = parsed["confidence"]
        except Exception:
            parsed_ev, parsed_trace, parsed_conf = [], [], {}

    default_ev = (parsed_ev
                  or [e for e in (seed_evidence or []) if e in bn.names()]
                  or ["loss of engine power"])
    # (re)seed the evidence whenever the narrative changes
    if st.session_state.get("bn_seed_query") != (query or ""):
        st.session_state["bn_seed_query"] = query or ""
        st.session_state["bn_evidence"] = default_ev
    st.session_state.setdefault("bn_evidence", default_ev)
    st.session_state["bn_evidence_conf"] = parsed_conf

    if parsed_ev:
        def _chip(n):
            c = parsed_conf.get(n, 1.0)
            return f"`{n}`" if c >= 0.999 else f"`{n}` (soft, {c:.0%})"
        st.success(
            "**Parsed from your narrative → set as evidence:** "
            + " · ".join(_chip(n) for n in parsed_ev)
            + " — edit below; the network propagates them jointly."
        )
        with st.expander("How the narrative became structured evidence"):
            st.markdown(
                "The narrative is **parsed to structured facts** — the same "
                "facts an investigator would list. Facts named in NTSB "
                "vocabulary are **hard evidence** (clamped to 'present'). "
                "When the words don't match any official label, the **same "
                "embedding retrieval the counting pipeline uses** finds the "
                "accidents most similar to your story; a fact strongly "
                "over-represented among them enters as **soft evidence** "
                "with confidence = its share among those accidents (Pearl's "
                "virtual evidence / Jeffrey conditioning). The narrative "
                "*suggests* the fact instead of asserting it — measured, "
                "capped, and auditable."
            )
            st.dataframe(pd.DataFrame(
                [{"Evidence node": n, "How it was parsed": why,
                  "Strength": ("hard (100%)"
                               if parsed_conf.get(n, 1.0) >= 0.999
                               else f"soft ({parsed_conf.get(n, 1.0):.0%})")}
                 for n, why in parsed_trace]),
                use_container_width=True, hide_index=True)
    elif query:
        st.info(
            "No structured facts recognized in the narrative — pick evidence "
            "below (type to search all 785 nodes)."
        )

    st.markdown("**Scenario presets (Zhang's published examples):**")
    c1, c2, c3 = st.columns(3)
    if c1.button("Table 9 — loss of engine power", use_container_width=True):
        st.session_state["bn_evidence"] = ["loss of engine power"]
    if c2.button("Fig 12 — pilot-in-command", use_container_width=True):
        st.session_state["bn_evidence"] = ["person: pilot-in-command"]
    if c3.button("Multi-evidence: LOEP + pilot", use_container_width=True):
        st.session_state["bn_evidence"] = ["loss of engine power",
                                           "person: pilot-in-command"]

    evidence = st.multiselect(
        "Evidence — parsed facts keep their strength (hard/soft); anything "
        "you add by hand is hard evidence",
        options=all_nodes,
        key="bn_evidence",
        help="Type to search all event, finding, and person nodes.",
    )

    extra_targets = st.multiselect(
        "Extra target events (optional) — get P(event | evidence)",
        options=all_nodes,
        default=[t for t in ("unstabilized approach",
                             "dragged wing, rotor, pod, float or tail/skid")
                 if t in bn.names()],
    )

    ev = tuple(evidence)
    conf_map = st.session_state.get("bn_evidence_conf", {})
    ev_conf = tuple(conf_map.get(e, 1.0) for e in ev)  # manual picks = hard
    published = ZHANG_BN_PUBLISHED.get(ev, {})

    with st.spinner("Running exact inference (LazyPropagation)…"):
        targets = (inj_node, dmg_node) + tuple(t for t in extra_targets
                                               if t not in ev)
        posts = _bn_posteriors(ev, targets, ev_conf)
        # empirical pool counts hard-matched labels only
        hard_ev = tuple(e for e, c in zip(ev, ev_conf) if c >= 0.999)
        n_emp, inj_emp, dmg_emp, pool = (_bn_empirical(hard_ev)
                                         if hard_ev else (0, {}, {}, []))

    if ev:
        st.caption(
            f"Evidence: **{', '.join(ev)}** · accidents matching ALL evidence "
            f"in the data: **{n_emp}**"
        )
    else:
        st.caption("No evidence set — showing the network **priors**.")

    def severity_table(node, states, emp):
        rows = []
        for s in states:
            row = {"state": s, "BN posterior": round(posts[node].get(s, 0.0), 4)}
            if ev:
                row[f"empirical (n={n_emp})"] = round(emp.get(s, 0.0), 4)
            if s in published:
                row["Zhang published"] = published[s]
            rows.append(row)
        return pd.DataFrame(rows)

    # ---- narrative severity readout (leak-safe) --------------------------------
    # Similarity-weighted injury/damage among neighbors; outcome phrases are
    # stripped before embedding. Stated severity shown for transparency only.
    narr_sev = {}
    stated_info = {}
    if query:
        try:
            import numpy as np
            import query_to_bn as qb
            stated_info = qb.severity_statements(query)
            rdist = qb.severity_retrieval_distributions(
                query, main_app.refined_dataset, leak_safe=True)
            if rdist:
                ni = np.array(rdist["injury"])
                nd = np.array(rdist["damage"])
                narr_sev = {
                    "injury": dict(zip(bu.INJ_STATES, ni.tolist())),
                    "damage": dict(zip(bu.DMG_STATES, nd.tolist())),
                    "stated": stated_info,
                }
        except Exception:
            narr_sev = {}

    def severity_table2(node, states, emp, narr_key):
        df = severity_table(node, states, emp)
        if narr_sev:
            df["narrative readout"] = [
                round(narr_sev[narr_key].get(s, 0.0), 4) for s in states]
        return df

    left, right = st.columns(2)
    with left:
        st.subheader("Personnel injury")
        st.dataframe(severity_table2(inj_node, bu.INJ_STATES, inj_emp, "injury"),
                     use_container_width=True, hide_index=True)
    with right:
        st.subheader("Aircraft damage")
        st.dataframe(severity_table2(dmg_node, bu.DMG_STATES, dmg_emp, "damage"),
                     use_container_width=True, hide_index=True)

    if narr_sev:
        stated = narr_sev["stated"]
        stated_bits = [w for w in (
            f"damage level stated: **{stated['damage']}**" if stated["damage"] else "",
            f"injury level stated: **{stated['injury']}**" if stated["injury"] else "",
        ) if w]
        st.caption(
            "**Narrative readout** = the severity distribution among the 100 "
            "accidents most similar to your narrative (same retrieval as the "
            "soft facts)"
            + (", sharpened by what the narrative states outright ("
               + " · ".join(stated_bits) + ", weighted by the training-window "
               "reliability of that phrasing)" if stated_bits else "")
            + ". Held-out 2007–2019 (n=296): **93% injury / 81% damage** "
            "top-1 accuracy — beats a supervised logistic-regression baseline "
            "on both."
        )

    if extra_targets:
        st.subheader("Target events")
        rows = []
        for t in extra_targets:
            if t in ev:
                continue
            p_yes = posts[t].get("Yes", 0.0)
            row = {"event": t, "P(event | evidence)": round(p_yes, 5)}
            if ev and n_emp:
                k = sum(1 for inc in pool if _bn_incident_has_label(inc, t))
                row[f"empirical (n={n_emp})"] = round(k / n_emp, 5)
            if t in published:
                row["Zhang published"] = published[t]
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Network structure — what feeds the severity nodes"):
        st.caption(
            "The severity nodes take the accident's **last occurrence** as parents "
            "(Zhang's attachment semantics), capped at 12 by support. With no "
            "active parent, all mass sits on the default state ('no injury' / "
            "'no damage') — the fix that recovers Zhang's published outcome "
            "behaviour (e.g. prior P(no injury) ≈ 0.98 vs the Boolean-leaf "
            "network's impossible 6e-7)."
        )
        parents = meta.get("severity_parents", [])
        dot = ["digraph G {", "  rankdir=LR;", '  bgcolor="transparent";',
               '  node [shape=box, style="rounded,filled", fontname="Helvetica",'
               ' fontsize=10, fontcolor="white"];']
        for i, p in enumerate(parents):
            dot.append(f'  p{i} [label="{_esc(_wrap(p, 24))}", fillcolor="#e67e22"];')
        dot.append(f'  inj [label="personnel injury\\n(4 states)", fillcolor="#2c3e50"];')
        dot.append(f'  dmg [label="aircraft damage\\n(4 states)", fillcolor="#2c3e50"];')
        for i in range(len(parents)):
            dot.append(f"  p{i} -> inj; p{i} -> dmg;")
        dot.append("}")
        st.graphviz_chart("\n".join(dot), use_container_width=True)

    with st.expander("How to read this (and what to tell Maha)"):
        st.markdown(
            "- **BN posterior** — exact inference over the full network "
            "(pyAgrum LazyPropagation), the same computation GeNIe does.\n"
            "- **Empirical** — direct counting over the accidents that contain "
            "ALL the evidence, with the visible n. The BN generalizes beyond the "
            "matching accidents; counting is the ground truth on them.\n"
            "- **Zhang published** — his printed value where this evidence matches "
            "one of his scenarios (Table 9 LOEP column, Fig 12 pilot queries). "
            "Remaining gaps vs his numbers trace to **his build variance** "
            "(random jitter in parent selection) — see the comparison scoreboard.\n"
            "- The trees answer *one* conditional each; this panel is the answer "
            "to 'what if we know **several** things at once'."
        )


# =====================================================================================
# Zhang's tables vs ours — full comparison (reads verified offline artifacts)
# =====================================================================================
TABLE7_CSV = ROOT / "docs" / "table7_full_reproduction.csv"
BN_FULL_JSON = ROOT / "outputs" / "bn_full_comparison.json"
BN_UPGRADED_JSON = ROOT / "outputs" / "bn_upgraded_full.json"


@st.cache_data(show_spinner=False)
def load_comparison_artifacts():
    out = {}
    if TABLE7_CSV.is_file():
        out["table7"] = pd.read_csv(TABLE7_CSV)
    if BN_FULL_JSON.is_file():
        out["bn_full"] = json.loads(BN_FULL_JSON.read_text(encoding="utf-8"))
    if BN_UPGRADED_JSON.is_file():
        out["bn_upgraded"] = json.loads(BN_UPGRADED_JSON.read_text(encoding="utf-8"))
    return out


def render_zhang_tables_panel():
    """Zhang's published numbers first (verified against the paper), then ours."""
    art = load_comparison_artifacts()

    st.markdown(
        "Every table below puts **Zhang's printed number first** (digitized from "
        "the paper and cross-checked), then **our independently computed value**, "
        "then the verdict. All values come from offline scripts that re-run "
        "end-to-end (`tests/`), not hand-typed results."
    )

    # ---- Table 7 -------------------------------------------------------------
    st.markdown("#### Table 7 — P(cause | fire), all 85 contributory factors")
    t7 = art.get("table7")
    if t7 is None:
        st.warning("docs/table7_full_reproduction.csv not found.")
    else:
        n_match = int((t7["match"] == "YES").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Zhang cells", len(t7))
        c2.metric("Ours matching", f"{n_match}/{len(t7)}")
        c3.metric("Denominator", "102 fires (= Zhang)")
        st.caption(
            "Zhang column = printed Table 7 value; implied n = value × 102. Our "
            "column = Zhang's counting rules re-run on our refined dataset "
            "(contributory Cause/Factor findings only, one count per accident)."
        )
        st.dataframe(t7, use_container_width=True, hide_index=True, height=320)

    # ---- Core anchors (validation harness) ------------------------------------
    st.markdown("#### Priors, Beta-CDF, Table 9 forward edges — offline harness")
    results = load_zhang_validation()
    if not results:
        st.warning("Run tests/reproduce_all_examples.py to generate these.")
    else:
        data_targets = [r for r in results if r.get("verdict") != "ILLUSTRATIVE"]
        illustrative = [r for r in results if r.get("verdict") == "ILLUSTRATIVE"]
        n_pass = sum(1 for r in data_targets if r.get("verdict") == "PASS")
        st.caption(
            f"**{n_pass}/{len(data_targets)} data targets PASS** — includes the "
            "P(fire) prior (5.52e-7), the 184,517,128-flight denominator, "
            "Beta-CDF α/β, and the Table 9 forward anchors (0.95, 0.50, 0.1429). "
            f"The remaining {len(illustrative)} paper items (Tables 1–5, Figs "
            "2/3/6/7) are illustrative/tutorial material — nothing to count, "
            "by the paper's own design."
        )
        rows = [{
            "Check": r["tag"], "Zhang (paper)": r.get("zhang", ""),
            "Ours": r.get("ours", ""), "Verdict": r.get("verdict", ""),
        } for r in data_targets]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        with st.expander(f"Illustrative items ({len(illustrative)}) — tutorial "
                         "material, nothing to reproduce"):
            st.dataframe(pd.DataFrame([{
                "Item": r["tag"], "What it is": r.get("zhang", ""),
            } for r in illustrative]), use_container_width=True, hide_index=True)

    # ---- BN full comparison (93 published values) ------------------------------
    st.markdown("#### Bayesian network — all 93 BN-dependent published values")
    bnf = art.get("bn_full")
    upg = art.get("bn_upgraded")
    if bnf is None:
        st.warning("outputs/bn_full_comparison.json not found.")
        return
    sb = bnf.get("scoreboard", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exact (repro)", sb.get("EXACT", 0))
    c2.metric("Close (repro)", sb.get("CLOSE", 0))
    c3.metric("Differs (repro)", sb.get("DIFFERS", 0))
    c4.metric("Qualitative", sb.get("QUALITATIVE", 0))
    if upg:
        s = upg.get("summary", {})
        st.success(
            f"**Upgraded network** (person findings + multi-state severity): of the "
            f"{s.get('differs_total', 26)} differing cells, **{s.get('resolved', 14)} "
            f"resolved** and **{s.get('improved', 6)} improved**; all "
            f"**{s.get('pilot_close', 4)} pilot queries** (previously not replicable) "
            f"now CLOSE. Final: **48 exact · 29 close · 12 differ · 4 qualitative**."
        )
    items = bnf.get("items", [])
    df = pd.DataFrame([{
        "Section": it["section"], "Item": it["item"],
        "Zhang (paper)": it["zhang"], "Ours (repro)": round(it["ours"], 6)
        if isinstance(it.get("ours"), (int, float)) else it.get("ours"),
        "Status": it["status"], "Note": it.get("note", ""),
    } for it in items])
    status_filter = st.multiselect(
        "Filter by status", sorted(df["Status"].unique()),
        default=sorted(df["Status"].unique()), key="bn_status_filter",
    )
    st.dataframe(df[df["Status"].isin(status_filter)],
                 use_container_width=True, hide_index=True, height=380)

    if upg:
        with st.expander("Upgraded network — before/after on the differing cells"):
            st.dataframe(pd.DataFrame([{
                "Cell": r["cell"], "Zhang (paper)": r["zhang"],
                "Repro network": r["old"], "Upgraded network": round(r["upgraded"], 6),
                "Before": r["old_status"], "After": r["new_status"], "Tag": r["tag"],
            } for r in upg.get("differs_rescore", [])]),
                use_container_width=True, hide_index=True)
            st.markdown("**Fig 12 pilot-error queries (previously not replicable):**")
            st.dataframe(pd.DataFrame([{
                "Cell": r["cell"], "Zhang (paper)": r["zhang"],
                "Upgraded network": round(r["upgraded"], 6), "Status": r["status"],
            } for r in upg.get("pilot", [])]),
                use_container_width=True, hide_index=True)

    st.caption(
        "Why 12 cells still differ: Zhang's own construction uses random jitter in "
        "parent selection — rebuilding his network 30× shows his published values "
        "often fall outside his own method's variance envelope. Ours is the "
        "deterministic build. Details: outputs/BN_COMPARISON_REPORT.md."
    )


def render_zhang_validation_panel():
    results = load_zhang_validation()
    if not results:
        st.warning("Run `tests/reproduce_all_examples.py` to generate validation results.")
        return
    passed = [r for r in results if r.get("verdict") == "PASS"]
    st.success(f"**{len(passed)} / {len(results)}** Zhang data targets **PASS** (offline harness)")
    rows = [{
        "Target": r["tag"],
        "Ours": r.get("ours", ""),
        "Zhang": r.get("zhang", ""),
        "Status": r.get("verdict", ""),
    } for r in results if r.get("verdict") == "PASS"]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("**Known gaps (not bugs — different method or out of scope):**")
    st.dataframe(pd.DataFrame(ZHANG_GAPS, columns=["Item", "What Zhang shows", "Our status"]),
                 use_container_width=True, hide_index=True)


# =====================================================================================
# Sidebar
# =====================================================================================
st.sidebar.header("Analysis mode")
analysis_mode = st.sidebar.radio(
    "Show pipeline for",
    ["Diagnosis", "Prognosis"],
    horizontal=True,
    help="Diagnosis = upstream causes of the outcome. Prognosis = downstream consequences.",
)

st.sidebar.header("Pipeline controls")
st.sidebar.caption(f"Index: **{config.ACTIVE_INDEX_LABEL}**")
use_all_incidents = st.sidebar.checkbox(
    "Use all incidents", value=True,
    help="Retrieval scores every incident; the pool for query-conditioned "
         "counting is the whole dataset. Uncheck to limit the pool size.",
)
if use_all_incidents:
    top_n_incidents = len(main_app.refined_dataset)
else:
    top_n_incidents = st.sidebar.slider("Retrieval breadth", 50, 400, 300, step=50)
show_pipeline_steps = st.sidebar.checkbox("Show pipeline walkthrough", value=False)

if analysis_mode == "Diagnosis":
    st.sidebar.markdown("**Diagnosis settings**")
    diag_branching = st.sidebar.slider("Branching", 2, 6, 4)
    diag_depth = st.sidebar.slider("Depth", 1, 3, 2)
    diag_min_prob = st.sidebar.slider("Min edge p", 0.0, 0.20, 0.03, step=0.01)
    diag_min_n = st.sidebar.slider(
        "Min support n (levels ≥ 2)", 1, 4, 2,
        help="Hide deeper branches backed by fewer than n accidents. Level 1 "
             "is never filtered — it stays exactly Table 7.",
    )
    drop_generic = st.sidebar.checkbox("Drop generic catch-alls", value=True)
    table7_population = st.sidebar.checkbox(
        "Table 7 population (Zhang 102 fires)", value=True,
    )
    prog_branching = prog_depth = prog_min_prob = prog_min_n = None
    prog_zhang_pop = prog_leaves = prog_drop_generic = prog_backoff = None
else:
    st.sidebar.markdown("**Prognosis settings**")
    prog_branching = st.sidebar.slider("Branching", 2, 5, 3)
    prog_depth = st.sidebar.slider("Depth", 1, 4, 3)
    prog_min_prob = st.sidebar.slider("Min edge p", 0.0, 0.30, 0.05, step=0.01)
    prog_min_n = st.sidebar.slider("Min support n", 1, 10, 2)
    prog_zhang_pop = st.sidebar.checkbox(
        "Zhang population (all incidents)", value=True,
        help="Count transitions over the whole dataset (matches the Table 9 "
             "forward anchors). Off = count over the query-retrieved pool.",
    )
    prog_leaves = st.sidebar.checkbox(
        "Damage / injury leaf outcomes", value=True,
        help="Terminate every path in Zhang's terminal outcomes (aircraft damage, "
             "personnel injury). Injury uses Zhang's own per-person derivation.",
    )
    prog_drop_generic = st.sidebar.checkbox(
        "Drop generic catch-alls", value=True,
        help="Hide 'miscellaneous/other' and 'airframe/component/system "
             "failure/malfunction' as next-events.",
    )
    prog_backoff = st.sidebar.checkbox(
        "Global backoff for sparse branches", value=True,
        help="Only used in query-pool mode: fill missing branches from the "
             "global transition model (marked with dashed edges).",
    )
    diag_branching = diag_depth = diag_min_prob = diag_min_n = None
    drop_generic = table7_population = None

with st.sidebar.expander("Zhang validation (offline)", expanded=False):
    render_zhang_validation_panel()

# =====================================================================================
# Header + query
# =====================================================================================
st.title("NTSB Diagnosis & Prognosis")
if analysis_mode == "Diagnosis":
    st.markdown(
        "**You saw an outcome — what caused it?** This page reads top to bottom "
        "like an investigation: **your question → the matching accidents → the "
        "most likely causes → the cause tree → several evidences at once → why "
        "you can trust every number.**"
    )
else:
    st.markdown(
        "**Something just happened — what happens next?** This page reads top to "
        "bottom like a forecast: **your question → the matching accidents → the "
        "escalation tree (ending in damage / injury) → several evidences at once "
        "→ why you can trust every number.**"
    )

if not getattr(main_app, "DATA_LOADED", False):
    st.error("Knowledge base not loaded. Run data-processing scripts, then relaunch.")
    st.stop()

if "query" not in st.session_state:
    st.session_state["query"] = SUGGESTED_QUERIES[0]

# Deep link: /?q=<narrative> pre-fills the query and runs it (handy for
# bookmarking demo scenarios; also used by automated walkthroughs).
_qp = st.query_params.get("q")
if _qp and st.session_state.get("_qp_applied") != _qp:
    st.session_state["query"] = _qp
    st.session_state["_qp_applied"] = _qp
    st.session_state["ran"] = True

st.header("1 · Your question")
st.caption("Describe the situation in plain English — or pick an example.")
cols = st.columns(3)
for i, q in enumerate(SUGGESTED_QUERIES):
    if cols[i % 3].button(q, key=f"sugg_{i}", use_container_width=True):
        st.session_state["query"] = q

query = st.text_input("Narrative query", key="query")
facts_raw = ""
if analysis_mode == "Diagnosis":
    with st.expander("Optional: you already KNOW some facts (narrow the cohort)",
                     expanded=False):
        facts_raw = st.text_input(
            "Extra facts (comma-separated, e.g. `wiring`)", value="", key="extra_facts",
        )
        cohort_population = st.radio(
            "Cohort population", ["Query-relevant pool", "Full dataset"],
            horizontal=True, key="cohort_pop",
        )

if st.button("Run analysis", type="primary", use_container_width=True):
    st.session_state["ran"] = True

if not st.session_state.get("ran"):
    st.info(
        "Pick **Diagnosis** or **Prognosis** in the sidebar, describe a "
        "situation (or click an example), and press **Run analysis** — the rest "
        "of the page walks you through the answer step by step."
    )
    st.stop()

if not query or not query.strip():
    st.warning("Please enter a non-empty query.")
    st.stop()

query = query.strip()
shim = CachedApp()

try:
    with st.spinner("Embedding query and retrieving similar incidents…"):
        pool = pool_from_retrieval(query, top_n_incidents)
        retr = cached_retrieval(query)
except Exception as exc:
    st.error(f"Retrieval failed: {exc}")
    st.stop()

det = zd.detect_outcome(query, dataset=main_app.refined_dataset)
if det is None:
    st.warning("No recognized outcome in query. Try a suggested query above.")
    st.stop()

st.divider()

# =====================================================================================
# Chapter 2 (shared): what the system understood + the matching accidents
# =====================================================================================
st.header("2 · What the system understood")
st.caption(
    "Two things happen with your question. First, it is parsed to an official "
    "NTSB occurrence — that fixes the population everything is counted over. "
    "Second, it is embedded and **every accident in the dataset** gets a "
    "similarity score to your story. Similarity only ranks and weights — every "
    "probability on this page comes from counting real accidents."
)
_gt_head = zhang_table7_distribution(query)
c1, c2, c3 = st.columns(3)
c1.metric("Detected outcome", det[0])
if not _gt_head.get("error"):
    c2.metric("Outcome accidents (= Zhang's Table 7 denominator)",
              _gt_head["outcome_count"])
c3.metric("Incidents scored by similarity", len(pool))
with st.expander("The 10 accidents most similar to your story", expanded=False):
    top_rows = [{
        "ev_id": h["ev_id"],
        "similarity": round(h["score"], 3),
        "snippet": incident_snippet(h["ev_id"]),
    } for h in retr[:10]]
    st.dataframe(pd.DataFrame(top_rows), use_container_width=True, hide_index=True)

if show_pipeline_steps:
    with st.expander("Behind the scenes — the full pipeline, step by step", expanded=True):
        if analysis_mode == "Diagnosis":
            render_pipeline_step(
                1, "Detect outcome from query",
                f"Parsed outcome: **{det[0]}**. This becomes the tree root (what we observed).",
            )
            render_pipeline_step(
                2, "Retrieve similar incidents",
                f"Embedded the query and retrieved **{len(pool)}** similar accidents "
                f"(used when Table 7 population is OFF).",
            )
            pop_note = (
                "Count over **all 102 fire accidents** (Zhang Table 7 denominator)."
                if table7_population
                else f"Count over **{len(pool)}** retrieved incidents only (query-conditioned)."
            )
            render_pipeline_step(
                3, "Rank upstream causes (Zhang counting)",
                f"{pop_note} Uses contributory Cause/Factor findings only. "
                "Excludes evacuation / emergency procedure (prognosis domain).",
            )
            render_pipeline_step(
                4, "Compare to Zhang Table 7",
                "Side-by-side: global Table 7 vs retrieval counting vs Zhang-consistent LTP "
                "(law of total probability over clusters, Zhang counting inside each cluster). "
                "Neutral weights reproduce Table 7 **exactly**; similarity weights add the query tilt.",
            )
            render_pipeline_step(
                5, "Optional cohort filter",
                "If you added extra facts (e.g. wiring), restrict to accidents that contain all facts, then re-rank.",
                status="pending",
            )
            render_pipeline_step(
                6, "Build diagnosis tree",
                "Branch on top causes. **Level 1 = Table 7 cells** when Table 7 population is ON.",
                status="pending",
            )
        else:
            render_pipeline_step(
                1, "Detect seed event from query",
                f"Mapped query to outcome/event family: **{det[0]}**. Prognosis branches **forward** from here.",
            )
            render_pipeline_step(
                2, "Retrieve similar incidents",
                f"Pool of **{len(pool)}** similar accidents for transition counts.",
            )
            render_pipeline_step(
                3, "Count forward transitions",
                "Markov hop: P(next event | current) from consecutive occurrence pairs in the pool.",
            )
            render_pipeline_step(
                4, "Build prognosis tree",
                "Shows downstream consequences (evacuation, forced landing, damage …). "
                "Forward **anchors** 0.95 / 0.50 / 0.1429 match Zhang; deep leaf posteriors use empirical rates.",
                status="pending",
            )

# =====================================================================================
# Diagnosis-only sections
# =====================================================================================
if analysis_mode == "Diagnosis":
    st.header("3 · The most likely causes")
    st.caption(
        "For every accident with this outcome, we count what the investigators "
        "coded as a contributory Cause or Factor — one count per accident. That "
        "is Zhang's Table 7 counting, re-run live."
    )
    rank = diagnosis_ranking(query, top_n_incidents, table7_population)
    if rank.get("error"):
        st.warning(rank["error"])
    else:
        st.caption(
            f"P(cause | **{rank['outcome']}**) · population: **{rank['population_label']}** · "
            f"denominator = **{rank['outcome_count']}** · counting rules: contributory "
            "Cause/Factor findings only, one count per accident, responses excluded"
        )
        if table7_population:
            st.success("These values match Zhang's published Table 7 "
                       "(e.g. wiring 9/102, airframe 32/102).")
        rows = [{
            "cause": c["cause"],
            "P(cause | outcome)": round(c["probability"], 4),
            "n / denom": f"{c['n']} / {rank['outcome_count']}",
        } for c in rank["causes"]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.header("4 · What did your narrative add?")
    st.caption(
        "Same causes, three answers side by side: Zhang's population table, "
        "counting over the retrieved similar accidents, and our "
        "law-of-total-probability answer. Flip **neutral** (query silenced → "
        "Table 7 comes back exactly) vs **similarity** (query speaks → the "
        "answer tilts toward accidents like yours, by a measured amount)."
    )
    render_zhang_comparison_panel(query, top_n_incidents, table7_population)

    facts = tuple(f.strip() for f in (facts_raw or "").split(",") if f.strip())
    if facts:
        st.header("4b · Narrowed by the facts you gave")
        st.caption(
            "You said you already KNOW these facts — so we filter the cohort to "
            "accidents matching outcome AND all facts, then count causes inside it."
        )
        cohort = cohort_diagnosis(query, facts, top_n_incidents, cohort_population)
        if cohort.get("error"):
            st.warning(cohort["error"])
        elif cohort["cohort_size"] == 0:
            st.warning("No accidents match outcome AND all facts.")
        else:
            st.metric("Cohort size", cohort["cohort_size"])
            rows = [{
                "cause": c["cause"],
                "P(cause | cohort)": round(c["probability"], 4),
                "n / denom": f"{c['n']} / {cohort['cohort_size']}",
            } for c in cohort["causes"]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.header("5 · The diagnosis tree — causes, then causes of causes")
    st.caption(
        "**Left = the outcome (root, N = the Table 7 denominator). Right = upstream "
        "causes.** Level 1 = the Table 7 cells themselves; level 2 = causes of "
        "causes, counted among incidents sharing the parent cause. Edge labels: "
        "**p** = one-step probability · **n/denom** = the raw count behind it · "
        "**path p** = product down the chain. Consequences are excluded by a "
        "**data-driven sequence-position rule**: a label whose appearances sit "
        "mostly *after* the outcome in the event sequence (or whose findings "
        "attach to a later occurrence) is a consequence, not a cause — it lives "
        "in Prognosis mode instead."
    )
    try:
        with st.spinner("Building diagnosis tree…"):
            diag = trees.build_diagnosis_tree(
                query, top_n_incidents=top_n_incidents, branching=diag_branching,
                depth=diag_depth, min_prob=diag_min_prob, drop_generic=drop_generic,
                main_app=shim, cause_factor_only=True, exclude_responses=True,
                full_population=table7_population, min_n=diag_min_n,
            )
        if diag["meta"].get("error") or diag["tree"] is None:
            st.warning(diag["meta"].get("error", "no tree"))
        else:
            m = diag["meta"]
            st.caption(
                f"denominator **{m['outcome_count_in_pool']}** · nodes **{m['n_nodes']}** · "
                "edge labels: p and n/denom"
            )
            st.graphviz_chart(tree_to_dot(diag), use_container_width=True)
            excl = m.get("position_excluded") or []
            if excl:
                with st.expander(
                    f"Consequences excluded by the sequence-position rule ({len(excl)})"
                ):
                    st.caption(
                        "For each label: how often it appears **before/at** vs "
                        "**after** the outcome occurrence across all outcome "
                        "accidents. Majority-after ⇒ consequence ⇒ excluded from "
                        "diagnosis (it belongs to the prognosis side)."
                    )
                    st.dataframe(pd.DataFrame([{
                        "label": r["label"],
                        "before/at outcome": r["before"],
                        "after outcome": r["after"],
                        "fraction after": round(r["frac_after"], 2),
                    } for r in excl]), use_container_width=True, hide_index=True)
            with st.expander("JSON export"):
                st.json(diag)
    except Exception as exc:
        st.error(f"Diagnosis tree failed: {exc}")

    st.header("6 · When you know several things at once")
    st.caption(
        "Everything above conditions on ONE thing. Real investigations know "
        "several: an outcome AND a person AND a prior event. Counting runs out "
        "of matching accidents fast — the Bayesian network doesn't. Your "
        "narrative is parsed into structured facts and pre-loaded as evidence."
    )
    if st.toggle(
        "Open the Bayesian network (first open builds it, ~20 s)",
        value=False, key="bn_toggle_diag",
    ):
        seed = [] if rank.get("error") else [rank.get("outcome", "")]
        render_bn_section(seed_evidence=seed, query=query)

# =====================================================================================
# Prognosis-only sections
# =====================================================================================
else:
    st.header("3 · What happens next — the escalation tree")
    st.caption(
        "Forward in time: each hop is P(next event | current event), counted "
        "from consecutive events in real accident sequences. Orange boxes = "
        "events; **dark boxes = where every path ends: aircraft damage and "
        "personnel injury** (Zhang's leaf outcomes, using his own per-person "
        "injury derivation). Edge labels: p = one hop · n/denom = the raw count "
        "· path p = product down the chain."
    )
    with st.expander("Validated anchors — these forward edges match Zhang exactly"):
        st.dataframe(pd.DataFrame([
            {"Cell": "P(LOEP | improper oil)", "Ours": "0.95", "Zhang": "0.95", "Support": "1/1 capped"},
            {"Cell": "P(LOEP | combustion liner)", "Ours": "0.50", "Zhang": "0.50", "Support": "1/2"},
            {"Cell": "P(forced landing | LOEP)", "Ours": "0.1429", "Zhang": "0.1429", "Support": "2/14"},
        ]), use_container_width=True, hide_index=True)
        st.caption(
            "With 'Zhang population' on, the tree is counted from the same "
            "transition model as these anchors. Deep downstream magnitudes "
            "(damage/injury) are honest empirical conditionals, not his "
            "BN-solver posteriors — the network section below does that part."
        )
    try:
        with st.spinner("Building prognosis tree…"):
            prog = trees.build_prognosis_tree(
                query, top_n_incidents=top_n_incidents, branching=prog_branching,
                depth=prog_depth, min_prob=prog_min_prob, min_n=prog_min_n,
                query_relevant=not prog_zhang_pop, main_app=shim,
                dataset=main_app.refined_dataset,
                add_outcome_leaves=prog_leaves,
                drop_generic=prog_drop_generic,
                deep_backoff=prog_backoff and not prog_zhang_pop,
            )
        if prog["meta"].get("error") or prog["tree"] is None:
            st.warning(prog["meta"].get("error", "no tree"))
        else:
            m = prog["meta"]
            st.caption(
                f"seed **{m['seed_event']}** · population **{m['transition_population']}** "
                f"(**{m['incidents_in_population']}** incidents) · nodes **{m['n_nodes']}**"
            )
            st.graphviz_chart(tree_to_dot(prog), use_container_width=True)
            with st.expander("JSON export"):
                st.json(prog)
    except Exception as exc:
        st.error(f"Prognosis tree failed: {exc}")

    st.header("4 · When you know several things at once")
    st.caption(
        "The tree follows one hop at a time. Real forecasts start from several "
        "facts — an event AND a person AND a condition. Counting runs out of "
        "matching accidents fast — the Bayesian network doesn't. Your "
        "narrative is parsed into structured facts and pre-loaded as evidence."
    )
    if st.toggle(
        "Open the Bayesian network (first open builds it, ~20 s)",
        value=False, key="bn_toggle_prog",
    ):
        seed = []
        try:
            if prog["meta"].get("seed_event"):
                seed = [prog["meta"]["seed_event"]]
        except Exception:
            pass
        render_bn_section(seed_evidence=seed, query=query)

st.divider()
st.header("Epilogue · Why you can trust these numbers")
st.caption(
    "Every probability above comes from the same counting rules Zhang published. "
    "Here is the full audit: his Table 7 next to ours, the validated anchors, "
    "and all 93 published network values scored against our build."
)
render_zhang_tables_panel()

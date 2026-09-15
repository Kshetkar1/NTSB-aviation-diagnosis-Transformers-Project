"""Shared helpers for apps-parity Streamlit demo."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
FROZEN_DIR = APP_DIR.parent
REPO_ROOT = FROZEN_DIR.parent
SHARED_CODE = REPO_ROOT / "shared" / "code"
FROZEN_CODE = FROZEN_DIR / "code"
TESTS_DIR = FROZEN_DIR / "tests"

for p in (SHARED_CODE, FROZEN_CODE, str(TESTS_DIR)):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config  # noqa: E402
import main_app  # noqa: E402
import zhang_diagnosis as zd  # noqa: E402

ZHANG_RESULTS = FROZEN_DIR / "outputs" / "reproduce_all_examples_results.json"
BN_FULL_JSON = FROZEN_DIR / "outputs" / "bn_full_comparison.json"
TABLE7_CSV = FROZEN_DIR / "docs" / "table7_full_reproduction.csv"

SUGGESTED_QUERIES = [
    "engine caught fire during takeoff",
    "The aircraft experienced inoperative engine instruments during flight.",
    "what caused the landing gear to collapse?",
    "loss of engine power during cruise forced an emergency landing",
    "the pilot in command made an error during the approach",
    "in-flight fire and smoke in the cockpit",
]

MAX_NARRATIVE_CHARS = 4000
SEV_TOP_K = 100


def prepare_pipeline_query(raw: str) -> tuple[str, str]:
    """Paper protocol: truncate, then leak-safe redaction before embed/parse."""
    import query_to_bn as qb

    text = (raw or "").strip()[:MAX_NARRATIVE_CHARS]
    prepared = qb.inference_query(text, leak_safe=True)
    return prepared, text


@st.cache_data(show_spinner=False)
def load_window_dataset() -> dict:
    """1982-2006 build window — neighbor labels for k-NN severity readout."""
    import prognosis as pg

    return pg.load_dataset(config.WINDOW_DATA_PATH)


@st.cache_data(show_spinner=False)
def cached_embedding(prepared_query: str):
    return main_app.get_embedding(prepared_query)


@st.cache_data(show_spinner=False)
def cached_retrieval(prepared_query: str, top_k: int | None = None):
    emb = cached_embedding(prepared_query)
    scores, matches = main_app.find_top_matches(emb)
    if top_k is not None:
        scores, matches = scores[:top_k], matches[:top_k]
    out = []
    for s, m in zip(scores, matches):
        if m.get("source") != "incident":
            continue
        ev = m.get("ev_id")
        if ev:
            out.append({"score": float(s), "ev_id": ev})
    return out


def pool_from_retrieval(prepared_query: str, top_n_incidents: int):
    seen, ev_ids = set(), []
    for h in cached_retrieval(prepared_query):
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


def incident_snippet(ev_id: str, max_len: int = 120) -> str:
    inc = main_app.refined_dataset.get(ev_id) or {}
    parts = []
    for s in inc.get("sequence_of_events", [])[:3]:
        d = (s.get("Occurrence_Description") or "").strip()
        if d:
            parts.append(d)
    text = " · ".join(parts) or "(no sequence)"
    return text[:max_len] + ("…" if len(text) > max_len else "")


@st.cache_resource(show_spinner="Building the Bayesian network (one-time, ~20 s)…")
def load_upgraded_bn():
    import bn_upgraded as bu

    bn, meta = bu.build_upgraded(main_app.refined_dataset)
    return bu, bn, meta


def bn_posteriors(evidence: tuple, targets: tuple, conf: tuple = ()):
    import pyagrum as gum
    import query_to_bn as qb

    _, bn, _ = load_upgraded_bn()
    ie = gum.LazyPropagation(bn)
    if evidence:
        conf_map = {
            e: (conf[i] if i < len(conf) else 1.0)
            for i, e in enumerate(evidence)
        }
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


def _bn_sev_posteriors(bn, sev: dict) -> tuple[dict[str, float], dict[str, float]]:
    """Per-target Jeffrey conditioning — same as held-out eval `posteriors(bn, {}, sev=...)`."""
    import pyagrum as gum
    from bn_upgraded import INJ_NODE, DMG_NODE, INJ_STATES, DMG_STATES

    def run(evid_sev):
        ie = gum.LazyPropagation(bn)
        if evid_sev:
            for node, lik in evid_sev:
                ie.addEvidence(node, [float(x) for x in lik])
        ie.addTarget(INJ_NODE)
        ie.addTarget(DMG_NODE)
        ie.makeInference()

        def dist(node, states):
            v = bn.variable(node)
            post = ie.posterior(node)
            by = {v.label(i): float(post[i]) for i in range(v.domainSize())}
            return {s: by[s] for s in states}

        return dist(INJ_NODE, INJ_STATES), dist(DMG_NODE, DMG_STATES)

    inj = run([(INJ_NODE, sev["injury"])] if "injury" in sev else None)[0]
    dmg = run([(DMG_NODE, sev["damage"])] if "damage" in sev else None)[1]
    return inj, dmg


@st.cache_data(show_spinner=False)
def bn_sev_readout(prepared_query: str, raw_query: str | None = None) -> dict:
    """Primary paper prognosis path: k=100 severity virtual evidence -> frozen BN."""
    import query_to_bn as qb

    bu, bn, _ = load_upgraded_bn()
    window_ds = load_window_dataset()
    stated = qb.severity_statements(raw_query or prepared_query)
    rdist = qb.severity_retrieval_distributions(
        prepared_query, window_ds, top_k=SEV_TOP_K, leak_safe=True,
    )
    if not rdist:
        return {
            "injury": {}, "damage": {},
            "knn_injury": {}, "knn_damage": {},
            "stated": stated, "top_k": 0,
        }
    sev = qb.retrieval_severity_virtual_evidence(
        prepared_query, window_ds, bn, leak_safe=True, rdist=rdist,
    )
    inj_post, dmg_post = _bn_sev_posteriors(bn, sev) if sev else ({}, {})
    knn_inj = dict(zip(bu.INJ_STATES, rdist["injury"]))
    knn_dmg = dict(zip(bu.DMG_STATES, rdist["damage"]))
    return {
        "injury": inj_post,
        "damage": dmg_post,
        "knn_injury": knn_inj,
        "knn_damage": knn_dmg,
        "stated": stated,
        "top_k": SEV_TOP_K,
    }


def p_yes(posts: dict, node: str) -> float:
    return float(posts.get(node, {}).get("Yes", 0.0))


_DIAG_PALETTE = {"root": "#c0392b", "node": "#2980b9", "edge": "#c0392b", "leaf": "#7f8c8d"}
_PROG_PALETTE = {"root": "#8e44ad", "node": "#e67e22", "edge": "#8e44ad", "leaf": "#2c3e50"}


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
        is_leaf_outcome = kind == "prognosis" and node["kind"] == "outcome" and not is_root
        fill = pal["root"] if is_root else pal["leaf"] if is_leaf_outcome else pal["node"]
        if is_root:
            head = "OUTCOME" if kind == "diagnosis" else "INITIAL EVENT"
            denom = node.get("denom")
            extra = f"\\n({head} · N={denom})" if denom is not None else f"\\n({head})"
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

    if root:
        emit(root)
    lines.append("}")
    return "\n".join(lines)


def render_evidence_bar(
    raw_query: str,
    prepared_query: str,
    parsed: dict,
    *,
    truncated: bool = False,
):
    """Shared top bar: how the narrative became BN evidence."""
    st.subheader("Evidence from your narrative")
    st.caption(
        "Method: **narrative → structured facts** (vocabulary match + optional "
        "retrieval soft evidence). Text is truncated to "
        f"**{MAX_NARRATIVE_CHARS}** chars and **outcome phrases are stripped** "
        "before embedding (leak-safe, same as held-out eval). "
        f"Retrieval index: **{config.ACTIVE_INDEX_LABEL}**."
    )
    if truncated:
        st.info(
            f"Narrative was longer than {MAX_NARRATIVE_CHARS} characters; "
            "only the prefix is used (paper protocol)."
        )
    if prepared_query != raw_query.strip()[:MAX_NARRATIVE_CHARS]:
        with st.expander("Leak-safe redaction (severity phrases removed)"):
            st.text(prepared_query)
    ev = parsed.get("evidence") or []
    conf = parsed.get("confidence") or {}
    trace = parsed.get("trace") or []
    if not ev:
        st.info("No structured facts recognized — use the manual pickers below.")
        return
    chips = []
    for n in ev:
        c = conf.get(n, 1.0)
        chips.append(f"`{n}`" if c >= 0.999 else f"`{n}` (soft {c:.0%})")
    st.success("Parsed evidence: " + " · ".join(chips))
    with st.expander("Evidence trace"):
        st.dataframe(
            pd.DataFrame([
                {
                    "Node": n,
                    "How parsed": why,
                    "Strength": "hard" if conf.get(n, 1.0) >= 0.999 else f"soft ({conf.get(n, 1.0):.0%})",
                }
                for n, why in trace
            ]),
            use_container_width=True,
            hide_index=True,
        )


@st.cache_data(show_spinner=False)
def zhang_table7_for_outcome(outcome_name: str, targets: set):
    ds = main_app.refined_dataset
    res = zd.empirical_cause_distribution(
        outcome_name, targets=targets, dataset=ds,
        cause_factor_only=True,
    )
    dist = {}
    for c in res["causes"]:
        if c["cause"].lower() in zd.DIAGNOSIS_RESPONSE_LABELS:
            continue
        dist[c["cause"]] = c["probability"]
    return {"outcome_count": res["outcome_count"], "dist": dist, "causes": res["causes"]}


def render_zhang_audit_panel():
    st.markdown("#### Offline Zhang audit (regenerate with `tests/bn_full_comparison.py`)")
    if BN_FULL_JSON.is_file():
        bnf = json.loads(BN_FULL_JSON.read_text(encoding="utf-8"))
        sb = bnf.get("scoreboard", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("BN exact", sb.get("EXACT", "—"))
        c2.metric("BN close", sb.get("CLOSE", "—"))
        c3.metric("BN differs", sb.get("DIFFERS", "—"))
    else:
        st.warning(f"Missing `{BN_FULL_JSON.name}` — run offline comparison first.")
    if TABLE7_CSV.is_file():
        t7 = pd.read_csv(TABLE7_CSV)
        n_match = int((t7["match"] == "YES").sum()) if "match" in t7.columns else "?"
        st.success(f"Table 7 offline: **{n_match}/{len(t7)}** cells match Zhang.")

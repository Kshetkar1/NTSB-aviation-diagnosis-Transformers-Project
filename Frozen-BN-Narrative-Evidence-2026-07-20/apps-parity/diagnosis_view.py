"""Diagnosis mode — upstream causes with Zhang Table 7 comparison."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import trees
import zhang_diagnosis as zd
import zhang_reference as zr

from demo_common import (
    CachedApp,
    TABLE7_CSV,
    bn_posteriors,
    load_upgraded_bn,
    main_app,
    pool_from_retrieval,
    tree_to_dot,
    zhang_table7_for_outcome,
)


def _query_for_tree(query: str, outcome_name: str, source: str) -> str:
    """Ensure tree builders can detect outcome (inject core if manual)."""
    if source == "manual picker" and outcome_name.lower() not in query.lower():
        return f"{query} {outcome_name}"
    return query


@st.cache_data(show_spinner=False)
def _rank_causes(outcome_name, targets_tuple, query, top_n_incidents, full_population, top_n):
    ds = main_app.refined_dataset
    targets = set(targets_tuple)
    pool = list(ds.keys()) if full_population else pool_from_retrieval(query, top_n_incidents)
    res = zd.empirical_cause_distribution(
        outcome_name, targets=targets, dataset=ds, restrict_ev_ids=pool,
        cause_factor_only=True,
    )
    causes = [
        c for c in res["causes"]
        if c["cause"].lower() not in zd.DIAGNOSIS_RESPONSE_LABELS
    ]
    return {
        "outcome": outcome_name,
        "outcome_count": res["outcome_count"],
        "population_label": "Table 7 (all outcome accidents)" if full_population else "query pool",
        "causes": causes[:top_n],
    }


def _load_table7_zhang_lookup() -> dict[str, float]:
    """Fire Table 7 published values from offline CSV if available."""
    if not TABLE7_CSV.is_file():
        return {}
    df = pd.read_csv(TABLE7_CSV)
    out = {}
    for _, row in df.iterrows():
        cause = str(row.get("cause") or row.get("Cause") or "").strip()
        zhang = row.get("zhang") or row.get("Zhang")
        if cause and zhang is not None:
            try:
                out[cause.lower()] = float(zhang)
            except (TypeError, ValueError):
                pass
    return out


def _table7_comparison_rows(causes, zhang_dist: dict, zhang_csv: dict, method: str):
    rows = []
    for c in causes:
        lab = c["cause"]
        ours = float(c["probability"])
        zhang = zhang_csv.get(lab.lower())
        if zhang is None:
            zhang = zhang_dist.get(lab)
        rows.append(zr.comparison_row(lab, ours, zhang, method))
    return rows


def render_diagnosis_page(
    query: str,
    det: tuple[str, set[str]],
    det_source: str,
    parsed: dict,
    pool: list,
    retr: list,
    *,
    top_n_incidents: int,
    table7_population: bool,
    diag_branching: int,
    diag_depth: int,
    diag_min_prob: float,
    diag_min_n: int,
    drop_generic: bool,
):
    outcome_name, targets = det
    st.header("2 · Outcome and similar accidents")
    gt = zhang_table7_for_outcome(outcome_name, targets)
    c1, c2, c3 = st.columns(3)
    c1.metric("Outcome", outcome_name)
    c2.metric("Outcome accidents (denominator)", gt["outcome_count"])
    c3.metric("Similar incidents scored", len(pool))
    st.caption(f"Outcome source: **{det_source}**")
    with st.expander("Top 10 similar accidents"):
        st.dataframe(
            pd.DataFrame([
                {"ev_id": h["ev_id"], "similarity": round(h["score"], 3)}
                for h in retr[:10]
            ]),
            use_container_width=True,
            hide_index=True,
        )

    method = (
        "Zhang Table 7 counting (full population)"
        if table7_population
        else "Zhang Table 7 counting (query pool)"
    )
    rank = _rank_causes(
        outcome_name, tuple(targets), query, top_n_incidents,
        table7_population, top_n=20,
    )
    zhang_csv = _load_table7_zhang_lookup()
    zhang_ref = zhang_table7_for_outcome(outcome_name, targets)["dist"]

    st.header("3 · Upstream causes — ours vs Zhang")
    st.caption(
        f"Estimand: **P(cause | {outcome_name})** · denominator **{rank['outcome_count']}** · "
        f"population: **{rank['population_label']}**"
    )
    if table7_population and outcome_name.lower() == "fire":
        st.success("Fire + full population reproduces Zhang Table 7 (85/85 offline).")

    rows = _table7_comparison_rows(
        rank["causes"], zhang_ref, zhang_csv, method,
    )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    tree_query = _query_for_tree(query, outcome_name, det_source)
    st.header("4 · Diagnosis tree")
    st.caption(
        "Level 1 edges = Table 7 cells. Level 2+ = P(cause | outcome ∧ parent) — "
        "**Method: exploratory counting (no Zhang benchmark)**."
    )
    try:
        with st.spinner("Building diagnosis tree…"):
            diag = trees.build_diagnosis_tree(
                tree_query,
                top_n_incidents=top_n_incidents,
                branching=diag_branching,
                depth=diag_depth,
                min_prob=diag_min_prob,
                drop_generic=drop_generic,
                main_app=CachedApp(),
                cause_factor_only=True,
                exclude_responses=True,
                full_population=table7_population,
                min_n=diag_min_n,
            )
        if diag["meta"].get("error") or diag["tree"] is None:
            st.warning(diag["meta"].get("error", "Could not build tree."))
        else:
            st.graphviz_chart(tree_to_dot(diag), use_container_width=True)
    except Exception as exc:
        st.error(f"Diagnosis tree failed: {exc}")

    st.header("5 · Multi-evidence BN (optional)")
    st.caption(
        "Method: **frozen BN propagation** — joint readout when several facts are known."
    )
    _render_diagnosis_bn(query, parsed, seed=[outcome_name])


def _render_diagnosis_bn(query: str, parsed: dict, seed: list):
    try:
        bu, bn, _ = load_upgraded_bn()
    except Exception as exc:
        st.error(f"BN build failed: {exc}")
        return

    all_nodes = sorted(n for n in bn.names() if n not in (bu.INJ_NODE, bu.DMG_NODE))
    ev_list = list(parsed.get("evidence") or [])
    for s in seed:
        if s in bn.names() and s not in ev_list:
            ev_list.append(s)

    evidence = st.multiselect(
        "Evidence nodes (edit parsed facts)",
        options=all_nodes,
        default=[e for e in ev_list if e in all_nodes],
        key="diag_bn_evidence",
    )
    if not evidence:
        st.info("Select at least one evidence node to run BN inference.")
        return

    conf = parsed.get("confidence") or {}
    ev_conf = tuple(conf.get(e, 1.0) for e in evidence)
    targets = (bu.INJ_NODE, bu.DMG_NODE)
    with st.spinner("BN inference…"):
        posts = bn_posteriors(tuple(evidence), targets, ev_conf)

    rows = []
    for state in bu.INJ_STATES:
        ours = posts[bu.INJ_NODE].get(state, 0.0)
        zhang = zr.fig12_zhang_value(set(e.lower() for e in evidence), state)
        rows.append(zr.comparison_row(
            f"injury: {state}", ours, zhang, "BN propagation (LazyPropagation)",
        ))
    for state in bu.DMG_STATES:
        ours = posts[bu.DMG_NODE].get(state, 0.0)
        zhang = zr.fig12_zhang_value(set(e.lower() for e in evidence), state)
        rows.append(zr.comparison_row(
            f"damage: {state}", ours, zhang, "BN propagation (LazyPropagation)",
        ))
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

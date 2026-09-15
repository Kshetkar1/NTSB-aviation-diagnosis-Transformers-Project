"""Prognosis mode — BN downstream (Zhang-comparable) + Markov tree (secondary)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import trees
import zhang_reference as zr

from demo_common import (
    CachedApp,
    bn_posteriors,
    bn_sev_readout,
    load_upgraded_bn,
    main_app,
    p_yes,
    tree_to_dot,
)
from evidence_bridge import hard_evidence_set


def render_bn_sev_panel(prepared_query: str, raw_query: str | None = None):
    """Primary paper prognosis — bn-sev (k=100 virtual severity -> frozen BN)."""
    st.header("2 · Severity prognosis — bn-sev (primary, paper protocol)")
    st.caption(
        "Estimand: **P(injury, damage | narrative)** via k=100 similarity-weighted "
        "severity among Zhang-window neighbors, entered as virtual evidence on the "
        "frozen BN (Jeffrey conditioning). Outcome phrases stripped before embedding. "
        "Held-out 2007–2019 (n=296): **90.9% injury / 77.4% damage** top-1."
    )
    try:
        bu, _, _ = load_upgraded_bn()
        readout = bn_sev_readout(prepared_query, raw_query=raw_query)
    except Exception as exc:
        st.error(f"bn-sev readout failed: {exc}")
        return

    if not readout.get("injury"):
        st.warning("No neighbors found — cannot compute bn-sev readout.")
        return

    def _top1(dist: dict) -> str:
        if not dist:
            return "—"
        return max(dist, key=dist.get)

    def _rows(states, post_key, knn_key, label):
        rows = []
        for s in states:
            rows.append({
                "state": s,
                "bn-sev posterior": round(readout[post_key].get(s, 0.0), 4),
                "k-NN readout (no BN)": round(readout[knn_key].get(s, 0.0), 4),
            })
        return rows, _top1(readout[post_key])

    inj_rows, inj_top = _rows(bu.INJ_STATES, "injury", "knn_injury", "injury")
    dmg_rows, dmg_top = _rows(bu.DMG_STATES, "damage", "knn_damage", "damage")

    c1, c2, c3 = st.columns(3)
    c1.metric("Neighbors pooled", readout.get("top_k", 100))
    c2.metric("Top-1 injury (bn-sev)", inj_top)
    c3.metric("Top-1 damage (bn-sev)", dmg_top)

    stated = readout.get("stated") or {}
    stated_bits = [w for w in (
        f"damage stated: **{stated['damage']}**" if stated.get("damage") else "",
        f"injury stated: **{stated['injury']}**" if stated.get("injury") else "",
    ) if w]
    if stated_bits:
        st.caption(
            "The raw narrative also states severity outright ("
            + " · ".join(stated_bits)
            + ") — shown for transparency; **not** used in bn-sev."
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Personnel injury")
        st.dataframe(pd.DataFrame(inj_rows), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Aircraft damage")
        st.dataframe(pd.DataFrame(dmg_rows), use_container_width=True, hide_index=True)


def render_bn_downstream(query: str, parsed: dict, manual_evidence: list | None = None):
    """Zhang Table 9 comparison — structured evidence propagation."""
    st.header("3 · Downstream readout — frozen BN (Zhang Table 9 scenarios)")
    st.caption(
        "Estimand: **P(target | evidence)** via exact BN propagation. "
        "Zhang column filled when your evidence matches a Table 9 / Fig 12 scenario."
    )

    try:
        bu, bn, _ = load_upgraded_bn()
    except Exception as exc:
        st.error(f"Could not build BN: {exc}")
        return

    all_nodes = sorted(n for n in bn.names() if n not in (bu.INJ_NODE, bu.DMG_NODE))
    default_ev = list(manual_evidence or parsed.get("evidence") or [])
    evidence = st.multiselect(
        "Evidence (from narrative — edit to match Zhang scenarios)",
        options=all_nodes,
        default=[e for e in default_ev if e in all_nodes],
        key="prog_bn_evidence",
    )
    if not evidence:
        st.warning("No evidence selected — pick nodes or parse a narrative with NTSB vocabulary.")
        return

    conf_map = parsed.get("confidence") or {}
    ev_conf = tuple(conf_map.get(e, 1.0) for e in evidence)
    hard = hard_evidence_set({"evidence": evidence, "confidence": conf_map})
    col_idx, col_label = zr.match_table9_column(hard)
    if col_idx is not None:
        st.success(f"Matched Zhang Table 9 scenario: **{col_label}** (column {col_idx + 1}/5)")
    else:
        st.info("No exact Table 9 match — Zhang column shows N/A (Fig 12 partial match may still apply).")

    event_targets = []
    for _disp, node, kind, _vals in zr.TABLE9_ROWS:
        if kind == "event" and node in bn.names() and node not in evidence:
            event_targets.append(node)

    targets = tuple(event_targets) + (bu.INJ_NODE, bu.DMG_NODE)
    with st.spinner("Running BN propagation…"):
        posts = bn_posteriors(tuple(evidence), targets, ev_conf)

    rows = []
    for i, (disp, node, kind, _vals) in enumerate(zr.TABLE9_ROWS):
        if kind == "event":
            if node not in posts:
                continue
            ours = p_yes(posts, node)
            zhang = zr.table9_zhang_value(col_idx, i)
            rows.append(zr.comparison_row(
                disp, ours, zhang, "BN propagation P(event=Yes | evidence)",
            ))
        elif kind == "injury":
            ours = posts[bu.INJ_NODE].get(node, 0.0)
            zhang = zr.table9_zhang_value(col_idx, i)
            if zhang is None:
                zhang = zr.fig12_zhang_value(hard, node)
            rows.append(zr.comparison_row(
                disp, ours, zhang, "BN propagation P(injury state | evidence)",
            ))
        elif kind == "damage":
            ours = posts[bu.DMG_NODE].get(node, 0.0)
            zhang = zr.table9_zhang_value(col_idx, i)
            if zhang is None:
                zhang = zr.fig12_zhang_value(hard, node)
            rows.append(zr.comparison_row(
                disp, ours, zhang, "BN propagation P(damage state | evidence)",
            ))

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_markov_tree(
    prepared_query: str,
    seed_event: str | None,
    *,
    top_n_incidents: int,
    prog_branching: int,
    prog_depth: int,
    prog_min_prob: float,
    prog_min_n: int,
    prog_zhang_pop: bool,
    prog_leaves: bool,
    prog_drop_generic: bool,
    prog_backoff: bool,
):
    """Secondary — Markov sequence explorer, explicitly not Zhang BN."""
    with st.expander("Sequence explorer (Markov — ours, not Zhang BN)", expanded=False):
        st.caption(
            "Estimand: **P(next event | current event)** from consecutive sequences. "
            "Leaf outcomes use **empirical reachability**, not BN posteriors."
        )
        tree_query = prepared_query
        if seed_event and seed_event.lower() not in prepared_query.lower():
            tree_query = f"{prepared_query} {seed_event}"
        try:
            with st.spinner("Building Markov tree…"):
                prog = trees.build_prognosis_tree(
                    tree_query,
                    top_n_incidents=top_n_incidents,
                    branching=prog_branching,
                    depth=prog_depth,
                    min_prob=prog_min_prob,
                    min_n=prog_min_n,
                    query_relevant=not prog_zhang_pop,
                    main_app=CachedApp(),
                    dataset=main_app.refined_dataset,
                    add_outcome_leaves=prog_leaves,
                    drop_generic=prog_drop_generic,
                    deep_backoff=prog_backoff and not prog_zhang_pop,
                )
            if prog["meta"].get("error") or prog["tree"] is None:
                st.warning(prog["meta"].get("error", "No tree."))
            else:
                m = prog["meta"]
                st.caption(
                    f"Seed: **{m.get('seed_event', '?')}** · "
                    f"population: **{m.get('transition_population', '?')}**"
                )
                st.graphviz_chart(tree_to_dot(prog), use_container_width=True)
        except Exception as exc:
            st.error(f"Markov tree failed: {exc}")


def render_prognosis_page(
    query: str,
    prepared_query: str,
    parsed: dict,
    seed_event: str | None,
    pool: list,
    retr: list,
    *,
    top_n_incidents: int,
    prog_branching: int,
    prog_depth: int,
    prog_min_prob: float,
    prog_min_n: int,
    prog_zhang_pop: bool,
    prog_leaves: bool,
    prog_drop_generic: bool,
    prog_backoff: bool,
):
    st.header("1 · Context")
    c1, c2 = st.columns(2)
    c1.metric("Similar incidents scored", len(pool))
    c2.metric("Seed event", seed_event or "(from narrative)")
    with st.expander("Top 10 similar accidents"):
        st.dataframe(
            pd.DataFrame([
                {"ev_id": h["ev_id"], "similarity": round(h["score"], 3)}
                for h in retr[:10]
            ]),
            use_container_width=True,
            hide_index=True,
        )

    render_bn_sev_panel(prepared_query, raw_query=query)
    render_bn_downstream(prepared_query, parsed, manual_evidence=parsed.get("evidence"))
    render_markov_tree(
        prepared_query, seed_event,
        top_n_incidents=top_n_incidents,
        prog_branching=prog_branching,
        prog_depth=prog_depth,
        prog_min_prob=prog_min_prob,
        prog_min_n=prog_min_n,
        prog_zhang_pop=prog_zhang_pop,
        prog_leaves=prog_leaves,
        prog_drop_generic=prog_drop_generic,
        prog_backoff=prog_backoff,
    )

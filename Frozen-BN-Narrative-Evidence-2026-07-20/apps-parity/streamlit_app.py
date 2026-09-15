#!/usr/bin/env python3
"""NTSB apps-parity demo — separate Diagnosis / Prognosis with Zhang comparison.

Launch:
  ./run_app.sh
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
# Paper-default index: Zhang build window 1982-2006 (same as held-out eval).
os.environ.pop("NTSB_FULL_CORPUS", None)
os.environ.pop("NTSB_USE_TRAIN_INDEX", None)

import streamlit as st

st.set_page_config(
    page_title="NTSB Parity Demo (Diagnosis / Prognosis)",
    layout="wide",
)

# Path setup via demo_common (must import first)
import config  # noqa: E402
import main_app  # noqa: E402
from demo_common import (  # noqa: E402
    SUGGESTED_QUERIES,
    cached_retrieval,
    load_upgraded_bn,
    pool_from_retrieval,
    prepare_pipeline_query,
    render_evidence_bar,
    render_zhang_audit_panel,
)
from diagnosis_view import render_diagnosis_page  # noqa: E402
from evidence_bridge import (  # noqa: E402
    list_outcome_cores,
    list_transition_seeds,
    parse_narrative_evidence,
    resolve_outcome,
    seed_events_from_parsed,
)
from prognosis_view import render_prognosis_page  # noqa: E402

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Analysis mode")
analysis_mode = st.sidebar.radio(
    "Show pipeline for",
    ["Diagnosis", "Prognosis"],
    horizontal=True,
    help="Diagnosis = upstream causes. Prognosis = downstream BN readout.",
)

st.sidebar.header("Pipeline controls")
st.sidebar.caption(f"Index: **{config.ACTIVE_INDEX_LABEL}**")
use_all = st.sidebar.checkbox(
    "Use all incidents for retrieval pool",
    value=False,
    help="Off = query-scoped pool (paper default). On = every incident in the active index.",
)
if use_all:
    top_n_incidents = len(main_app.refined_dataset)
else:
    top_n_incidents = st.sidebar.slider("Retrieval breadth", 50, 400, 300, step=50)

if analysis_mode == "Diagnosis":
    st.sidebar.markdown("**Diagnosis settings**")
    table7_population = st.sidebar.checkbox("Table 7 population (full dataset)", value=True)
    diag_branching = st.sidebar.slider("Tree branching", 2, 6, 4)
    diag_depth = st.sidebar.slider("Tree depth", 1, 3, 2)
    diag_min_prob = st.sidebar.slider("Min edge p", 0.0, 0.20, 0.03, step=0.01)
    diag_min_n = st.sidebar.slider("Min support n (depth ≥ 2)", 1, 4, 2)
    drop_generic = st.sidebar.checkbox("Drop generic catch-alls", value=True)
else:
    st.sidebar.markdown("**Prognosis settings**")
    prog_branching = st.sidebar.slider("Markov tree branching", 2, 5, 3)
    prog_depth = st.sidebar.slider("Markov tree depth", 1, 4, 3)
    prog_min_prob = st.sidebar.slider("Min edge p", 0.0, 0.30, 0.05, step=0.01)
    prog_min_n = st.sidebar.slider("Min support n", 1, 10, 2)
    prog_zhang_pop = st.sidebar.checkbox("Markov: Zhang population (global)", value=True)
    prog_leaves = st.sidebar.checkbox("Markov: damage/injury leaves", value=True)
    prog_drop_generic = st.sidebar.checkbox("Markov: drop generic catch-alls", value=True)
    prog_backoff = st.sidebar.checkbox("Markov: global backoff", value=True)

with st.sidebar.expander("Zhang offline audit"):
    render_zhang_audit_panel()

# ---------------------------------------------------------------------------
# Header + query
# ---------------------------------------------------------------------------
st.title("NTSB Parity Demo")
st.markdown(
    "**Separate Diagnosis and Prognosis modes.** Every probability is labeled. "
    "Zhang published values appear where they exist. Original demo unchanged in `apps/`."
)

if not getattr(main_app, "DATA_LOADED", False):
    st.error("Knowledge base not loaded. Run data-processing scripts, then relaunch.")
    st.stop()

if "query" not in st.session_state:
    st.session_state["query"] = SUGGESTED_QUERIES[0]

st.header("Your narrative")
cols = st.columns(3)
for i, q in enumerate(SUGGESTED_QUERIES):
    if cols[i % 3].button(q, key=f"sugg_{i}", use_container_width=True):
        st.session_state["query"] = q

query = st.text_input("Describe the situation", key="query")

if st.button("Run analysis", type="primary", use_container_width=True):
    st.session_state["ran"] = True

if not st.session_state.get("ran"):
    st.info("Pick a mode in the sidebar, enter a narrative, and press **Run analysis**.")
    st.stop()

if not query or not query.strip():
    st.warning("Enter a non-empty narrative.")
    st.stop()

query = query.strip()
raw_query = query
prepared_query, truncated_text = prepare_pipeline_query(raw_query)
truncated = len(raw_query) > len(truncated_text)
ds = main_app.refined_dataset

# BN node list for parsing
try:
    _, bn, _ = load_upgraded_bn()
    bn_nodes = sorted(bn.names())
except Exception:
    bn_nodes = []

parsed = parse_narrative_evidence(prepared_query, bn_nodes, ds) if bn_nodes else {
    "evidence": [], "trace": [], "confidence": {},
}

with st.spinner("Retrieving similar incidents…"):
    pool = pool_from_retrieval(prepared_query, top_n_incidents)
    retr = cached_retrieval(prepared_query)

st.divider()
render_evidence_bar(raw_query, prepared_query, parsed, truncated=truncated)

# ---------------------------------------------------------------------------
# Mode-specific fallbacks (no hard stop)
# ---------------------------------------------------------------------------
if analysis_mode == "Diagnosis":
    det, det_source = resolve_outcome(prepared_query, None, ds)
    if det is None:
        st.warning("No outcome auto-detected in narrative.")
        cores = list_outcome_cores(ds)
        manual = st.selectbox(
            "Pick outcome manually (required for diagnosis)",
            options=[""] + cores[:80],
            format_func=lambda x: "(select…)" if not x else x,
            key="manual_outcome",
        )
        det, det_source = resolve_outcome(prepared_query, manual or None, ds)
    if det is None:
        st.error("Select an outcome to run diagnosis.")
        st.stop()

    render_diagnosis_page(
        prepared_query, det, det_source, parsed, pool, retr,
        top_n_incidents=top_n_incidents,
        table7_population=table7_population,
        diag_branching=diag_branching,
        diag_depth=diag_depth,
        diag_min_prob=diag_min_prob,
        diag_min_n=diag_min_n,
        drop_generic=drop_generic,
    )
else:
    seeds = seed_events_from_parsed(parsed)
    seed_event = seeds[0] if seeds else None
    if not seed_event:
        st.warning("No seed event parsed from narrative.")
        options = list_transition_seeds(ds)
        seed_event = st.selectbox(
            "Pick initial event manually (for Markov tree seed)",
            options=[""] + options,
            format_func=lambda x: "(select…)" if not x else x,
            key="manual_seed",
        ) or None

    render_prognosis_page(
        raw_query, prepared_query, parsed, seed_event, pool, retr,
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

st.divider()
st.header("Epilogue · Zhang parity audit")
render_zhang_audit_panel()

"""
Streamlit demo for the paper worked examples.

Pre-computed pipeline outputs from `precompute_a0.py` and `precompute_a2.py`.
Table-first layout for live walkthroughs (you explain; the app shows numbers).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE / "data"
A0_PATH = DATA_DIR / "a0.json"
A2_PATH = DATA_DIR / "a2.json"

st.set_page_config(page_title="NTSB Worked Examples", layout="wide")

PROB_FMT = "%.4f"


@st.cache_data
def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def step(title: str) -> None:
    st.markdown(f"**{title}**")


def show_df(df: pd.DataFrame, *, prob_cols: dict[str, str] | None = None) -> None:
    cfg = {}
    if prob_cols:
        for col, label in prob_cols.items():
            if col in df.columns:
                cfg[col] = st.column_config.NumberColumn(label, format=PROB_FMT)
    st.dataframe(df, use_container_width=True, hide_index=True, column_config=cfg or None)


def diagnosis_summary(ltp_causes: list[dict], *, label: str = "Diagnosis result") -> None:
    """Highlight top-ranked causes — LTP *is* the final P(C|Q) distribution."""
    if not ltp_causes:
        return
    top = ltp_causes[0]
    st.success(
        f"**{label}** — The LTP table below is the final output: a ranked list of "
        f"P(cause | query) over {len(ltp_causes)} causes (sums to 1.0). "
        f"There is no single winner unless you read **rank #1**."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Top-1 P(C|Q)", f"{top['probability']:.4f}")
    c2.metric("Rank-1 cause (preview)", top["cause"][:55] + "…")
    if len(ltp_causes) >= 3:
        c3.metric("Top-3 mass", f"{sum(c['probability'] for c in ltp_causes[:3]):.4f}")
    with st.expander("Top 5 causes (final diagnosis ranking)", expanded=True):
        show_df(
            pd.DataFrame(ltp_causes[:5]),
            prob_cols={"probability": "P(C|Q)"},
        )


def prognosis_summary(prognosis: list[dict]) -> None:
    if not prognosis:
        return
    top = prognosis[0]
    st.success(
        "**Prognosis result** — Final output is P(next event | query). "
        f"Most likely next event: **{top['probability']:.1%}**."
    )
    c1, c2 = st.columns(2)
    c1.metric("Top-1 P(next|Q)", f"{top['probability']:.4f}")
    c2.metric("Top next event", top["event"][:70] + ("…" if len(top["event"]) > 70 else ""))


def cluster_shift_table(a0_clusters: list[dict], a2_clusters: list[dict]) -> pd.DataFrame:
    a0_map = {c["cluster"]: c["p_k_given_q"] for c in a0_clusters}
    rows = []
    for c in sorted(a2_clusters, key=lambda x: -x["p_k_given_q"]):
        a0p = a0_map.get(c["cluster"], 0.0)
        a2p = c["p_k_given_q"]
        rows.append({
            "cluster": c["cluster"],
            "n": c["n_incidents"],
            "A0 P(K|Q)": a0p,
            "A2 P(K|Q)": a2p,
            "shift": a2p - a0p,
        })
    return pd.DataFrame(rows)


# ----- Sidebar -----------------------------------------------------------------

st.sidebar.title("Worked examples")
st.sidebar.caption("Pre-computed from production pipeline code.")
show_codes = st.sidebar.checkbox("Show Zhang 54-code mapping", value=False)
st.sidebar.code(
    ".venv/bin/python Worked_Examples/precompute_a0.py\n"
    ".venv/bin/python Worked_Examples/precompute_a2.py\n"
    ".venv/bin/python -m streamlit run Worked_Examples/app.py",
    language="bash",
)

st.title("NTSB pipeline — worked examples")
st.caption("Same numbers as the paper examples. Tables only; explain live.")

tab_a0, tab_a2 = st.tabs(["Example 1 — A0 (Sec 7)", "Example 2 — A2 (Sec 8.5.1)"])

# =============================================================================
# Example 1 — A0
# =============================================================================
with tab_a0:
    a0 = load_json(A0_PATH)
    if a0 is None:
        st.error(f"Missing `{A0_PATH.name}`. Run precompute_a0.py first.")
    else:
        diag_tab, prog_tab = st.tabs(["Diagnosis (7.1)", "Prognosis (7.2)"])

        with diag_tab:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mode", a0["mode"])
            c2.metric("Corpus", f"{a0['corpus_size']:,}")
            c3.metric("Retrieved", len(a0["retrieval"]))
            c4.metric("LTP causes", len(a0["ltp_causes"]))

            step("1 — Query")
            st.text(a0["query"])

            step(f"2 — Top {len(a0['retrieval'])} retrieval (cosine similarity)")
            show_df(pd.DataFrame(a0["retrieval"]), prob_cols={"score": "cosine sim"})

            step(f"3 — Clusters + P(K|Q) ({len(a0['clusters'])} active)")
            show_df(
                pd.DataFrame(a0["clusters"]),
                prob_cols={
                    "avg_similarity": "avg sim",
                    "weight": "weight",
                    "p_k_given_q": "P(K|Q)",
                },
            )
            skipped = a0.get("skipped_clusters") or []
            if skipped:
                st.caption(
                    f"Excluded from LTP (no recorded causes): "
                    + ", ".join(c["cluster"] for c in skipped)
                )

            step("4 — P(C|K) per cluster")
            for label, rows in a0["cluster_causes"].items():
                st.markdown(f"*{label}* ({len(rows)} causes)")
                show_df(pd.DataFrame(rows), prob_cols={"p_c_given_k": "P(C|K)"})

            step(f"5 — LTP: P(C|Q) ({len(a0['ltp_causes'])} causes, Σ={a0.get('ltp_sum', 0):.4f})")
            diagnosis_summary(a0["ltp_causes"])
            show_df(pd.DataFrame(a0["ltp_causes"]), prob_cols={"probability": "P(C|Q)"})

            if show_codes:
                cd = a0["coded_distribution"]
                step(f"6 — Zhang code mapping ({len(cd)} codes)")
                if cd:
                    st.info(
                        f"Optional aggregation: top mapped code **{cd[0]['code']}** "
                        f"at **{cd[0]['probability']:.1%}** ({cd[0]['label']})."
                    )
                show_df(
                    pd.DataFrame(cd),
                    prob_cols={"probability": "P(code|Q)"},
                )

        with prog_tab:
            meta = a0.get("prognosis_meta") or {}
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sequences w/ next", meta.get("sequences_analyzed", "—"))
            c2.metric("Next-event labels", len(a0.get("prognosis") or []))
            c3.metric("LTP Σ", f"{a0.get('prognosis_ltp_sum', 0):.4f}")
            c4.metric("Prognosis clusters", len(a0.get("prognosis_clusters") or []))

            step("P1 — Prognosis query (truncated)")
            st.text(a0.get("prognosis_query") or a0["query"])
            st.caption("Retrieve → cluster → P(Next|K) → LTP (same structure as diagnosis).")

            prog_ret = a0.get("prognosis_retrieval") or []
            if prog_ret:
                step(f"P2 — Retrieve top {len(prog_ret)} (cosine similarity)")
                show_df(pd.DataFrame(prog_ret), prob_cols={"score": "cosine sim"})

            prog_clusters = a0.get("prognosis_clusters") or []
            if prog_clusters:
                step(f"P3 — Clusters + P(K|Q) ({len(prog_clusters)} active)")
                show_df(
                    pd.DataFrame(prog_clusters),
                    prob_cols={
                        "avg_similarity": "avg sim",
                        "weight": "weight",
                        "p_k_given_q": "P(K|Q)",
                    },
                )
                skipped = a0.get("prognosis_skipped_clusters") or []
                if skipped:
                    st.caption(
                        "Excluded from LTP: "
                        + ", ".join(c["cluster"] for c in skipped)
                    )

            prog_next = a0.get("prognosis_cluster_next_events") or {}
            if prog_next:
                step("P4 — P(Next event | Cluster) per cluster")
                for label, rows in prog_next.items():
                    st.markdown(f"*{label}* ({len(rows)} next-event labels)")
                    show_df(
                        pd.DataFrame(rows),
                        prob_cols={"p_next_given_k": "P(Next|K)"},
                    )

            if a0.get("prognosis"):
                step(
                    f"P5 — LTP: P(Next event | Query) "
                    f"({len(a0['prognosis'])} labels, Σ={a0.get('prognosis_ltp_sum', 0):.4f})"
                )
                prognosis_summary(a0["prognosis"])
                show_df(
                    pd.DataFrame(a0["prognosis"]),
                    prob_cols={"probability": "P(Next|Q)"},
                )

            aligned = a0.get("prognosis_aligned") or []
            if aligned:
                with st.expander(f"Defining-event rows ({len(aligned)} incidents)"):
                    show_df(
                        pd.DataFrame(aligned),
                        prob_cols={"incident_similarity": "cosine sim"},
                    )

            for step_row in a0.get("prognosis_multi_step") or []:
                if step_row.get("step", 1) <= 1:
                    continue
                step(f"P5 — Multi-step chain step {step_row.get('step')} (LTP over transitions)")
                show_df(
                    pd.DataFrame(step_row.get("events") or []),
                    prob_cols={"probability": "P(next|Q)"},
                )

# =============================================================================
# Example 2 — A2
# =============================================================================
with tab_a2:
    a2 = load_json(A2_PATH)
    if a2 is None:
        st.error(f"Missing `{A2_PATH.name}`. Run precompute_a2.py first.")
    else:
        gt = a2["ground_truth"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mode", a2["mode"])
        c2.metric("Corpus", f"{a2['corpus_size']:,}")
        c3.metric("ev_id", a2["ev_id"])
        c4.metric("Truth code", gt["code"])
        c5.metric("A0 / A2 hit", f"{'✓' if a2['hit']['a0'] else '✗'} / {'✓' if a2['hit']['a2'] else '✗'}")

        step("1 — Query + ground truth")
        st.text(a2["query"][:1200] + ("…" if len(a2["query"]) > 1200 else ""))
        st.caption(f"Truth: {gt['text'][:300]}")

        step(f"2 — Top {len(a2['retrieval'])} retrieval")
        show_df(pd.DataFrame(a2["retrieval"]), prob_cols={"score": "cosine sim"})

        step("3 — A0 clusters + P(K|Q)")
        show_df(
            pd.DataFrame(a2["a0"]["clusters"]),
            prob_cols={"avg_similarity": "avg sim", "p_k_given_q": "P(K|Q)"},
        )

        step("4 — A2 clusters after structural reranking (α=2.0)")
        show_df(
            pd.DataFrame(a2["a2"]["clusters"]),
            prob_cols={"avg_similarity": "avg rerank", "p_k_given_q": "P(K|Q)"},
        )

        step("5 — Cluster weight shift (A2 − A0)")
        show_df(
            cluster_shift_table(a2["a0"]["clusters"], a2["a2"]["clusters"]),
            prob_cols={"A0 P(K|Q)": "A0", "A2 P(K|Q)": "A2", "shift": "Δ"},
        )

        col_a0, col_a2 = st.columns(2)
        with col_a0:
            step(f"6a — A0 LTP ({len(a2['a0']['ltp_causes'])} causes)")
            diagnosis_summary(a2["a0"]["ltp_causes"], label="A0 diagnosis result")
            show_df(
                pd.DataFrame(a2["a0"]["ltp_causes"]),
                prob_cols={"probability": "P(C|Q)"},
            )
        with col_a2:
            step(f"6b — A2 LTP ({len(a2['a2']['ltp_causes'])} causes)")
            diagnosis_summary(a2["a2"]["ltp_causes"], label="A2 diagnosis result")
            show_df(
                pd.DataFrame(a2["a2"]["ltp_causes"]),
                prob_cols={"probability": "P(C|Q)"},
            )
        st.metric("Final top-1 code (A0 / A2)", f"{a2['a0']['top1_code']} / {a2['a2']['top1_code']}")

        if show_codes:
            col_a0, col_a2 = st.columns(2)
            with col_a0:
                step("7a — A0 coded distribution")
                show_df(
                    pd.DataFrame(a2["a0"]["coded_distribution"]),
                    prob_cols={"probability": "P(code|Q)"},
                )
            with col_a2:
                step("7b — A2 coded distribution")
                show_df(
                    pd.DataFrame(a2["a2"]["coded_distribution"]),
                    prob_cols={"probability": "P(code|Q)"},
                )

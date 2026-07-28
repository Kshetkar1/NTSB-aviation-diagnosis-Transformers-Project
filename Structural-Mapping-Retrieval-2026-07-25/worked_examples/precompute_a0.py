"""
Precompute A0 worked example for the paper Section 7.

Runs the A0 (embedding-only) pipeline in PRODUCTION mode (full 2,243 corpus)
on a synthetic narrative query. Calls only production functions in
`main_app` and `cause_code_mapper` — no parallel math here. Saves both:
  - `data/a0.json`  (structured output for the Streamlit demo + HTML)
  - `data/trace_synthetic_a0.txt`  (human-readable trace, paper artifact)
"""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

# Production mode = full 2,243 corpus. Must be unset BEFORE import main_app.
os.environ.pop("NTSB_USE_TRAIN_INDEX", None)

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
SCRIPTS_DIR = PROJECT_ROOT / "Testing_Structural_Mapping_Slides" / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

import main_app  # noqa: E402
from cause_code_mapper import build_code_embeddings, map_distribution_to_codes  # noqa: E402


class _Tee:
    """Write to multiple streams at once."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

# ----- Synthetic narrative queries (paper Section 7) -----
# Diagnosis (7.1): full narrative including outcome — realistic production query.
QUERY_DIAGNOSIS = (
    "During cruise flight at FL340, the No. 2 engine experienced a sudden loss "
    "of power accompanied by elevated EGT, vibration, and an associated master "
    "caution. The crew completed the engine shutdown procedure and declared an "
    "emergency. The aircraft diverted to the nearest suitable airport and "
    "landed without further incident. No injuries were reported among the "
    "passengers or crew."
)
# Prognosis (7.2): same scenario but stops before crew actions / outcome so
# forward predictions are genuine forecasts, not echoes of the query ending.
QUERY_PROGNOSIS = (
    "During cruise flight at FL340, the No. 2 engine experienced a sudden loss "
    "of power accompanied by elevated EGT, vibration, and an associated master "
    "caution."
)
QUERY = QUERY_DIAGNOSIS  # backward-compatible alias

OUT_PATH = _HERE / "data" / "a0.json"
TRACE_PATH = _HERE / "data" / "trace_synthetic_a0.txt"


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _run_pipeline() -> None:
    if not main_app.DATA_LOADED:
        raise RuntimeError("main_app failed to load data.")

    print("=" * 80)
    print("END-TO-END TRACE: A0 — synthetic engine-failure query (production mode)")
    print("=" * 80)
    print(f"[precompute_a0] Mode: PRODUCTION (full corpus, {len(main_app.refined_dataset)} incidents)")
    print(f"[precompute_a0] Diagnosis query: {QUERY_DIAGNOSIS[:80]}...")
    print(f"[precompute_a0] Prognosis query: {QUERY_PROGNOSIS[:80]}...")

    # ----- Step 1: embed (diagnosis) -----
    print("[precompute_a0] Embedding diagnosis query...")
    emb = main_app.get_embedding(QUERY_DIAGNOSIS)

    # ----- Step 2: retrieve top-50 -----
    print("[precompute_a0] Retrieving top matches...")
    all_scores, all_matches = main_app.find_top_matches(emb)
    top_scores = list(all_scores[:50])
    top_matches = list(all_matches[:50])

    retrieval_rows = []
    for rank, (s, m) in enumerate(zip(top_scores, top_matches), start=1):
        text = (m.get("text") or "")[:160].replace("\n", " ")
        retrieval_rows.append({
            "rank": rank,
            "ev_id": m.get("ev_id", ""),
            "source": m.get("source", ""),
            "score": _safe_float(s),
            "snippet": text,
        })

    # ----- Step 3: cluster top-50 (production function) -----
    print("[precompute_a0] Clustering top 50...")
    clusters = main_app.cluster_incidents_by_type(top_scores, top_matches, 50)

    # ----- Step 4: P(Cause | Cluster) (production function) -----
    print("[precompute_a0] Computing P(Cause|Cluster)...")
    cluster_analysis = main_app.calculate_cause_probabilities_per_cluster(clusters)

    # ----- Step 5: Law of Total Probability (production function, fixed) -----
    # main_app.calculate_chain_rule_diagnosis was patched to restrict the LTP
    # partition to clusters with recorded causes; Σ P(C|Q) is now exactly 1.0.
    # Per-cluster P(K|Q) values are written back into `cluster_analysis[k]`
    # under the key 'p_cluster_query' (set to 0.0 for skipped clusters).
    print("[precompute_a0] Applying law of total probability...")
    diagnosis = main_app.calculate_chain_rule_diagnosis(clusters, cluster_analysis)
    weighted_causes = diagnosis.get("weighted_causes", []) or []

    s = sum(c["probability"] for c in weighted_causes)
    print(f"  Σ P(C|Q) = {s:.6f}  (expect 1.0)")
    if abs(s - 1.0) > 1e-3:
        print(f"  WARNING: LTP did not sum to 1; difference = {1.0 - s:+.6f}")

    skipped_clusters = [
        {
            "cluster": label,
            "n_incidents": ca.get("total_incidents", 0),
            "reason": "no recorded causes",
        }
        for label, ca in cluster_analysis.items()
        if not (ca.get("causes") or {})
    ]

    cluster_rows = []
    for label, ca in sorted(
        cluster_analysis.items(),
        key=lambda kv: kv[1].get("p_cluster_query", 0.0),
        reverse=True,
    ):
        if not (ca.get("causes") or {}):
            continue  # skipped clusters surfaced separately
        n = ca.get("total_incidents", 0)
        avg = _safe_float(ca.get("avg_similarity", 0.0))
        cluster_rows.append({
            "cluster": label,
            "n_incidents": n,
            "avg_similarity": avg,
            "weight": avg * n,
            "p_k_given_q": _safe_float(ca.get("p_cluster_query", 0.0)),
        })

    # All causes per active cluster (full P(C|K) tables).
    cluster_cause_rows = {}
    for label, ca in cluster_analysis.items():
        causes_dict = ca.get("causes") or {}
        if not causes_dict:
            continue
        ranked = sorted(causes_dict.items(), key=lambda kv: kv[1], reverse=True)
        cluster_cause_rows[label] = [
            {"rank": i + 1, "cause": str(c), "p_c_given_k": _safe_float(p)}
            for i, (c, p) in enumerate(ranked)
        ]

    cause_rows = []
    for rank, wc in enumerate(weighted_causes, start=1):
        cause_rows.append({
            "rank": rank,
            "probability": _safe_float(wc.get("probability", 0.0)),
            "cause": str(wc.get("cause", "")),
        })

    # ----- Step 6: map to Zhang's 54 codes -----
    print("[precompute_a0] Mapping to 54 codes...")
    codes_list, labels_list, code_embs = build_code_embeddings()
    dist = [
        {"cause": wc["cause"], "probability": wc["probability"]}
        for wc in weighted_causes
    ]
    coded_dist, _mappings = map_distribution_to_codes(
        dist, codes_list, labels_list, code_embs
    )
    code_rows = []
    for rank, cd in enumerate(coded_dist, start=1):
        code = cd.get("code")
        code_rows.append({
            "rank": rank,
            "code": "—" if code is None else str(code),
            "label": str(cd.get("label") or ""),
            "probability": _safe_float(cd.get("probability", 0.0)),
        })

    # ----- Step 7: prognosis (truncated query — forward forecast) -----
    print("[precompute_a0] Prognosis (truncated query)...")
    prog_emb = main_app.get_embedding(QUERY_PROGNOSIS)
    prog_all_scores, prog_all_matches = main_app.find_top_matches(prog_emb)
    prog_top_scores = list(prog_all_scores[:50])
    prog_top_matches = list(prog_all_matches[:50])
    prognosis_retrieval = []
    for rank, (s, m) in enumerate(zip(prog_top_scores, prog_top_matches), start=1):
        prognosis_retrieval.append({
            "rank": rank,
            "ev_id": m.get("ev_id", ""),
            "source": m.get("source", ""),
            "score": _safe_float(s),
            "snippet": (m.get("text") or "")[:160].replace("\n", " "),
        })

    prog = main_app.predict_future_events_lotp(
        QUERY_PROGNOSIS, top_n_incidents=50, max_chain_steps=3
    )
    future_rows = []
    for rank, fe in enumerate(prog.get("future_events") or [], start=1):
        future_rows.append({
            "rank": rank,
            "probability": _safe_float(fe.get("probability", 0.0)),
            "event": str(fe.get("event") or ""),
        })
    multi_step_rows = []
    for step in prog.get("multi_step") or []:
        dist = step.get("distribution") or []
        multi_step_rows.append({
            "step": step.get("step"),
            "conditioned_on": step.get("conditioned_on") or [],
            "total_weight": _safe_float(step.get("total_weight", 0.0)),
            "events": [
                {
                    "rank": i + 1,
                    "probability": _safe_float(fe.get("probability", 0.0)),
                    "event": str(fe.get("event") or ""),
                }
                for i, fe in enumerate(dist)
            ],
        })
    prognosis_aligned = []
    for i, row in enumerate(prog.get("aligned_incidents") or [], start=1):
        prognosis_aligned.append({
            "rank": i,
            "ev_id": row.get("ev_id", ""),
            "incident_similarity": _safe_float(row.get("incident_similarity", 0.0)),
            "defining_event": str(row.get("defining_event") or ""),
            "next_event": str(row.get("next_event") or ""),
            "has_downstream": bool(row.get("has_downstream")),
        })

    prog_cluster_rows = []
    prog_ca = prog.get("cluster_analysis") or {}
    for label, ca in sorted(
        prog_ca.items(),
        key=lambda kv: kv[1].get("p_cluster_query", 0.0),
        reverse=True,
    ):
        if not (ca.get("events") or {}):
            continue
        n = ca.get("total_incidents", 0)
        avg = _safe_float(ca.get("avg_similarity", 0.0))
        prog_cluster_rows.append({
            "cluster": label,
            "n_incidents": n,
            "avg_similarity": avg,
            "weight": avg * n,
            "p_k_given_q": _safe_float(ca.get("p_cluster_query", 0.0)),
            "sequences_with_next": ca.get("sequences_with_next", 0),
        })

    prog_cluster_next_events = {}
    for label, ca in prog_ca.items():
        events_dict = ca.get("events") or {}
        if not events_dict:
            continue
        ranked = sorted(events_dict.items(), key=lambda kv: kv[1], reverse=True)
        prog_cluster_next_events[label] = [
            {"rank": i + 1, "event": str(ev), "p_next_given_k": _safe_float(p)}
            for i, (ev, p) in enumerate(ranked)
        ]

    prog_skipped = [
        {"cluster": k, "reason": "no post-defining next event"}
        for k in (prog.get("skipped_clusters") or [])
    ]

    # ----- Save -----
    payload = {
        "mode": "production",
        "corpus_size": len(main_app.refined_dataset),
        "query": QUERY_DIAGNOSIS,
        "prognosis_query": QUERY_PROGNOSIS,
        "retrieval": retrieval_rows,
        "clusters": cluster_rows,
        "skipped_clusters": skipped_clusters,
        "cluster_causes": cluster_cause_rows,
        "ltp_causes": cause_rows,
        "ltp_sum": sum(r["probability"] for r in cause_rows),
        "coded_distribution": code_rows,
        "prognosis_retrieval": prognosis_retrieval,
        "prognosis_clusters": prog_cluster_rows,
        "prognosis_skipped_clusters": prog_skipped,
        "prognosis_cluster_next_events": prog_cluster_next_events,
        "prognosis_aligned": prognosis_aligned,
        "prognosis": future_rows,
        "prognosis_ltp_sum": _safe_float(prog.get("ltp_sum", 0.0)),
        "prognosis_multi_step": multi_step_rows,
        "prognosis_meta": {
            "max_chain_steps": 3,
            "methodology": prog.get("methodology", ""),
            "sequences_analyzed": prog.get("sequences_analyzed", 0),
            "aligned_incident_count": len(prog.get("aligned_incidents") or []),
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"[precompute_a0] Saved {OUT_PATH}")


def main() -> None:
    """Run the production pipeline and tee stdout into a permanent trace file."""
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRACE_PATH.open("w", encoding="utf-8") as trace_file:
        tee = _Tee(sys.stdout, trace_file)
        with redirect_stdout(tee):
            _run_pipeline()
    print(f"[precompute_a0] Trace saved to {TRACE_PATH}")


if __name__ == "__main__":
    main()

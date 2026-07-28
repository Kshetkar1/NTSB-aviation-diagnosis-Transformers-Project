"""
Precompute A2 worked example for the paper Section 8.5.1.

Runs the A2 (embedding + structural reranking, α=2.0) pipeline in HELD-OUT mode
(177 train index) on real test incident 20100114X11754. Calls only production
functions in `main_app` — no parallel math. Captures both A0 and A2 paths.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Held-out mode = 177 train index. Must be set BEFORE import main_app.
os.environ["NTSB_USE_TRAIN_INDEX"] = "1"

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
SCRIPTS_DIR = PROJECT_ROOT / "Testing_Structural_Mapping_Slides" / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

import main_app  # noqa: E402
from config import REFINED_DATA_PATH, LLM_MODEL  # noqa: E402
from cause_code_mapper import (  # noqa: E402
    build_code_embeddings,
    map_cause_to_code,
    map_distribution_to_codes,
)
from extract_struct_v2 import extract_struct_v2  # noqa: E402
from io_cache import load_query_struct_jsonl, load_struct_jsonl, sha256_text  # noqa: E402
from paths import DEFAULT_QUERY_CACHE_V2_PATH, DEFAULT_STRUCT_CACHE_V2_PATH  # noqa: E402
from struct_hooks import make_score_adjust_fn  # noqa: E402
from struct_score_v2 import structural_similarity as structural_similarity_v2  # noqa: E402

EV_ID = "20100114X11754"
OUT_PATH = _HERE / "data" / "a2.json"


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _load_query_and_truth(ev_id: str) -> tuple[str, str]:
    """Load query narrative and M1 ground truth (Cause_Factor == 'C' finding)."""
    refined = json.loads(Path(REFINED_DATA_PATH).read_text(encoding="utf-8"))
    rec = refined.get(ev_id) or {}
    query = str(rec.get("narr_accp") or rec.get("narr_accf") or "").strip()
    findings = rec.get("findings") or []
    cause_findings = [
        str(f.get("finding_description", "")).strip()
        for f in findings
        if isinstance(f, dict)
        and (f.get("Cause_Factor") or "").strip() == "C"
        and (f.get("finding_description") or "").strip()
    ]
    narr_cause = str(rec.get("narr_cause") or "").strip()
    truth = cause_findings[0] if cause_findings else narr_cause
    return query, truth


def _cluster_rows(cluster_analysis: dict) -> list[dict]:
    """Build cluster rows using production P(K|Q) from calculate_chain_rule_diagnosis."""
    rows = []
    for label, ca in sorted(
        cluster_analysis.items(),
        key=lambda kv: kv[1].get("p_cluster_query", 0.0),
        reverse=True,
    ):
        if not (ca.get("causes") or {}):
            continue
        n = ca.get("total_incidents", 0)
        avg = _safe_float(ca.get("avg_similarity", 0.0))
        rows.append({
            "cluster": label,
            "n_incidents": n,
            "avg_similarity": avg,
            "weight": avg * n,
            "p_k_given_q": _safe_float(ca.get("p_cluster_query", 0.0)),
        })
    return rows


def main() -> None:
    if not main_app.DATA_LOADED:
        raise RuntimeError("main_app failed to load data.")

    print(
        f"[precompute_a2] Mode: HELD-OUT "
        f"(train-only index, {len(main_app.refined_dataset)} incidents)"
    )
    query, truth = _load_query_and_truth(EV_ID)
    if not query:
        raise RuntimeError(f"Could not find query narrative for {EV_ID}")
    print(f"[precompute_a2] Query: ev_id={EV_ID}, len={len(query)}")
    print(f"[precompute_a2] Ground truth: {truth[:140]}")

    emb = main_app.get_embedding(query)
    all_scores, all_matches = main_app.find_top_matches(emb)
    top_scores = list(all_scores[:50])
    top_matches = list(all_matches[:50])

    retrieval_rows = []
    for rank, (s, m) in enumerate(zip(top_scores, top_matches), start=1):
        retrieval_rows.append({
            "rank": rank,
            "ev_id": m.get("ev_id", ""),
            "source": m.get("source", ""),
            "score": _safe_float(s),
            "snippet": (m.get("text") or "")[:160].replace("\n", " "),
        })

    # ----- A0 path (production) -----
    print("[precompute_a2] A0 path...")
    a0_clusters = main_app.cluster_incidents_by_type(top_scores, top_matches, 50)
    a0_ca = main_app.calculate_cause_probabilities_per_cluster(a0_clusters)
    a0_diag = main_app.calculate_chain_rule_diagnosis(a0_clusters, a0_ca)
    a0_causes = a0_diag.get("weighted_causes", []) or []
    a0_sum = sum(c["probability"] for c in a0_causes)
    print(f"  A0 Σ P(C|Q) = {a0_sum:.6f}")

    # ----- A2 path (production, α=2.0 per paper Section 8.5.1) -----
    print("[precompute_a2] A2 path (α=2.0)...")
    struct_by_eid = load_struct_jsonl(DEFAULT_STRUCT_CACHE_V2_PATH)
    query_cache = load_query_struct_jsonl(DEFAULT_QUERY_CACHE_V2_PATH)
    q_hash = sha256_text(query)
    if q_hash in query_cache:
        q_struct = query_cache[q_hash]
        print("[precompute_a2] Using cached query structural representation (v2).")
    else:
        q_struct = extract_struct_v2(query, ev_id=EV_ID, model=LLM_MODEL)
        print("[precompute_a2] WARNING: query struct not in cache; LLM output may vary between runs.")
    score_adjust_fn = make_score_adjust_fn(
        q_struct, struct_by_eid, alpha=2.0, similarity_fn=structural_similarity_v2,
    )
    a2_scores = [_safe_float(score_adjust_fn(s, m)) for s, m in zip(top_scores, top_matches)]
    a2_clusters = main_app.cluster_incidents_by_type(a2_scores, top_matches, 50)
    a2_ca = main_app.calculate_cause_probabilities_per_cluster(a2_clusters)
    a2_diag = main_app.calculate_chain_rule_diagnosis(a2_clusters, a2_ca)
    a2_causes = a2_diag.get("weighted_causes", []) or []
    a2_sum = sum(c["probability"] for c in a2_causes)
    print(f"  A2 Σ P(C|Q) = {a2_sum:.6f}")

    codes_list, labels_list, code_embs = build_code_embeddings()
    truth_code, truth_label, truth_sim = map_cause_to_code(
        truth, codes_list, labels_list, code_embs
    )

    def _to_dist(causes):
        return [{"cause": c["cause"], "probability": c["probability"]} for c in causes]

    a0_coded, _ = map_distribution_to_codes(_to_dist(a0_causes), codes_list, labels_list, code_embs)
    a2_coded, _ = map_distribution_to_codes(_to_dist(a2_causes), codes_list, labels_list, code_embs)

    def _cluster_cause_rows(cluster_analysis: dict) -> dict[str, list[dict]]:
        out = {}
        for label, ca in cluster_analysis.items():
            causes_dict = ca.get("causes") or {}
            if not causes_dict:
                continue
            ranked = sorted(causes_dict.items(), key=lambda kv: kv[1], reverse=True)
            out[label] = [
                {"rank": i + 1, "cause": str(c), "p_c_given_k": _safe_float(p)}
                for i, (c, p) in enumerate(ranked)
            ]
        return out

    def _code_rows(coded):
        return [
            {
                "rank": rank,
                "code": "—" if cd.get("code") is None else str(cd.get("code")),
                "label": str(cd.get("label") or ""),
                "probability": _safe_float(cd.get("probability", 0.0)),
            }
            for rank, cd in enumerate(coded, start=1)
        ]

    def _cause_rows(causes):
        return [
            {
                "rank": rank,
                "probability": _safe_float(wc.get("probability", 0.0)),
                "cause": str(wc.get("cause", "")),
            }
            for rank, wc in enumerate(causes, start=1)
        ]

    a0_top1 = a0_coded[0].get("code") if a0_coded else None
    a2_top1 = a2_coded[0].get("code") if a2_coded else None

    payload = {
        "mode": "held_out",
        "corpus_size": len(main_app.refined_dataset),
        "ev_id": EV_ID,
        "query": query,
        "ground_truth": {
            "text": truth,
            "code": "—" if truth_code is None else str(truth_code),
            "label": truth_label,
            "similarity_to_label": _safe_float(truth_sim),
        },
        "retrieval": retrieval_rows,
        "a0": {
            "clusters": _cluster_rows(a0_ca),
            "cluster_causes": _cluster_cause_rows(a0_ca),
            "ltp_causes": _cause_rows(a0_causes),
            "ltp_sum": a0_sum,
            "coded_distribution": _code_rows(a0_coded),
            "top1_code": "—" if a0_top1 is None else str(a0_top1),
        },
        "a2": {
            "clusters": _cluster_rows(a2_ca),
            "cluster_causes": _cluster_cause_rows(a2_ca),
            "ltp_causes": _cause_rows(a2_causes),
            "ltp_sum": a2_sum,
            "coded_distribution": _code_rows(a2_coded),
            "top1_code": "—" if a2_top1 is None else str(a2_top1),
        },
        "hit": {
            "a0": (str(a0_top1) == str(truth_code) and truth_code is not None),
            "a2": (str(a2_top1) == str(truth_code) and truth_code is not None),
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"[precompute_a2] Saved {OUT_PATH}")
    print(
        f"  truth={truth_code}, A0 top1={a0_top1}, A2 top1={a2_top1}, "
        f"A0 hit={payload['hit']['a0']}, A2 hit={payload['hit']['a2']}"
    )


if __name__ == "__main__":
    main()

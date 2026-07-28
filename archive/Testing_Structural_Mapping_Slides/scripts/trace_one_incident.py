#!/usr/bin/env python3
"""
Trace one test incident end-to-end through the diagnosis pipeline.

Jesse: "I would trace this through carefully."

Prints every step so you can walk your advisors through:
  1. Query text
  2. Embedding → cosine similarities
  3. Top 50 matches
  4. Cluster assignments
  5. P(Cluster|Query)
  6. P(Cause|Cluster) within each cluster
  7. Chain rule: P(Cause|Query) = sum P(Cause|Cluster) * P(Cluster|Query)
  8. Final distribution (A0)
  9. Code mapping → Zhang's 54 codes
  10. Probability sum verification
  11. A2 structural reranking — how scores change with structural similarity
  12. Prognosis — predicted downstream events

Usage:
    python trace_one_incident.py                   # traces first test incident
    python trace_one_incident.py --ev_id 20100101X # traces specific incident
"""
from __future__ import annotations

import os
os.environ["NTSB_USE_TRAIN_INDEX"] = "1"

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPTS))

import main_app  # noqa: E402
from config import REFINED_DATA_PATH, LLM_MODEL, get_openai_api_key  # noqa: E402
from openai import OpenAI  # noqa: E402
from paths import EVAL_OUTPUT_DIR, TEST_IDS_PATH  # noqa: E402
from cause_code_mapper import (  # noqa: E402
    ZHANG_OCCURRENCE_CODES,
    build_code_embeddings,
    map_cause_to_code,
    map_distribution_to_codes,
)

from extract_struct_v2 import extract_struct_v2  # noqa: E402
from io_cache import load_struct_jsonl  # noqa: E402
from paths import DEFAULT_STRUCT_CACHE_V2_PATH  # noqa: E402
from struct_hooks import make_score_adjust_fn  # noqa: E402
from struct_score_v2 import structural_similarity as structural_similarity_v2  # noqa: E402

import numpy as np  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ev_id", default=None, help="Specific ev_id to trace")
    args = parser.parse_args()

    if not main_app.DATA_LOADED:
        raise RuntimeError("Train index not loaded.")

    with open(REFINED_DATA_PATH, "r", encoding="utf-8") as f:
        full_dataset = json.load(f)

    with open(TEST_IDS_PATH, "r") as f:
        test_ids = [line.strip() for line in f if line.strip()]

    ev_id = args.ev_id or test_ids[0]
    inc = full_dataset.get(ev_id)
    if not inc:
        print(f"Incident {ev_id} not found in dataset.")
        return

    output_dir = EVAL_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace_incident_detail.txt"

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 80)
    log(f"END-TO-END TRACE: {ev_id}")
    log("=" * 80)

    # ---- Step 1: Query text ----
    query = str(inc.get("narr_accp") or inc.get("narr_accf") or "").strip()
    log(f"\n{'='*60}")
    log("STEP 1: Query Text")
    log(f"{'='*60}")
    log(f"Source: {'narr_accp' if inc.get('narr_accp') else 'narr_accf'}")
    log(f"Text: {query[:500]}")

    # Ground truth
    findings = inc.get("findings") or []
    cause_findings = [
        str(f.get("finding_description", "")).strip()
        for f in findings
        if isinstance(f, dict)
        and (f.get("Cause_Factor") or "").strip() == "C"
        and (f.get("finding_description") or "").strip()
    ]
    narr_cause = str(inc.get("narr_cause") or "").strip()
    ground_truth = cause_findings[0] if cause_findings else narr_cause
    log(f"\nGround truth cause: {ground_truth[:300]}")

    # ---- Step 2: Embedding ----
    log(f"\n{'='*60}")
    log("STEP 2: Query Embedding")
    log(f"{'='*60}")
    query_embedding = main_app.get_embedding(query)
    emb_arr = np.array(query_embedding)
    log(f"Embedding dimension: {len(query_embedding)}")
    log(f"Embedding norm: {np.linalg.norm(emb_arr):.6f}")
    log(f"First 10 values: {emb_arr[:10].tolist()}")

    # ---- Step 3: Top matches ----
    log(f"\n{'='*60}")
    log("STEP 3: Top 50 Matches (cosine similarity)")
    log(f"{'='*60}")
    top_scores, top_matches = main_app.find_top_matches(query_embedding)

    log(f"Total matches retrieved: {len(top_scores)}")
    log(f"\n{'Rank':<6} {'Score':<10} {'ev_id':<20} {'Description':<60}")
    log("-" * 96)
    for rank, (score, match) in enumerate(zip(top_scores[:50], top_matches[:50]), 1):
        desc = str(match.get("text", ""))[:55]
        eid = str(match.get("ev_id", ""))
        log(f"{rank:<6} {score:<10.6f} {eid:<20} {desc}")

    # ---- Step 4: Cluster assignments ----
    log(f"\n{'='*60}")
    log("STEP 4: Cluster Assignments")
    log(f"{'='*60}")
    clusters = main_app.cluster_incidents_by_type(top_scores, top_matches, 50)

    log(f"Number of clusters: {len(clusters)}")
    for ci, (label, members) in enumerate(clusters.items()):
        total_weight = sum(m.get("score", 0) for m in members)
        avg_sim = total_weight / len(members) if members else 0.0
        log(f"\n  Cluster {ci}: {label}")
        log(f"    Members: {len(members)}, Total weight: {total_weight:.6f}, Avg sim: {avg_sim:.6f}")
        for m in members[:3]:
            log(f"      ev_id={m.get('ev_id','?')}, score={m.get('score',0):.6f}")
        if len(members) > 3:
            log(f"      ... and {len(members)-3} more")

    # ---- Step 5: P(Cluster|Query) ----
    log(f"\n{'='*60}")
    log("STEP 5: P(Cluster|Query) = avg_sim_j * n_j / sum_k (avg_sim_k * n_k)")
    log(f"{'='*60}")
    cluster_stats = []
    for label, members in clusters.items():
        n = len(members)
        avg_sim = (sum(m.get("score", 0) for m in members) / n) if n > 0 else 0.0
        w = avg_sim * n
        cluster_stats.append((label, n, avg_sim, w))
    total_w = sum(w for _, _, _, w in cluster_stats)
    log(f"Total weight across all clusters: {total_w:.6f}")
    log(f"\n{'Cluster':<45} {'N':<5} {'AvgSim':<10} {'Weight':<10} {'P(Cluster|Q)':<14}")
    log("-" * 84)
    for label, n, avg_sim, w in cluster_stats:
        p = w / total_w if total_w > 0 else 0
        log(f"{label[:42]:<45} {n:<5d} {avg_sim:<10.6f} {w:<10.6f} {p:<14.6f}")

    # ---- Step 6: P(Cause|Cluster) ----
    log(f"\n{'='*60}")
    log("STEP 6: P(Cause|Cluster) — top causes per cluster")
    log(f"{'='*60}")
    cluster_analysis = main_app.calculate_cause_probabilities_per_cluster(clusters)

    for ci, (label, ca) in enumerate(cluster_analysis.items()):
        causes_dict = ca.get("causes", {})
        sorted_causes = sorted(causes_dict.items(), key=lambda x: x[1], reverse=True)
        log(f"\n  Cluster {ci}: {label}  (n={ca.get('total_incidents',0)}, avg_sim={ca.get('avg_similarity',0):.4f})")
        for cname, cprob in sorted_causes[:5]:
            log(f"    P({cname[:80]}) = {cprob:.6f}")

    # ---- Step 7: Chain rule ----
    log(f"\n{'='*60}")
    log("STEP 7: Chain Rule — P(Cause|Query) = sum P(Cause|Cluster) * P(Cluster|Query)")
    log(f"{'='*60}")
    diagnosis = main_app.calculate_chain_rule_diagnosis(clusters, cluster_analysis)
    weighted_causes = diagnosis.get("weighted_causes", [])

    log(f"Total unique causes in final distribution: {len(weighted_causes)}")
    log(f"\n{'Rank':<6} {'Probability':<14} {'Cause':<80}")
    log("-" * 100)
    for rank, wc in enumerate(weighted_causes[:20], 1):
        cause = str(wc.get("cause", ""))[:75]
        prob = wc.get("probability", 0)
        log(f"{rank:<6} {prob:<14.6f} {cause}")

    total_prob = sum(wc.get("probability", 0) for wc in weighted_causes)
    log(f"\nSum of all probabilities: {total_prob:.6f}")
    log(f"  (Should be ~1.0. Deviation: {abs(1.0 - total_prob):.6f})")

    # ---- Step 8: Code mapping ----
    log(f"\n{'='*60}")
    log("STEP 8: Map to Zhang's 54 Occurrence Codes")
    log(f"{'='*60}")

    codes_list, labels_list, code_embs = build_code_embeddings()
    dist_for_mapping = [{"cause": wc["cause"], "probability": wc["probability"]} for wc in weighted_causes]
    coded_dist, mappings = map_distribution_to_codes(dist_for_mapping, codes_list, labels_list, code_embs)

    def _fmt_code(cd):
        c = cd.get("code")
        l = cd.get("label") or ""
        p = cd.get("probability") or 0.0
        c_str = "?" if c is None else str(c)
        return c_str, str(l)[:37], float(p)

    log(f"\nTop 15 coded probabilities:")
    log(f"{'Code':<8} {'Label':<40} {'Probability':<14}")
    log("-" * 62)
    for cd in coded_dist[:15]:
        c_str, l_str, p_val = _fmt_code(cd)
        log(f"{c_str:<8} {l_str:<40} {p_val:<14.6f}")

    coded_total = sum((cd.get("probability") or 0.0) for cd in coded_dist)
    log(f"\nSum of coded probabilities: {coded_total:.6f}")

    # Ground truth code
    if ground_truth:
        gt_code, gt_label, gt_sim = map_cause_to_code(ground_truth, codes_list, labels_list, code_embs)
        log(f"\nGround truth maps to: code {gt_code} = '{gt_label}' (similarity: {gt_sim:.4f})")

        if coded_dist:
            top1_code = coded_dist[0].get("code")
            top1_label = coded_dist[0].get("label") or ""
            log(f"Top-1 prediction:     code {top1_code} = '{top1_label}'")
            log(f"Match: {'YES' if top1_code == gt_code else 'NO'}")

    # ---- Step 9: Sample mappings ----
    log(f"\n{'='*60}")
    log("STEP 9: Sample Cause → Code Mappings (first 10)")
    log(f"{'='*60}")
    log(f"\n{'Original Cause':<55} {'→ Code':<8} {'Label':<30} {'Sim':<8}")
    log("-" * 101)
    for m in mappings[:10]:
        orig = str(m.get("original_cause") or "")[:52]
        code = m.get("mapped_code")
        code_str = "?" if code is None else str(code)
        label = str(m.get("mapped_label") or "")[:27]
        sim = float(m.get("similarity") or 0.0)
        log(f"{orig:<55} {code_str:<8} {label:<30} {sim:<8.4f}")

    # ---- Step 10: A2 Structural Reranking ----
    log(f"\n{'='*60}")
    log("STEP 10: A2 — Structural Mapping Reranking")
    log(f"{'='*60}")
    log("A2 fuses embedding similarity with structural similarity:")
    log("  new_score = max(0, cosine) × exp(α × struct_sim),  α=2.0 (matches paper Section 6.7)")

    try:
        struct_by_eid = load_struct_jsonl(DEFAULT_STRUCT_CACHE_V2_PATH)
        log(f"  Loaded {len(struct_by_eid)} cached structural representations.")

        q_struct = extract_struct_v2(query, ev_id="QUERY", model=LLM_MODEL)
        log(f"  Query structural representation extracted.")
        log(f"    Components: {list(q_struct.keys()) if isinstance(q_struct, dict) else 'N/A'}")

        score_adjust_fn = make_score_adjust_fn(
            q_struct, struct_by_eid, alpha=2.0,
            similarity_fn=structural_similarity_v2,
        )

        # A2 intermediate: reweight per-incident scores, recluster, recompute P(K|Q)
        a2_scores = [float(score_adjust_fn(s, m)) for s, m in zip(top_scores, top_matches)]
        a2_clusters = main_app.cluster_incidents_by_type(a2_scores, top_matches, 50)
        a2_cluster_analysis_pre = main_app.calculate_cause_probabilities_per_cluster(a2_clusters)

        log(f"\nA2 cluster weights P(K|Q) (compare to A0 Step 5):")
        log(f"{'Cluster':<45} {'N':<5} {'A2 AvgSim':<12} {'A2 Weight':<12} {'A2 P(K|Q)':<14}")
        log("-" * 90)
        a2_total_w = sum(a['avg_similarity'] * a['total_incidents'] for a in a2_cluster_analysis_pre.values())
        a2_p_k = {}
        for label, a in sorted(a2_cluster_analysis_pre.items(), key=lambda kv: kv[1]['avg_similarity']*kv[1]['total_incidents'], reverse=True):
            avg_sim = a['avg_similarity']
            n = a['total_incidents']
            w = avg_sim * n
            p = w / a2_total_w if a2_total_w > 0 else 0
            a2_p_k[label] = p
            log(f"{label[:42]:<45} {n:<5d} {avg_sim:<12.6f} {w:<12.6f} {p:<14.6f}")
        log(f"Total A2 weight: {a2_total_w:.6f}")

        # Re-run diagnosis with A2
        a2_result = main_app.diagnose_with_conditional_probabilities(
            query, top_n=10, top_n_incidents=50, score_adjust_fn=score_adjust_fn,
        )
        a2_causes = a2_result.get("weighted_causes", [])

        log(f"\nA2 top 20 causes (compare to A0 Step 7):")
        log(f"{'Rank':<6} {'Probability':<14} {'Cause':<80}")
        log("-" * 100)
        for rank, wc in enumerate(a2_causes[:20], 1):
            cause = str(wc.get("cause", ""))[:75]
            prob = wc.get("probability", 0)
            log(f"{rank:<6} {prob:<14.6f} {cause}")

        a2_total = sum(wc.get("probability", 0) for wc in a2_causes)
        log(f"\nA2 sum of probabilities: {a2_total:.6f}")

        # Map A2 to codes
        a2_dist_for_mapping = [{"cause": wc["cause"], "probability": wc["probability"]} for wc in a2_causes]
        a2_coded, _ = map_distribution_to_codes(a2_dist_for_mapping, codes_list, labels_list, code_embs)

        log(f"\nA2 top 10 coded probabilities:")
        log(f"{'Code':<8} {'Label':<40} {'Probability':<14}")
        log("-" * 62)
        for cd in a2_coded[:10]:
            c = cd.get("code")
            c_str = "?" if c is None else str(c)
            l_str = (str(cd.get("label") or ""))[:37]
            p_val = float(cd.get("probability") or 0.0)
            log(f"{c_str:<8} {l_str:<40} {p_val:<14.6f}")

        a2_top1 = a2_coded[0].get("code") if a2_coded else None
        a0_top1 = coded_dist[0].get("code") if coded_dist else None
        if coded_dist:
            log(f"\nA0 top-1: code {a0_top1} = '{coded_dist[0].get('label') or ''}'")
        if a2_coded:
            log(f"A2 top-1: code {a2_top1} = '{a2_coded[0].get('label') or ''}'")
        log(f"A2 changed top-1: {'YES' if a0_top1 != a2_top1 else 'NO'}")

    except Exception as e:
        log(f"  A2 trace skipped: {e}")

    # ---- Step 11: Prognosis ----
    log(f"\n{'='*60}")
    log("STEP 11: Prognosis — Predicted Downstream Events")
    log(f"{'='*60}")
    try:
        prog_result = main_app.predict_future_events(
            query, top_n_incidents=50, max_chain_steps=1,
        )
        future_events = prog_result.get("future_events", [])

        log(f"Total predicted events: {len(future_events)}")
        log(f"\n{'Rank':<6} {'Probability':<14} {'Event':<80}")
        log("-" * 100)
        for rank, fe in enumerate(future_events[:15], 1):
            event = str(fe.get("event") or "")[:75]
            prob = float(fe.get("probability") or 0.0)
            log(f"{rank:<6} {prob:<14.6f} {event}")

        prog_total = sum(float(fe.get("probability") or 0.0) for fe in future_events)
        log(f"\nSum of prognosis probabilities: {prog_total:.6f}")
    except Exception as e:
        log(f"  Prognosis skipped: {e}")

    log(f"\n{'='*80}")
    log("TRACE COMPLETE")
    log(f"{'='*80}")

    # Save to file
    with open(trace_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nTrace saved to: {trace_path}")


if __name__ == "__main__":
    main()

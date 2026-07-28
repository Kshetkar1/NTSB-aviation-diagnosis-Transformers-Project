"""
Precompute both worked examples for the paper, including BOTH diagnosis AND
prognosis with BOTH A0 (embedding-only) and A2 (structural reranking, alpha=2.0).

Generates two output files:

  data/example_after.json   - "After-incident" scenario:
        diagnosis_query  = full incident narrative (investigator view)
        prognosis_query  = truncated narrative (first observation sentence only)

  data/example_during.json  - "During-incident" scenario (Maha's pilot use case):
        diagnosis_query = prognosis_query =
            "I just got a left-engine fire warning on takeoff. I can see flames.
             What could be causing this?"

Both scenarios use the same real test incident (20100114X11754, ATR72 engine
fire, St. Croix, 2010) and the same held-out 177-incident training index.
Calls only production functions in `main_app` and the structural-mapping
helpers under `Testing_Structural_Mapping_Slides/scripts`.
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

ALPHA = 2.0
OUT_DIR = _HERE / "data"

# Add new test cases by appending to this list. The first entry uses the
# original filenames (example_after.json / example_during.json) for backward
# compatibility; subsequent entries get a slug-suffixed filename so we can run
# many side-by-side incidents through the same pipeline.
INCIDENTS = [
    {
        "ev_id": "20100114X11754",
        "slug": "",
        "label": "ATR72 left-engine fire on takeoff (St. Croix, 2010)",
        "during_query": (
            "I just got a left-engine fire warning on takeoff. "
            "I can see flames. What could be causing this?"
        ),
    },
    {
        "ev_id": "20081116X33137",
        "slug": "landing_gear",
        "label": "DHC-8 nose-gear retracted landing (Philadelphia, 2008)",
        "during_query": (
            "I just got an unsafe nose-gear indication on approach. "
            "Three greens on the mains but nothing on the nose. "
            "What could be wrong?"
        ),
    },
]


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _load_full_narrative_and_truth(ev_id: str) -> tuple[str, str]:
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


def _truncate_to_first_observation(narrative: str) -> str:
    """
    Cut narrative at the first sentence that finishes an observation but
    BEFORE any pilot action. For the engine-fire incident this stops after
    'experienced a No. 1 (left) engine fire during takeoff ... United States
    Virgin Islands.' which is exactly the observation moment we want for
    prognosis.
    """
    for i, ch in enumerate(narrative):
        if ch == "." and (i + 1 >= len(narrative) or narrative[i + 1] in (" ", "\n", "\t", "\r")):
            return narrative[: i + 1].strip()
    return narrative.strip()


def _get_score_adjust_fn(
    query: str,
    ev_id: str | None,
    struct_by_eid: dict,
    query_cache: dict,
):
    """Resolve a structural-reranking function for the given query. Uses the
    cache when possible; falls back to a deterministic (temperature=0) LLM
    extraction otherwise."""
    q_hash = sha256_text(query)
    if q_hash in query_cache:
        q_struct = query_cache[q_hash]
        print(f"    using cached query struct (hash={q_hash[:10]}...)")
    else:
        print(f"    cache miss - extracting query struct via LLM (temp=0)")
        q_struct = extract_struct_v2(query, ev_id=ev_id, model=LLM_MODEL)
    return make_score_adjust_fn(
        q_struct, struct_by_eid, alpha=ALPHA, similarity_fn=structural_similarity_v2,
    )


def _retrieval_rows(top_scores, top_matches) -> list[dict]:
    return [
        {
            "rank": rank,
            "ev_id": m.get("ev_id", ""),
            "source": m.get("source", ""),
            "score": _safe_float(s),
            "snippet": (m.get("text") or "")[:200].replace("\n", " "),
        }
        for rank, (s, m) in enumerate(zip(top_scores, top_matches), start=1)
    ]


def _cluster_rows(cluster_analysis: dict, kind: str = "diagnosis") -> list[dict]:
    """kind in {'diagnosis', 'prognosis'}: causes vs events presence check."""
    key = "causes" if kind == "diagnosis" else "events"
    rows = []
    for label, ca in sorted(
        cluster_analysis.items(),
        key=lambda kv: kv[1].get("p_cluster_query", 0.0),
        reverse=True,
    ):
        if not (ca.get(key) or {}):
            continue
        n = ca.get("total_incidents", 0)
        avg = _safe_float(ca.get("avg_similarity", 0.0))
        row = {
            "cluster": label,
            "n_incidents": n,
            "avg_similarity": avg,
            "weight": avg * n,
            "p_k_given_q": _safe_float(ca.get("p_cluster_query", 0.0)),
        }
        if kind == "prognosis":
            row["sequences_with_next"] = ca.get("sequences_with_next", 0)
        rows.append(row)
    return rows


def _cluster_member_rows(cluster_analysis: dict, kind: str = "diagnosis") -> dict[str, list[dict]]:
    """Per-cluster P(C|K) or P(NextEvent|K) tables."""
    key = "causes" if kind == "diagnosis" else "events"
    out: dict[str, list[dict]] = {}
    for label, ca in cluster_analysis.items():
        d = ca.get(key) or {}
        if not d:
            continue
        ranked = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
        if kind == "diagnosis":
            out[label] = [
                {"rank": i + 1, "cause": str(k), "p_c_given_k": _safe_float(v)}
                for i, (k, v) in enumerate(ranked)
            ]
        else:
            out[label] = [
                {"rank": i + 1, "event": str(k), "p_next_given_k": _safe_float(v)}
                for i, (k, v) in enumerate(ranked)
            ]
    return out


def _ltp_rows(weighted_items: list[dict], item_key: str) -> list[dict]:
    return [
        {
            "rank": rank,
            "probability": _safe_float(w.get("probability", 0.0)),
            item_key: str(w.get(item_key, "")),
        }
        for rank, w in enumerate(weighted_items, start=1)
    ]


def _coded_rows(coded_dist: list[dict]) -> list[dict]:
    return [
        {
            "rank": rank,
            "code": "-" if cd.get("code") is None else str(cd.get("code")),
            "label": str(cd.get("label") or ""),
            "probability": _safe_float(cd.get("probability", 0.0)),
        }
        for rank, cd in enumerate(coded_dist, start=1)
    ]


def _diagnosis_section(top_scores, top_matches, score_adjust_fn, codes_list, labels_list, code_embs):
    """Run diagnosis pipeline on the given top-50. If score_adjust_fn is not
    None, returns the A2 view; otherwise the A0 view."""
    if score_adjust_fn is not None:
        eff_scores = [_safe_float(score_adjust_fn(s, m)) for s, m in zip(top_scores, top_matches)]
    else:
        eff_scores = list(top_scores)
    clusters = main_app.cluster_incidents_by_type(eff_scores, top_matches, 50)
    ca = main_app.calculate_cause_probabilities_per_cluster(clusters)
    diag = main_app.calculate_chain_rule_diagnosis(clusters, ca)
    causes = diag.get("weighted_causes", []) or []
    ltp_sum = sum(c["probability"] for c in causes)

    dist = [{"cause": c["cause"], "probability": c["probability"]} for c in causes]
    coded, _ = map_distribution_to_codes(dist, codes_list, labels_list, code_embs)
    top1_code = coded[0].get("code") if coded else None

    return {
        "clusters": _cluster_rows(ca, kind="diagnosis"),
        "cluster_causes": _cluster_member_rows(ca, kind="diagnosis"),
        "ltp_causes": _ltp_rows(causes, "cause"),
        "ltp_sum": ltp_sum,
        "coded_distribution": _coded_rows(coded),
        "top1_code": "-" if top1_code is None else str(top1_code),
    }


def _prognosis_section(query: str, score_adjust_fn) -> dict:
    """Run prognosis pipeline via predict_future_events_lotp. Returns
    cluster + LTP + multi-step + aligned-incident rows."""
    prog = main_app.predict_future_events_lotp(
        query, top_n_incidents=50, max_chain_steps=3, score_adjust_fn=score_adjust_fn,
    )
    future_rows = _ltp_rows(prog.get("future_events") or [], "event")

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

    aligned_rows = []
    for i, row in enumerate(prog.get("aligned_incidents") or [], start=1):
        aligned_rows.append({
            "rank": i,
            "ev_id": row.get("ev_id", ""),
            "incident_similarity": _safe_float(row.get("incident_similarity", 0.0)),
            "defining_event": str(row.get("defining_event") or ""),
            "next_event": str(row.get("next_event") or ""),
            "has_downstream": bool(row.get("has_downstream")),
        })

    ca = prog.get("cluster_analysis") or {}
    skipped = [
        {"cluster": k, "reason": "no post-defining next event"}
        for k in (prog.get("skipped_clusters") or [])
    ]

    return {
        "clusters": _cluster_rows(ca, kind="prognosis"),
        "skipped_clusters": skipped,
        "cluster_next_events": _cluster_member_rows(ca, kind="prognosis"),
        "ltp_events": future_rows,
        "ltp_sum": _safe_float(prog.get("ltp_sum", 0.0)),
        "multi_step": multi_step_rows,
        "aligned_incidents": aligned_rows,
        "sequences_analyzed": prog.get("sequences_analyzed", 0),
        "methodology": prog.get("methodology", ""),
    }


def run_example(
    scenario: str,
    diagnosis_query: str,
    prognosis_query: str,
    ev_id: str,
    truth: str,
    out_path: Path,
    struct_by_eid: dict,
    query_cache: dict,
    codes_list,
    labels_list,
    code_embs,
):
    """Run a complete worked example (diagnosis + prognosis, A0 + A2) and save."""
    print(f"\n{'=' * 80}")
    print(f"SCENARIO: {scenario}")
    print(f"{'=' * 80}")
    print(f"  ev_id            = {ev_id}")
    print(f"  diagnosis_query  = {diagnosis_query[:100]}{'...' if len(diagnosis_query) > 100 else ''}")
    print(f"  prognosis_query  = {prognosis_query[:100]}{'...' if len(prognosis_query) > 100 else ''}")
    print(f"  ground_truth     = {truth[:140]}")

    # --- Diagnosis ---
    print(f"\n[{scenario}] Diagnosis retrieval (top-50 cosine)")
    diag_emb = main_app.get_embedding(diagnosis_query)
    diag_scores_all, diag_matches_all = main_app.find_top_matches(diag_emb)
    diag_scores = list(diag_scores_all[:50])
    diag_matches = list(diag_matches_all[:50])
    diag_retrieval = _retrieval_rows(diag_scores, diag_matches)

    print(f"[{scenario}] Diagnosis - resolving structural reranking function")
    diag_score_adjust = _get_score_adjust_fn(diagnosis_query, ev_id, struct_by_eid, query_cache)

    print(f"[{scenario}] Diagnosis A0")
    a0_diag = _diagnosis_section(diag_scores, diag_matches, None,
                                  codes_list, labels_list, code_embs)
    print(f"  A0 Sigma P(C|Q) = {a0_diag['ltp_sum']:.6f}")
    print(f"  A0 top-1 code   = {a0_diag['top1_code']}")

    print(f"[{scenario}] Diagnosis A2 (alpha={ALPHA})")
    a2_diag = _diagnosis_section(diag_scores, diag_matches, diag_score_adjust,
                                  codes_list, labels_list, code_embs)
    print(f"  A2 Sigma P(C|Q) = {a2_diag['ltp_sum']:.6f}")
    print(f"  A2 top-1 code   = {a2_diag['top1_code']}")

    # --- Prognosis ---
    if prognosis_query == diagnosis_query:
        prog_score_adjust = diag_score_adjust
        prog_retrieval = diag_retrieval
        prog_scores = diag_scores
        prog_matches = diag_matches
    else:
        print(f"\n[{scenario}] Prognosis retrieval (top-50 cosine on truncated query)")
        prog_emb = main_app.get_embedding(prognosis_query)
        prog_scores_all, prog_matches_all = main_app.find_top_matches(prog_emb)
        prog_scores = list(prog_scores_all[:50])
        prog_matches = list(prog_matches_all[:50])
        prog_retrieval = _retrieval_rows(prog_scores, prog_matches)
        print(f"[{scenario}] Prognosis - resolving structural function for prognosis query")
        prog_score_adjust = _get_score_adjust_fn(prognosis_query, ev_id, struct_by_eid, query_cache)

    print(f"[{scenario}] Prognosis A0")
    a0_prog = _prognosis_section(prognosis_query, None)
    print(f"  A0 Sigma P(Next|Q) = {a0_prog['ltp_sum']:.6f}")

    print(f"[{scenario}] Prognosis A2 (alpha={ALPHA})")
    a2_prog = _prognosis_section(prognosis_query, prog_score_adjust)
    print(f"  A2 Sigma P(Next|Q) = {a2_prog['ltp_sum']:.6f}")

    # --- Ground truth code ---
    truth_code, truth_label, truth_sim = map_cause_to_code(
        truth, codes_list, labels_list, code_embs
    )
    truth_section = {
        "text": truth,
        "code": "-" if truth_code is None else str(truth_code),
        "label": truth_label,
        "similarity_to_label": _safe_float(truth_sim),
    }

    hit = {
        "a0_diagnosis": (str(a0_diag["top1_code"]) == str(truth_code) and truth_code is not None),
        "a2_diagnosis": (str(a2_diag["top1_code"]) == str(truth_code) and truth_code is not None),
    }

    payload = {
        "scenario": scenario,
        "mode": "held_out",
        "corpus_size": len(main_app.refined_dataset),
        "ev_id": ev_id,
        "alpha": ALPHA,
        "diagnosis_query": diagnosis_query,
        "prognosis_query": prognosis_query,
        "ground_truth": truth_section,
        "diagnosis": {
            "retrieval": diag_retrieval,
            "a0": a0_diag,
            "a2": a2_diag,
        },
        "prognosis": {
            "retrieval": prog_retrieval,
            "a0": a0_prog,
            "a2": a2_prog,
        },
        "hit": hit,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[{scenario}] Saved {out_path}")
    print(
        f"  truth_code={truth_code}  A0_top1={a0_diag['top1_code']}  A2_top1={a2_diag['top1_code']}  "
        f"A0_hit={hit['a0_diagnosis']}  A2_hit={hit['a2_diagnosis']}"
    )


def _paths_for(slug: str) -> tuple[Path, Path]:
    """Return (after_path, during_path) for an incident slug."""
    suffix = f"_{slug}" if slug else ""
    return (
        OUT_DIR / f"example_after{suffix}.json",
        OUT_DIR / f"example_during{suffix}.json",
    )


def main() -> None:
    if not main_app.DATA_LOADED:
        raise RuntimeError("main_app failed to load data.")

    force = os.environ.get("NTSB_FORCE_PRECOMPUTE", "0") == "1"
    print(f"Mode: HELD-OUT (train-only index, {len(main_app.refined_dataset)} incidents)")
    print(f"Force recompute (overrides skip-if-exists): {force}")

    struct_by_eid = load_struct_jsonl(DEFAULT_STRUCT_CACHE_V2_PATH)
    query_cache = load_query_struct_jsonl(DEFAULT_QUERY_CACHE_V2_PATH)
    print(f"Loaded struct cache: {len(struct_by_eid)} incidents, {len(query_cache)} cached queries")

    codes_list, labels_list, code_embs = build_code_embeddings()
    print(f"Loaded Zhang code embeddings ({len(codes_list)} codes)")

    for cfg in INCIDENTS:
        ev_id = cfg["ev_id"]
        slug = cfg.get("slug", "")
        label = cfg.get("label", ev_id)
        during_query = cfg["during_query"]

        out_after, out_during = _paths_for(slug)
        if not force and out_after.exists() and out_during.exists():
            print(f"\n[SKIP] {label} - both outputs already exist:")
            print(f"  {out_after.name}, {out_during.name}")
            print(f"  (set NTSB_FORCE_PRECOMPUTE=1 to rerun)")
            continue

        print(f"\n{'#' * 80}")
        print(f"# Incident: {label}")
        print(f"# ev_id   : {ev_id}")
        print(f"{'#' * 80}")

        full_narrative, truth = _load_full_narrative_and_truth(ev_id)
        if not full_narrative:
            raise RuntimeError(f"Could not find narrative for {ev_id}")
        truncated = _truncate_to_first_observation(full_narrative)
        print(f"Full narrative length: {len(full_narrative)} chars")
        print(f"Truncated prognosis query: '{truncated[:160]}'")

        run_example(
            scenario="after_incident",
            diagnosis_query=full_narrative,
            prognosis_query=truncated,
            ev_id=ev_id,
            truth=truth,
            out_path=out_after,
            struct_by_eid=struct_by_eid,
            query_cache=query_cache,
            codes_list=codes_list,
            labels_list=labels_list,
            code_embs=code_embs,
        )
        run_example(
            scenario="during_incident",
            diagnosis_query=during_query,
            prognosis_query=during_query,
            ev_id=ev_id,
            truth=truth,
            out_path=out_during,
            struct_by_eid=struct_by_eid,
            query_cache=query_cache,
            codes_list=codes_list,
            labels_list=labels_list,
            code_embs=code_embs,
        )

    print("\n" + "=" * 80)
    print("DONE - precompute step finished")
    print("=" * 80)


if __name__ == "__main__":
    main()

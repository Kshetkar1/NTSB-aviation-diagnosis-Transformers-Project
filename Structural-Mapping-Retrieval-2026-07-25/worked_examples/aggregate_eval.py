"""
Aggregate validation: run the full NTSB pipeline (A0 + A2) on every held-out
test incident and compare each prediction against the NTSB probable cause
text.

For each test incident we capture:
  - Top cluster label                  (A0 and A2)
  - P(top cluster | Query)             (A0 and A2)
  - Top free-text cause label (LTP)    (A0 and A2)
  - cosine(top cluster, NTSB cause)    (A0 and A2)
  - cosine(top cause,   NTSB cause)    (A0 and A2)

This produces a defensible aggregate metric that does NOT depend on Zhang's
54-code taxonomy. Results stream to data/aggregate_results.jsonl so the
script is resumable - re-running picks up from the last completed incident.

Usage:
  python Worked_Examples/aggregate_eval.py
  python Worked_Examples/aggregate_eval.py --force      # re-run from scratch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Held-out mode = 177 train index. Must be set BEFORE import main_app.
os.environ["NTSB_USE_TRAIN_INDEX"] = "1"

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
SCRIPTS_DIR = PROJECT_ROOT / "Testing_Structural_Mapping_Slides" / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np  # noqa: E402

import main_app  # noqa: E402
from config import REFINED_DATA_PATH, LLM_MODEL  # noqa: E402
from extract_struct_v2 import extract_struct_v2  # noqa: E402
from io_cache import load_query_struct_jsonl, load_struct_jsonl, sha256_text  # noqa: E402
from paths import DEFAULT_QUERY_CACHE_V2_PATH, DEFAULT_STRUCT_CACHE_V2_PATH  # noqa: E402
from struct_hooks import make_score_adjust_fn  # noqa: E402
from struct_score_v2 import structural_similarity as structural_similarity_v2  # noqa: E402

ALPHA = 2.0
TEST_IDS_PATH = PROJECT_ROOT / "data" / "Testing_Data_Metrics" / "splits" / "test_ev_ids.txt"
RESULTS_PATH = _HERE / "data" / "aggregate_results.jsonl"


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _cosine(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(av)
    nb = np.linalg.norm(bv)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def _load_test_ids() -> list[str]:
    return [l.strip() for l in TEST_IDS_PATH.read_text().splitlines() if l.strip()]


def _load_truth(ev_id: str, refined: dict) -> tuple[str, str]:
    """Return (full narrative, NTSB probable-cause text)."""
    rec = refined.get(ev_id) or {}
    narrative = str(rec.get("narr_accp") or rec.get("narr_accf") or "").strip()
    findings = rec.get("findings") or []
    cause_findings = [
        str(f.get("finding_description", "")).strip()
        for f in findings
        if isinstance(f, dict)
        and (f.get("Cause_Factor") or "").strip() == "C"
        and (f.get("finding_description") or "").strip()
    ]
    narr_cause = str(rec.get("narr_cause") or "").strip()
    truth = narr_cause or (cause_findings[0] if cause_findings else "")
    return narrative, truth


def _load_done() -> set[str]:
    if not RESULTS_PATH.exists():
        return set()
    done = set()
    for line in RESULTS_PATH.read_text().splitlines():
        try:
            d = json.loads(line)
            done.add(str(d.get("ev_id", "")))
        except Exception:
            continue
    return done


def _diagnose_one(top_scores, top_matches, score_adjust_fn) -> dict:
    """Run a single A0 or A2 diagnosis pass on a fixed retrieval set.
    Returns top_cluster_label, p_top_cluster, top_cause_text, top_cause_prob."""
    if score_adjust_fn is not None:
        eff_scores = [_safe_float(score_adjust_fn(s, m)) for s, m in zip(top_scores, top_matches)]
    else:
        eff_scores = list(top_scores)
    clusters = main_app.cluster_incidents_by_type(eff_scores, top_matches, 50)
    ca = main_app.calculate_cause_probabilities_per_cluster(clusters)
    diag = main_app.calculate_chain_rule_diagnosis(clusters, ca)
    weighted_causes = diag.get("weighted_causes", []) or []

    # Top cluster by P(K|Q)
    top_cluster_label = ""
    top_cluster_p = 0.0
    if ca:
        items = sorted(ca.items(), key=lambda kv: _safe_float(kv[1].get("p_cluster_query", 0.0)), reverse=True)
        top_cluster_label = str(items[0][0])
        top_cluster_p = _safe_float(items[0][1].get("p_cluster_query", 0.0))

    top_cause_text = ""
    top_cause_p = 0.0
    if weighted_causes:
        top_cause_text = str(weighted_causes[0].get("cause", ""))
        top_cause_p = _safe_float(weighted_causes[0].get("probability", 0.0))

    return {
        "top_cluster_label": top_cluster_label,
        "top_cluster_p": top_cluster_p,
        "top_cause_text": top_cause_text,
        "top_cause_p": top_cause_p,
    }


def _resolve_struct_fn(query: str, ev_id: str, struct_by_eid: dict, query_cache: dict):
    q_hash = sha256_text(query)
    if q_hash in query_cache:
        q_struct = query_cache[q_hash]
    else:
        q_struct = extract_struct_v2(query, ev_id=ev_id, model=LLM_MODEL)
    return make_score_adjust_fn(
        q_struct, struct_by_eid, alpha=ALPHA, similarity_fn=structural_similarity_v2,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-run from scratch")
    parser.add_argument("--limit", type=int, default=None, help="limit number of incidents (for smoke testing)")
    args = parser.parse_args()

    if args.force and RESULTS_PATH.exists():
        print(f"--force: deleting existing {RESULTS_PATH.name}")
        RESULTS_PATH.unlink()

    test_ids = _load_test_ids()
    if args.limit:
        test_ids = test_ids[: args.limit]
    print(f"Loaded {len(test_ids)} held-out test incident IDs")

    refined = json.loads(Path(REFINED_DATA_PATH).read_text(encoding="utf-8"))
    struct_by_eid = load_struct_jsonl(DEFAULT_STRUCT_CACHE_V2_PATH)
    query_cache = load_query_struct_jsonl(DEFAULT_QUERY_CACHE_V2_PATH)
    print(f"struct cache: {len(struct_by_eid)} incidents, {len(query_cache)} cached queries")

    done = _load_done()
    print(f"Already-done incidents in {RESULTS_PATH.name}: {len(done)}")

    skipped_no_data = 0
    skipped_no_truth = 0
    fail_count = 0
    out = open(RESULTS_PATH, "a", encoding="utf-8")
    t_global = time.time()
    for i, ev_id in enumerate(test_ids, start=1):
        if ev_id in done:
            continue
        narrative, truth = _load_truth(ev_id, refined)
        if not narrative:
            skipped_no_data += 1
            continue
        if not truth:
            skipped_no_truth += 1
            continue

        t0 = time.time()
        try:
            # Embed query once, retrieve top 50
            emb = main_app.get_embedding(narrative)
            scores_all, matches_all = main_app.find_top_matches(emb)
            scores = list(scores_all[:50])
            matches = list(matches_all[:50])

            # A0
            a0 = _diagnose_one(scores, matches, score_adjust_fn=None)
            # A2
            score_fn = _resolve_struct_fn(narrative, ev_id, struct_by_eid, query_cache)
            a2 = _diagnose_one(scores, matches, score_adjust_fn=score_fn)

            # Embed the four predicted strings + ground-truth, compute cosine sims
            truth_emb = main_app.get_embedding(truth)
            sims = {}
            for tag, text in (
                ("cluster_a0", a0["top_cluster_label"]),
                ("cluster_a2", a2["top_cluster_label"]),
                ("cause_a0", a0["top_cause_text"]),
                ("cause_a2", a2["top_cause_text"]),
            ):
                if not text:
                    sims[tag] = 0.0
                    continue
                e = main_app.get_embedding(text)
                sims[tag] = _cosine(truth_emb, e)

            row = {
                "ev_id": ev_id,
                "narrative_chars": len(narrative),
                "truth_text": truth,
                "a0_top_cluster": a0["top_cluster_label"],
                "a2_top_cluster": a2["top_cluster_label"],
                "a0_top_p_k": a0["top_cluster_p"],
                "a2_top_p_k": a2["top_cluster_p"],
                "a0_top_cause": a0["top_cause_text"],
                "a2_top_cause": a2["top_cause_text"],
                "sim_cluster_a0": sims["cluster_a0"],
                "sim_cluster_a2": sims["cluster_a2"],
                "sim_cause_a0":   sims["cause_a0"],
                "sim_cause_a2":   sims["cause_a2"],
                "elapsed_sec": round(time.time() - t0, 2),
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(
                f"[{i:>3}/{len(test_ids)}]  {ev_id}  "
                f"A0_p_k={a0['top_cluster_p']*100:5.1f}%  A2_p_k={a2['top_cluster_p']*100:5.1f}%  "
                f"sim_clust_A0={sims['cluster_a0']:.3f} A2={sims['cluster_a2']:.3f}  "
                f"({row['elapsed_sec']:.1f}s)",
                flush=True,
            )
        except Exception as e:
            fail_count += 1
            print(f"[{i:>3}/{len(test_ids)}]  {ev_id}  FAIL: {e}", flush=True)
            continue

    out.close()
    elapsed = time.time() - t_global
    print(f"\nDONE - {elapsed:.1f}s elapsed")
    print(f"  skipped (no narrative): {skipped_no_data}")
    print(f"  skipped (no truth)    : {skipped_no_truth}")
    print(f"  failures              : {fail_count}")
    print(f"  results               : {RESULTS_PATH}")


if __name__ == "__main__":
    main()

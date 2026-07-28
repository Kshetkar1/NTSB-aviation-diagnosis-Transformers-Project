#!/usr/bin/env python3
from __future__ import annotations

# Train-only index before config/main_app.
import os

os.environ["NTSB_USE_TRAIN_INDEX"] = "1"

import argparse
import json
import math
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPTS))
_METRICS = PROJECT_ROOT / "data" / "Testing_Data_Metrics" / "scripts"
sys.path.insert(0, str(_METRICS))

import eval_common  # noqa: E402
from config import REFINED_DATA_PATH, LLM_MODEL, get_openai_api_key  # noqa: E402
from openai import OpenAI  # noqa: E402

import main_app  # noqa: E402

from extract_struct import extract_struct_from_text, load_prompt_template  # noqa: E402
from extract_struct_v2 import extract_struct_v2  # noqa: E402
from io_cache import append_jsonl, load_query_struct_jsonl, load_struct_jsonl, sha256_text  # noqa: E402
from paths import (  # noqa: E402
    DEFAULT_PROMPT_PATH,
    DEFAULT_QUERY_CACHE_PATH,
    DEFAULT_QUERY_CACHE_V2_PATH,
    DEFAULT_STRUCT_CACHE_PATH,
    DEFAULT_STRUCT_CACHE_V2_PATH,
    EVAL_OUTPUT_DIR,
    TEST_IDS_PATH,
)
from struct_hooks import make_score_adjust_fn  # noqa: E402
from struct_score_v2 import structural_similarity as structural_similarity_v2  # noqa: E402

MATCH_THRESHOLD = 0.75


def max_cosine_sim(pred: str, truths: list[str]) -> float:
    return eval_common.max_cosine_sim(pred, truths, main_app.get_embedding)


def recall_at_k(preds: list[str], truths: list[str], k: int) -> bool:
    return eval_common.recall_at_k(preds, truths, k, MATCH_THRESHOLD, main_app.get_embedding)


def mrr_score(preds: list[str], truths: list[str], k: int) -> float:
    return eval_common.mrr_score(preds, truths, k, MATCH_THRESHOLD, main_app.get_embedding)


def build_query(inc: dict):
    accp = inc.get("narr_accp") or ""
    accf = inc.get("narr_accf") or ""
    if accp and str(accp).strip():
        return str(accp).strip(), "narr_accp"
    if accf and str(accf).strip():
        return str(accf).strip(), "narr_accf"
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Diagnosis eval with optional structural reweighting (train-only index)."
    )
    ap.add_argument("--n", default="10", help='Test count or "all".')
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument(
        "--structural",
        action="store_true",
        help="Enable structural reweighting (requires --struct-cache).",
    )
    ap.add_argument("--alpha", type=float, default=0.5, help="Blend strength for structural term.")
    ap.add_argument(
        "--struct-version",
        choices=("v1", "v2"),
        default="v1",
        help="v1=flat enums (A1); v2=causal_chain alignment (A2). Only with --structural.",
    )
    ap.add_argument(
        "--struct-cache",
        type=Path,
        default=None,
        help="Train JSONL cache (default: v1 or v2 path from --struct-version).",
    )
    ap.add_argument(
        "--query-cache",
        type=Path,
        default=None,
        help="Per-query struct JSONL (default: v1 or v2 path).",
    )
    ap.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    ap.add_argument("--model", default=LLM_MODEL)
    ap.add_argument("--output-stem", default="eval_diagnosis_structural")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip ev_ids already present in the output CSV (same --output-stem).",
    )
    args = ap.parse_args()

    if args.struct_cache is None:
        args.struct_cache = (
            DEFAULT_STRUCT_CACHE_V2_PATH if args.structural and args.struct_version == "v2" else DEFAULT_STRUCT_CACHE_PATH
        )
    if args.query_cache is None:
        args.query_cache = (
            DEFAULT_QUERY_CACHE_V2_PATH if args.structural and args.struct_version == "v2" else DEFAULT_QUERY_CACHE_PATH
        )

    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EVAL_OUTPUT_DIR / f"{args.output_stem}.csv"
    json_path = EVAL_OUTPUT_DIR / f"{args.output_stem}.json"
    summary_path = EVAL_OUTPUT_DIR / f"{args.output_stem}_summary.json"

    if not main_app.DATA_LOADED:
        raise RuntimeError("Train index not loaded. See data/Testing_Data_Metrics/README.md.")
    if not REFINED_DATA_PATH.is_file():
        raise FileNotFoundError(f"Full dataset not found: {REFINED_DATA_PATH}")

    struct_by_eid: dict[str, dict] = {}
    if args.structural:
        struct_by_eid = load_struct_jsonl(args.struct_cache)
        if not struct_by_eid:
            raise FileNotFoundError(
                f"No structs in {args.struct_cache}. Run build_struct_cache_train.py "
                f"or build_struct_cache_train_v2.py (for v2)."
            )

    query_disk = load_query_struct_jsonl(args.query_cache)
    tmpl = load_prompt_template(args.prompt)
    use_v1_llm = args.structural and args.struct_version == "v1"
    client = OpenAI(api_key=get_openai_api_key()) if use_v1_llm else None

    def get_query_struct(q: str, ev_id_for_query: str) -> dict | None:
        if not args.structural:
            return None
        h = sha256_text(q)
        if h in query_disk:
            return query_disk[h]
        if args.struct_version == "v2":
            st = extract_struct_v2(q, ev_id_for_query, model=args.model)
        else:
            if client is None:
                raise RuntimeError("v1 structural mode requires OpenAI client")
            st = extract_struct_from_text(client, args.model, q, tmpl)
        append_jsonl(
            args.query_cache,
            {"query_hash": h, "struct": st, "preview": q[:240]},
        )
        query_disk[h] = st
        return st

    test_ids = eval_common.load_test_ids(TEST_IDS_PATH)
    with open(REFINED_DATA_PATH, "r", encoding="utf-8") as f:
        full_dataset: dict = json.load(f)

    if args.n == "all":
        batch = test_ids[args.offset :]
    else:
        batch = test_ids[args.offset : args.offset + int(args.n)]

    if args.resume:
        done = eval_common.ev_ids_in_output_csv(csv_path)
        before = len(batch)
        batch = [eid for eid in batch if eid not in done]
        print(
            f"[resume] {csv_path.name}: {len(done)} ev_ids on disk; "
            f"{before} -> {len(batch)} to run",
            flush=True,
        )

    fieldnames = [
        "ev_id",
        "mode",
        "query_source",
        "query_full_text",
        "top_prediction_cause",
        "top_prediction_prob",
        "m1_truth_full",
        "m1_match_pct",
        "m1_hit",
        "m1_recall5",
        "mrr_m1",
        "m2_truth_full",
        "m2_match_pct",
        "m2_hit",
        "m2_recall5",
        "mrr_m2",
    ]

    if args.structural:
        mode = "A2_causal_chain" if args.struct_version == "v2" else "A1_structural"
    else:
        mode = "A0_baseline"

    for ev_id in batch:
        inc = full_dataset.get(ev_id)
        if not inc:
            print(f"[skip] {ev_id}: not in full dataset", flush=True)
            continue
        query, qsrc = build_query(inc)
        if not query:
            print(f"[skip] {ev_id}: no narr_accp / narr_accf", flush=True)
            continue

        q_struct = get_query_struct(query, ev_id)
        adj_fn = None
        if args.structural:
            adj_fn = make_score_adjust_fn(
                q_struct,
                struct_by_eid,
                args.alpha,
                similarity_fn=structural_similarity_v2 if args.struct_version == "v2" else None,
            )
        diag = main_app.diagnose_with_conditional_probabilities(
            query, top_n=10, top_n_incidents=50, score_adjust_fn=adj_fn
        )
        causes = diag.get("weighted_causes") or []
        if diag.get("error"):
            print(f"[warn] {ev_id}: {diag.get('error')}", flush=True)

        top5 = [str(c.get("cause", "")).strip() for c in causes[:5] if c.get("cause")]
        top1 = top5[0] if top5 else ""
        top1_prob = causes[0].get("probability", 0.0) if causes else 0.0

        findings = inc.get("findings") or []
        m1_list = [
            str(f.get("finding_description", "")).strip()
            for f in findings
            if isinstance(f, dict)
            and (f.get("Cause_Factor") or "").strip() == "C"
            and (f.get("finding_description") or "").strip()
        ]
        narr_cause = inc.get("narr_cause") or ""
        narr_s = str(narr_cause).strip() if narr_cause else ""
        m2_chunks = eval_common.narr_cause_truth_chunks(narr_s)

        def score_metric(truths: list[str]):
            if not truths:
                return "N/A", "", "", ""
            pct = max_cosine_sim(top1, truths) * 100.0 if top1 else float("nan")
            pct_s = "" if math.isnan(pct) else f"{pct:.4f}"
            hit = (
                ""
                if not top1 or math.isnan(pct)
                else ("1" if pct / 100.0 >= MATCH_THRESHOLD else "0")
            )
            r5 = "1" if recall_at_k(top5, truths, 5) else "0"
            mrr = mrr_score(top5, truths, 5)
            return pct_s, hit, r5, f"{mrr:.6f}"

        m1_truth_s = eval_common.pipe_join(m1_list) if m1_list else "N/A"
        m2_truth_s = narr_s if m2_chunks else "N/A"

        if not m1_list:
            m1_pct, m1_hit, m1_r5, m1_mrr = ("N/A", "", "", "")
        else:
            m1_pct, m1_hit, m1_r5, m1_mrr = score_metric(m1_list)

        if not m2_chunks:
            m2_pct, m2_hit, m2_r5, m2_mrr = ("N/A", "", "", "")
        else:
            m2_pct, m2_hit, m2_r5, m2_mrr = score_metric(m2_chunks)

        row = {
            "ev_id": ev_id,
            "mode": mode,
            "query_source": qsrc or "",
            "query_full_text": query,
            "top_prediction_cause": top1,
            "top_prediction_prob": f"{top1_prob:.6f}",
            "m1_truth_full": m1_truth_s,
            "m1_match_pct": m1_pct,
            "m1_hit": m1_hit,
            "m1_recall5": m1_r5,
            "mrr_m1": m1_mrr,
            "m2_truth_full": m2_truth_s,
            "m2_match_pct": m2_pct,
            "m2_hit": m2_hit,
            "m2_recall5": m2_r5,
            "mrr_m2": m2_mrr,
        }
        eval_common.append_csv_row(row, fieldnames, csv_path, EVAL_OUTPUT_DIR)
        # Build full-distribution record for JSON (includes all cause probabilities)
        json_row = dict(row)
        json_row["probability_distribution"] = [
            {"cause": c.get("cause", ""), "probability": round(c.get("probability", 0.0), 6)}
            for c in causes
        ]
        eval_common.append_json_record(json_row, json_path, EVAL_OUTPUT_DIR)
        print(f"[ok] {ev_id} {mode}  ({len(causes)} causes in distribution)", flush=True)

    summary = eval_common.recompute_summary_structural_only(csv_path, MATCH_THRESHOLD)
    summary["structural"] = bool(args.structural)
    summary["struct_version"] = args.struct_version if args.structural else None
    summary["alpha"] = args.alpha if args.structural else None
    summary["struct_cache"] = str(args.struct_cache) if args.structural else None
    summary["query_cache"] = str(args.query_cache) if args.structural else None
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path} and {summary_path}")


if __name__ == "__main__":
    main()

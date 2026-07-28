#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ["NTSB_USE_TRAIN_INDEX"] = "1"

_SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPTS))

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


def load_test_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run create_splits.py first.")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [x for x in lines if x]


def sequence_descriptions(inc: dict) -> list[str]:
    seq = inc.get("sequence_of_events") or []
    return [str(e.get("Occurrence_Description") or "").strip() for e in seq]


def norm(s: str) -> str:
    return main_app._normalize_occurrence_text(s)


def eval_transition(
    query: str,
    truth_next: str,
    top_n_incidents: int,
    score_adjust_fn,
):
    r = main_app.predict_future_events(
        query,
        top_n_incidents=top_n_incidents,
        max_chain_steps=1,
        score_adjust_fn=score_adjust_fn,
    )
    dist = r.get("future_events") or []
    tnorm = norm(truth_next)
    if not dist or not tnorm:
        return False, False, "", float("nan"), len(r.get("aligned_incidents") or []), r.get(
            "downstream_incident_count", 0
        )

    top1 = dist[0]["event"]
    p1 = float(dist[0]["probability"])
    top5_events = [d["event"] for d in dist[:5]]
    exact = norm(top1) == tnorm
    r5 = any(norm(x) == tnorm for x in top5_events)
    return exact, r5, top1, p1, len(r.get("aligned_incidents") or []), r.get("downstream_incident_count", 0)


def soft_hit(truth_next: str, pred: str, threshold: float) -> bool:
    if not truth_next or not pred:
        return False
    a = main_app.get_embedding(truth_next)
    b = main_app.get_embedding(pred)
    import numpy as np

    return float(np.dot(np.asarray(a), np.asarray(b))) >= threshold


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prognosis eval with optional structural reweighting (train-only index)."
    )
    ap.add_argument("--n", default="20", help='Test incidents (int) or "all".')
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-positions", type=int, default=8)
    ap.add_argument("--top-n-incidents", type=int, default=50)
    ap.add_argument("--soft-threshold", type=float, default=None)
    ap.add_argument("--structural", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument(
        "--struct-version",
        choices=("v1", "v2"),
        default="v1",
        help="v1=flat enums (A1); v2=causal_chain alignment (A2). Only with --structural.",
    )
    ap.add_argument("--struct-cache", type=Path, default=None)
    ap.add_argument("--query-cache", type=Path, default=None)
    ap.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    ap.add_argument("--model", default=LLM_MODEL)
    ap.add_argument("--output-stem", default="eval_prognosis_structural")
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
    summary_path = EVAL_OUTPUT_DIR / f"{args.output_stem}_summary.json"

    if not main_app.DATA_LOADED:
        raise RuntimeError("Train index not loaded.")
    if not REFINED_DATA_PATH.is_file():
        raise FileNotFoundError(f"Full dataset not found: {REFINED_DATA_PATH}")

    struct_by_eid: dict[str, dict] = {}
    if args.structural:
        struct_by_eid = load_struct_jsonl(args.struct_cache)
        if not struct_by_eid:
            raise FileNotFoundError(f"No structs in {args.struct_cache}.")

    query_disk = load_query_struct_jsonl(args.query_cache)
    tmpl = load_prompt_template(args.prompt)
    use_v1_llm = args.structural and args.struct_version == "v1"
    client = OpenAI(api_key=get_openai_api_key()) if use_v1_llm else None

    def get_query_struct(q: str, query_ev_tag: str):
        if not args.structural:
            return None
        h = sha256_text(q)
        if h in query_disk:
            return query_disk[h]
        if args.struct_version == "v2":
            st = extract_struct_v2(q, query_ev_tag, model=args.model)
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

    test_ids = load_test_ids(TEST_IDS_PATH)
    with open(REFINED_DATA_PATH, "r", encoding="utf-8") as f:
        full_dataset: dict = json.load(f)

    if args.n == "all":
        batch = test_ids[args.offset :]
    else:
        batch = test_ids[args.offset : args.offset + int(args.n)]

    if args.structural:
        mode = "A2_causal_chain" if args.struct_version == "v2" else "A1_structural"
    else:
        mode = "A0_baseline"
    rows_out = []
    n_exact = n_r5 = n_soft = n_soft_evaluated = n_total = n_skipped_empty = 0

    fieldnames = [
        "mode",
        "ev_id",
        "step_index",
        "query_event",
        "truth_next",
        "top1_pred",
        "top1_prob",
        "exact_hit",
        "recall5_hit",
        "n_aligned",
        "n_downstream",
        "soft_hit",
    ]

    for ev_id in batch:
        inc = full_dataset.get(ev_id)
        if not inc:
            continue
        descs = sequence_descriptions(inc)
        if len(descs) < 2:
            n_skipped_empty += 1
            continue
        max_i = min(len(descs) - 2, args.max_positions - 1)
        for i in range(max_i + 1):
            q = descs[i]
            truth = descs[i + 1]
            if not q or not truth:
                continue
            qs = get_query_struct(q, f"{ev_id}_{i}")
            adj_fn = None
            if args.structural:
                adj_fn = make_score_adjust_fn(
                    qs,
                    struct_by_eid,
                    args.alpha,
                    similarity_fn=structural_similarity_v2 if args.struct_version == "v2" else None,
                )
            exact, r5, top1, p1, n_al, n_dn = eval_transition(
                q, truth, args.top_n_incidents, adj_fn
            )
            sh = ""
            if args.soft_threshold is not None and top1:
                n_soft_evaluated += 1
                s = soft_hit(truth, top1, args.soft_threshold)
                sh = "1" if s else "0"
                if s:
                    n_soft += 1
            n_total += 1
            n_exact += 1 if exact else 0
            n_r5 += 1 if r5 else 0
            rows_out.append(
                {
                    "mode": mode,
                    "ev_id": ev_id,
                    "step_index": i,
                    "query_event": q,
                    "truth_next": truth,
                    "top1_pred": top1,
                    "top1_prob": "" if math.isnan(p1) else f"{p1:.6f}",
                    "exact_hit": "1" if exact else "0",
                    "recall5_hit": "1" if r5 else "0",
                    "n_aligned": n_al,
                    "n_downstream": n_dn,
                    "soft_hit": sh,
                }
            )

    exact_acc = n_exact / n_total if n_total else 0.0
    r5_acc = n_r5 / n_total if n_total else 0.0
    soft_acc = (n_soft / n_soft_evaluated) if n_soft_evaluated else None

    summary = {
        "mode": mode,
        "structural": bool(args.structural),
        "struct_version": args.struct_version if args.structural else None,
        "alpha": args.alpha if args.structural else None,
        "struct_cache": str(args.struct_cache) if args.structural else None,
        "query_cache": str(args.query_cache) if args.structural else None,
        "n_test_incidents_requested": len(batch),
        "n_transition_rows": n_total,
        "n_skipped_fewer_than_2_events": n_skipped_empty,
        "exact_accuracy": exact_acc,
        "recall5_accuracy": r5_acc,
        "soft_accuracy": soft_acc,
        "n_soft_evaluated": n_soft_evaluated,
        "soft_threshold": args.soft_threshold,
        "train_only_index": True,
        "max_positions_per_incident": args.max_positions,
    }

    import csv

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()

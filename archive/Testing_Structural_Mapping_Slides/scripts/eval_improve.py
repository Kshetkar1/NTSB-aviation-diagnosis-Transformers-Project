#!/usr/bin/env python3
"""
Improvement ladder for structural-mapping reranking (A2 family) vs A0 on the
paper's held-out split. Companion to eval_diagnosis_structural_refined.py;
reuses the same protocol, caches, and evaluation code paths.

Subcommands:
  pairs    -- precompute (query, candidate) cosine + slide-formula struct_sim
              for every query x its cosine top-50; cached to
              Testing_Structural_Mapping/cache_refined/pair_scores_improve.jsonl.
              Zero API cost (query embeddings come from the retest emb cache).
  run      -- run one reranking variant pass through the full
              main_app.diagnose_with_conditional_probabilities pipeline and
              write outputs/structmap_improve/eval_diagnosis_<variant>.csv.
              Variants (all reweight the SAME cosine top-50 pool, exactly like
              the retest A2):
                mult_a0.5 / mult_a1 / mult_a4 / mult_a8  : max(0,cos)*exp(a*s)
                struct_only                              : s (cosine ignored)
                add_w0.1 / add_w0.3 / add_w0.5           : max(0,cos) + w*s
                rrf                                      : 1/(60+rank_cos) + 1/(60+rank_struct)
              --pairs-file lets Phase 2 swap in coded-chain similarities.
  analyze  -- paired comparison of every variant CSV found in the output dir
              against the retest A0 CSV: avg match%, top-1/R@5 at 0.75,
              McNemar exact, bootstrap CI on delta avg match%, and a
              threshold sweep 0.40-0.75; writes variants_summary.json.

Never modifies existing files: reads the retest caches read-only and appends
new embeddings to a NEW cache file (emb_cache_improve.jsonl).

Run from repo root, e.g.:
  python3.11 Testing_Structural_Mapping_Slides/scripts/eval_improve.py pairs
  python3.11 Testing_Structural_Mapping_Slides/scripts/eval_improve.py run --variant mult_a4
  python3.11 Testing_Structural_Mapping_Slides/scripts/eval_improve.py analyze
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPTS))
_METRICS = PROJECT_ROOT / "data" / "Testing_Data_Metrics" / "scripts"
sys.path.insert(0, str(_METRICS))

import eval_common  # noqa: E402
from config import REFINED_DATA_PATH, WINDOW_DATA_PATH, USE_ZHANG_WINDOW  # noqa: E402

import main_app  # noqa: E402

from io_cache import append_jsonl, load_struct_jsonl, sha256_text  # noqa: E402
from struct_score_v2 import structural_similarity as structural_similarity_v2  # noqa: E402

MATCH_THRESHOLD = 0.75
QUERY_TRUNC = 4000
TOP_N_INCIDENTS = 50
RRF_K = 60

CACHE_DIR = PROJECT_ROOT / "Testing_Structural_Mapping" / "cache_refined"
OUT_DIR = PROJECT_ROOT / "outputs" / "structmap_improve"
STRUCT_TRAIN_CACHE = CACHE_DIR / "struct_train_v2_refined.jsonl"
QUERY_STRUCT_CACHE = CACHE_DIR / "query_struct_v2_refined.jsonl"
EMB_CACHE_OLD = CACHE_DIR / "emb_cache.jsonl"            # retest cache, READ-ONLY
EMB_CACHE_NEW = CACHE_DIR / "emb_cache_improve.jsonl"    # new appends go here
RETRIEVED_TOP50 = CACHE_DIR / "retrieved_top50.jsonl"
PAIR_SCORES_DEFAULT = CACHE_DIR / "pair_scores_improve.jsonl"

A0_CSV = PROJECT_ROOT / "outputs" / "structmap_refined" / "eval_diagnosis_A0_refined.csv"

FIELDNAMES = [
    "ev_id", "mode", "query_source", "query_full_text",
    "top_prediction_cause", "top_prediction_prob",
    "m1_truth_full", "m1_match_pct", "m1_hit", "m1_recall5", "mrr_m1",
    "m2_truth_full", "m2_match_pct", "m2_hit", "m2_recall5", "mrr_m2",
]

SWEEP_THRESHOLDS = [0.75, 0.60, 0.55, 0.50, 0.45, 0.40]


# ---------------------------------------------------------------------------
# Embedding cache: load retest cache read-only; append new vectors to a NEW file.
# ---------------------------------------------------------------------------
_emb_mem: dict[str, list] = {}
_orig_get_embedding = main_app.get_embedding
N_EMB_API_CALLS = 0


def _load_emb_file(path: Path) -> int:
    n = 0
    if not path.is_file():
        return 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            h, v = row.get("h"), row.get("v")
            if h and isinstance(v, list):
                _emb_mem[h] = v
                n += 1
    return n


def load_emb_caches() -> None:
    n1 = _load_emb_file(EMB_CACHE_OLD)
    n2 = _load_emb_file(EMB_CACHE_NEW)
    print(f"[emb-cache] loaded {len(_emb_mem)} vectors ({n1} retest + {n2} improve)", flush=True)


def cached_get_embedding(text):
    global N_EMB_API_CALLS
    h = sha256_text(str(text))
    if h in _emb_mem:
        return _emb_mem[h]
    v = _orig_get_embedding(text)
    v = list(v)
    _emb_mem[h] = v
    append_jsonl(EMB_CACHE_NEW, {"h": h, "v": v})
    N_EMB_API_CALLS += 1
    return v


main_app.get_embedding = cached_get_embedding


# ---------------------------------------------------------------------------
# Shared loading (identical population logic to the retest runner).
# ---------------------------------------------------------------------------
def load_heldout() -> list[tuple[str, dict, str]]:
    full = json.loads(REFINED_DATA_PATH.read_text(encoding="utf-8"))
    window_ids = set(json.loads(WINDOW_DATA_PATH.read_text(encoding="utf-8")).keys())
    held = []
    for k, inc in full.items():
        if k in window_ids:
            continue
        narr = str(inc.get("narr_accf") or "").strip()
        if len(narr) < 100:
            continue
        held.append((k, inc, narr))
    held.sort(key=lambda t: t[0])
    return held


def load_retrieved_top50() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with open(RETRIEVED_TOP50, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = row.get("query_ev_id")
            ids = row.get("retrieved_ev_ids")
            if q and isinstance(ids, list):
                out[q] = ids
    return out


def load_pair_scores(path: Path) -> dict[str, list[dict]]:
    """query_ev_id -> [{'ev_id','cos','struct'}] in cosine-retrieval order."""
    out: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = row.get("query_ev_id")
            pairs = row.get("pairs")
            if q and isinstance(pairs, list):
                out[q] = pairs
    return out


# ---------------------------------------------------------------------------
# Scoring row (identical to the retest runner).
# ---------------------------------------------------------------------------
def max_cosine_sim(pred: str, truths: list[str]) -> float:
    return eval_common.max_cosine_sim(pred, truths, main_app.get_embedding)


def score_row(ev_id: str, mode: str, query: str, inc: dict, diag: dict) -> dict:
    causes = diag.get("weighted_causes") or []
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
    narr_s = str(inc.get("narr_cause") or "").strip()
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
        r5 = "1" if eval_common.recall_at_k(top5, truths, 5, MATCH_THRESHOLD, main_app.get_embedding) else "0"
        mrr = eval_common.mrr_score(top5, truths, 5, MATCH_THRESHOLD, main_app.get_embedding)
        return pct_s, hit, r5, f"{mrr:.6f}"

    m1 = score_metric(m1_list) if m1_list else ("N/A", "", "", "")
    m2 = score_metric(m2_chunks) if m2_chunks else ("N/A", "", "", "")

    return {
        "ev_id": ev_id,
        "mode": mode,
        "query_source": "narr_accf",
        "query_full_text": query,
        "top_prediction_cause": top1,
        "top_prediction_prob": f"{top1_prob:.6f}",
        "m1_truth_full": eval_common.pipe_join(m1_list) if m1_list else "N/A",
        "m1_match_pct": m1[0], "m1_hit": m1[1], "m1_recall5": m1[2], "mrr_m1": m1[3],
        "m2_truth_full": narr_s if m2_chunks else "N/A",
        "m2_match_pct": m2[0], "m2_hit": m2[1], "m2_recall5": m2[2], "mrr_m2": m2[3],
    }


# ---------------------------------------------------------------------------
# pairs: precompute cosine + struct_sim for every (query, top-50 candidate).
# ---------------------------------------------------------------------------
def cmd_pairs(args) -> None:
    held = load_heldout()
    retrieved = load_retrieved_top50()
    struct_by_eid = load_struct_jsonl(STRUCT_TRAIN_CACHE)
    query_structs = load_struct_jsonl(QUERY_STRUCT_CACHE)

    # ev_id -> embedding row of the window index
    eid_to_vec: dict[str, np.ndarray] = {}
    for i, info in enumerate(main_app.embeddings_map):
        eid = info.get("ev_id")
        if eid and eid not in eid_to_vec:
            eid_to_vec[eid] = np.asarray(main_app.embeddings[i], dtype=np.float64)

    done: set[str] = set(load_pair_scores(args.out).keys()) if args.out.is_file() else set()
    todo = [(k, narr) for (k, _, narr) in held if k not in done]
    print(f"[pairs] {len(done)} done; {len(todo)} to compute", flush=True)

    n_missing_struct = 0
    for i, (k, narr) in enumerate(todo, 1):
        query = narr[:QUERY_TRUNC]
        h = sha256_text(str(query))
        if h not in _emb_mem:
            raise RuntimeError(f"query embedding for {k} not in retest emb cache")
        q_emb = np.asarray(_emb_mem[h], dtype=np.float64)
        q_struct = query_structs.get(k)
        pairs = []
        for eid in retrieved.get(k, []):
            vec = eid_to_vec.get(eid)
            cos = float(np.dot(q_emb, vec)) if vec is not None else float("nan")
            cand = struct_by_eid.get(eid)
            if q_struct and cand:
                s = float(structural_similarity_v2(q_struct, cand))
            else:
                s = None
                n_missing_struct += 1
            pairs.append({"ev_id": eid, "cos": cos, "struct": s})
        append_jsonl(args.out, {"query_ev_id": k, "pairs": pairs})
        if i % 50 == 0 or i == len(todo):
            print(f"[pairs] {i}/{len(todo)}", flush=True)
    print(f"[pairs] complete -> {args.out} (missing-struct pairs: {n_missing_struct})", flush=True)


# ---------------------------------------------------------------------------
# run: one variant pass through the full diagnosis pipeline.
# ---------------------------------------------------------------------------
def build_weight_maps(variant: str, pair_scores: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    """query_ev_id -> {cand_ev_id: replacement retrieval weight}."""
    out: dict[str, dict[str, float]] = {}
    for q, pairs in pair_scores.items():
        rows = [(p["ev_id"], float(p["cos"]), float(p["struct"]) if p["struct"] is not None else 0.0)
                for p in pairs]
        wm: dict[str, float] = {}
        if variant.startswith("mult_a"):
            a = float(variant.split("mult_a")[1])
            for eid, cos, s in rows:
                wm[eid] = max(0.0, cos) * math.exp(a * s)
        elif variant == "struct_only":
            for eid, cos, s in rows:
                wm[eid] = max(0.0, s) + 1e-9
        elif variant.startswith("add_w"):
            w = float(variant.split("add_w")[1])
            for eid, cos, s in rows:
                wm[eid] = max(0.0, cos) + w * s
        elif variant == "rrf":
            by_cos = sorted(rows, key=lambda r: -r[1])
            by_s = sorted(rows, key=lambda r: -r[2])
            rc = {eid: i for i, (eid, _, _) in enumerate(by_cos, start=1)}
            rs = {eid: i for i, (eid, _, _) in enumerate(by_s, start=1)}
            for eid, cos, s in rows:
                wm[eid] = 1.0 / (RRF_K + rc[eid]) + 1.0 / (RRF_K + rs[eid])
        else:
            raise ValueError(f"unknown variant: {variant}")
        out[q] = wm
    return out


def cmd_run(args) -> None:
    variant = args.variant
    pair_scores = load_pair_scores(args.pairs_file)
    weight_maps = build_weight_maps(variant, pair_scores)
    held = load_heldout()
    tag = args.tag or variant
    csv_path = OUT_DIR / f"eval_diagnosis_{tag}.csv"

    done = eval_common.ev_ids_in_output_csv(csv_path)
    todo = [(k, inc, narr) for (k, inc, narr) in held if k not in done]
    print(f"[{tag}] {len(done)} done on disk; {len(todo)} to run", flush=True)
    t0 = time.time()
    for i, (k, inc, narr) in enumerate(todo, 1):
        query = narr[:QUERY_TRUNC]
        wm = weight_maps.get(k, {})

        def adj_fn(cosine_score: float, match: dict) -> float:
            if match.get("source") != "incident":
                return float(cosine_score)
            eid = match.get("ev_id")
            if eid in wm:
                return wm[eid]
            return float(cosine_score)

        diag = main_app.diagnose_with_conditional_probabilities(
            query, top_n=10, top_n_incidents=TOP_N_INCIDENTS, score_adjust_fn=adj_fn
        )
        if diag.get("error"):
            print(f"[warn] {k}: {diag.get('error')}", flush=True)
        row = score_row(k, tag, query, inc, diag)
        eval_common.append_csv_row(row, FIELDNAMES, csv_path, OUT_DIR)
        if i % 50 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"[{tag}] {i}/{len(todo)} ({el/60:.1f} min, {el/i:.2f} s/query)", flush=True)
    print(f"[{tag}] pass complete -> {csv_path} "
          f"(embedding API calls this run: {N_EMB_API_CALLS})", flush=True)


# ---------------------------------------------------------------------------
# analyze: paired stats for every variant CSV vs the retest A0 CSV.
# ---------------------------------------------------------------------------
def analyze_variant(a0_csv: Path, var_csv: Path) -> dict:
    import analyze_a0_vs_a2 as az

    df = az.load_and_join(a0_csv, var_csv)
    df = df[df["m1_truth_full_A0"].notna()].reset_index(drop=True)
    summary = az.compute_summary(df)

    a0p = df["m1_match_pct_A0"].fillna(0).astype(float).to_numpy()
    vp = df["m1_match_pct_A2"].fillna(0).astype(float).to_numpy()
    sweep = []
    for t in SWEEP_THRESHOLDS:
        h0 = (a0p >= t * 100.0).astype(int)
        h2 = (vp >= t * 100.0).astype(int)
        b = int(((h0 == 1) & (h2 == 0)).sum())  # A0-only
        c = int(((h0 == 0) & (h2 == 1)).sum())  # variant-only
        sweep.append({
            "threshold": t,
            "a0_hits": int(h0.sum()),
            "var_hits": int(h2.sum()),
            "a0_only": b,
            "var_only": c,
            "mcnemar_p": az.mcnemar_exact(b, c),
        })
    summary["threshold_sweep"] = sweep
    summary["n_top1_changed"] = None  # filled below by caller if desired
    return summary


def cmd_analyze(args) -> None:
    import pandas as pd

    a0_df = pd.read_csv(A0_CSV, low_memory=False)[["ev_id", "top_prediction_cause"]]
    results: dict[str, dict] = {}
    for csv_path in sorted(OUT_DIR.glob("eval_diagnosis_*.csv")):
        tag = csv_path.stem.replace("eval_diagnosis_", "")
        try:
            summary = analyze_variant(A0_CSV, csv_path)
        except Exception as ex:  # noqa: BLE001
            print(f"[analyze] {tag}: FAILED ({ex})", flush=True)
            continue
        v_df = pd.read_csv(csv_path, low_memory=False)[["ev_id", "top_prediction_cause"]]
        m = a0_df.merge(v_df, on="ev_id", suffixes=("_a0", "_v"))
        summary["n_top1_changed"] = int(
            (m["top_prediction_cause_a0"].fillna("") != m["top_prediction_cause_v"].fillna("")).sum()
        )
        summary["n_rows"] = int(len(m))
        results[tag] = summary
        d = summary["delta_avg_match_pct"]
        print(
            f"[analyze] {tag:>16}: n={summary['n']}  "
            f"avg match% {summary['A2']['avg_match_pct']:.2f} vs A0 {summary['A0']['avg_match_pct']:.2f}  "
            f"delta {d['mean']:+.2f} pp CI [{d['ci95'][0]:+.2f},{d['ci95'][1]:+.2f}]  "
            f"McNemar(0.75) p={summary['mcnemar']['p_value_two_sided']:.3f}  "
            f"top1 changed {summary['n_top1_changed']}/{summary['n_rows']}",
            flush=True,
        )
    out = OUT_DIR / "variants_summary.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[analyze] wrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pairs")
    p.add_argument("--out", type=Path, default=PAIR_SCORES_DEFAULT)

    p = sub.add_parser("run")
    p.add_argument("--variant", required=True)
    p.add_argument("--pairs-file", type=Path, default=PAIR_SCORES_DEFAULT)
    p.add_argument("--tag", default=None, help="output CSV tag (default: variant name)")

    sub.add_parser("analyze")

    args = ap.parse_args()

    if not main_app.DATA_LOADED:
        raise RuntimeError("Window retrieval index not loaded")
    if not USE_ZHANG_WINDOW:
        raise RuntimeError("config did not select the 1982-2006 window index")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    load_emb_caches()

    if args.cmd == "pairs":
        cmd_pairs(args)
    elif args.cmd == "run":
        cmd_run(args)
    else:
        cmd_analyze(args)


if __name__ == "__main__":
    main()

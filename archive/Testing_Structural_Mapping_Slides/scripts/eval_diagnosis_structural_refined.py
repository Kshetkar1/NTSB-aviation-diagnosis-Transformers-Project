#!/usr/bin/env python3
"""
A0 vs A2 diagnosis retest on the CURRENT refined dataset (paper's held-out split).

Re-runs the old Testing_Structural_Mapping experiment (A0 = embedding-only
diagnosis; A2 = causal-chain structural reranking, slide formulas, alpha=2.0)
on the paper's main evaluation split:

  * Retrieval index : 1982-2006 window artifact (data/processed/
    refined_dataset_1982_2006.json + embeddings_1982_2006.*), loaded by
    main_app with default config -- the exact index the paper's held-out
    narrative->BN eval uses.
  * Test queries    : accidents in the full refined corpus but NOT in the
    window, with a usable factual narrative (narr_accf, >=100 chars),
    truncated to 4000 chars -- same population logic as
    tests/heldout_narrative_bn_eval.py (n=296).

Phases (all resumable):
  a0       -- baseline diagnosis for each query; also records each query's
              top-50 retrieved incident ev_ids.
  extract  -- gpt-4o-mini causal-chain extraction for (a) every incident
              retrieved in some query's top-50 and (b) the 296 queries.
              Cached to Testing_Structural_Mapping/cache_refined/.
  a2       -- structurally reweighted diagnosis (slide struct_score_v2,
              fusion max(0,cos)*exp(2.0*struct_sim)).
  analyze  -- paired A0-vs-A2 analysis (reuses analyze_a0_vs_a2).

Run from repo root:
  python3.11 Testing_Structural_Mapping_Slides/scripts/eval_diagnosis_structural_refined.py --phase a0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPTS))
_METRICS = PROJECT_ROOT / "data" / "Testing_Data_Metrics" / "scripts"
sys.path.insert(0, str(_METRICS))

import eval_common  # noqa: E402
from config import REFINED_DATA_PATH, WINDOW_DATA_PATH, USE_ZHANG_WINDOW  # noqa: E402

import main_app  # noqa: E402

from extract_struct_v2 import extract_struct_v2, extract_struct_from_incident_v2  # noqa: E402
from io_cache import append_jsonl, load_struct_jsonl, sha256_text  # noqa: E402
from struct_hooks import make_score_adjust_fn  # noqa: E402
from struct_score_v2 import structural_similarity as structural_similarity_v2  # noqa: E402

MATCH_THRESHOLD = 0.75
ALPHA = 2.0
STRUCT_MODEL = "gpt-4o-mini"
QUERY_TRUNC = 4000  # same truncation as tests/heldout_narrative_bn_eval.py
TOP_N_INCIDENTS = 50

CACHE_DIR = PROJECT_ROOT / "Testing_Structural_Mapping" / "cache_refined"
OUT_DIR = PROJECT_ROOT / "outputs" / "structmap_refined"
STRUCT_TRAIN_CACHE = CACHE_DIR / "struct_train_v2_refined.jsonl"
QUERY_STRUCT_CACHE = CACHE_DIR / "query_struct_v2_refined.jsonl"
EMB_CACHE = CACHE_DIR / "emb_cache.jsonl"
RETRIEVED_TOP50 = CACHE_DIR / "retrieved_top50.jsonl"

A0_CSV = OUT_DIR / "eval_diagnosis_A0_refined.csv"
A2_CSV = OUT_DIR / "eval_diagnosis_A2_refined.csv"

FIELDNAMES = [
    "ev_id", "mode", "query_source", "query_full_text",
    "top_prediction_cause", "top_prediction_prob",
    "m1_truth_full", "m1_match_pct", "m1_hit", "m1_recall5", "mrr_m1",
    "m2_truth_full", "m2_match_pct", "m2_hit", "m2_recall5", "mrr_m2",
]


# ---------------------------------------------------------------------------
# Embedding cache (runtime patch of main_app.get_embedding; no file edits).
# Identical text -> identical vector, so A0/A2 scoring is deterministic and
# repeated metric embeddings are free.
# ---------------------------------------------------------------------------
_emb_mem: dict[str, list] = {}
_orig_get_embedding = main_app.get_embedding
N_EMB_API_CALLS = 0


def _load_emb_cache() -> None:
    if not EMB_CACHE.is_file():
        return
    with open(EMB_CACHE, encoding="utf-8") as f:
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
    print(f"[emb-cache] loaded {len(_emb_mem)} cached embeddings", flush=True)


def cached_get_embedding(text):
    global N_EMB_API_CALLS
    h = sha256_text(str(text))
    if h in _emb_mem:
        return _emb_mem[h]
    v = _orig_get_embedding(text)
    v = list(v)
    _emb_mem[h] = v
    append_jsonl(EMB_CACHE, {"h": h, "v": v})
    N_EMB_API_CALLS += 1
    return v


main_app.get_embedding = cached_get_embedding


# ---------------------------------------------------------------------------
# Held-out population: EXACT logic of tests/heldout_narrative_bn_eval.py.
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


# ---------------------------------------------------------------------------
# Scoring: identical to eval_diagnosis_structural.py (0.75 threshold, M1 = 'C'
# findings, M2 = narr_cause chunks; max-cosine match%, R@5, MRR).
# ---------------------------------------------------------------------------
def max_cosine_sim(pred: str, truths: list[str]) -> float:
    return eval_common.max_cosine_sim(pred, truths, main_app.get_embedding)


def recall_at_k(preds: list[str], truths: list[str], k: int) -> bool:
    return eval_common.recall_at_k(preds, truths, k, MATCH_THRESHOLD, main_app.get_embedding)


def mrr_score(preds: list[str], truths: list[str], k: int) -> float:
    return eval_common.mrr_score(preds, truths, k, MATCH_THRESHOLD, main_app.get_embedding)


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
        r5 = "1" if recall_at_k(top5, truths, 5) else "0"
        mrr = mrr_score(top5, truths, 5)
        return pct_s, hit, r5, f"{mrr:.6f}"

    if m1_list:
        m1_pct, m1_hit, m1_r5, m1_mrr = score_metric(m1_list)
    else:
        m1_pct, m1_hit, m1_r5, m1_mrr = ("N/A", "", "", "")
    if m2_chunks:
        m2_pct, m2_hit, m2_r5, m2_mrr = score_metric(m2_chunks)
    else:
        m2_pct, m2_hit, m2_r5, m2_mrr = ("N/A", "", "", "")

    return {
        "ev_id": ev_id,
        "mode": mode,
        "query_source": "narr_accf",
        "query_full_text": query,
        "top_prediction_cause": top1,
        "top_prediction_prob": f"{top1_prob:.6f}",
        "m1_truth_full": eval_common.pipe_join(m1_list) if m1_list else "N/A",
        "m1_match_pct": m1_pct,
        "m1_hit": m1_hit,
        "m1_recall5": m1_r5,
        "mrr_m1": m1_mrr,
        "m2_truth_full": narr_s if m2_chunks else "N/A",
        "m2_match_pct": m2_pct,
        "m2_hit": m2_hit,
        "m2_recall5": m2_r5,
        "mrr_m2": m2_mrr,
    }


def load_retrieved_top50() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not RETRIEVED_TOP50.is_file():
        return out
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


def record_top50(query_ev_id: str, query: str) -> list[str]:
    """Top-50 incident ev_ids by cosine for this query (pure retrieval)."""
    q_emb = main_app.get_embedding(query)
    _, matches = main_app.find_top_matches(q_emb)
    ids: list[str] = []
    for m in matches[:TOP_N_INCIDENTS]:
        if m.get("source") != "incident":
            continue
        eid = m.get("ev_id")
        if eid:
            ids.append(eid)
    append_jsonl(RETRIEVED_TOP50, {"query_ev_id": query_ev_id, "retrieved_ev_ids": ids})
    return ids


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
def run_eval_pass(mode: str, csv_path: Path, held, adj_fn_factory=None) -> None:
    done = eval_common.ev_ids_in_output_csv(csv_path)
    retrieved = load_retrieved_top50()
    todo = [(k, inc, narr) for (k, inc, narr) in held if k not in done]
    print(f"[{mode}] {len(done)} done on disk; {len(todo)} to run", flush=True)
    t0 = time.time()
    for i, (k, inc, narr) in enumerate(todo, 1):
        query = narr[:QUERY_TRUNC]
        if mode == "A0_refined" and k not in retrieved:
            record_top50(k, query)
        adj_fn = adj_fn_factory(k, query) if adj_fn_factory else None
        diag = main_app.diagnose_with_conditional_probabilities(
            query, top_n=10, top_n_incidents=TOP_N_INCIDENTS, score_adjust_fn=adj_fn
        )
        if diag.get("error"):
            print(f"[warn] {k}: {diag.get('error')}", flush=True)
        row = score_row(k, mode, query, inc, diag)
        eval_common.append_csv_row(row, FIELDNAMES, csv_path, OUT_DIR)
        if i % 25 == 0 or i == len(todo):
            el = time.time() - t0
            print(
                f"[{mode}] progress {i}/{len(todo)} "
                f"({el/60:.1f} min elapsed, {el/i:.1f} s/query)",
                flush=True,
            )
    print(f"[{mode}] pass complete -> {csv_path}", flush=True)


def phase_a0(held) -> None:
    run_eval_pass("A0_refined", A0_CSV, held)


def phase_extract(held) -> None:
    window_ds = main_app.refined_dataset  # window dataset = retrieval pool
    retrieved = load_retrieved_top50()
    missing_q = [k for (k, _, _) in held if k not in retrieved]
    if missing_q:
        raise RuntimeError(
            f"{len(missing_q)} queries have no recorded top-50 (run --phase a0 first)"
        )
    needed: set[str] = set()
    for ids in retrieved.values():
        needed.update(ids)
    needed &= set(window_ds.keys())

    n_llm = 0
    # -- train-side incident structs --
    train_cache = load_struct_jsonl(STRUCT_TRAIN_CACHE)
    todo = sorted(needed - set(train_cache.keys()))
    print(
        f"[extract] train incidents needed: {len(needed)}; cached: "
        f"{len(needed) - len(todo)}; to extract: {len(todo)}",
        flush=True,
    )
    t0 = time.time()
    for i, eid in enumerate(todo, 1):
        try:
            st = extract_struct_from_incident_v2(window_ds[eid], eid, model=STRUCT_MODEL)
            append_jsonl(STRUCT_TRAIN_CACHE, {"ev_id": eid, "struct": st, "error": None})
        except Exception as ex:  # noqa: BLE001
            append_jsonl(STRUCT_TRAIN_CACHE, {"ev_id": eid, "struct": None, "error": str(ex)})
            print(f"[extract] ERR {eid}: {ex}", flush=True)
        n_llm += 1
        if i % 25 == 0 or i == len(todo):
            el = time.time() - t0
            print(
                f"[extract-train] {i}/{len(todo)} ({el/60:.1f} min, {el/i:.2f} s/call)",
                flush=True,
            )

    # -- query structs (raw narrative text, same as old eval's get_query_struct) --
    query_cache = load_struct_jsonl(QUERY_STRUCT_CACHE)
    q_todo = [(k, narr) for (k, _, narr) in held if k not in query_cache]
    print(
        f"[extract] queries: {len(held)}; cached: {len(held) - len(q_todo)}; "
        f"to extract: {len(q_todo)}",
        flush=True,
    )
    t0 = time.time()
    for i, (k, narr) in enumerate(q_todo, 1):
        query = narr[:QUERY_TRUNC]
        try:
            st = extract_struct_v2(query, k, model=STRUCT_MODEL)
            append_jsonl(
                QUERY_STRUCT_CACHE,
                {"ev_id": k, "query_hash": sha256_text(query), "struct": st, "error": None},
            )
        except Exception as ex:  # noqa: BLE001
            append_jsonl(
                QUERY_STRUCT_CACHE,
                {"ev_id": k, "query_hash": sha256_text(query), "struct": None,
                 "error": str(ex)},
            )
            print(f"[extract] ERR query {k}: {ex}", flush=True)
        n_llm += 1
        if i % 25 == 0 or i == len(q_todo):
            el = time.time() - t0
            print(
                f"[extract-query] {i}/{len(q_todo)} ({el/60:.1f} min, {el/i:.2f} s/call)",
                flush=True,
            )
    print(f"[extract] complete; {n_llm} LLM calls this run", flush=True)


def phase_a2(held) -> None:
    struct_by_eid = load_struct_jsonl(STRUCT_TRAIN_CACHE)
    query_structs = load_struct_jsonl(QUERY_STRUCT_CACHE)
    if not struct_by_eid:
        raise RuntimeError(f"No train structs in {STRUCT_TRAIN_CACHE}; run --phase extract")
    n_missing_q = sum(1 for (k, _, _) in held if k not in query_structs)
    print(
        f"[A2] train structs: {len(struct_by_eid)}; query structs: "
        f"{len(query_structs)} ({n_missing_q} queries missing -> cosine fallback)",
        flush=True,
    )

    def adj_fn_factory(k: str, query: str):
        q_struct = query_structs.get(k)
        return make_score_adjust_fn(
            q_struct, struct_by_eid, ALPHA, similarity_fn=structural_similarity_v2
        )

    run_eval_pass("A2_refined", A2_CSV, held, adj_fn_factory=adj_fn_factory)


def phase_analyze() -> None:
    import analyze_a0_vs_a2 as az

    out_dir = OUT_DIR / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = az.load_and_join(A0_CSV, A2_CSV)
    # pandas reads the literal "N/A" truth marker as NaN, so load_and_join's
    # string filter misses those rows; drop them here (M1-eligible rows only).
    df = df[df["m1_truth_full_A0"].notna()].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Joined dataframe is empty")
    summary = az.compute_summary(df)
    az.write_text_summary(summary, out_dir / "summary.txt")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    az.plot_bar_with_ci(summary, out_dir / "bar_with_ci.png")
    az.plot_paired_delta(df, out_dir / "paired_delta.png")
    az.plot_match_pct_distribution(df, out_dir / "match_pct_distribution.png")
    az.plot_mcnemar_table(summary, out_dir / "mcnemar_table.png")
    # Some rows have no top-1 prediction (empty m1_hit) -> NaN; the summary
    # treats these as misses (fillna(0)), so do the same for the rank plot.
    num_cols = [c for c in df.columns if c not in ("ev_id", "m1_truth_full_A0", "m1_truth_full_A2")]
    df_plot = df.copy()
    df_plot[num_cols] = df_plot[num_cols].fillna(0)
    az.plot_rank_distribution(df_plot, out_dir / "rank_distribution.png")
    df.to_csv(out_dir / "paired_per_incident.csv", index=False)
    print((out_dir / "summary.txt").read_text())
    print(f"Wrote analysis artifacts to: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("a0", "extract", "a2", "analyze"), required=True)
    ap.add_argument("--limit", type=int, default=None, help="Cap queries (smoke test)")
    args = ap.parse_args()

    if not main_app.DATA_LOADED:
        raise RuntimeError("Window retrieval index not loaded")
    if not USE_ZHANG_WINDOW:
        raise RuntimeError(
            "config did not select the 1982-2006 window index; "
            "unset NTSB_FULL_CORPUS / NTSB_USE_TRAIN_INDEX"
        )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _load_emb_cache()

    held = load_heldout()
    print(
        f"held-out queries: {len(held)}; retrieval index: "
        f"{len(main_app.embeddings_map)} embedded incidents "
        f"({len(main_app.refined_dataset)} window accidents)",
        flush=True,
    )
    if args.limit:
        held = held[: args.limit]

    if args.phase == "a0":
        phase_a0(held)
    elif args.phase == "extract":
        phase_extract(held)
    elif args.phase == "a2":
        phase_a2(held)
    else:
        phase_analyze()
    print(f"[done] embedding API calls this run: {N_EMB_API_CALLS}", flush=True)


if __name__ == "__main__":
    main()

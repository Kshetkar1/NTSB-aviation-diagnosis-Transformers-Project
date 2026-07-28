#!/usr/bin/env python3
"""
FINAL exhaustive structural-mapping experiment (see outputs/structmap_final_verdict/).

Tests the levers the two prior null experiments left untested:
  V1  best-match selection (bypass cluster/LTP aggregation)   [phase 0, free]
  V2  struct-weighted severity vote (paper's actual task)     [phase 0, free]
  V3  wider pool (top-200) -> struct rerank -> trim to 50     [phase 1]
  V4  struct-first retrieval over ALL window incidents        [phase 1]
  V5  hybrid narrative+chain embedding retrieval              [phase 1]

Subcommands:
  extract    -- gpt-4o-mini chain extraction for embedded window incidents not
                in struct_train_v2_refined.jsonl; appends to NEW cache
                struct_train_v2_final.jsonl (~488 calls).
  structmat  -- 296x1703 struct-sim matrix (queries x window incidents) -> npz.
  chainemb   -- text-embedding-3-small embeddings of rendered chain texts for
                all window incidents + queries -> npz (vectors cached to
                emb_cache_final.jsonl).
  v1         -- best-match selection variants (rankers cos/struct/fused x k=1,3,5)
                -> eval_diagnosis_bm_*.csv (same M1 metric as the retest).
  diag       -- one pool-variant diagnosis pass (v3_a2, v3_a4, v3_rrf,
                v4_struct, v4_rrf, v5_w0.3, v5_w0.5) through the identical
                cluster+LTP pipeline -> eval_diagnosis_<variant>.csv.
  sev        -- severity-vote per-item predictions for one variant
                -> severity_<variant>.json.
  analyze    -- paired stats (bootstrap CI on delta avg match%, McNemar exact,
                threshold sweep) of every eval_diagnosis_*.csv vs the retest A0.
  sevanalyze -- severity head-to-head table: every severity_*.json vs
                narr-sev / full / soft+stated / LR baselines (McNemar + bootstrap).

Protocol identical to the retest (paper held-out split, 296 queries, window
retrieval index, M1 = Cause_Factor 'C' findings, 0.75 max-cosine threshold).
Never truncates existing caches; new embeddings append to emb_cache_final.jsonl.

Run from repo root with
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      Testing_Structural_Mapping_Slides/scripts/final_verdict.py <cmd> [...]
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
sys.path.insert(0, str(PROJECT_ROOT / "tests"))
_METRICS = PROJECT_ROOT / "data" / "Testing_Data_Metrics" / "scripts"
sys.path.insert(0, str(_METRICS))

import eval_common  # noqa: E402
from config import REFINED_DATA_PATH, WINDOW_DATA_PATH, USE_ZHANG_WINDOW  # noqa: E402

import main_app  # noqa: E402

from io_cache import append_jsonl, load_struct_jsonl, sha256_text  # noqa: E402
from struct_score_v2 import structural_similarity as struct_sim_v2  # noqa: E402

MATCH_THRESHOLD = 0.75
QUERY_TRUNC = 4000
TOP_N = 50
RRF_K = 60
STRUCT_MODEL = "gpt-4o-mini"

CACHE_DIR = PROJECT_ROOT / "Testing_Structural_Mapping" / "cache_refined"
OUT_DIR = PROJECT_ROOT / "outputs" / "structmap_final_verdict"
STRUCT_TRAIN_OLD = CACHE_DIR / "struct_train_v2_refined.jsonl"   # READ-ONLY
STRUCT_TRAIN_FINAL = CACHE_DIR / "struct_train_v2_final.jsonl"   # new appends
QUERY_STRUCT_CACHE = CACHE_DIR / "query_struct_v2_refined.jsonl"  # READ-ONLY
EMB_CACHES_RO = [CACHE_DIR / "emb_cache.jsonl", CACHE_DIR / "emb_cache_improve.jsonl"]
EMB_CACHE_FINAL = CACHE_DIR / "emb_cache_final.jsonl"            # new appends
RETRIEVED_TOP50 = CACHE_DIR / "retrieved_top50.jsonl"
PAIR_SCORES_IMPROVE = CACHE_DIR / "pair_scores_improve.jsonl"

STRUCT_MAT_NPZ = OUT_DIR / "struct_sim_matrix.npz"
CHAIN_EMB_NPZ = OUT_DIR / "chain_emb.npz"

A0_CSV = PROJECT_ROOT / "outputs" / "structmap_refined" / "eval_diagnosis_A0_refined.csv"
HELDOUT_PER_ITEM = PROJECT_ROOT / "outputs" / "heldout_per_item.json"
LR_PER_ITEM = PROJECT_ROOT / "outputs" / "lr_per_item.json"

FIELDNAMES = [
    "ev_id", "mode", "query_source", "query_full_text",
    "top_prediction_cause", "top_prediction_prob",
    "m1_truth_full", "m1_match_pct", "m1_hit", "m1_recall5", "mrr_m1",
    "m2_truth_full", "m2_match_pct", "m2_hit", "m2_recall5", "mrr_m2",
]
SWEEP_THRESHOLDS = [0.75, 0.60, 0.55, 0.50, 0.45, 0.40]


# ---------------------------------------------------------------------------
# Embedding cache: prior caches read-only, new vectors -> emb_cache_final.jsonl.
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
    total = 0
    for p in EMB_CACHES_RO + [EMB_CACHE_FINAL]:
        total += _load_emb_file(p)
    print(f"[emb-cache] loaded {len(_emb_mem)} vectors from {total} rows", flush=True)


def cached_get_embedding(text):
    global N_EMB_API_CALLS
    h = sha256_text(str(text))
    if h in _emb_mem:
        return _emb_mem[h]
    v = list(_orig_get_embedding(text))
    _emb_mem[h] = v
    append_jsonl(EMB_CACHE_FINAL, {"h": h, "v": v})
    N_EMB_API_CALLS += 1
    return v


main_app.get_embedding = cached_get_embedding


# ---------------------------------------------------------------------------
# Shared loading
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


def load_struct_merged() -> dict[str, dict]:
    """Chains: retest cache + final cache (final wins on conflict; there are none)."""
    out = load_struct_jsonl(STRUCT_TRAIN_OLD)
    out.update(load_struct_jsonl(STRUCT_TRAIN_FINAL))
    return out


def window_index():
    """(eids list, E matrix, eid -> embeddings_map entry) for incident rows."""
    eids, rows, eid_to_match = [], [], {}
    for i, info in enumerate(main_app.embeddings_map):
        if info.get("source") != "incident":
            continue
        eid = info.get("ev_id")
        if not eid or eid in eid_to_match:
            continue
        eids.append(eid)
        rows.append(i)
        eid_to_match[eid] = info
    E = np.asarray(main_app.embeddings, dtype=np.float64)[rows]
    return eids, E, eid_to_match


def query_embedding_from_cache(query: str) -> np.ndarray:
    h = sha256_text(str(query))
    if h not in _emb_mem:
        raise RuntimeError("query embedding not cached (retest should have cached all 296)")
    return np.asarray(_emb_mem[h], dtype=np.float64)


def candidate_causes(match: dict) -> list[str]:
    """Recorded causes of a retrieved incident, with the same truncation-repair
    logic as main_app.calculate_cause_probabilities_per_cluster."""
    dd = match.get("diagnostic_data") or match.get("bayesian_data", {})
    if not dd.get("has_diagnostic_data"):
        return []
    causes = list(dd.get("all_causes", []))
    narr_cause = dd.get("narr_cause")
    if narr_cause and len(narr_cause) > 100:
        causes = [c for c in causes if not (len(c) == 100 and narr_cause.startswith(c))]
        if narr_cause not in causes:
            causes.append(narr_cause)
    return [c.strip().lower() for c in causes if c and c.strip()]


# ---------------------------------------------------------------------------
# Scoring row (identical to the retest / improve runners)
# ---------------------------------------------------------------------------
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
        pct = (eval_common.max_cosine_sim(top1, truths, main_app.get_embedding) * 100.0
               if top1 else float("nan"))
        pct_s = "" if math.isnan(pct) else f"{pct:.4f}"
        hit = ("" if not top1 or math.isnan(pct)
               else ("1" if pct / 100.0 >= MATCH_THRESHOLD else "0"))
        r5 = "1" if eval_common.recall_at_k(top5, truths, 5, MATCH_THRESHOLD, main_app.get_embedding) else "0"
        mrr = eval_common.mrr_score(top5, truths, 5, MATCH_THRESHOLD, main_app.get_embedding)
        return pct_s, hit, r5, f"{mrr:.6f}"

    m1 = score_metric(m1_list) if m1_list else ("N/A", "", "", "")
    m2 = score_metric(m2_chunks) if m2_chunks else ("N/A", "", "", "")

    return {
        "ev_id": ev_id, "mode": mode, "query_source": "narr_accf",
        "query_full_text": query,
        "top_prediction_cause": top1, "top_prediction_prob": f"{top1_prob:.6f}",
        "m1_truth_full": eval_common.pipe_join(m1_list) if m1_list else "N/A",
        "m1_match_pct": m1[0], "m1_hit": m1[1], "m1_recall5": m1[2], "mrr_m1": m1[3],
        "m2_truth_full": narr_s if m2_chunks else "N/A",
        "m2_match_pct": m2[0], "m2_hit": m2[1], "m2_recall5": m2[2], "mrr_m2": m2[3],
    }


# ---------------------------------------------------------------------------
# extract: chains for embedded window incidents missing from the retest cache
# ---------------------------------------------------------------------------
def cmd_extract(args) -> None:
    from extract_struct_v2 import extract_struct_from_incident_v2

    eids, _, _ = window_index()
    have = set(load_struct_merged().keys())
    todo = sorted(set(eids) - have)
    if args.limit:
        todo = todo[: args.limit]
    print(f"[extract] embedded incidents: {len(eids)}; cached: {len(have)}; "
          f"to extract: {len(todo)}", flush=True)
    t0 = time.time()
    n_calls = 0
    for i, eid in enumerate(todo, 1):
        try:
            st = extract_struct_from_incident_v2(
                main_app.refined_dataset[eid], eid, model=STRUCT_MODEL)
            append_jsonl(STRUCT_TRAIN_FINAL, {"ev_id": eid, "struct": st, "error": None})
        except Exception as ex:  # noqa: BLE001
            append_jsonl(STRUCT_TRAIN_FINAL, {"ev_id": eid, "struct": None, "error": str(ex)})
            print(f"[extract] ERR {eid}: {ex}", flush=True)
        n_calls += 1
        if i % 25 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"[extract] {i}/{len(todo)} ({el/60:.1f} min, {el/i:.2f} s/call)", flush=True)
    print(f"[extract] complete; {n_calls} gpt-4o-mini calls this run", flush=True)


# ---------------------------------------------------------------------------
# structmat: 296x1703 chain-similarity matrix
# ---------------------------------------------------------------------------
def cmd_structmat(args) -> None:
    held = load_heldout()
    eids, _, _ = window_index()
    chains = load_struct_merged()
    q_structs = load_struct_jsonl(QUERY_STRUCT_CACHE)

    n_missing_cand = sum(1 for e in eids if e not in chains)
    n_missing_q = sum(1 for (k, _, _) in held if k not in q_structs)
    print(f"[structmat] queries {len(held)} (missing struct {n_missing_q}); "
          f"candidates {len(eids)} (missing struct {n_missing_cand})", flush=True)

    S = np.zeros((len(held), len(eids)), dtype=np.float32)
    mask = np.zeros_like(S, dtype=bool)  # True = real similarity computed
    t0 = time.time()
    for qi, (k, _, _) in enumerate(held):
        qs = q_structs.get(k)
        if not qs:
            continue
        for ci, eid in enumerate(eids):
            cand = chains.get(eid)
            if not cand:
                continue
            S[qi, ci] = struct_sim_v2(qs, cand)
            mask[qi, ci] = True
        if (qi + 1) % 50 == 0:
            print(f"[structmat] {qi+1}/{len(held)} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(
        STRUCT_MAT_NPZ, S=S, mask=mask,
        query_eids=np.array([k for (k, _, _) in held]),
        cand_eids=np.array(eids),
    )
    print(f"[structmat] wrote {STRUCT_MAT_NPZ} "
          f"(coverage {mask.mean()*100:.1f}%)", flush=True)


def load_structmat(held, eids):
    d = np.load(STRUCT_MAT_NPZ, allow_pickle=False)
    assert list(d["query_eids"]) == [k for (k, _, _) in held], "query order mismatch"
    assert list(d["cand_eids"]) == list(eids), "candidate order mismatch"
    return d["S"].astype(np.float64), d["mask"]


# ---------------------------------------------------------------------------
# chainemb: embeddings of rendered chain texts
# ---------------------------------------------------------------------------
def render_chain_text(struct: dict | None) -> str:
    if not struct:
        return "no causal chain"
    steps = struct.get("causal_chain") or []
    parts = []
    for s in steps:
        parts.append(f"{s.get('role','unknown')}: {s.get('element','')} "
                     f"[{s.get('system','aircraft')}/{s.get('mechanism','unknown')}]")
    txt = " -> ".join(parts) if parts else "no causal chain"
    fp = struct.get("failure_pattern")
    if fp and fp != "unknown":
        txt += f" | pattern: {fp}"
    return txt


def cmd_chainemb(args) -> None:
    held = load_heldout()
    eids, _, _ = window_index()
    chains = load_struct_merged()
    q_structs = load_struct_jsonl(QUERY_STRUCT_CACHE)

    inc_vecs, t0 = [], time.time()
    for i, eid in enumerate(eids, 1):
        inc_vecs.append(cached_get_embedding(render_chain_text(chains.get(eid))))
        if i % 200 == 0:
            print(f"[chainemb] incidents {i}/{len(eids)} "
                  f"(api calls so far {N_EMB_API_CALLS})", flush=True)
    q_vecs = []
    for i, (k, _, _) in enumerate(held, 1):
        q_vecs.append(cached_get_embedding(render_chain_text(q_structs.get(k))))
    inc_arr = np.asarray(inc_vecs, dtype=np.float32)
    q_arr = np.asarray(q_vecs, dtype=np.float32)
    # L2-normalize each part (OpenAI vectors are ~normalized already; be exact)
    inc_arr /= np.linalg.norm(inc_arr, axis=1, keepdims=True)
    q_arr /= np.linalg.norm(q_arr, axis=1, keepdims=True)
    np.savez_compressed(
        CHAIN_EMB_NPZ, inc=inc_arr, q=q_arr,
        query_eids=np.array([k for (k, _, _) in held]), cand_eids=np.array(eids),
    )
    print(f"[chainemb] wrote {CHAIN_EMB_NPZ}; embedding API calls this run: "
          f"{N_EMB_API_CALLS} ({time.time()-t0:.0f}s)", flush=True)


def load_chainemb(held, eids):
    d = np.load(CHAIN_EMB_NPZ, allow_pickle=False)
    assert list(d["query_eids"]) == [k for (k, _, _) in held]
    assert list(d["cand_eids"]) == list(eids)
    return d["inc"].astype(np.float64), d["q"].astype(np.float64)


# ---------------------------------------------------------------------------
# V1: best-match selection (bypass cluster/LTP aggregation)
# ---------------------------------------------------------------------------
def load_pair_scores(path: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q, pairs = row.get("query_ev_id"), row.get("pairs")
            if q and isinstance(pairs, list):
                out[q] = pairs
    return out


def cmd_v1(args) -> None:
    held = load_heldout()
    pair_scores = load_pair_scores(PAIR_SCORES_IMPROVE)
    _, _, eid_to_match = window_index()

    rankers = {
        "cos": lambda cos, s: cos,
        "struct": lambda cos, s: s,
        "fused": lambda cos, s: max(0.0, cos) * math.exp(2.0 * s),
    }
    ks = [1, 3, 5]
    variants = [(rn, k) for rn in rankers for k in ks]
    if args.variant:
        want = set(args.variant.split(","))
        variants = [(rn, k) for (rn, k) in variants if f"bm_{rn}_k{k}" in want]

    for rn, k in variants:
        tag = f"bm_{rn}_k{k}"
        csv_path = OUT_DIR / f"eval_diagnosis_{tag}.csv"
        done = eval_common.ev_ids_in_output_csv(csv_path)
        todo = [(kk, inc, narr) for (kk, inc, narr) in held if kk not in done]
        print(f"[{tag}] {len(done)} done; {len(todo)} to run", flush=True)
        rank_fn = rankers[rn]
        t0 = time.time()
        for i, (kk, inc, narr) in enumerate(todo, 1):
            query = narr[:QUERY_TRUNC]
            pairs = pair_scores.get(kk, [])
            scored = []
            for p in pairs:
                cos = float(p["cos"])
                s = float(p["struct"]) if p["struct"] is not None else 0.0
                scored.append((p["ev_id"], rank_fn(cos, s)))
            scored.sort(key=lambda t: -t[1])
            top = scored[:k]
            # weighted vote over the recorded causes of the top-k candidates
            weights: dict[str, float] = {}
            order: dict[str, int] = {}
            for rank_i, (eid, w) in enumerate(top):
                m = eid_to_match.get(eid)
                if not m:
                    continue
                for j, c in enumerate(candidate_causes(m)):
                    weights[c] = weights.get(c, 0.0) + max(w, 1e-9)
                    order.setdefault(c, rank_i * 1000 + j)
            ranked = sorted(weights.items(), key=lambda t: (-t[1], order[t[0]]))
            tot = sum(w for _, w in ranked) or 1.0
            diag = {"weighted_causes": [
                {"cause": c, "probability": w / tot} for c, w in ranked]}
            row = score_row(kk, tag, query, inc, diag)
            eval_common.append_csv_row(row, FIELDNAMES, csv_path, OUT_DIR)
            if i % 50 == 0 or i == len(todo):
                print(f"[{tag}] {i}/{len(todo)} ({(time.time()-t0)/60:.1f} min)", flush=True)
        print(f"[{tag}] complete -> {csv_path}", flush=True)
    print(f"[v1] embedding API calls this run: {N_EMB_API_CALLS}", flush=True)


# ---------------------------------------------------------------------------
# Pool builders for V3/V4/V5 (and severity pools)
# ---------------------------------------------------------------------------
def build_pool(variant: str, cos: np.ndarray, s: np.ndarray,
               chain_cos: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, weights) of the 50-incident pool for this variant."""
    if variant.startswith("v3_a"):
        a = float(variant.split("v3_a")[1])
        idx200 = np.argsort(-cos)[:200]
        w = np.maximum(0.0, cos[idx200]) * np.exp(a * s[idx200])
        sel = np.argsort(-w)[:TOP_N]
        return idx200[sel], w[sel]
    if variant == "v3_rrf":
        idx200 = np.argsort(-cos)[:200]
        rc = np.empty(len(idx200)); rc[np.argsort(-cos[idx200])] = np.arange(1, len(idx200) + 1)
        rs = np.empty(len(idx200)); rs[np.argsort(-s[idx200])] = np.arange(1, len(idx200) + 1)
        w = 1.0 / (RRF_K + rc) + 1.0 / (RRF_K + rs)
        sel = np.argsort(-w)[:TOP_N]
        return idx200[sel], w[sel]
    if variant == "v4_struct":
        idx = np.argsort(-s)[:TOP_N]
        return idx, s[idx] + 1e-9
    if variant == "v4_rrf":
        n = len(cos)
        rc = np.empty(n); rc[np.argsort(-cos)] = np.arange(1, n + 1)
        rs = np.empty(n); rs[np.argsort(-s)] = np.arange(1, n + 1)
        w = 1.0 / (RRF_K + rc) + 1.0 / (RRF_K + rs)
        idx = np.argsort(-w)[:TOP_N]
        return idx, w[idx]
    if variant.startswith("v5_w"):
        w_chain = float(variant.split("v5_w")[1])
        assert chain_cos is not None
        score = (1.0 - w_chain) * cos + w_chain * chain_cos
        idx = np.argsort(-score)[:TOP_N]
        return idx, score[idx]
    if variant == "cos":  # baseline pool (sanity / severity)
        idx = np.argsort(-cos)[:TOP_N]
        return idx, cos[idx]
    raise ValueError(f"unknown pool variant: {variant}")


def cmd_diag(args) -> None:
    variant = args.variant
    held = load_heldout()
    eids, E, eid_to_match = window_index()
    S, _ = load_structmat(held, eids)
    chain_inc = chain_q = None
    if variant.startswith("v5_"):
        chain_inc, chain_q = load_chainemb(held, eids)

    csv_path = OUT_DIR / f"eval_diagnosis_{variant}.csv"
    pool_json = OUT_DIR / f"pool_overlap_{variant}.json"
    done = eval_common.ev_ids_in_output_csv(csv_path)
    todo = [(k, inc, narr) for (k, inc, narr) in held if k not in done]
    print(f"[{variant}] {len(done)} done; {len(todo)} to run", flush=True)

    overlaps = {}
    if pool_json.is_file():
        overlaps = json.loads(pool_json.read_text())

    t0 = time.time()
    for i, (k, inc, narr) in enumerate(todo, 1):
        qi = [j for j, (kk, _, _) in enumerate(held) if kk == k][0]
        query = narr[:QUERY_TRUNC]
        q_emb = query_embedding_from_cache(query)
        cos = E @ q_emb
        s = S[qi]
        ccos = None
        if chain_inc is not None:
            ccos = chain_inc @ chain_q[qi]
        idx, w = build_pool(variant, cos, s, ccos)

        # pool overlap vs cosine top-50 (diagnostic)
        base = set(np.argsort(-cos)[:TOP_N].tolist())
        overlaps[k] = len(base & set(idx.tolist()))

        top_scores = [float(x) for x in w]
        top_matches = [eid_to_match[eids[j]] for j in idx]
        clusters = main_app.cluster_incidents_by_type(top_scores, top_matches, TOP_N)
        if not clusters:
            diag = {"error": "no clusters", "weighted_causes": []}
        else:
            analysis = main_app.calculate_cause_probabilities_per_cluster(clusters)
            diag = main_app.calculate_chain_rule_diagnosis(clusters, analysis)
        row = score_row(k, variant, query, inc, diag)
        eval_common.append_csv_row(row, FIELDNAMES, csv_path, OUT_DIR)
        if i % 25 == 0 or i == len(todo):
            print(f"[{variant}] {i}/{len(todo)} ({(time.time()-t0)/60:.1f} min, "
                  f"emb api {N_EMB_API_CALLS})", flush=True)
            pool_json.write_text(json.dumps(overlaps))
    pool_json.write_text(json.dumps(overlaps))
    ov = np.array(list(overlaps.values()))
    print(f"[{variant}] complete -> {csv_path}; pool overlap with cosine top-50: "
          f"mean {ov.mean():.1f}/50, min {ov.min()}", flush=True)


# ---------------------------------------------------------------------------
# V2: severity vote (the paper's actual reported task)
# ---------------------------------------------------------------------------
def cmd_sev(args) -> None:
    """Variants:
      sev_cos_k100      exact narr-sev protocol reproduction (weight = cosine)
      sev_cos_k50       same, k=50
      sev_struct_k50 / sev_struct_k100          weight = struct sim
      sev_fused_a{1,2,4}_k{50,100}              weight = max(0,cos)*e^(a*s)
      sev_pool_v4struct  pool = struct-first top-50, weight = struct
      sev_pool_v4rrf     pool = RRF(cos, struct) over all, weight = rrf
      sev_pool_v3a2      pool = top-200 cos -> fused a2 -> top-50, weight = fused
      sev_pool_v5w0.3 / sev_pool_v5w0.5         pool + weight = hybrid embedding
    """
    import prognosis as pg
    import query_to_bn as qb

    variant = args.variant
    held = load_heldout()
    eids, E, _ = window_index()
    ds = main_app.refined_dataset  # window dataset (same file pg.load_dataset reads)

    need_struct = not variant.startswith("sev_cos")
    S = None
    if need_struct:
        S, _mask = load_structmat(held, eids)
    chain_inc = chain_q = None
    if "v5w" in variant:
        chain_inc, chain_q = load_chainemb(held, eids)

    inj_idx = {c: i for i, c in enumerate(qb.INJ_CODES)}
    dmg_idx = {c: i for i, c in enumerate(qb.DMG_CODES)}

    def pool_and_weights(qi: int, cos: np.ndarray):
        if variant.startswith("sev_cos_k") or variant.startswith("sev_struct_k") \
                or variant.startswith("sev_fused_"):
            k = int(variant.rsplit("_k", 1)[1])
            idx = np.argsort(-cos)[:k]
            if variant.startswith("sev_cos"):
                w = cos[idx]
            elif variant.startswith("sev_struct"):
                w = S[qi][idx] + 1e-9
            else:
                a = float(variant.split("sev_fused_a")[1].split("_")[0])
                w = np.maximum(0.0, cos[idx]) * np.exp(a * S[qi][idx])
            return idx, w
        pool_map = {"sev_pool_v4struct": "v4_struct", "sev_pool_v4rrf": "v4_rrf",
                    "sev_pool_v3a2": "v3_a2", "sev_pool_v5w0.3": "v5_w0.3",
                    "sev_pool_v5w0.5": "v5_w0.5"}
        pv = pool_map[variant]
        ccos = chain_inc @ chain_q[qi] if chain_inc is not None else None
        return build_pool(pv, cos, S[qi], ccos)

    items = []
    t0 = time.time()
    for qi, (k, inc, narr) in enumerate(held):
        text = narr[:QUERY_TRUNC]
        inj_y = {"FATL": 0, "SERS": 1, "MINR": 2, "NONE": 3}[pg.zhang_injury_code(inc)]
        from bn_upgraded import DMG_BY_CODE
        dmg_code = str(inc.get("damage") or "").upper()
        dmg_y = DMG_BY_CODE.get(dmg_code)

        q_emb = query_embedding_from_cache(text)
        cos = E @ q_emb
        idx, w = pool_and_weights(qi, cos)

        inj = [0.5] * 4
        dmg = [0.5] * 4
        for j, wt in zip(idx, w):
            inc_c = ds.get(eids[j])
            if inc_c is None:
                continue
            wt = float(wt)
            inj[inj_idx[pg.zhang_injury_code(inc_c)]] += wt
            d = str(inc_c.get("damage") or "").upper()
            if d in dmg_idx:
                dmg[dmg_idx[d]] += wt
        ni = np.array(inj) / sum(inj)
        nd = np.array(dmg) / sum(dmg)

        sev = qb.severity_virtual_evidence(text, ds)
        if "injury" in sev:
            v = ni * np.array(sev["injury"]); ni = v / v.sum()
        if "damage" in sev:
            v = nd * np.array(sev["damage"]); nd = v / v.sum()

        items.append({"id": k, "inj_true": inj_y, "dmg_true": dmg_y,
                      "inj_pred": int(ni.argmax()), "dmg_pred": int(nd.argmax()),
                      "inj_dist": [float(x) for x in ni],
                      "dmg_dist": [float(x) for x in nd]})
        if (qi + 1) % 50 == 0:
            print(f"[{variant}] {qi+1}/{len(held)} ({time.time()-t0:.0f}s)", flush=True)

    out = OUT_DIR / f"severity_{variant}.json"
    out.write_text(json.dumps({"variant": variant, "items": items}, indent=1))
    acc_i = np.mean([it["inj_pred"] == it["inj_true"] for it in items])
    accd = [it for it in items if it["dmg_true"] is not None]
    acc_d = np.mean([it["dmg_pred"] == it["dmg_true"] for it in accd])
    print(f"[{variant}] injury acc {acc_i*100:.1f}%  damage acc {acc_d*100:.1f}% "
          f"(n={len(items)}/{len(accd)}) -> {out}", flush=True)


# ---------------------------------------------------------------------------
# analyze: paired diagnosis stats vs A0
# ---------------------------------------------------------------------------
def analyze_variant_csv(a0_csv: Path, var_csv: Path) -> dict:
    import analyze_a0_vs_a2 as az
    import pandas as pd

    df = az.load_and_join(a0_csv, var_csv)
    df = df[df["m1_truth_full_A0"].notna()].reset_index(drop=True)
    summary = az.compute_summary(df)

    a0p = df["m1_match_pct_A0"].fillna(0).astype(float).to_numpy()
    vp = df["m1_match_pct_A2"].fillna(0).astype(float).to_numpy()
    sweep = []
    for t in SWEEP_THRESHOLDS:
        h0 = (a0p >= t * 100.0).astype(int)
        h2 = (vp >= t * 100.0).astype(int)
        b = int(((h0 == 1) & (h2 == 0)).sum())
        c = int(((h0 == 0) & (h2 == 1)).sum())
        sweep.append({"threshold": t, "a0_hits": int(h0.sum()), "var_hits": int(h2.sum()),
                      "a0_only": b, "var_only": c, "mcnemar_p": az.mcnemar_exact(b, c)})
    summary["threshold_sweep"] = sweep

    a0_df = pd.read_csv(a0_csv, low_memory=False)[["ev_id", "top_prediction_cause"]]
    v_df = pd.read_csv(var_csv, low_memory=False)[["ev_id", "top_prediction_cause"]]
    m = a0_df.merge(v_df, on="ev_id", suffixes=("_a0", "_v"))
    summary["n_top1_changed"] = int(
        (m["top_prediction_cause_a0"].fillna("") != m["top_prediction_cause_v"].fillna("")).sum())
    summary["n_rows"] = int(len(m))
    return summary


def passes_gate(summary: dict) -> bool:
    """Pre-registered gate: McNemar p < 0.2 at any threshold 0.40-0.75 with
    positive net wins, OR delta avg-match% CI excludes 0 on the positive side."""
    for sw in summary["threshold_sweep"]:
        if sw["var_only"] > sw["a0_only"] and sw["mcnemar_p"] < 0.2:
            return True
    lo, hi = summary["delta_avg_match_pct"]["ci95"]
    return lo > 0


def cmd_analyze(args) -> None:
    results = {}
    for csv_path in sorted(OUT_DIR.glob("eval_diagnosis_*.csv")):
        tag = csv_path.stem.replace("eval_diagnosis_", "")
        try:
            summary = analyze_variant_csv(A0_CSV, csv_path)
        except Exception as ex:  # noqa: BLE001
            print(f"[analyze] {tag}: FAILED ({ex})", flush=True)
            continue
        summary["passes_gate"] = passes_gate(summary)
        results[tag] = summary
        d = summary["delta_avg_match_pct"]
        best_sweep = min(summary["threshold_sweep"], key=lambda s: s["mcnemar_p"])
        print(f"[analyze] {tag:>16}: n={summary['n']} "
              f"avg {summary['A2']['avg_match_pct']:.2f} vs A0 {summary['A0']['avg_match_pct']:.2f} "
              f"delta {d['mean']:+.2f}pp CI[{d['ci95'][0]:+.2f},{d['ci95'][1]:+.2f}] "
              f"minP {best_sweep['mcnemar_p']:.3f}@{best_sweep['threshold']} "
              f"(var {best_sweep['var_only']} vs a0 {best_sweep['a0_only']}) "
              f"top1chg {summary['n_top1_changed']}/{summary['n_rows']} "
              f"GATE={'PASS' if summary['passes_gate'] else 'fail'}", flush=True)
    out = OUT_DIR / "final_variants_summary.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[analyze] wrote {out}", flush=True)


# ---------------------------------------------------------------------------
# sevanalyze: severity head-to-head vs baselines
# ---------------------------------------------------------------------------
def _boot_acc_ci(correct: np.ndarray, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(correct)
    boots = np.array([correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _boot_paired_delta(c_var: np.ndarray, c_base: np.ndarray, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(c_var)
    d = c_var.astype(float) - c_base.astype(float)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def cmd_sevanalyze(args) -> None:
    import analyze_a0_vs_a2 as az

    per = json.loads(HELDOUT_PER_ITEM.read_text())
    lr = json.loads(LR_PER_ITEM.read_text())
    base_by_id = {it["id"]: it for it in per["items"]}
    lr_by_id = {it["id"]: it for it in lr["items"]}

    baselines = {
        "narr-sev": ("narr-sev:inj_pred", "narr-sev:dmg_pred", base_by_id),
        "full-BN": ("full:inj_pred", "full:dmg_pred", base_by_id),
        "soft+stated": ("soft+stated:inj_pred", "soft+stated:dmg_pred", base_by_id),
        "lr": ("lr:inj_pred", "lr:dmg_pred", lr_by_id),
    }

    results = {}
    for f in sorted(OUT_DIR.glob("severity_sev_*.json")):
        d = json.loads(f.read_text())
        tag = d["variant"]
        items = d["items"]
        ids = [it["id"] for it in items]
        inj_true = np.array([it["inj_true"] for it in items])
        inj_pred = np.array([it["inj_pred"] for it in items])
        dmg_ok = [it for it in items if it["dmg_true"] is not None]
        dmg_true = np.array([it["dmg_true"] for it in dmg_ok])
        dmg_pred = np.array([it["dmg_pred"] for it in dmg_ok])
        dmg_ids = [it["id"] for it in dmg_ok]

        # sanity: truths must agree with the paper's per-item file
        for it in items[:5]:
            b = base_by_id.get(it["id"])
            assert b is None or (b["inj_true"] == it["inj_true"]), "injury truth mismatch"

        ci = inj_pred == inj_true
        cd = dmg_pred == dmg_true
        entry = {
            "n_inj": len(ci), "n_dmg": len(cd),
            "inj_acc": float(ci.mean()), "inj_acc_ci95": _boot_acc_ci(ci),
            "dmg_acc": float(cd.mean()), "dmg_acc_ci95": _boot_acc_ci(cd),
            "vs": {},
        }
        for bname, (ik, dk, src) in baselines.items():
            b_inj = np.array([src[i][ik] for i in ids])
            b_ci = b_inj == inj_true
            bi_only = int((b_ci & ~ci).sum()); vi_only = int((~b_ci & ci).sum())
            b_dmg = np.array([src[i][dk] for i in dmg_ids])
            b_cd = b_dmg == dmg_true
            bd_only = int((b_cd & ~cd).sum()); vd_only = int((~b_cd & cd).sum())
            di, di_lo, di_hi = _boot_paired_delta(ci, b_ci)
            dd, dd_lo, dd_hi = _boot_paired_delta(cd, b_cd)
            entry["vs"][bname] = {
                "inj": {"base_only": bi_only, "var_only": vi_only,
                        "mcnemar_p": az.mcnemar_exact(bi_only, vi_only),
                        "delta_acc": di, "ci95": [di_lo, di_hi]},
                "dmg": {"base_only": bd_only, "var_only": vd_only,
                        "mcnemar_p": az.mcnemar_exact(bd_only, vd_only),
                        "delta_acc": dd, "ci95": [dd_lo, dd_hi]},
            }
        results[tag] = entry
        v = entry["vs"]["narr-sev"]
        print(f"[sev] {tag:>22}: inj {entry['inj_acc']*100:.1f}% dmg {entry['dmg_acc']*100:.1f}% | "
              f"vs narr-sev inj p={v['inj']['mcnemar_p']:.3f} ({v['inj']['var_only']}-{v['inj']['base_only']}) "
              f"dmg p={v['dmg']['mcnemar_p']:.3f} ({v['dmg']['var_only']}-{v['dmg']['base_only']})",
              flush=True)
    out = OUT_DIR / "severity_summary.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[sevanalyze] wrote {out}", flush=True)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract"); p.add_argument("--limit", type=int, default=None)
    sub.add_parser("structmat")
    sub.add_parser("chainemb")
    p = sub.add_parser("v1"); p.add_argument("--variant", default=None)
    p = sub.add_parser("diag"); p.add_argument("--variant", required=True)
    p = sub.add_parser("sev"); p.add_argument("--variant", required=True)
    sub.add_parser("analyze")
    sub.add_parser("sevanalyze")
    args = ap.parse_args()

    if not main_app.DATA_LOADED:
        raise RuntimeError("Window retrieval index not loaded")
    if not USE_ZHANG_WINDOW:
        raise RuntimeError("config did not select the 1982-2006 window index")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    load_emb_caches()

    {"extract": cmd_extract, "structmat": cmd_structmat, "chainemb": cmd_chainemb,
     "v1": cmd_v1, "diag": cmd_diag, "sev": cmd_sev,
     "analyze": cmd_analyze, "sevanalyze": cmd_sevanalyze}[args.cmd](args)


if __name__ == "__main__":
    main()

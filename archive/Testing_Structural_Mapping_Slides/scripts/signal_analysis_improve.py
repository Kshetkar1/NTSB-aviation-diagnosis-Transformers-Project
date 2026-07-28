#!/usr/bin/env python3
"""
Phase 1c signal analysis: does slide-formula struct_sim carry information
orthogonal to cosine that predicts diagnostic correctness?

Inputs (all cached, zero API):
  * pair_scores_improve.jsonl  -- (query, top-50 candidate) cosine + struct_sim
  * outputs/structmap_refined/eval_diagnosis_A0_refined.csv -- per-query A0 match%
  * window index diagnostic_data + retest emb cache -- per-CANDIDATE oracle
    quality: for each (query, candidate) pair, the best cosine between any of
    the candidate's recorded causes and the query's M1 truth texts ("cand_truth
    match"). This is the quantity a good reranker should predict: candidates
    whose causes match the query's truth should be up-weighted.

Outputs: outputs/structmap_improve/signal_analysis.json + printed summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

_SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPTS))

from io_cache import sha256_text  # noqa: E402
from config import REFINED_DATA_PATH, WINDOW_DATA_PATH  # noqa: E402

CACHE_DIR = PROJECT_ROOT / "Testing_Structural_Mapping" / "cache_refined"
PAIRS = CACHE_DIR / "pair_scores_improve.jsonl"
EMB_CACHES = [CACHE_DIR / "emb_cache.jsonl", CACHE_DIR / "emb_cache_improve.jsonl"]
A0_CSV = PROJECT_ROOT / "outputs" / "structmap_refined" / "eval_diagnosis_A0_refined.csv"
EMB_MAP = PROJECT_ROOT / "data" / "processed" / "embeddings_map_1982_2006.json"
OUT = PROJECT_ROOT / "outputs" / "structmap_improve" / "signal_analysis.json"

QUERY_TRUNC = 4000


def load_emb_mem() -> dict[str, list]:
    mem: dict[str, list] = {}
    for p in EMB_CACHES:
        if not p.is_file():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("h") and isinstance(row.get("v"), list):
                    mem[row["h"]] = row["v"]
    return mem


def main() -> None:
    emb_mem = load_emb_mem()

    def vec(text: str) -> np.ndarray | None:
        v = emb_mem.get(sha256_text(str(text)))
        return np.asarray(v, dtype=np.float64) if v is not None else None

    # --- pairs ---
    pair_rows = []
    with open(PAIRS, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            q = row["query_ev_id"]
            for rank, p in enumerate(row["pairs"], start=1):
                if p["struct"] is None:
                    continue
                pair_rows.append((q, p["ev_id"], rank, float(p["cos"]), float(p["struct"])))
    dfp = pd.DataFrame(pair_rows, columns=["query", "cand", "rank_cos", "cos", "struct"])
    print(f"pairs: {len(dfp)} across {dfp['query'].nunique()} queries")

    out: dict = {"n_pairs": int(len(dfp)), "n_queries": int(dfp["query"].nunique())}

    # (1) struct vs cosine correlation, pooled and within-query
    pr = pearsonr(dfp["cos"], dfp["struct"])
    sr = spearmanr(dfp["cos"], dfp["struct"])
    out["pooled_pearson_cos_struct"] = {"r": float(pr[0]), "p": float(pr[1])}
    out["pooled_spearman_cos_struct"] = {"rho": float(sr[0]), "p": float(sr[1])}

    within = []
    for q, g in dfp.groupby("query"):
        if g["struct"].nunique() > 1 and g["cos"].nunique() > 1:
            within.append(spearmanr(g["cos"], g["struct"])[0])
    out["within_query_spearman_cos_struct"] = {
        "mean": float(np.mean(within)),
        "median": float(np.median(within)),
        "n_queries": len(within),
        "frac_positive": float(np.mean([w > 0 for w in within])),
    }
    out["struct_distribution"] = {
        "mean": float(dfp["struct"].mean()),
        "std": float(dfp["struct"].std()),
        "p5": float(dfp["struct"].quantile(0.05)),
        "p50": float(dfp["struct"].quantile(0.50)),
        "p95": float(dfp["struct"].quantile(0.95)),
        "within_query_std_mean": float(dfp.groupby("query")["struct"].std().mean()),
    }

    # (2) per-candidate oracle quality: max cosine between candidate's recorded
    # causes and query's M1 truth texts (all vectors are in the emb cache
    # because A0/A2 scored every prediction and truth text).
    full = json.loads(REFINED_DATA_PATH.read_text(encoding="utf-8"))
    emap = json.loads(EMB_MAP.read_text(encoding="utf-8"))
    cand_causes: dict[str, list[str]] = {}
    for info in emap:
        if info.get("source") != "incident":
            continue
        eid = info.get("ev_id")
        dd = info.get("diagnostic_data") or info.get("bayesian_data") or {}
        causes = [c.strip().lower() for c in (dd.get("all_causes") or []) if c and c.strip()]
        nc = dd.get("narr_cause")
        if nc and len(nc) > 100:
            causes = [c for c in causes if not (len(c) == 100 and nc.lower().startswith(c))]
            causes.append(nc.strip().lower())
        if eid and causes:
            cand_causes.setdefault(eid, causes)

    truth_by_q: dict[str, list[str]] = {}
    for k in dfp["query"].unique():
        inc = full.get(k) or {}
        truths = [
            str(f.get("finding_description", "")).strip()
            for f in (inc.get("findings") or [])
            if isinstance(f, dict)
            and (f.get("Cause_Factor") or "").strip() == "C"
            and (f.get("finding_description") or "").strip()
        ]
        if truths:
            truth_by_q[k] = truths

    cause_vec_cache: dict[str, np.ndarray | None] = {}

    def cvec(text: str) -> np.ndarray | None:
        if text not in cause_vec_cache:
            cause_vec_cache[text] = vec(text)
        return cause_vec_cache[text]

    oracle = []
    n_no_vec = 0
    for q, g in dfp.groupby("query"):
        truths = truth_by_q.get(q)
        if not truths:
            continue
        tvecs = [cvec(t) for t in truths]
        tvecs = [t for t in tvecs if t is not None]
        if not tvecs:
            continue
        T = np.stack(tvecs)
        for _, r in g.iterrows():
            causes = cand_causes.get(r["cand"]) or []
            best = np.nan
            got = False
            for c in causes:
                v = cvec(c)
                if v is None:
                    continue
                got = True
                s = float(np.max(T @ v))
                best = s if np.isnan(best) else max(best, s)
            if not got:
                n_no_vec += 1
                continue
            oracle.append((q, r["cand"], r["cos"], r["struct"], best))
    dfo = pd.DataFrame(oracle, columns=["query", "cand", "cos", "struct", "cand_truth"])
    out["oracle_pairs"] = {"n": int(len(dfo)), "n_skipped_no_vec": int(n_no_vec)}
    print(f"oracle pairs: {len(dfo)} (skipped {n_no_vec} with no cached cause vectors)")

    if len(dfo):
        for col in ("cos", "struct"):
            pr = pearsonr(dfo[col], dfo["cand_truth"])
            out[f"pooled_pearson_{col}_vs_candtruth"] = {"r": float(pr[0]), "p": float(pr[1])}
        # within-query: which score ranks truth-matching candidates higher?
        w_cos, w_str, w_res = [], [], []
        for q, g in dfo.groupby("query"):
            if g["cand_truth"].nunique() < 2:
                continue
            if g["cos"].nunique() > 1:
                w_cos.append(spearmanr(g["cos"], g["cand_truth"])[0])
            if g["struct"].nunique() > 1:
                w_str.append(spearmanr(g["struct"], g["cand_truth"])[0])
                # residual struct after removing linear cos effect
                b = np.polyfit(g["cos"], g["struct"], 1)
                resid = g["struct"] - np.polyval(b, g["cos"])
                if np.std(resid) > 1e-12:
                    w_res.append(spearmanr(resid, g["cand_truth"])[0])
        out["within_query_spearman_vs_candtruth"] = {
            "cos": {"mean": float(np.mean(w_cos)), "n": len(w_cos),
                    "frac_positive": float(np.mean([w > 0 for w in w_cos]))},
            "struct": {"mean": float(np.mean(w_str)), "n": len(w_str),
                       "frac_positive": float(np.mean([w > 0 for w in w_str]))},
            "struct_residualized_on_cos": {"mean": float(np.mean(w_res)), "n": len(w_res),
                                           "frac_positive": float(np.mean([w > 0 for w in w_res]))},
        }

    # (3) query-level: does mean/max struct_sim of the retrieved pool predict
    # the A0 outcome (m1_match_pct)?
    a0 = pd.read_csv(A0_CSV, low_memory=False)[["ev_id", "m1_match_pct"]]
    a0["m1_match_pct"] = pd.to_numeric(a0["m1_match_pct"], errors="coerce")
    qstats = dfp.groupby("query").agg(
        struct_mean=("struct", "mean"), struct_max=("struct", "max"),
        cos_mean=("cos", "mean"),
    ).reset_index().rename(columns={"query": "ev_id"})
    j = qstats.merge(a0.dropna(), on="ev_id")
    out["query_level_vs_a0_match"] = {
        "n": int(len(j)),
        "pearson_structmean_match": float(pearsonr(j["struct_mean"], j["m1_match_pct"])[0]),
        "pearson_structmax_match": float(pearsonr(j["struct_max"], j["m1_match_pct"])[0]),
        "pearson_cosmean_match": float(pearsonr(j["cos_mean"], j["m1_match_pct"])[0]),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

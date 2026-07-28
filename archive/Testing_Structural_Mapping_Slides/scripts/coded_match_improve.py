#!/usr/bin/env python3
"""
Phase 2a: coded-step matching. Map every LLM-extracted chain step (train +
query) onto the nearest NTSB finding/occurrence code description via
text-embedding-3-small, then compute chain similarity over CODE IDs
(exact = 1.0, same 3-digit subject category = 0.6, else 0.0) with the same
Needleman-Wunsch alignment (gap -0.05) and F3 chain score as the slide
formulas. Emits a pairs file compatible with eval_improve.py run
--pairs-file so the same fusion variants can be evaluated.

Also (subcommand `oracle`) emits an oracle pairs file where "struct" is the
candidate's true quality (max cosine between any of the candidate's recorded
causes and the query's M1 truth texts) - the upper bound for ANY reranker of
the fixed top-50 pool.

Embedding calls are batched and cached (step_code_emb_cache.jsonl, new file).
No chat-model calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPTS))

from io_cache import append_jsonl, load_struct_jsonl, sha256_text  # noqa: E402
from config import REFINED_DATA_PATH, WINDOW_DATA_PATH, EMBEDDING_MODEL, get_openai_api_key  # noqa: E402

CACHE_DIR = PROJECT_ROOT / "Testing_Structural_Mapping" / "cache_refined"
STRUCT_TRAIN_CACHE = CACHE_DIR / "struct_train_v2_refined.jsonl"
QUERY_STRUCT_CACHE = CACHE_DIR / "query_struct_v2_refined.jsonl"
PAIRS_BASE = CACHE_DIR / "pair_scores_improve.jsonl"
PAIRS_CODED = CACHE_DIR / "pair_scores_coded.jsonl"
PAIRS_ORACLE = CACHE_DIR / "pair_scores_oracle.jsonl"
STEP_EMB_CACHE = CACHE_DIR / "step_code_emb_cache.jsonl"
EMB_CACHES_RO = [CACHE_DIR / "emb_cache.jsonl", CACHE_DIR / "emb_cache_improve.jsonl"]
EMB_MAP = PROJECT_ROOT / "data" / "processed" / "embeddings_map_1982_2006.json"

GAP_PENALTY = -0.05
STRONG_THRESHOLD = 0.70
PARTIAL_THRESHOLD = 0.35
UNMAPPED_PENALTY_COEF = 0.10
PARTIAL_CODE_SCORE = 0.6

N_EMB_API_CALLS = 0


# ---------------------------------------------------------------------------
# Batched, cached embeddings
# ---------------------------------------------------------------------------
_emb_mem: dict[str, list] = {}


def _load_cache(path: Path) -> None:
    if not path.is_file():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("h") and isinstance(row.get("v"), list):
                _emb_mem[row["h"]] = row["v"]


def embed_texts(texts: list[str], batch: int = 256) -> dict[str, np.ndarray]:
    """text -> unit vector; caches to STEP_EMB_CACHE (new file)."""
    global N_EMB_API_CALLS
    out: dict[str, np.ndarray] = {}
    todo: list[str] = []
    seen: set[str] = set()
    for t in texts:
        h = sha256_text(t)
        if h in _emb_mem:
            out[t] = np.asarray(_emb_mem[h], dtype=np.float64)
        elif t not in seen:
            seen.add(t)
            todo.append(t)
    if todo:
        from openai import OpenAI

        client = OpenAI(api_key=get_openai_api_key())
        for i in range(0, len(todo), batch):
            chunk = todo[i : i + batch]
            resp = client.embeddings.create(input=[t.replace("\n", " ") for t in chunk],
                                            model=EMBEDDING_MODEL)
            N_EMB_API_CALLS += len(chunk)
            for t, d in zip(chunk, resp.data):
                v = list(d.embedding)
                h = sha256_text(t)
                _emb_mem[h] = v
                append_jsonl(STEP_EMB_CACHE, {"h": h, "v": v})
                out[t] = np.asarray(v, dtype=np.float64)
            print(f"[embed] {min(i + batch, len(todo))}/{len(todo)} new", flush=True)
    return out


# ---------------------------------------------------------------------------
# Code dictionary from the window (retrieval-index) dataset
# ---------------------------------------------------------------------------
def build_code_dictionary() -> list[tuple[str, str]]:
    """[(code_id, description)]: finding Subj_Codes + occurrence codes."""
    window = json.loads(WINDOW_DATA_PATH.read_text(encoding="utf-8"))
    by_desc: dict[str, str] = {}
    for inc in window.values():
        for f in inc.get("findings") or []:
            if not isinstance(f, dict):
                continue
            code = str(f.get("Subj_Code") or "").strip()
            desc = str(f.get("finding_description") or "").strip()
            if code and desc and desc not in by_desc:
                by_desc[desc] = f"SUBJ:{code}"
        for s in inc.get("sequence_of_events") or []:
            if not isinstance(s, dict):
                continue
            code = str(s.get("Occurrence_Code") or "").strip()
            desc = str(s.get("Occurrence_Description") or "").strip()
            if desc and desc not in by_desc:
                by_desc[desc] = f"OCC:{code or desc[:20]}"
    return [(cid, desc) for desc, cid in by_desc.items()]


def code_category(code_id: str) -> str:
    """Coarse category: SUBJ 3-digit prefix; OCC code group."""
    kind, _, raw = code_id.partition(":")
    if kind == "SUBJ":
        return f"SUBJ:{raw[:3]}"
    return f"OCC:{raw[:3]}"


# ---------------------------------------------------------------------------
# Step -> code assignment
# ---------------------------------------------------------------------------
def step_text(step: dict) -> str:
    el = str(step.get("element") or "").strip()
    role = str(step.get("role") or "unknown")
    system = str(step.get("system") or "unknown")
    mech = str(step.get("mechanism") or "unknown")
    return f"{el} (role: {role}; system: {system}; mechanism: {mech})"


def assign_codes(structs: dict[str, dict], code_ids: list[str],
                 code_mat: np.ndarray, code_texts: list[str]) -> dict[str, list[dict]]:
    """ev_id -> [{'code','category','sim'}] per chain step."""
    all_texts: list[str] = []
    for st in structs.values():
        for step in st.get("causal_chain") or []:
            if isinstance(step, dict):
                all_texts.append(step_text(step))
    vecs = embed_texts(all_texts)

    out: dict[str, list[dict]] = {}
    for eid, st in structs.items():
        coded = []
        for step in st.get("causal_chain") or []:
            if not isinstance(step, dict):
                continue
            v = vecs[step_text(step)]
            v = v / (np.linalg.norm(v) + 1e-12)
            sims = code_mat @ v
            j = int(np.argmax(sims))
            coded.append({
                "code": code_ids[j],
                "category": code_category(code_ids[j]),
                "sim": float(sims[j]),
                "desc": code_texts[j],
            })
        out[eid] = coded
    return out


# ---------------------------------------------------------------------------
# Coded chain similarity (NW + F3 over code IDs)
# ---------------------------------------------------------------------------
def coded_step_sim(a: dict, b: dict) -> float:
    if a["code"] == b["code"]:
        return 1.0
    if a["category"] == b["category"]:
        return PARTIAL_CODE_SCORE
    return 0.0


def coded_chain_similarity(ca: list[dict], cb: list[dict]) -> float:
    n, m = len(ca), len(cb)
    if not n and not m:
        return 1.0
    if not n or not m:
        return 0.0
    neg_inf = -1e18
    dp = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    bp = [[""] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + GAP_PENALTY
        bp[i][0] = "up"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + GAP_PENALTY
        bp[0][j] = "left"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i - 1][j - 1] + coded_step_sim(ca[i - 1], cb[j - 1])
            up = dp[i - 1][j] + GAP_PENALTY
            left = dp[i][j - 1] + GAP_PENALTY
            best = max(match, up, left)
            dp[i][j] = best
            bp[i][j] = "diag" if best == match else ("up" if best == up else "left")
    strong = partial = unmapped = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bp[i][j] == "diag":
            s = coded_step_sim(ca[i - 1], cb[j - 1])
            if s >= STRONG_THRESHOLD:
                strong += 1
            elif s >= PARTIAL_THRESHOLD:
                partial += 1
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or bp[i][j] == "up"):
            unmapped += 1
            i -= 1
        else:
            unmapped += 1
            j -= 1
    total = n + m
    raw = (strong + 0.5 * partial) / total - UNMAPPED_PENALTY_COEF * (unmapped / total)
    return max(0.0, min(1.0, raw))


def load_pairs_base() -> list[dict]:
    rows = []
    with open(PAIRS_BASE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cmd_coded(_args) -> None:
    for p in EMB_CACHES_RO + [STEP_EMB_CACHE]:
        _load_cache(p)
    print(f"[cache] {len(_emb_mem)} embeddings in memory", flush=True)

    codes = build_code_dictionary()
    code_ids = [c for c, _ in codes]
    code_texts = [d for _, d in codes]
    print(f"[codes] {len(codes)} unique code descriptions", flush=True)
    cvecs = embed_texts(code_texts)
    code_mat = np.stack([cvecs[t] / (np.linalg.norm(cvecs[t]) + 1e-12) for t in code_texts])

    train = load_struct_jsonl(STRUCT_TRAIN_CACHE)
    query = load_struct_jsonl(QUERY_STRUCT_CACHE)
    coded_train = assign_codes(train, code_ids, code_mat, code_texts)
    coded_query = assign_codes(query, code_ids, code_mat, code_texts)
    print(f"[codes] assigned: {len(coded_train)} train, {len(coded_query)} query chains "
          f"({N_EMB_API_CALLS} embedding API calls)", flush=True)

    # diagnostics: assignment quality + code diversity
    sims = [s["sim"] for ch in list(coded_train.values()) + list(coded_query.values()) for s in ch]
    from collections import Counter
    top_codes = Counter(s["code"] for ch in coded_train.values() for s in ch).most_common(10)
    diag = {
        "n_codes": len(codes),
        "assign_sim_mean": float(np.mean(sims)),
        "assign_sim_p10": float(np.percentile(sims, 10)),
        "assign_sim_p90": float(np.percentile(sims, 90)),
        "top10_assigned_codes": top_codes,
    }
    print(json.dumps(diag, indent=2), flush=True)
    (PROJECT_ROOT / "outputs" / "structmap_improve" / "coded_assignment_diag.json").write_text(
        json.dumps(diag, indent=2), encoding="utf-8")

    if PAIRS_CODED.is_file():
        PAIRS_CODED.unlink()
    for row in load_pairs_base():
        q = row["query_ev_id"]
        qc = coded_query.get(q)
        pairs = []
        for p in row["pairs"]:
            tc = coded_train.get(p["ev_id"])
            s = coded_chain_similarity(qc, tc) if (qc is not None and tc is not None) else None
            pairs.append({"ev_id": p["ev_id"], "cos": p["cos"], "struct": s})
        append_jsonl(PAIRS_CODED, {"query_ev_id": q, "pairs": pairs})
    print(f"[coded] wrote {PAIRS_CODED}", flush=True)


def cmd_oracle(_args) -> None:
    for p in EMB_CACHES_RO:
        _load_cache(p)
    full = json.loads(REFINED_DATA_PATH.read_text(encoding="utf-8"))
    emap = json.loads(EMB_MAP.read_text(encoding="utf-8"))

    def vec(text: str) -> np.ndarray | None:
        v = _emb_mem.get(sha256_text(str(text)))
        return np.asarray(v, dtype=np.float64) if v is not None else None

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

    if PAIRS_ORACLE.is_file():
        PAIRS_ORACLE.unlink()
    n_missing = 0
    for row in load_pairs_base():
        q = row["query_ev_id"]
        inc = full.get(q) or {}
        truths = [
            str(f.get("finding_description", "")).strip()
            for f in (inc.get("findings") or [])
            if isinstance(f, dict)
            and (f.get("Cause_Factor") or "").strip() == "C"
            and (f.get("finding_description") or "").strip()
        ]
        tvecs = [v for v in (vec(t) for t in truths) if v is not None]
        T = np.stack(tvecs) if tvecs else None
        pairs = []
        for p in row["pairs"]:
            s = None
            if T is not None:
                best = None
                for c in cand_causes.get(p["ev_id"]) or []:
                    v = vec(c)
                    if v is None:
                        continue
                    x = float(np.max(T @ v))
                    best = x if best is None else max(best, x)
                s = best
            if s is None:
                n_missing += 1
            pairs.append({"ev_id": p["ev_id"], "cos": p["cos"], "struct": s})
        append_jsonl(PAIRS_ORACLE, {"query_ev_id": q, "pairs": pairs})
    print(f"[oracle] wrote {PAIRS_ORACLE} ({n_missing} pairs without oracle score)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("coded")
    sub.add_parser("oracle")
    args = ap.parse_args()
    (PROJECT_ROOT / "outputs" / "structmap_improve").mkdir(parents=True, exist_ok=True)
    if args.cmd == "coded":
        cmd_coded(args)
    else:
        cmd_oracle(args)


if __name__ == "__main__":
    main()

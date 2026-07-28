#!/usr/bin/env python3
"""Leak-free, baseline-compared, calibrated evaluation of the DIAGNOSIS engine.

Runs the REAL production pipeline (main_app: embed -> retrieve -> cluster ->
P(cause|cluster) -> Law of Total Probability) on the held-out test incidents,
entirely OFFLINE:

  * Query vectors are read from the cached FULL embedding index (the test
    incident's own 'narrative' row) — no OpenAI call.
  * Retrieval runs against the TRAIN-ONLY index (NTSB_USE_TRAIN_INDEX=1), and we
    additionally pass exclude_ev_ids={query} so the engine can never see the case
    it is diagnosing. This closes the leakage hole (Hole 1).

Scoring is deliberately NON-CIRCULAR (Hole 2): predictions are reduced to NTSB
finding CATEGORIES (top-level + level-2 of the finding taxonomy) and matched to
the incident's true Cause_Factor='C' categories by string equality — NOT by the
embedding model that did the retrieval.

We compare the method against BASELINES (Hole 3): base-rate (always predict the
most frequent train categories) and random. We also compute calibration (Hole 4):
Brier score + a reliability table on the category-level probabilities.

Outputs: outputs/leakfree_eval.{json,md}
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# --- offline setup: train index + dummy key (no real API call happens) ---
os.environ["NTSB_USE_TRAIN_INDEX"] = "1"
os.environ.setdefault("OPENAI_API_KEY", "offline-dummy-key")

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
SPLITS = REPO / "data" / "Testing_Data_Metrics" / "splits"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO))
import main_app  # noqa: E402  (imports load the TRAIN index per env above)

TOP_LEVELS = {
    "personnel issues", "environmental issues", "aircraft",
    "organizational issues", "not determined",
}


def norm(s: str) -> str:
    return " ".join(str(s or "").lower().split())


def categories(finding_desc: str, lvl: int = 1) -> str | None:
    segs = [norm(x) for x in str(finding_desc or "").split("-") if x.strip()]
    if not segs:
        return None
    return "-".join(segs[:lvl])


def truth_categories(inc: dict, lvl: int = 1) -> set[str]:
    out = set()
    for f in inc.get("findings") or []:
        if isinstance(f, dict) and f.get("Cause_Factor") == "C":
            c = categories(f.get("finding_description"), lvl)
            if c:
                out.add(c)
    return out


def cause_to_category(cause: str, lvl: int = 1) -> str | None:
    """Reduce a predicted free-text cause to an NTSB top-level/level-2 category.

    Only finding-taxonomy-style causes map cleanly; narrative paragraphs return
    None and are ignored for the category metric (non-circular by construction)."""
    c = norm(cause)
    seg0 = c.split("-")[0].strip()
    # match the leading segment to a known top-level category
    top = None
    for tl in TOP_LEVELS:
        if seg0 == tl or seg0.startswith(tl):
            top = tl
            break
    if top is None:
        return None
    if lvl == 1:
        return top
    return categories(cause, lvl)


def predicted_category_dist(weighted_causes: list[dict], lvl: int = 1) -> dict[str, float]:
    """Aggregate weighted_causes probabilities into category-level probabilities."""
    agg: dict[str, float] = defaultdict(float)
    for wc in weighted_causes:
        cat = cause_to_category(wc.get("cause", ""), lvl)
        if cat is None:
            continue
        agg[cat] += float(wc.get("probability", 0.0))
    total = sum(agg.values())
    if total > 0:
        agg = {k: v / total for k, v in agg.items()}
    return dict(agg)


def load_full_index() -> tuple[np.ndarray, list]:
    emb = np.load(PROC / "embeddings.npy")
    fmap = json.loads((PROC / "embeddings_map.json").read_text())
    return emb, fmap


def query_vectors_from_full(emb: np.ndarray, fmap: list) -> dict[str, np.ndarray]:
    """ev_id -> narrative embedding from the cached FULL index (offline query source)."""
    out: dict[str, np.ndarray] = {}
    for i, row in enumerate(fmap):
        if row.get("type") == "narrative" and row.get("source") == "incident":
            ev = row.get("ev_id")
            if ev and ev not in out:
                out[ev] = emb[i]
    return out


def self_leak_probe(emb: np.ndarray, fmap: list, qvecs: dict, test_ids: list,
                    top_n: int = 50) -> dict:
    """Quantify the leakage the NAIVE full-corpus setup (no exclude-self) would have.

    For each test query, retrieve top_n rows from the FULL index and check whether
    the incident's OWN ev_id shows up — i.e. the engine citing the answer as its
    own evidence. This is exactly what exclude-self + train-index prevents."""
    leaked = 0
    for ev in test_ids:
        qv = qvecs.get(ev)
        if qv is None:
            continue
        sims = np.dot(emb, qv)
        top_idx = np.argsort(sims)[::-1][:top_n]
        if any(fmap[i].get("ev_id") == ev for i in top_idx):
            leaked += 1
    n = sum(1 for ev in test_ids if qvecs.get(ev) is not None)
    return {"top_n": top_n, "n": n, "self_leaked": leaked,
            "leak_rate": round(leaked / n, 3) if n else None}


def brier_multi(prob_dist: dict[str, float], truth: set[str], classes: list[str]) -> float:
    """Multiclass Brier over the category vocabulary."""
    s = 0.0
    for c in classes:
        p = prob_dist.get(c, 0.0)
        y = 1.0 if c in truth else 0.0
        s += (p - y) ** 2
    return s / len(classes)


def main() -> None:
    rng = random.Random(42)
    test_ids = [t for t in (SPLITS / "test_ev_ids.txt").read_text().split() if t]
    full = json.loads((PROC / "refined_dataset.json").read_text())
    full_emb, full_map = load_full_index()
    qvecs = query_vectors_from_full(full_emb, full_map)
    leak = self_leak_probe(full_emb, full_map, qvecs, test_ids)

    # base-rate priors from TRAIN
    train = json.loads((PROC / "merged_dataset_train.json").read_text())
    base_l1, base_l2 = Counter(), Counter()
    for v in train.values():
        for c in truth_categories(v, 1):
            base_l1[c] += 1
        for c in truth_categories(v, 2):
            base_l2[c] += 1
    base_l1_rank = [c for c, _ in base_l1.most_common()]
    base_l2_rank = [c for c, _ in base_l2.most_common()]
    classes_l1 = sorted(base_l1)
    base_l1_dist = {c: base_l1[c] / sum(base_l1.values()) for c in base_l1}

    records = []
    hit = {"method_l1": 0, "method_l2": 0, "baserate_l1": 0, "random_l1": 0}
    brier = {"method": [], "baserate": []}
    evaluated = 0
    skipped = []

    for ev in test_ids:
        inc = full.get(ev)
        qv = qvecs.get(ev)
        if inc is None or qv is None:
            skipped.append(ev)
            continue
        truth1 = truth_categories(inc, 1)
        truth2 = truth_categories(inc, 2)
        if not truth1:
            skipped.append(ev)
            continue

        # --- REAL pipeline, leak-free (exclude the query incident itself) ---
        top_scores, top_matches = main_app.find_top_matches(qv, exclude_ev_ids={ev})
        clusters = main_app.cluster_incidents_by_type(top_scores, top_matches, 50)
        if not clusters:
            skipped.append(ev)
            continue
        analysis = main_app.calculate_cause_probabilities_per_cluster(clusters)
        result = main_app.calculate_chain_rule_diagnosis(clusters, analysis)
        wcauses = result.get("weighted_causes", [])

        dist1 = predicted_category_dist(wcauses, 1)
        dist2 = predicted_category_dist(wcauses, 2)
        top1 = max(dist1, key=dist1.get) if dist1 else None
        top2 = max(dist2, key=dist2.get) if dist2 else None

        m1 = top1 in truth1
        m2 = top2 in truth2
        b1 = base_l1_rank[0] in truth1 if base_l1_rank else False
        r1 = rng.choice(classes_l1) in truth1 if classes_l1 else False

        evaluated += 1
        hit["method_l1"] += int(m1)
        hit["method_l2"] += int(m2)
        hit["baserate_l1"] += int(b1)
        hit["random_l1"] += int(r1)
        brier["method"].append(brier_multi(dist1, truth1, classes_l1))
        brier["baserate"].append(brier_multi(base_l1_dist, truth1, classes_l1))

        records.append({
            "ev_id": ev, "truth_l1": sorted(truth1), "pred_l1": top1,
            "pred_l1_p": round(dist1.get(top1, 0.0), 3) if top1 else None,
            "method_hit_l1": m1, "method_hit_l2": m2, "baserate_hit_l1": b1,
            "n_retrieved_clusters": len(clusters),
        })

    n = evaluated
    acc = {k: (v / n if n else 0.0) for k, v in hit.items()}

    # reliability table (method, level-1): bin predicted top-1 prob vs hit rate
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    rel = []
    for lo, hi in bins:
        sel = [r for r in records if r["pred_l1_p"] is not None and lo <= r["pred_l1_p"] < hi]
        if sel:
            conf = sum(r["pred_l1_p"] for r in sel) / len(sel)
            obs = sum(1 for r in sel if r["method_hit_l1"]) / len(sel)
            rel.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": len(sel),
                        "avg_conf": round(conf, 3), "obs_acc": round(obs, 3)})

    summary = {
        "protocol": {
            "n_test_total": len(test_ids), "n_evaluated": n, "n_skipped": len(skipped),
            "retrieval_index": "train-only (NTSB_USE_TRAIN_INDEX=1)",
            "exclude_self": True,
            "query_source": "cached full-index narrative vector (offline, no API)",
            "metric": "NON-circular NTSB finding-category match (string equality, not embedding)",
        },
        "leakage_probe": leak,
        "accuracy_top1": {
            "method_level1": round(acc["method_l1"], 3),
            "method_level2": round(acc["method_l2"], 3),
            "baserate_level1": round(acc["baserate_l1"], 3),
            "random_level1": round(acc["random_l1"], 3),
            "method_lift_over_baserate": round(acc["method_l1"] - acc["baserate_l1"], 3),
        },
        "calibration": {
            "brier_method": round(float(np.mean(brier["method"])), 4) if brier["method"] else None,
            "brier_baserate": round(float(np.mean(brier["baserate"])), 4) if brier["baserate"] else None,
            "reliability": rel,
        },
        "base_rate_top_category": base_l1_rank[0] if base_l1_rank else None,
        "records": records,
    }
    (OUT / "leakfree_eval.json").write_text(json.dumps(summary, indent=2))

    L = [
        "# Leak-free, baseline-compared, calibrated diagnosis evaluation",
        "",
        f"- Test incidents evaluated: **{n}** / {len(test_ids)} (skipped {len(skipped)})",
        "- Retrieval: **train-only index**, **exclude-self ON** (no leakage).",
        "- Query vectors: cached full-index narrative rows (fully offline).",
        "- Metric: **non-circular** NTSB finding-category match (string equality).",
        "",
        "## Leakage probe (why exclude-self matters)",
        "",
        f"- In the NAIVE full-corpus setup, **{leak['self_leaked']}/{leak['n']}** "
        f"({leak['leak_rate']*100:.0f}%) of test incidents would retrieve their OWN record "
        f"in the top {leak['top_n']} — i.e. the engine citing the answer as evidence.",
        f"- This evaluation eliminates that: train-only index + exclude-self → **0%** self-leakage.",
        "",
        "## Does the method beat the baselines? (top-1 accuracy)",
        "",
        "| System | Top-1 accuracy (level-1 cause category) |",
        "|--------|------------------------------------------|",
        f"| **Method (retrieval + LTP)** | **{acc['method_l1']:.3f}** |",
        f"| Base-rate (always '{base_l1_rank[0]}') | {acc['baserate_l1']:.3f} |",
        f"| Random | {acc['random_l1']:.3f} |",
        "",
        f"**Lift over base-rate: {acc['method_l1'] - acc['baserate_l1']:+.3f}**  "
        f"(method level-2 accuracy: {acc['method_l2']:.3f})",
        "",
        "## Calibration",
        "",
        f"- Brier (method): **{summary['calibration']['brier_method']}**  ·  "
        f"Brier (base-rate): {summary['calibration']['brier_baserate']}  (lower = better)",
        "",
        "| confidence bin | n | avg confidence | observed accuracy |",
        "|----------------|---|----------------|-------------------|",
    ]
    for r in rel:
        L.append(f"| {r['bin']} | {r['n']} | {r['avg_conf']} | {r['obs_acc']} |")
    L += [
        "",
        "## How to read this",
        "- **Lift over base-rate** is the scientific contribution: if positive, the narrative-driven",
        "  retrieval genuinely adds information beyond 'always guess the most common cause'.",
        "- **Exclude-self + train-only index** means no test incident can retrieve itself — the",
        "  leakage objection is closed by construction.",
        "- Metric is **not** the embedding model that did the retrieval, so it is not circular.",
    ]
    (OUT / "leakfree_eval.md").write_text("\n".join(L) + "\n")

    print("\n=== LEAK-FREE EVAL DONE ===")
    print(f"evaluated {n}/{len(test_ids)}")
    print(f"method L1 acc   = {acc['method_l1']:.3f}")
    print(f"method L2 acc   = {acc['method_l2']:.3f}")
    print(f"base-rate L1    = {acc['baserate_l1']:.3f}  (always '{base_l1_rank[0]}')")
    print(f"random L1       = {acc['random_l1']:.3f}")
    print(f"LIFT over base  = {acc['method_l1']-acc['baserate_l1']:+.3f}")
    print(f"Brier method    = {summary['calibration']['brier_method']}")
    print(f"→ outputs/leakfree_eval.md")


if __name__ == "__main__":
    main()

"""LOO with LIFT-style re-ranking: does distinctiveness beat raw frequency?

Clone of tests/loo_specific.py. The leave-one-out harness, outcome detection,
self-excluded neighbor retrieval, generic-label stripping, real-narrative filter
and metrics (top-1/3/5 + MRR) are IDENTICAL. The ONLY change is how the predicted
causes are RANKED. We evaluate every ranking variant on the same incident set so
they are directly comparable:

  raw   : P(cause|neighbors)                                  (control == loo_specific)
  lift  : P(cause|neighbors) / P(cause|global)
  pmi   : log( P(cause|neighbors) / P(cause|global) )
  slift : (P(cause|neighbors)+eps) / (P(cause|global)+eps),  eps=0.01
  blend : P(cause|neighbors) * lift

Plus the base-rate baseline (global order, generics stripped).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main_app  # noqa: E402
import zhang_diagnosis as zd  # noqa: E402

TOPK = (1, 3, 5)
GENERIC = {
    "airframe/component/system failure/malfunction",
    "miscellaneous/other",
}
EPS = 0.01
GLOBAL_FLOOR = 1e-6  # global_prob for causes never seen in the global population

VARIANTS = ("raw", "lift", "pmi", "slift", "blend")


def real_narr(inc):
    for k in ("narr_accp", "narr_accf"):
        t = (inc.get(k) or "").strip()
        if len(t) >= 80:
            return t
    return ""


def specific(labels):
    return [c for c in labels if c.lower() not in GENERIC]


def first_rank(pred, true_set):
    for i, p in enumerate(pred, 1):
        if p in true_set:
            return i
    return 0


def rank_variant(variant, neigh_causes, global_prob):
    """Return a list of cause names ranked best-first for the given variant.

    neigh_causes: list of (cause, p_neigh) for causes present in the neighborhood
                  (already generic-stripped, all p_neigh > 0).
    global_prob:  dict cause -> P(cause|global), floored.
    Ties are broken deterministically by p_neigh descending, then cause name.
    """
    scored = []
    for cause, p_neigh in neigh_causes:
        g = global_prob.get(cause, GLOBAL_FLOOR) or GLOBAL_FLOOR
        if variant == "raw":
            score = p_neigh
        elif variant == "lift":
            score = p_neigh / g
        elif variant == "pmi":
            # all p_neigh > 0 here; guard anyway
            score = math.log(p_neigh / g) if p_neigh > 0 else float("-inf")
        elif variant == "slift":
            score = (p_neigh + EPS) / (g + EPS)
        elif variant == "blend":
            score = p_neigh * (p_neigh / g)
        else:
            raise ValueError(variant)
        scored.append((score, p_neigh, cause))
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    return [c for _, _, c in scored]


def eval_outcome(query_word, top_n_incidents=100):
    ds = main_app.refined_dataset
    name, targets = zd.detect_outcome(query_word, ds)
    out_ids = [k for k, v in ds.items() if zd._causes_into_outcome(v, targets)]

    g = zd.empirical_cause_distribution(name, targets=targets, dataset=ds)
    global_spec = specific([c["cause"] for c in g["causes"]])
    global_prob = {c["cause"]: c["probability"] for c in g["causes"]}

    methods = ("base",) + VARIANTS
    acc = {m: {k: 0 for k in TOPK} for m in methods}
    mrr = {m: 0.0 for m in methods}
    n = 0
    for ev in out_ids:
        inc = ds[ev]
        narr = real_narr(inc)
        if not narr:
            continue
        true_spec = set(specific(zd._causes_into_outcome(inc, targets)))
        if not true_spec:  # only the catch-all was the cause -> skip
            continue
        q = main_app.get_embedding(narr)
        _, matches = main_app.find_top_matches(q)
        ev_ids, seen = [], set()
        for m in matches:
            if m.get("source") != "incident":
                continue
            e = m.get("ev_id")
            if e and e != ev and e not in seen:
                seen.add(e)
                ev_ids.append(e)
            if len(ev_ids) >= top_n_incidents:
                break
        r = zd.empirical_cause_distribution(name, targets=targets, dataset=ds,
                                            restrict_ev_ids=ev_ids)
        # (cause, p_neigh) for specific causes present in the neighborhood
        neigh_causes = [(c["cause"], c["probability"])
                        for c in r["causes"] if c["cause"].lower() not in GENERIC]

        n += 1
        preds = {"base": global_spec}
        for v in VARIANTS:
            preds[v] = rank_variant(v, neigh_causes, global_prob)

        for mth, pred in preds.items():
            for k in TOPK:
                if any(p in true_spec for p in pred[:k]):
                    acc[mth][k] += 1
            rr = first_rank(pred, true_spec)
            mrr[mth] += (1.0 / rr) if rr else 0.0

        if n % 20 == 0:
            print(f"   ...{name}: {n}", file=sys.stderr)
    return name, n, acc, mrr


def main():
    for word in ["fire", "loss of engine power", "gear collapse"]:
        name, n, acc, mrr = eval_outcome(word)
        print(f"\n=== {name}  (n={n}) ===")
        if n == 0:
            print("  (no scorable incidents)")
            continue
        print(f"{'method':10} {'top-1':>7} {'top-3':>7} {'top-5':>7} {'MRR':>7}")
        print("-" * 42)
        order = ("base",) + VARIANTS
        for mth in order:
            print(f"{mth:10} "
                  f"{acc[mth][1]/n:7.1%} {acc[mth][3]/n:7.1%} "
                  f"{acc[mth][5]/n:7.1%} {mrr[mth]/n:7.3f}")

        # winners (excluding the baseline, which is the bar to beat)
        best_mrr = max(VARIANTS, key=lambda v: mrr[v])
        best_t3 = max(VARIANTS, key=lambda v: acc[v][3])
        print(f"winner(MRR)  : {best_mrr} ({mrr[best_mrr]/n:.3f} "
              f"vs base {mrr['base']/n:.3f})")
        print(f"winner(top-3): {best_t3} ({acc[best_t3][3]/n:.1%} "
              f"vs base {acc['base'][3]/n:.1%})")


if __name__ == "__main__":
    main()

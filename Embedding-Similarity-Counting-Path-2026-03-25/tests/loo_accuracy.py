"""Leave-one-out predictive accuracy of diagnosis.

For each outcome accident with a REAL narrative (synthesized narratives are
skipped to avoid leakage), use its narrative as the query, diagnose causes from
the OTHER incidents (self excluded), and check whether the incident's true
cause(s) appear in the top-k. Compared against a majority-class baseline
(always predict the globally most common causes) to show whether query-specific
retrieval adds predictive value.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main_app  # noqa: E402
import zhang_diagnosis as zd  # noqa: E402

TOPK = (1, 3, 5)


def real_narr(inc):
    for k in ("narr_accp", "narr_accf"):
        t = (inc.get(k) or "").strip()
        if len(t) >= 80:
            return t
    return ""


def eval_outcome(query_word, top_n_incidents=100):
    ds = main_app.refined_dataset
    name, targets = zd.detect_outcome(query_word, ds)
    out_ids = [k for k, v in ds.items() if zd._causes_into_outcome(v, targets)]
    g = zd.empirical_cause_distribution(name, targets=targets, dataset=ds)
    global_order = [c["cause"] for c in g["causes"]]

    retr = {k: 0 for k in TOPK}
    base = {k: 0 for k in TOPK}
    n = 0
    for ev in out_ids:
        inc = ds[ev]
        narr = real_narr(inc)
        if not narr:
            continue
        true = zd._causes_into_outcome(inc, targets)
        if not true:
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
        pred = [c["cause"] for c in r["causes"]]
        n += 1
        for k in TOPK:
            if any(p in true for p in pred[:k]):
                retr[k] += 1
            if any(p in true for p in global_order[:k]):
                base[k] += 1
        if n % 20 == 0:
            print(f"   ...{name}: {n} evaluated", file=sys.stderr)
    return name, n, retr, base


def main():
    print(f"{'outcome':24} {'n':>4}  {'method':9} {'top-1':>7} {'top-3':>7} {'top-5':>7}")
    print("-" * 70)
    for word in ["fire", "loss of engine power", "gear collapse"]:
        name, n, retr, base = eval_outcome(word)
        if n == 0:
            print(f"{name:24} {n:>4}  (no evaluable incidents)")
            continue
        print(f"{name:24} {n:>4}  {'retrieval':9} "
              f"{retr[1]/n:7.1%} {retr[3]/n:7.1%} {retr[5]/n:7.1%}")
        print(f"{'':24} {'':>4}  {'baseline':9} "
              f"{base[1]/n:7.1%} {base[3]/n:7.1%} {base[5]/n:7.1%}")
        print()


if __name__ == "__main__":
    main()

"""Fairer leave-one-out: can diagnosis predict the SPECIFIC cause?

Strips the dominant catch-all labels from both ground truth and predictions, so
the metric measures whether the model identifies the specific mechanism (wiring,
fuel, turbine blade, ...) rather than scoring easy hits on generic causes.
Reports top-k AND mean reciprocal rank (MRR), retrieval vs majority-class baseline.
"""
from __future__ import annotations

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


def eval_outcome(query_word, top_n_incidents=100):
    ds = main_app.refined_dataset
    name, targets = zd.detect_outcome(query_word, ds)
    out_ids = [k for k, v in ds.items() if zd._causes_into_outcome(v, targets)]
    g = zd.empirical_cause_distribution(name, targets=targets, dataset=ds)
    global_spec = specific([c["cause"] for c in g["causes"]])

    acc = {m: {k: 0 for k in TOPK} for m in ("retr", "base")}
    mrr = {"retr": 0.0, "base": 0.0}
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
        pred_spec = specific([c["cause"] for c in r["causes"]])
        n += 1
        for k in TOPK:
            if any(p in true_spec for p in pred_spec[:k]):
                acc["retr"][k] += 1
            if any(p in true_spec for p in global_spec[:k]):
                acc["base"][k] += 1
        rr = first_rank(pred_spec, true_spec)
        rb = first_rank(global_spec, true_spec)
        mrr["retr"] += (1.0 / rr) if rr else 0.0
        mrr["base"] += (1.0 / rb) if rb else 0.0
        if n % 20 == 0:
            print(f"   ...{name}: {n}", file=sys.stderr)
    return name, n, acc, mrr


def main():
    print(f"{'outcome':22} {'n':>4} {'method':9} {'top-1':>7} {'top-3':>7} {'top-5':>7} {'MRR':>7}")
    print("-" * 74)
    for word in ["fire", "loss of engine power", "gear collapse"]:
        name, n, acc, mrr = eval_outcome(word)
        if n == 0:
            print(f"{name:22} {n:>4}  (none)")
            continue
        for mth in ("retr", "base"):
            lbl = "retrieval" if mth == "retr" else "baseline"
            print(f"{name if mth=='retr' else '':22} {n if mth=='retr' else '':>4} {lbl:9} "
                  f"{acc[mth][1]/n:7.1%} {acc[mth][3]/n:7.1%} {acc[mth][5]/n:7.1%} {mrr[mth]/n:7.3f}")
        print()


if __name__ == "__main__":
    main()

"""Selective-prediction & distinctiveness-stratified LOO.

Clones tests/loo_specific.py's fairer leave-one-out harness (self-excluded
top-100 retrieval, generic catch-all stripping, real-narrative filter) but adds
per-incident diagnostics so we can ask TWO new questions that re-ranking by lift
did not answer:

  EXP 1  Selective prediction: if we only commit predictions on the highest
         CONFIDENCE incidents (neighborhood prob of the top-1 specific cause),
         does retrieval beat the majority-class base rate on that subset?
  EXP 2  Distinctiveness stratification: split incidents by the GLOBAL frequency
         of their true specific cause ("common" >= 0.10 vs "rare/specific"
         < 0.10). Does retrieval beat base rate on rare/specific causes?

Confidence = probability of the model's top-1 specific cause.
Margin      = prob(top1) - prob(top2) over specific causes.
A null result (a wash at every operating point) is a legitimate finding.
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
OUTCOMES = ["fire", "loss of engine power", "gear collapse"]
COVERAGE_LEVELS = (1.00, 0.75, 0.50, 0.25, 0.10)
COMMON_THRESHOLD = 0.10  # global P(cause|outcome) >= this -> "common"
GLOBAL_FLOOR = 1e-6


def real_narr(inc):
    for k in ("narr_accp", "narr_accf"):
        t = (str(inc.get(k) or "")).strip()
        if len(t) >= 80:
            return t
    return ""


def specific(labels):
    return [c for c in labels if str(c or "").lower() not in GENERIC]


def first_rank(pred, true_set):
    for i, p in enumerate(pred, 1):
        if p in true_set:
            return i
    return 0


def cause_prob_map(dist_causes):
    """{cause_label: probability} from an empirical_cause_distribution result."""
    m = {}
    for c in dist_causes:
        lab = str(c.get("cause") or "")
        try:
            p = float(c.get("probability") or 0.0)
        except (TypeError, ValueError):
            p = 0.0
        m[lab] = p
    return m


def eval_outcome(query_word, top_n_incidents=100):
    """Return (name, records) where each record is a dict of per-incident diagnostics."""
    ds = main_app.refined_dataset
    det = zd.detect_outcome(query_word, ds)
    if det is None:
        return query_word, []
    name, targets = det
    out_ids = [k for k, v in ds.items() if zd._causes_into_outcome(v, targets)]

    g = zd.empirical_cause_distribution(name, targets=targets, dataset=ds)
    global_spec = specific([c["cause"] for c in g["causes"]])
    global_prob = cause_prob_map(g["causes"])  # full-population P(cause|outcome)

    records = []
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
        if not ev_ids:  # empty neighbor list -> skip
            continue

        r = zd.empirical_cause_distribution(name, targets=targets, dataset=ds,
                                            restrict_ev_ids=ev_ids)
        pred_spec = specific([c["cause"] for c in r["causes"]])
        pred_prob = cause_prob_map(r["causes"])

        # confidence = neighborhood prob of predicted top-1 specific cause;
        # margin = prob(top1) - prob(top2) over specific causes.
        conf = pred_prob.get(pred_spec[0], 0.0) if pred_spec else 0.0
        p2 = pred_prob.get(pred_spec[1], 0.0) if len(pred_spec) >= 2 else 0.0
        margin = conf - p2

        retr_top1 = bool(pred_spec[:1]) and (pred_spec[0] in true_spec)
        base_top1 = bool(global_spec[:1]) and (global_spec[0] in true_spec)

        # distinctiveness: most-common true specific cause's global frequency
        # (floor for causes absent from the global dict).
        true_global = max((global_prob.get(c, GLOBAL_FLOOR) for c in true_spec),
                          default=GLOBAL_FLOOR)

        rec = {
            "ev": ev,
            "confidence": conf,
            "margin": margin,
            "true_global": true_global,
            "retr_correct": {k: any(p in true_spec for p in pred_spec[:k]) for k in TOPK},
            "base_correct": {k: any(p in true_spec for p in global_spec[:k]) for k in TOPK},
            "retr_top1": retr_top1,
            "base_top1": base_top1,
        }
        records.append(rec)
        n += 1
        if n % 20 == 0:
            print(f"   ...{name}: {n}", file=sys.stderr)
    return name, records


def _acc(records, method, k):
    if not records:
        return 0.0
    key = "retr_correct" if method == "retr" else "base_correct"
    return sum(1 for r in records if r[key][k]) / len(records)


def coverage_table(name, records, sort_key, label):
    print(f"\n[EXP 1 :: {label}] {name}  (n={len(records)})")
    print(f"  {'coverage':>8} {'n':>4} {'retr top-1':>11} {'base top-1':>11} {'lift(pp)':>9}")
    print("  " + "-" * 48)
    ordered = sorted(records, key=lambda r: r[sort_key], reverse=True)
    total = len(ordered)
    for c in COVERAGE_LEVELS:
        m = max(1, round(total * c)) if total else 0
        sub = ordered[:m]
        retr = _acc(sub, "retr", 1)
        base = _acc(sub, "base", 1)
        lift = (retr - base) * 100.0
        print(f"  {c:>7.0%} {len(sub):>4} {retr:>11.1%} {base:>11.1%} {lift:>+9.1f}")


def stratify_table(name, records):
    common = [r for r in records if r["true_global"] >= COMMON_THRESHOLD]
    rare = [r for r in records if r["true_global"] < COMMON_THRESHOLD]
    print(f"\n[EXP 2] {name}  (n={len(records)})")
    print(f"  {'bucket':>14} {'n':>4} {'retr t1':>8} {'base t1':>8} {'retr t3':>8} {'base t3':>8}")
    print("  " + "-" * 56)
    for lbl, sub in (("common(>=.10)", common), ("rare(<.10)", rare)):
        if not sub:
            print(f"  {lbl:>14} {0:>4} {'-':>8} {'-':>8} {'-':>8} {'-':>8}")
            continue
        print(f"  {lbl:>14} {len(sub):>4} "
              f"{_acc(sub,'retr',1):>8.1%} {_acc(sub,'base',1):>8.1%} "
              f"{_acc(sub,'retr',3):>8.1%} {_acc(sub,'base',3):>8.1%}")


def main():
    all_records = {}
    for word in OUTCOMES:
        name, records = eval_outcome(word)
        all_records[name] = records

    print("\n" + "=" * 64)
    print("EXPERIMENT 1 — Accuracy vs coverage (selective prediction)")
    print("=" * 64)
    for name, records in all_records.items():
        if not records:
            print(f"\n{name}: (no evaluable incidents)")
            continue
        coverage_table(name, records, "confidence", "gate=confidence")
        coverage_table(name, records, "margin", "gate=margin")

    print("\n" + "=" * 64)
    print("EXPERIMENT 2 — Stratify by cause distinctiveness")
    print("=" * 64)
    for name, records in all_records.items():
        if not records:
            print(f"\n{name}: (no evaluable incidents)")
            continue
        stratify_table(name, records)


if __name__ == "__main__":
    main()

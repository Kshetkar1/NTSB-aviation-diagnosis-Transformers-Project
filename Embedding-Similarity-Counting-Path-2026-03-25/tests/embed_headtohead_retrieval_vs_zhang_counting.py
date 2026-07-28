"""HEAD-TO-HEAD PREDICTIVE SCORES: narrative method vs Zhang's counting baseline.

THE QUESTION (the publishability crux)
--------------------------------------
Does the NARRATIVE method produce BETTER predictive SCORES than ZHANG's counting
method?  Reproducing Zhang's *descriptive* tables (Table 7 P(cause|fire), Table 9
edges) is **parity by definition** -- you cannot "beat" someone on their own
summary statistic.  "Better" can only be claimed on a HELD-OUT PREDICTIVE task
where neither method saw the test item.  This script measures exactly that, on two
tasks, honestly, with significance tests, and prints a per-task verdict.

  TASK 1 -- DIAGNOSIS.  Given a held-out incident's NARRATIVE, predict its true
            cause/finding label.
              * NARRATIVE (ours): embed the held-out narrative -> retrieve similar
                TRAINING incidents (self EXCLUDED) -> rank causes over the
                outcome accidents in that neighbourhood (diagnose_retrieval style).
              * ZHANG baseline:   leave-self-out P(cause | outcome) over ALL outcome
                accidents -- the population prior, no narrative conditioning. This
                is Zhang's counting answer (Table-7 method), and its top-1 is the
                majority-cause-given-outcome.
              * MAJORITY baseline: global cause frequency (outcome-agnostic) -- the
                dumbest floor.

  TASK 2 -- PROGNOSIS.  Given the current event (and the incident's narrative),
            predict the NEXT event on held-out transitions.
              * NARRATIVE (ours): forward Markov transitions counted only over the
                narrative-retrieved neighbour incidents (self excluded).
              * GLOBAL MARKOV baseline: leave-self-out global transition counts.

METRICS
-------
Diagnosis: top-1 acc, top-3 recall, MRR, log-loss, Brier -- reported with the
generic catch-all causes BOTH included AND excluded (the "fairer" metric strips
"Airframe/component/system failure/malfunction" and "Miscellaneous/other").
Prognosis: next-event top-1 acc, top-3 recall, log-loss, Brier.

log-loss / Brier use the probability MASS each method assigns to the true label:
  p_hit = sum(prob over the true labels);  log-loss = -log(clip(p_hit));
  Brier = (1 - p_hit)^2.  Identical scoring for every method.

SIGNIFICANCE (narrative vs the Zhang/global baseline):
  * top-1: McNemar exact (paired binomial on discordant pairs).
  * MRR / log-loss: Wilcoxon signed-rank + paired bootstrap 95% CI on the
    per-item difference.

LEAKAGE CONTROL
---------------
Diagnosis queries are the FACTUAL narrative (narr_accf/narr_accp) -- the
sequence-of-events report, NOT NTSB's probable-cause prose -- reusing the
validated leakage-free stratum A from report Section 14.  We additionally report
the A-clean subset (factual text does not echo the cause; containment < 0.7).

OFFLINE / CACHING
-----------------
Runs with NO network by default: diagnosis reuses the cached factual-narrative
query embeddings in docs/qc_embed_*.{npy,json} (freshly-embedded narr_accf, report
Section 14); prognosis reuses each incident's narrative vector from the live
retrieval index (main_app.embeddings).  Pass --embed-missing to fill any gaps via
the OpenAI API (needs OPENAI_API_KEY + network).  Engine files
(zhang_diagnosis.py, prognosis.py, trees.py, main_app.py) are NOT modified.

Usage:
  PY=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11
  $PY tests/headtohead_scores.py                 # both tasks, offline
  $PY tests/headtohead_scores.py --task diagnosis --top-n-incidents 100
  $PY tests/headtohead_scores.py --task prognosis --top-k-neighbors 100
Outputs: docs/head_to_head_scores_results.json (machine-readable).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))

import main_app  # noqa: E402  (loads the 1982-2006 window retrieval index)
import zhang_diagnosis as zd  # noqa: E402
import prognosis as pg  # noqa: E402
import query_conditioning_validation as qcv  # noqa: E402

DOCS = ROOT / "docs"
RESULTS_PATH = DOCS / "head_to_head_scores_results.json"

GENERIC = {c.lower() for c in zd.GENERIC_CAUSES}
LOGLOSS_FLOOR = 1e-3          # clip for log-loss so a total miss is finite (~6.9)
LEAK_CONTAINMENT = qcv.LEAK_CONTAINMENT  # 0.7 (A-clean threshold, report Section 14)


# ============================================================================ #
# Generic scoring helpers (identical for every method)
# ============================================================================ #
def _strip_generic_order(order):
    return [c for c in order if c not in GENERIC]


def _strip_generic_prob(prob):
    sub = {c: p for c, p in prob.items() if c not in GENERIC}
    s = sum(sub.values())
    return {c: p / s for c, p in sub.items()} if s > 0 else {}


def top1(order, true_set):
    return bool(order) and order[0] in true_set


def topk(order, true_set, k):
    return any(p in true_set for p in order[:k])


def mrr(order, true_set):
    for i, p in enumerate(order, 1):
        if p in true_set:
            return 1.0 / i
    return 0.0


def p_hit(prob, true_set):
    return float(sum(prob.get(c, 0.0) for c in true_set))


def log_loss(prob, true_set):
    return -math.log(min(max(p_hit(prob, true_set), LOGLOSS_FLOOR), 1.0))


def brier(prob, true_set):
    return (1.0 - min(max(p_hit(prob, true_set), 0.0), 1.0)) ** 2


def score_item(order, prob, true_set):
    """All five metrics for one (ranked order, prob map, true label set)."""
    return {
        "top1": float(top1(order, true_set)),
        "top3": float(topk(order, true_set, 3)),
        "mrr": mrr(order, true_set),
        "logloss": log_loss(prob, true_set),
        "brier": brier(prob, true_set),
    }


# ============================================================================ #
# Significance
# ============================================================================ #
def mcnemar_exact(a_correct, b_correct):
    """Exact McNemar on paired binary outcomes a (narrative) vs b (baseline).

    Returns dict(b=narr-right/base-wrong, c=narr-wrong/base-right, p_value)."""
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    try:
        from scipy.stats import binomtest
        p = binomtest(min(b, c), n, 0.5).pvalue
    except Exception:  # noqa: BLE001
        from scipy.stats import binom
        k = min(b, c)
        p = min(1.0, 2.0 * binom.cdf(k, n, 0.5))
    return {"b": b, "c": c, "p_value": float(p)}


def paired_diff_stats(a_vals, b_vals, n_boot=10000, seed=0):
    """Paired difference a-b: mean, 95% bootstrap CI, Wilcoxon p (lower=better aware)."""
    d = np.asarray(a_vals, float) - np.asarray(b_vals, float)
    n = len(d)
    out = {"n": n, "mean_diff": float(d.mean()) if n else 0.0,
           "frac_pos": float((d > 0).mean()) if n else 0.0,
           "frac_neg": float((d < 0).mean()) if n else 0.0}
    if n >= 2:
        rng = np.random.default_rng(seed)
        means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
        out["ci95"] = [float(np.percentile(means, 2.5)),
                       float(np.percentile(means, 97.5))]
    else:
        out["ci95"] = [0.0, 0.0]
    try:
        from scipy.stats import wilcoxon
        if np.any(d != 0):
            stat, p = wilcoxon(d)
            out["wilcoxon_p"] = float(p)
        else:
            out["wilcoxon_p"] = 1.0
    except Exception as e:  # noqa: BLE001
        out["wilcoxon_p"] = float("nan")
        out["wilcoxon_err"] = str(e)
    return out


def _mean(vals):
    return float(np.mean(vals)) if vals else float("nan")


# ============================================================================ #
# TASK 1 -- DIAGNOSIS
# ============================================================================ #
def _det_order(prob):
    """Deterministic ranking: by -probability then label (hash-seed independent)."""
    return sorted(prob, key=lambda c: (-prob[c], c))


def build_global_majority(ds, exclude_generic_floor=False):
    """Outcome-agnostic global cause frequency (the majority-class floor).

    P(cause) = (# incidents whose node set contains the cause) / (# incidents).
    Same cause-label space as the diagnosis true labels (lowercased findings +
    occurrences).  Returned as (ordered list, prob map)."""
    count = defaultdict(int)
    n = 0
    for inc in ds.values():
        labs = zd._incident_cause_labels(inc)
        labs = {c for c in labs if c}
        if not labs:
            continue
        n += 1
        for c in labs:
            count[c] += 1
    total = sum(count.values()) or 1
    prob = {c: v / total for c, v in count.items()}
    return _det_order(prob), prob


def run_diagnosis(top_n_incidents=100, embed_missing=False, limit=None):
    qcv._DS = main_app.refined_dataset
    ds = main_app.refined_dataset

    cache = qcv.load_emb_cache()
    items = qcv.build_eval_set(limit=limit)
    items = [it for it in items if it["stratum"] == "A"]  # factual narratives only
    if embed_missing:
        qcv.embed_missing({x["key"]: x["query"] for x in items}, cache)
    items = [it for it in items if it["key"] in cache]
    print(f"[diagnosis] eval incidents (factual, cached): {len(items)}",
          file=sys.stderr)

    maj_order_full, maj_prob_full = build_global_majority(ds)
    maj_order_spec = _strip_generic_order(maj_order_full)
    maj_prob_spec = _strip_generic_prob(maj_prob_full)

    records = []
    for j, it in enumerate(items, 1):
        ev = it["ev"]
        vec = cache[it["key"]]
        pop = qcv._population(it["targets"])
        neighbors = qcv.neighbor_ev_ids(vec, ev, top_n_incidents)

        _, narr_prob, pool_n = qcv._dist_restricted(pop, neighbors, ev)
        _, zhang_prob = qcv._dist_excluding(pop, ev)
        narr_order = _det_order(narr_prob)
        zhang_order = _det_order(zhang_prob)

        true_all = set(it["true_all"])
        true_spec = set(it["true_spec"])

        rec = {"ev": ev, "containment": it["containment"],
               "clean": it["containment"] < LEAK_CONTAINMENT,
               "generic_only": it["generic_only"], "pool_cond": pool_n}

        # generic-INCLUDED scoring
        rec["incl"] = {
            "narr": score_item(narr_order, narr_prob, true_all),
            "zhang": score_item(zhang_order, zhang_prob, true_all),
            "maj": score_item(maj_order_full, maj_prob_full, true_all),
        }
        # generic-EXCLUDED scoring (only if a specific true cause exists)
        if true_spec:
            n_o, z_o = _strip_generic_order(narr_order), _strip_generic_order(zhang_order)
            n_p, z_p = _strip_generic_prob(narr_prob), _strip_generic_prob(zhang_prob)
            rec["excl"] = {
                "narr": score_item(n_o, n_p, true_spec),
                "zhang": score_item(z_o, z_p, true_spec),
                "maj": score_item(maj_order_spec, maj_prob_spec, true_spec),
            }
        records.append(rec)
        if j % 200 == 0:
            print(f"   ...diagnosis {j}/{len(items)}", file=sys.stderr)

    return _aggregate_diagnosis(records, top_n_incidents)


def _agg_block(records, space, subset_pred):
    sub = [r for r in records if space in r and subset_pred(r)]
    if not sub:
        return None
    metrics = ("top1", "top3", "mrr", "logloss", "brier")
    means = {m: {meth: _mean([r[space][meth][m] for r in sub])
                 for meth in ("narr", "zhang", "maj")} for m in metrics}
    # significance narrative vs zhang
    sig = {
        "top1_mcnemar": mcnemar_exact(
            [bool(r[space]["narr"]["top1"]) for r in sub],
            [bool(r[space]["zhang"]["top1"]) for r in sub]),
        "mrr_paired": paired_diff_stats(
            [r[space]["narr"]["mrr"] for r in sub],
            [r[space]["zhang"]["mrr"] for r in sub]),
        # for log-loss lower is better: report narr-zhang (negative => narr better)
        "logloss_paired": paired_diff_stats(
            [r[space]["narr"]["logloss"] for r in sub],
            [r[space]["zhang"]["logloss"] for r in sub]),
    }
    return {"n": len(sub), "means": means, "sig_narr_vs_zhang": sig}


def _aggregate_diagnosis(records, top_n_incidents):
    out = {"meta": {"task": "diagnosis", "n_records": len(records),
                    "top_n_incidents": top_n_incidents,
                    "logloss_floor": LOGLOSS_FLOOR,
                    "mean_pool_cond": _mean([r["pool_cond"] for r in records])}}
    allp = lambda r: True               # noqa: E731
    cleanp = lambda r: r["clean"]       # noqa: E731
    out["incl_all"] = _agg_block(records, "incl", allp)
    out["incl_clean"] = _agg_block(records, "incl", cleanp)
    out["excl_all"] = _agg_block(records, "excl", allp)
    out["excl_clean"] = _agg_block(records, "excl", cleanp)
    return out


# ============================================================================ #
# TASK 2 -- PROGNOSIS
# ============================================================================ #
def _incident_pairs(inc):
    """Unique consecutive (a, b) occurrence transitions in one incident (ordered)."""
    descs, _ = pg._ordered_occurrences(inc)
    pairs = []
    seen = set()
    for i in range(len(descs) - 1):
        ab = (descs[i], descs[i + 1])
        if ab not in seen:
            seen.add(ab)
            pairs.append(ab)
    return pairs


def _index_vectors():
    """ev_id -> narrative embedding from the live retrieval index (offline)."""
    vecs = {}
    for i, m in enumerate(main_app.embeddings_map):
        ev = m.get("ev_id")
        if ev and ev not in vecs:
            vecs[ev] = main_app.embeddings[i]
    return vecs


def _dist_from_adj(counter):
    """(ordered next-events, prob map) from a Counter of next-event counts."""
    total = sum(counter.values())
    if total <= 0:
        return [], {}
    prob = {b: c / total for b, c in counter.items()}
    order = sorted(prob, key=lambda b: (-prob[b], b))
    return order, prob


def run_prognosis(top_k_neighbors=100, embed_missing=False, limit=None):
    ds = main_app.refined_dataset
    indexed = {m.get("ev_id") for m in main_app.embeddings_map}

    # per-incident transition pairs + global leave-self-out adjacency
    inc_pairs = {}
    global_adj = defaultdict(Counter)       # a -> Counter(b) over ALL incidents
    self_adj = {}                            # ev -> {a -> Counter(b)} for subtraction
    for ev, inc in ds.items():
        pairs = _incident_pairs(inc)
        if len(pairs) < 1:
            continue
        inc_pairs[ev] = pairs
        sadj = defaultdict(Counter)
        for a, b in pairs:
            global_adj[a][b] += 1
            sadj[a][b] += 1
        self_adj[ev] = sadj

    vecs = _index_vectors()

    # eval incidents: have >=1 transition, a narrative vector, and a usable narrative
    eval_evs = [ev for ev in inc_pairs
                if ev in indexed and ev in vecs
                and qcv.factual_narrative(ds[ev])]
    if limit:
        eval_evs = eval_evs[:limit]
    print(f"[prognosis] eval incidents (>=1 transition, narrative, indexed): "
          f"{len(eval_evs)}", file=sys.stderr)

    records = []
    for j, ev in enumerate(eval_evs, 1):
        vec = vecs[ev]
        # narrative neighbours (self excluded)
        _, matches = main_app.find_top_matches(vec, exclude_ev_ids={ev})
        neigh, seen = [], set()
        for m in matches:
            if m.get("source") != "incident":
                continue
            e = m.get("ev_id")
            if e and e != ev and e not in seen:
                seen.add(e)
                neigh.append(e)
            if len(neigh) >= top_k_neighbors:
                break
        neigh_set = set(neigh)

        # neighbour adjacency a -> Counter(b)
        neigh_adj = defaultdict(Counter)
        for e in neigh_set:
            for a, b in inc_pairs.get(e, ()):
                neigh_adj[a][b] += 1

        sadj = self_adj[ev]
        for (a, b) in inc_pairs[ev]:
            # GLOBAL Markov, leave-self-out
            g = global_adj[a].copy()
            for bb, cc in sadj[a].items():
                g[bb] -= cc
                if g[bb] <= 0:
                    del g[bb]
            if sum(g.values()) <= 0:
                continue  # ill-posed: no other incident transitions out of `a`
            g_order, g_prob = _dist_from_adj(g)

            # NARRATIVE-conditioned: neighbour transitions only (self already excluded)
            n_order, n_prob = _dist_from_adj(neigh_adj.get(a, Counter()))
            covered = bool(n_order)

            true_set = {b}
            records.append({
                "ev": ev, "covered": covered,
                "narr": score_item(n_order, n_prob, true_set),
                "global": score_item(g_order, g_prob, true_set),
            })
        if j % 100 == 0:
            print(f"   ...prognosis {j}/{len(eval_evs)}", file=sys.stderr)

    return _aggregate_prognosis(records, top_k_neighbors)


def _agg_prog_block(records):
    if not records:
        return None
    metrics = ("top1", "top3", "mrr", "logloss", "brier")
    means = {m: {meth: _mean([r[meth][m] for r in records])
                 for meth in ("narr", "global")} for m in metrics}
    sig = {
        "top1_mcnemar": mcnemar_exact(
            [bool(r["narr"]["top1"]) for r in records],
            [bool(r["global"]["top1"]) for r in records]),
        "logloss_paired": paired_diff_stats(
            [r["narr"]["logloss"] for r in records],
            [r["global"]["logloss"] for r in records]),
        "top3_mcnemar": mcnemar_exact(
            [bool(r["narr"]["top3"]) for r in records],
            [bool(r["global"]["top3"]) for r in records]),
    }
    return {"n": len(records), "means": means, "sig_narr_vs_global": sig}


def _aggregate_prognosis(records, top_k_neighbors):
    covered = [r for r in records if r["covered"]]
    out = {"meta": {"task": "prognosis", "n_transitions": len(records),
                    "n_covered": len(covered),
                    "coverage": (len(covered) / len(records)) if records else 0.0,
                    "top_k_neighbors": top_k_neighbors,
                    "logloss_floor": LOGLOSS_FLOOR}}
    out["all_transitions"] = _agg_prog_block(records)      # A penalised where empty
    out["covered_only"] = _agg_prog_block(covered)         # where narrative can speak
    return out


# ============================================================================ #
# Reporting
# ============================================================================ #
def _verdict(narr, base, sig_p, higher_is_better=True, lo=None, hi=None):
    """Verdict string from a paired CI on (narr - base)."""
    if lo is None or hi is None:
        return "n/a"
    better = lo > 0 if higher_is_better else hi < 0
    worse = hi < 0 if higher_is_better else lo > 0
    if better and sig_p < 0.05:
        return "BETTER"
    if worse and sig_p < 0.05:
        return "WORSE"
    return "TIE"


def print_diag_block(name, blk):
    if not blk:
        print(f"\n[{name}] (no incidents)")
        return
    m = blk["means"]
    print(f"\n[{name}]  n={blk['n']}")
    print(f"  {'metric':9} {'NARRATIVE':>11} {'ZHANG':>11} {'MAJORITY':>11}")
    for met, hib in (("top1", True), ("top3", True), ("mrr", True),
                     ("logloss", False), ("brier", False)):
        arrow = "↑" if hib else "↓"
        print(f"  {met+arrow:9} {m[met]['narr']:>11.4f} {m[met]['zhang']:>11.4f} "
              f"{m[met]['maj']:>11.4f}")
    s = blk["sig_narr_vs_zhang"]
    mr, ll = s["mrr_paired"], s["logloss_paired"]
    mc = s["top1_mcnemar"]
    print(f"  significance (narrative vs ZHANG):")
    print(f"    top-1  McNemar b={mc['b']} c={mc['c']}  p={mc['p_value']:.2e}")
    print(f"    MRR    Δ={mr['mean_diff']:+.4f} CI[{mr['ci95'][0]:+.4f},"
          f"{mr['ci95'][1]:+.4f}] p={mr['wilcoxon_p']:.2e}  "
          f"-> {_verdict(None,None,mr['wilcoxon_p'],True,mr['ci95'][0],mr['ci95'][1])}")
    print(f"    logloss Δ={ll['mean_diff']:+.4f} CI[{ll['ci95'][0]:+.4f},"
          f"{ll['ci95'][1]:+.4f}] p={ll['wilcoxon_p']:.2e}  "
          f"-> {_verdict(None,None,ll['wilcoxon_p'],False,ll['ci95'][0],ll['ci95'][1])}")


def print_prog_block(name, blk):
    if not blk:
        print(f"\n[{name}] (no transitions)")
        return
    m = blk["means"]
    print(f"\n[{name}]  n={blk['n']}")
    print(f"  {'metric':9} {'NARRATIVE':>11} {'GLOBAL-MARKOV':>14}")
    for met, hib in (("top1", True), ("top3", True), ("mrr", True),
                     ("logloss", False), ("brier", False)):
        arrow = "↑" if hib else "↓"
        print(f"  {met+arrow:9} {m[met]['narr']:>11.4f} {m[met]['global']:>14.4f}")
    s = blk["sig_narr_vs_global"]
    mc, ll = s["top1_mcnemar"], s["logloss_paired"]
    print(f"  significance (narrative vs GLOBAL):")
    print(f"    top-1  McNemar b={mc['b']} c={mc['c']}  p={mc['p_value']:.2e}")
    print(f"    logloss Δ={ll['mean_diff']:+.4f} CI[{ll['ci95'][0]:+.4f},"
          f"{ll['ci95'][1]:+.4f}] p={ll['wilcoxon_p']:.2e}  "
          f"-> {_verdict(None,None,ll['wilcoxon_p'],False,ll['ci95'][0],ll['ci95'][1])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["diagnosis", "prognosis", "both"],
                    default="both")
    ap.add_argument("--top-n-incidents", type=int, default=100,
                    help="diagnosis retrieval depth")
    ap.add_argument("--top-k-neighbors", type=int, default=100,
                    help="prognosis narrative-neighbour count")
    ap.add_argument("--embed-missing", action="store_true",
                    help="fill missing query embeddings via OpenAI API (network)")
    ap.add_argument("--limit", type=int, default=None, help="smoke cap")
    args = ap.parse_args()

    import config
    print(f"index: {getattr(config, 'ACTIVE_INDEX_LABEL', '?')}  "
          f"dataset incidents: {len(main_app.refined_dataset)}", file=sys.stderr)

    results = {"meta": {"index_label": getattr(config, "ACTIVE_INDEX_LABEL", "?"),
                        "n_dataset": len(main_app.refined_dataset)}}

    if args.task in ("diagnosis", "both"):
        diag = run_diagnosis(args.top_n_incidents, args.embed_missing, args.limit)
        results["diagnosis"] = diag
        print("\n" + "=" * 72)
        print("TASK 1 -- DIAGNOSIS  (predict held-out incident's true cause)")
        print("=" * 72)
        print(f"mean conditioned pool size ~ {diag['meta']['mean_pool_cond']:.0f}")
        print_diag_block("generic INCLUDED -- all factual", diag["incl_all"])
        print_diag_block("generic INCLUDED -- A-clean", diag["incl_clean"])
        print_diag_block("generic EXCLUDED (fairer) -- all factual", diag["excl_all"])
        print_diag_block("generic EXCLUDED (fairer) -- A-clean", diag["excl_clean"])

    if args.task in ("prognosis", "both"):
        prog = run_prognosis(args.top_k_neighbors, args.embed_missing, args.limit)
        results["prognosis"] = prog
        print("\n" + "=" * 72)
        print("TASK 2 -- PROGNOSIS  (predict next event on held-out transitions)")
        print("=" * 72)
        print(f"coverage (narrative had neighbour data for `a`): "
              f"{prog['meta']['coverage']:.1%} "
              f"({prog['meta']['n_covered']}/{prog['meta']['n_transitions']})")
        print_prog_block("all held-out transitions", prog["all_transitions"])
        print_prog_block("covered subset (narrative can speak)", prog["covered_only"])

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nresults -> {RESULTS_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()

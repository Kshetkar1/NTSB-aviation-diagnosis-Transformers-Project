"""Does query-conditioning in the narrative diagnosis method actually WORK?

The single validation that decides whether "narrative diagnosis" is a real
contribution or a capability demo. Counting (Zhang, Section 4-5) already gives the
population distribution P(cause | outcome). The narrative method's ONLY unique
value is *per-incident conditioning*: given one incident's narrative, does
conditioning on it identify THAT incident's true cause better than the
unconditioned population prior? We test this on held-out incidents, paired, with
significance, and with strict leakage control.

DESIGN (leave-one-out, paired, leakage-aware)
---------------------------------------------
For each incident i with a usable narrative and a Zhang-edge true cause:
  * Outcome  = i's NTSB defining event (Defining_ev==1), else its terminal
    occurrence; expanded to its occurrence FAMILY via zhang_diagnosis.detect_outcome.
  * True cause(s) = zhang_diagnosis._causes_into_outcome(i, family)  (the exact
    Zhang edge logic used for Table 7 and the prior LOO harnesses).
  * CONDITIONED  prediction = embed i's narrative -> cosine retrieval over the
    window index with SELF EXCLUDED -> restrict to retrieved incidents that had the
    outcome -> P(cause | outcome, narrative-neighbors). (This is the conditioning
    mechanism of the narrative method, isolated.)
  * UNCONDITIONED baseline   = P(cause | outcome) over ALL outcome accidents with
    SELF EXCLUDED -- i.e. the *same* diagnosis pipeline with the narrative removed.
    This is the prior with NO narrative (the key baseline). top-1 of this baseline
    is the majority-class-given-outcome, so it subsumes the prior majority baseline.

The headline question is the PAIRED difference conditioned - unconditioned, not
absolute accuracy.

METRICS (per incident, then aggregated): top-1, top-3, MRR, and mean LIFT in the
probability MASS assigned to the true cause(s) (conditioned - unconditioned).

SIGNIFICANCE: Wilcoxon signed-rank + paired bootstrap 95% CI on the per-incident
lift, computed per stratum.

LEAKAGE CONTROL (mandatory):
  Stratum A = query is the FACTUAL narrative (narr_accf, fallback narr_accp) -- the
              sequence-of-events report. The honest test.
  Stratum B = query is the NTSB PROBABLE-CAUSE prose (narr_cause) -- which states
              the cause. The leakage-inflated ceiling.
  Within A we further sub-split A-clean vs A-leak by token-containment of the
  probable-cause statement inside the factual narrative (reusing the catch-all
  decomposition idea that factual prose can still echo the cause).

Further stratification: generic vs specific true cause (GENERIC_CAUSES), so we can
see whether conditioning helps where it should (specific mechanisms).

This DIFFERS from the prior near-base-rate LOO (report Section 10.5,
tests/loo_specific.py) in three ways: (1) the baseline is the matched unconditioned
population *distribution* compared PAIRED per incident (not an unpaired
majority-class number); (2) it runs over ALL incidents/outcomes (defining-event
outcome), not 3 hand-picked outcomes; (3) it adds the A-vs-B leakage stratification
and paired significance on the per-incident lift.

A NULL / NEGATIVE result (conditioning does not beat the unconditioned baseline in
the leakage-free stratum A) is a valid, important outcome and is reported with
numbers, not inflated.

Usage
-----
  python3.11 tests/query_conditioning_validation.py            # full run (network)
  python3.11 tests/query_conditioning_validation.py --max 60   # smoke test
  python3.11 tests/query_conditioning_validation.py --top-n-incidents 100
Outputs: docs/query_conditioning_results.json, docs/query_conditioning_validation.png
Embeddings are cached to docs/qc_embed_*.{npy,json} (resumable; reused across runs).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zlib
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "docs" / ".mplcache"))

import main_app  # noqa: E402  (loads the window index)
import zhang_diagnosis as zd  # noqa: E402

DOCS = ROOT / "docs"
RESULTS_PATH = DOCS / "query_conditioning_results.json"
FIG_PATH = DOCS / "query_conditioning_validation.png"
EMB_VECS_PATH = DOCS / "qc_embed_vecs.npy"
EMB_KEYS_PATH = DOCS / "qc_embed_keys.json"

GENERIC = {c.lower() for c in zd.GENERIC_CAUSES}
MIN_NARR = 80          # min chars for a usable narrative (matches prior LOO)
MAX_QUERY_CHARS = 6000  # truncate long factual reports before embedding
LEAK_CONTAINMENT = 0.7  # >= this fraction of cause tokens inside factual -> A-leak

_STOP = {"the", "a", "an", "of", "was", "is", "were", "are", "in", "on", "to",
         "and", "or", "but", "with", "during", "flight", "aircraft", "there",
         "had", "has", "that", "this", "it", "its", "for", "from", "at", "by",
         "as", "be", "been", "being", "which", "resulted", "due", "into", "no"}


def _toks(t: str) -> set:
    return {w for w in re.sub(r"[^a-z0-9 ]", " ", (t or "").lower()).split()
            if w not in _STOP and len(w) > 2}


# --- narrative selection ------------------------------------------------------
def factual_narrative(inc: dict) -> str:
    for k in ("narr_accf", "narr_accp"):
        t = (inc.get(k) or "").strip()
        if len(t) >= MIN_NARR:
            return t
    return ""


def cause_prose(inc: dict) -> str:
    t = (inc.get("narr_cause") or "").strip()
    return t if len(t) >= 1 else ""


def containment(cause: str, factual: str) -> float:
    ct, ft = _toks(cause), _toks(factual)
    if not ct or not ft:
        return 0.0
    return len(ct & ft) / len(ct)


# --- outcome / true-cause -----------------------------------------------------
def defining_or_last_label(inc: dict) -> str | None:
    seq = sorted(inc.get("sequence_of_events", []),
                 key=lambda s: int(s.get("Occurrence_No") or 0))
    if not seq:
        return None
    for s in seq:
        if s.get("Defining_ev") in (1, "1", True):
            d = (s.get("Occurrence_Description") or "").strip()
            if d:
                return d
    for s in reversed(seq):
        d = (s.get("Occurrence_Description") or "").strip()
        if d:
            return d
    return None


@lru_cache(maxsize=None)
def _detect_outcome_cached(label: str):
    det = zd.detect_outcome(label, dataset=_DS)
    if det is None:
        return None
    name, targets = det
    return name, frozenset(targets)


# --- population cache (leave-self-out P(cause|outcome)) ------------------------
# Per outcome family (frozenset targets): {ev_id: frozenset(cause labels)} over all
# outcome accidents in the dataset, computed once. Leave-self-out distributions are
# derived by subtracting the self incident's contribution -- exact and fast.
_POP_CACHE: dict = {}


def _population(targets: frozenset) -> dict:
    if targets in _POP_CACHE:
        return _POP_CACHE[targets]
    tset = set(targets)
    ev_causes: dict[str, frozenset] = {}
    cause_count: dict[str, int] = defaultdict(int)
    for ev_id, inc in _DS.items():
        causes = zd._causes_into_outcome(inc, tset)
        if not causes:
            continue
        cl = frozenset(c.strip().lower() for c in causes if c.strip())
        ev_causes[ev_id] = cl
        for c in cl:
            cause_count[c] += 1
    _POP_CACHE[targets] = {"ev_causes": ev_causes, "count": dict(cause_count),
                           "total": len(ev_causes)}
    return _POP_CACHE[targets]


def _dist_excluding(pop: dict, exclude_ev: str) -> tuple[list, dict]:
    """Leave-self-out unconditioned P(cause|outcome): ordered list + prob map."""
    count = dict(pop["count"])
    total = pop["total"]
    self_causes = pop["ev_causes"].get(exclude_ev)
    if self_causes is not None:
        total -= 1
        for c in self_causes:
            count[c] -= 1
            if count[c] <= 0:
                del count[c]
    if total <= 0:
        return [], {}
    prob = {c: n / total for c, n in count.items()}
    order = sorted(prob, key=lambda c: -prob[c])
    return order, prob


def _dist_restricted(pop: dict, allow_ev_ids, exclude_ev: str) -> tuple[list, dict, int]:
    """Conditioned P(cause|outcome, neighbors): counts over (neighbors n outcome)."""
    count: dict[str, int] = defaultdict(int)
    total = 0
    ev_causes = pop["ev_causes"]
    for ev in allow_ev_ids:
        if ev == exclude_ev:
            continue
        cl = ev_causes.get(ev)
        if cl is None:
            continue
        total += 1
        for c in cl:
            count[c] += 1
    if total <= 0:
        return [], {}, 0
    prob = {c: n / total for c, n in count.items()}
    order = sorted(prob, key=lambda c: -prob[c])
    return order, prob, total


def _random_outcome_pool(pop: dict, size: int, exclude_ev: str, seed: int) -> list:
    """A random set of `size` outcome accidents (self excluded) -- the
    concentration control: same pool SIZE as the conditioned neighborhood but with
    NO narrative relevance, so it isolates true conditioning signal from the pure
    distribution-concentration artifact of diagnosing over a smaller pool."""
    pool = [e for e in pop["ev_causes"] if e != exclude_ev]
    if size <= 0 or not pool:
        return []
    rng = np.random.default_rng(seed)
    if size >= len(pool):
        return pool
    idx = rng.choice(len(pool), size=size, replace=False)
    return [pool[i] for i in idx]


# --- metrics ------------------------------------------------------------------
def _specific(seq):
    return [c for c in seq if c not in GENERIC]


def rank_top(pred_order, true_set, k):
    return any(p in true_set for p in pred_order[:k])


def mrr(pred_order, true_set):
    for i, p in enumerate(pred_order, 1):
        if p in true_set:
            return 1.0 / i
    return 0.0


def mass(prob_map, true_set):
    return sum(prob_map.get(c, 0.0) for c in true_set)


# --- embeddings (cached, resumable) -------------------------------------------
def load_emb_cache() -> dict:
    if EMB_VECS_PATH.is_file() and EMB_KEYS_PATH.is_file():
        keys = json.loads(EMB_KEYS_PATH.read_text())
        vecs = np.load(EMB_VECS_PATH)
        return {k: vecs[i] for i, k in enumerate(keys)}
    return {}


def save_emb_cache(cache: dict) -> None:
    keys = list(cache.keys())
    if not keys:
        return
    vecs = np.vstack([cache[k] for k in keys]).astype(np.float32)
    np.save(EMB_VECS_PATH, vecs)
    EMB_KEYS_PATH.write_text(json.dumps(keys))


def embed_missing(need: dict, cache: dict, batch: int = 128) -> None:
    """need: {key: text}. Fills cache for keys not present (network)."""
    todo = [(k, t) for k, t in need.items() if k not in cache]
    if not todo:
        return
    from config import EMBEDDING_MODEL
    client = main_app.get_client()
    print(f"embedding {len(todo)} query texts (model={EMBEDDING_MODEL})...",
          file=sys.stderr)
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        resp = client.embeddings.create(
            input=[t[:MAX_QUERY_CHARS].replace("\n", " ") for _, t in chunk],
            model=EMBEDDING_MODEL,
        )
        for (k, _), d in zip(chunk, resp.data):
            cache[k] = np.asarray(d.embedding, dtype=np.float32)
        if (i // batch) % 4 == 0:
            print(f"  ...{min(i + batch, len(todo))}/{len(todo)}", file=sys.stderr)
            save_emb_cache(cache)
    save_emb_cache(cache)


# --- retrieval ----------------------------------------------------------------
def neighbor_ev_ids(query_vec: np.ndarray, self_ev: str, top_n: int) -> set:
    _, matches = main_app.find_top_matches(query_vec, exclude_ev_ids={self_ev})
    out, seen = [], set()
    for m in matches:
        if m.get("source") != "incident":
            continue
        ev = m.get("ev_id")
        if ev and ev != self_ev and ev not in seen:
            seen.add(ev)
            out.append(ev)
        if len(out) >= top_n:
            break
    return set(out)


# --- significance -------------------------------------------------------------
def paired_stats(diffs, n_boot=10000, seed=0):
    d = np.asarray([x for x in diffs], dtype=float)
    n = len(d)
    res = {"n": n, "mean_diff": float(d.mean()) if n else 0.0,
           "frac_pos": float((d > 0).mean()) if n else 0.0,
           "frac_neg": float((d < 0).mean()) if n else 0.0,
           "frac_zero": float((d == 0).mean()) if n else 0.0}
    if n >= 2:
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        means = np.empty(n_boot)
        for b in range(n_boot):
            means[b] = d[rng.integers(0, n, n)].mean()
        res["boot_ci95"] = [float(np.percentile(means, 2.5)),
                            float(np.percentile(means, 97.5))]
    else:
        res["boot_ci95"] = [0.0, 0.0]
    try:
        from scipy.stats import wilcoxon
        if np.any(d != 0):
            stat, p = wilcoxon(d)
            res["wilcoxon_stat"], res["wilcoxon_p"] = float(stat), float(p)
        else:
            res["wilcoxon_stat"], res["wilcoxon_p"] = float("nan"), 1.0
    except Exception as e:  # noqa: BLE001
        res["wilcoxon_stat"], res["wilcoxon_p"] = float("nan"), float("nan")
        res["wilcoxon_err"] = str(e)
    return res


# --- main evaluation ----------------------------------------------------------
def build_eval_set(limit=None):
    """List of dicts: ev, stratum, query_text, key, targets, true_all, true_spec,
    containment, generic_only."""
    items = []
    indexed = {c.get("ev_id") for c in main_app.embeddings_map}
    for ev, inc in _DS.items():
        if ev not in indexed:
            continue
        label = defining_or_last_label(inc)
        if not label:
            continue
        det = _detect_outcome_cached(label)
        if det is None:
            continue
        name, targets = det
        true = zd._causes_into_outcome(inc, set(targets))
        true_all = {c.strip().lower() for c in true if c.strip()}
        if not true_all:
            continue
        true_spec = set(_specific(true_all))
        fact = factual_narrative(inc)
        cause = cause_prose(inc)
        cont = containment(cause, fact) if (cause and fact) else 0.0
        if fact:
            items.append({"ev": ev, "stratum": "A", "query": fact,
                          "key": f"{ev}|A", "name": name, "targets": targets,
                          "true_all": true_all, "true_spec": true_spec,
                          "containment": cont,
                          "generic_only": len(true_spec) == 0})
        if cause and len(cause) >= MIN_NARR:
            items.append({"ev": ev, "stratum": "B", "query": cause,
                          "key": f"{ev}|B", "name": name, "targets": targets,
                          "true_all": true_all, "true_spec": true_spec,
                          "containment": cont,
                          "generic_only": len(true_spec) == 0})
    if limit:
        # keep a balanced-ish smoke sample: first `limit` A and first `limit` B
        a = [x for x in items if x["stratum"] == "A"][:limit]
        b = [x for x in items if x["stratum"] == "B"][:limit]
        items = a + b
    return items


def evaluate(items, top_n_incidents, cache):
    records = []
    n = len(items)
    for j, it in enumerate(items, 1):
        vec = cache.get(it["key"])
        if vec is None:
            continue
        pop = _population(it["targets"])
        ev = it["ev"]
        neighbors = neighbor_ev_ids(vec, ev, top_n_incidents)

        u_order, u_prob = _dist_excluding(pop, ev)
        c_order, c_prob, pool_n = _dist_restricted(pop, neighbors, ev)
        # concentration control: random outcome pool of the SAME size as the
        # conditioned neighborhood (deterministic seed per incident).
        rand_pool = _random_outcome_pool(pop, pool_n, ev, seed=zlib.crc32(ev.encode()))
        r_order, r_prob, _ = _dist_restricted(pop, rand_pool, ev)

        rec = {"ev": ev, "stratum": it["stratum"], "name": it["name"],
               "containment": it["containment"], "generic_only": it["generic_only"],
               "pool_cond": pool_n, "pool_uncond": pop["total"] - (1 if ev in pop["ev_causes"] else 0)}

        for space, true_set, c_ord, u_ord, r_ord in (
            ("all", it["true_all"], c_order, u_order, r_order),
            ("spec", it["true_spec"], _specific(c_order), _specific(u_order), _specific(r_order)),
        ):
            if space == "spec" and not true_set:
                continue
            rec[space] = {
                "cond_top1": rank_top(c_ord, true_set, 1),
                "cond_top3": rank_top(c_ord, true_set, 3),
                "cond_mrr": mrr(c_ord, true_set),
                "cond_mass": mass(c_prob, true_set),
                "unc_top1": rank_top(u_ord, true_set, 1),
                "unc_top3": rank_top(u_ord, true_set, 3),
                "unc_mrr": mrr(u_ord, true_set),
                "unc_mass": mass(u_prob, true_set),
                "rand_top1": rank_top(r_ord, true_set, 1),
                "rand_top3": rank_top(r_ord, true_set, 3),
                "rand_mrr": mrr(r_ord, true_set),
                "rand_mass": mass(r_prob, true_set),
            }
        records.append(rec)
        if j % 100 == 0:
            print(f"  evaluated {j}/{n}", file=sys.stderr)
    return records


# --- aggregation --------------------------------------------------------------
def aggregate(records, predicate, space):
    sub = [r for r in records if space in r and predicate(r)]
    if not sub:
        return None
    out = {"n": len(sub)}
    for metric in ("top1", "top3", "mrr", "mass"):
        cond = np.array([r[space][f"cond_{metric}"] for r in sub], float)
        unc = np.array([r[space][f"unc_{metric}"] for r in sub], float)
        rand = np.array([r[space][f"rand_{metric}"] for r in sub], float)
        out[metric] = {
            "cond": float(cond.mean()),
            "unc": float(unc.mean()),
            "rand": float(rand.mean()),
            "lift_vs_unc": float((cond - unc).mean()),
            "lift_vs_rand": float((cond - rand).mean()),
        }
    # mass: conditioned vs unconditioned prior (confounded by concentration)
    out["mass_paired_vs_unc"] = paired_stats(
        [r[space]["cond_mass"] - r[space]["unc_mass"] for r in sub])
    # mass: conditioned vs random equal-size pool (concentration-controlled signal)
    out["mass_paired_vs_rand"] = paired_stats(
        [r[space]["cond_mass"] - r[space]["rand_mass"] for r in sub])
    # rank (MRR): conditioned vs unconditioned, and vs random pool
    out["mrr_paired_vs_unc"] = paired_stats(
        [r[space]["cond_mrr"] - r[space]["unc_mrr"] for r in sub])
    out["mrr_paired_vs_rand"] = paired_stats(
        [r[space]["cond_mrr"] - r[space]["rand_mrr"] for r in sub])
    out["mean_pool_cond"] = float(np.mean([r["pool_cond"] for r in sub]))
    return out


def make_groups():
    A = lambda r: r["stratum"] == "A"  # noqa: E731
    B = lambda r: r["stratum"] == "B"  # noqa: E731
    return {
        "A_factual_all": (A, "all"),
        "A_factual_specific": (A, "spec"),
        "A_clean_specific": (lambda r: A(r) and r["containment"] < LEAK_CONTAINMENT, "spec"),
        "A_leak_specific": (lambda r: A(r) and r["containment"] >= LEAK_CONTAINMENT, "spec"),
        "B_causeprose_all": (B, "all"),
        "B_causeprose_specific": (B, "spec"),
    }


def main():
    global _DS
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None, help="cap incidents per stratum (smoke)")
    ap.add_argument("--top-n-incidents", type=int, default=100)
    ap.add_argument("--no-embed", action="store_true", help="use only cached embeddings")
    args = ap.parse_args()

    _DS = main_app.refined_dataset
    print(f"dataset: {len(_DS)} incidents; index label: "
          f"{getattr(__import__('config'), 'ACTIVE_INDEX_LABEL', '?')}", file=sys.stderr)

    items = build_eval_set(limit=args.max)
    nA = sum(1 for x in items if x["stratum"] == "A")
    nB = sum(1 for x in items if x["stratum"] == "B")
    print(f"eval set: {len(items)} (A factual={nA}, B cause-prose={nB})", file=sys.stderr)

    cache = load_emb_cache()
    if not args.no_embed:
        embed_missing({x["key"]: x["query"] for x in items}, cache)
    have = sum(1 for x in items if x["key"] in cache)
    print(f"embeddings available: {have}/{len(items)}", file=sys.stderr)

    records = evaluate(items, args.top_n_incidents, cache)

    groups = make_groups()
    summary = {}
    for gname, (pred, space) in groups.items():
        summary[gname] = aggregate(records, pred, space)

    # generic vs specific TRUE cause (stratum A), all-space
    summary["A_true_generic_only_all"] = aggregate(
        records, lambda r: r["stratum"] == "A" and r["generic_only"], "all")
    summary["A_true_has_specific_all"] = aggregate(
        records, lambda r: r["stratum"] == "A" and not r["generic_only"], "all")

    meta = {
        "n_records": len(records),
        "top_n_incidents": args.top_n_incidents,
        "min_narrative_chars": MIN_NARR,
        "leak_containment_threshold": LEAK_CONTAINMENT,
        "generic_causes": sorted(GENERIC),
        "index_label": getattr(__import__("config"), "ACTIVE_INDEX_LABEL", "?"),
        "n_A": nA, "n_B": nB,
        "design": "LOO; conditioned=narrative-retrieval P(cause|outcome,neighbors) "
                  "vs unconditioned=P(cause|outcome) population prior; self excluded "
                  "from both; paired per incident.",
    }
    out = {"meta": meta, "summary": summary}
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {RESULTS_PATH}")

    _print_console(summary)
    try:
        _make_figure(summary)
        print(f"figure  -> {FIG_PATH}")
    except Exception as e:  # noqa: BLE001
        print(f"(figure skipped: {e})", file=sys.stderr)


def _fmt(g):
    if not g:
        return "  (no incidents)"
    mu = g["mass_paired_vs_unc"]
    mr = g["mass_paired_vs_rand"]
    return (
        f"  n={g['n']:>4}  pool~{g['mean_pool_cond']:.0f}\n"
        f"    top1  cond {g['top1']['cond']:.1%}  unc {g['top1']['unc']:.1%}  rand {g['top1']['rand']:.1%} "
        f"(vs unc {g['top1']['lift_vs_unc']:+.1%}, vs rand {g['top1']['lift_vs_rand']:+.1%})\n"
        f"    MRR   cond {g['mrr']['cond']:.3f}  unc {g['mrr']['unc']:.3f}  rand {g['mrr']['rand']:.3f} "
        f"(vs unc {g['mrr']['lift_vs_unc']:+.3f}, vs rand {g['mrr']['lift_vs_rand']:+.3f}, "
        f"p_rand={g['mrr_paired_vs_rand']['wilcoxon_p']:.2e})\n"
        f"    mass  cond-unc {mu['mean_diff']:+.4f} CI[{mu['boot_ci95'][0]:+.4f},{mu['boot_ci95'][1]:+.4f}] p={mu['wilcoxon_p']:.1e}  |  "
        f"cond-rand {mr['mean_diff']:+.4f} CI[{mr['boot_ci95'][0]:+.4f},{mr['boot_ci95'][1]:+.4f}] p={mr['wilcoxon_p']:.1e}")


def _print_console(summary):
    print("\n" + "=" * 88)
    print("QUERY-CONDITIONING VALIDATION  (cond / uncond ; lift = cond - uncond)")
    print("=" * 88)
    order = ["A_factual_all", "A_factual_specific", "A_clean_specific",
             "A_leak_specific", "B_causeprose_all", "B_causeprose_specific",
             "A_true_generic_only_all", "A_true_has_specific_all"]
    for k in order:
        print(f"\n[{k}]")
        print(_fmt(summary.get(k)))


def _make_figure(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("A clean\n(specific)", "A_clean_specific"),
              ("A all\n(specific)", "A_factual_specific"),
              ("A leak\n(specific)", "A_leak_specific"),
              ("B cause-prose\n(specific)", "B_causeprose_specific")]
    labels = []
    mrr_unc, mrr_rand = [], []   # MRR lift vs each baseline (the honest rank signal)
    for lab, key in groups:
        g = summary.get(key)
        if not g:
            continue
        labels.append(lab)
        mrr_unc.append(g["mrr"]["lift_vs_unc"])
        mrr_rand.append(g["mrr"]["lift_vs_rand"])
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, mrr_unc, w, label="vs unconditioned prior", color="#1f77b4")
    ax.bar(x + w / 2, mrr_rand, w, label="vs random equal-size pool", color="#ff7f0e")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("mean MRR lift for the TRUE cause\n(conditioned - baseline)")
    ax.set_title("Does query-conditioning rank the true cause higher?\n"
                 "(MRR lift; ~0 = conditioning adds no per-incident ranking value)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=130)


if __name__ == "__main__":
    main()

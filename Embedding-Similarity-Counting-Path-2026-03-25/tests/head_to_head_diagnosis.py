"""HEAD-TO-HEAD: Zhang counting (Method A) vs Narrative similarity-mass LTP (Method B).

Research question: does the student's NARRATIVE-driven diagnosis method give
similar or better results than Zhang's counting method?

Both methods estimate a distribution over causes for a given outcome.

  METHOD A -- ZHANG COUNTING (baseline, Table 7):
      zhang_diagnosis.empirical_cause_distribution(name, targets=, dataset=...)
      with NO restrict_ev_ids -> P(cause | outcome) over the FULL population.
      The query only selects the OUTCOME; wording is otherwise ignored.

  METHOD B -- NARRATIVE SIMILARITY-MASS LTP (student's original engine):
      main_app.diagnose_with_conditional_probabilities(query, top_n, top_n_incidents)
      = embed query -> cosine retrieve -> cluster -> Law of Total Probability
        P(C|Q) = sum_K P(C|K) * P(K|Q), with P(K|Q) weighted by SIMILARITY MASS.
      The query WORDING drives retrieval and therefore the distribution.

Neither engine file is modified; this script only reads from them.

Run (framework python + network for the embedding/clustering API):
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
        tests/head_to_head_diagnosis.py
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main_app  # noqa: E402  (loads the active retrieval index on import)
import zhang_diagnosis as zd  # noqa: E402
import sparse_cpt as sc  # noqa: E402


# --------------------------------------------------------------------------- #
# Data hygiene helpers
# --------------------------------------------------------------------------- #
def s(x) -> str:
    """NaN/None/float-safe string coercion."""
    return str(x or "").strip()


def normalize_label(label) -> str:
    """Fair-comparison normalization applied to BOTH methods' cause labels.

    - lowercase
    - strip everything after the first comma (drops the trailing modifier, e.g.
      'landing gear, main gear strut' -> 'landing gear')
    - collapse internal whitespace

    This is intentionally lossy and symmetric: it is the only way Method A's
    clean dictionary labels and Method B's modified finding labels can land on a
    common key. Narrative-prose causes (full sentences from Method B) almost
    never collapse onto a Method A label and are reported as 'unmatched'.
    """
    t = s(label).lower()
    t = t.split(",")[0]
    t = " ".join(t.split())
    return t


def aggregate_distribution(pairs) -> dict:
    """Collapse a list of (raw_label, prob) onto normalized labels by summing.

    Returns {normalized_label: summed_probability}. Empty labels are skipped.
    """
    agg: dict[str, float] = defaultdict(float)
    for raw, p in pairs:
        key = normalize_label(raw)
        if not key:
            continue
        agg[key] += float(p or 0.0)
    return dict(agg)


def renorm(dist: dict, support=None) -> dict:
    """Renormalize a {label: value} map to sum to 1 over `support` (or its keys)."""
    keys = list(support) if support is not None else list(dist)
    total = sum(max(dist.get(k, 0.0), 0.0) for k in keys)
    if total <= 0:
        return {k: 0.0 for k in keys}
    return {k: max(dist.get(k, 0.0), 0.0) / total for k in keys}


# --------------------------------------------------------------------------- #
# Method runners (engine code used AS-IS)
# --------------------------------------------------------------------------- #
def run_method_a(query: str):
    """Method A: Zhang counting over the FULL outcome population (Table 7 method).

    Returns (outcome_name, outcome_count, {raw_label: P(cause|outcome)}, raw_rows).
    """
    det = zd.detect_outcome(query, dataset=main_app.refined_dataset)
    if det is None:
        return None, 0, {}, []
    name, targets = det
    res = zd.empirical_cause_distribution(
        name, targets=targets, dataset=main_app.refined_dataset
    )  # no restrict_ev_ids -> full population
    raw = {s(c["cause"]): float(c["probability"]) for c in res["causes"]}
    return name, res["outcome_count"], raw, res["causes"]


def run_method_b(query: str, top_n=200, top_n_incidents=50):
    """Method B: narrative similarity-mass LTP (student's original engine).

    Returns ({raw_label: P(cause|query)}, weighted_causes_rows, meta).
    """
    res = main_app.diagnose_with_conditional_probabilities(
        query, top_n=top_n, top_n_incidents=top_n_incidents
    )
    rows = res.get("weighted_causes", []) or []
    raw = {s(r["cause"]): float(r["probability"]) for r in rows}
    meta = {
        "incidents_analyzed": res.get("total_incidents_analyzed"),
        "clusters": res.get("total_clusters"),
        "error": res.get("error"),
    }
    return raw, rows, meta


# --------------------------------------------------------------------------- #
# Comparison metrics
# --------------------------------------------------------------------------- #
def topk_labels(norm_dist: dict, k: int):
    return [lab for lab, _ in sorted(norm_dist.items(), key=lambda x: -x[1])[:k]]


def jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def compare(name_a: dict, name_b: dict):
    """Quantitative comparison of two normalized {label: prob} distributions."""
    A = name_a
    B = name_b
    shared = sorted(set(A) & set(B))
    union = sorted(set(A) | set(B))

    # Spearman over shared causes (rank by probability within each method).
    rho, pval, n_shared = float("nan"), float("nan"), len(shared)
    if n_shared >= 3:
        va = [A[k] for k in shared]
        vb = [B[k] for k in shared]
        rho, pval = spearmanr(va, vb)

    # Top-K overlap.
    overlap = {}
    for k in (5, 10):
        ta, tb = topk_labels(A, k), topk_labels(B, k)
        overlap[k] = {
            "shared": len(set(ta) & set(tb)),
            "jaccard": jaccard(ta, tb),
        }

    # L1 over the union, each side renormalized to a proper distribution on union.
    Ar = renorm(A, union)
    Br = renorm(B, union)
    l1 = sum(abs(Ar[k] - Br[k]) for k in union)

    return {
        "n_a": len(A),
        "n_b": len(B),
        "n_shared": n_shared,
        "n_union": len(union),
        "spearman_rho": rho,
        "spearman_p": pval,
        "overlap": overlap,
        "l1": l1,
    }


def fmt_dist_row(rank, label, prob, width=52):
    lab = (label[: width - 1] + "\u2026") if len(label) > width else label
    return f"  {rank:>2}. {lab:<{width}} {prob:>8.4f}"


def print_side_by_side(title, A_norm, B_norm, k=10):
    print(f"\n  TOP-{k} CAUSES  --  Method A (Zhang count)   |   Method B (Narrative LTP)")
    ta = sorted(A_norm.items(), key=lambda x: -x[1])[:k]
    tb = sorted(B_norm.items(), key=lambda x: -x[1])[:k]
    print(f"  {'#':>2}  {'A: P(cause|outcome)':<46} {'B: P(cause|query)':<46}")
    print("  " + "-" * 94)
    for i in range(k):
        la = f"{ta[i][0][:34]:<34} {ta[i][1]:>7.4f}" if i < len(ta) else ""
        lb = f"{tb[i][0][:34]:<34} {tb[i][1]:>7.4f}" if i < len(tb) else ""
        print(f"  {i+1:>2}  {la:<46} {lb:<46}")


def find_label(norm_dist: dict, *keywords):
    """Highest-prob normalized label containing ALL keywords; (label, rank, prob)."""
    ranked = sorted(norm_dist.items(), key=lambda x: -x[1])
    for rank, (lab, p) in enumerate(ranked, start=1):
        if all(kw in lab for kw in keywords):
            return lab, rank, p
    return None, None, 0.0


# --------------------------------------------------------------------------- #
# EXPERIMENT 1 -- agreement / validation
# --------------------------------------------------------------------------- #
def experiment_1():
    print("\n" + "=" * 98)
    print("EXPERIMENT 1 -- AGREEMENT (does Narrative LTP recover Zhang's population ranking?)")
    print("=" * 98)

    outcomes = [
        ("fire", "What is the probability of fire?"),
        ("loss of engine power", "What is the probability of loss of engine power?"),
        ("gear collapse", "What is the probability of gear collapse?"),
    ]

    summary = []
    for label, query in outcomes:
        print("\n" + "-" * 98)
        print(f"OUTCOME: {label!r}   QUERY: {query!r}")
        print("-" * 98)

        a_name, a_count, a_raw, _ = run_method_a(query)
        b_raw, _, b_meta = run_method_b(query)

        if a_name is None:
            print("  [Method A] no outcome detected -- skipping")
            continue
        if b_meta.get("error"):
            print(f"  [Method B] error: {b_meta['error']} -- skipping")
            continue

        A = aggregate_distribution(a_raw.items())
        B = aggregate_distribution(b_raw.items())

        print(f"  Method A: outcome={a_name!r}  outcome_accidents={a_count}  "
              f"raw_causes={len(a_raw)} -> normalized={len(A)}")
        print(f"  Method B: incidents_analyzed={b_meta['incidents_analyzed']}  "
              f"clusters={b_meta['clusters']}  raw_causes={len(b_raw)} -> normalized={len(B)}")

        m = compare(A, B)
        print(f"\n  ALIGNMENT: shared(normalized)={m['n_shared']}  union={m['n_union']}  "
              f"A-only={m['n_a']-m['n_shared']}  B-only={m['n_b']-m['n_shared']}")
        rho = m["spearman_rho"]
        print(f"  Spearman rho (shared causes) = "
              f"{rho:.3f}" if rho == rho else "  Spearman rho = n/a (<3 shared)")
        print(f"  Top-5  overlap: {m['overlap'][5]['shared']}/5  "
              f"(Jaccard {m['overlap'][5]['jaccard']:.3f})")
        print(f"  Top-10 overlap: {m['overlap'][10]['shared']}/10 "
              f"(Jaccard {m['overlap'][10]['jaccard']:.3f})")
        print(f"  L1 distance (renormalized over union) = {m['l1']:.3f}  (0=identical, 2=disjoint)")

        print_side_by_side(label, A, B, k=10)

        summary.append((label, m))

    # Compact summary table
    print("\n" + "=" * 98)
    print("EXPERIMENT 1 SUMMARY")
    print("=" * 98)
    print(f"  {'outcome':<24} {'shared':>7} {'rho':>8} {'top5':>6} {'top10':>7} {'L1':>7}")
    print("  " + "-" * 64)
    for label, m in summary:
        rho = m["spearman_rho"]
        rho_s = f"{rho:.3f}" if rho == rho else "n/a"
        print(f"  {label:<24} {m['n_shared']:>7} {rho_s:>8} "
              f"{m['overlap'][5]['shared']:>4}/5 {m['overlap'][10]['shared']:>5}/10 "
              f"{m['l1']:>7.3f}")
    return summary


# --------------------------------------------------------------------------- #
# EXPERIMENT 2 -- narrative added value (the novelty case)
# --------------------------------------------------------------------------- #
def experiment_2():
    print("\n\n" + "=" * 98)
    print("EXPERIMENT 2 -- NARRATIVE ADDED VALUE (does query wording shift the distribution?)")
    print("=" * 98)
    print("  Method A is STATIC per outcome (same answer regardless of wording).")
    print("  Method B should move the query-relevant cause UP if narratives add value.")

    # (label, detailed query, keywords identifying the query-relevant cause)
    cases = [
        ("fire / electrical wiring",
         "fire after electrical wiring arced behind the instrument panel",
         ["wiring"]),
        ("fire / fuel ignition",
         "fire when leaking fuel ignited near a hot engine exhaust",
         ["fuel"]),
        ("loss of engine power / fuel exhaustion",
         "engine lost power after fuel exhaustion on approach",
         ["fuel", "exhaustion"]),
    ]

    for label, query, kws in cases:
        print("\n" + "-" * 98)
        print(f"CASE: {label}\n  QUERY: {query!r}")
        print("-" * 98)

        a_name, a_count, a_raw, _ = run_method_a(query)
        b_raw, _, b_meta = run_method_b(query)
        if a_name is None or b_meta.get("error"):
            print(f"  skipped (A outcome={a_name}, B error={b_meta.get('error')})")
            continue

        A = aggregate_distribution(a_raw.items())
        B = aggregate_distribution(b_raw.items())

        print(f"  Detected outcome (both use same outcome): {a_name!r}  "
              f"(A population={a_count})")

        # Locate the query-relevant cause in each distribution.
        la, ra, pa = find_label(A, *kws)
        lb, rb, pb = find_label(B, *kws)
        print(f"\n  Query-relevant cause keyword(s): {kws}")
        print(f"   Method A: label={la!r}  rank={ra}  P={pa:.4f}")
        print(f"   Method B: label={lb!r}  rank={rb}  P={pb:.4f}")
        if ra and rb:
            print(f"   SHIFT: rank {ra} -> {rb}  ({'UP' if rb < ra else 'DOWN/SAME'} "
                  f"by {ra-rb:+d}),  prob {pa:.4f} -> {pb:.4f} ({pb-pa:+.4f})")
        elif rb and not ra:
            print(f"   Method B surfaces this cause (rank {rb}); Method A does not list it at all.")

        # Show each method's top-5 so the reader sees the divergence.
        print_side_by_side(label, A, B, k=5)

    # Direct demonstration: counting is wording-invariant, narrative is not.
    print("\n" + "-" * 98)
    print("WORDING-INVARIANCE CHECK: two DIFFERENT fire narratives")
    print("-" * 98)
    q1 = "fire after electrical wiring arced behind the instrument panel"
    q2 = "fire when leaking fuel ignited near a hot engine exhaust"
    _, _, a1, _ = run_method_a(q1)
    _, _, a2, _ = run_method_a(q2)
    b1, _, _ = run_method_b(q1)
    b2, _, _ = run_method_b(q2)
    A1, A2 = aggregate_distribution(a1.items()), aggregate_distribution(a2.items())
    B1, B2 = aggregate_distribution(b1.items()), aggregate_distribution(b2.items())
    union_a = sorted(set(A1) | set(A2))
    union_b = sorted(set(B1) | set(B2))
    l1_a = sum(abs(renorm(A1, union_a)[k] - renorm(A2, union_a)[k]) for k in union_a)
    l1_b = sum(abs(renorm(B1, union_b)[k] - renorm(B2, union_b)[k]) for k in union_b)
    print(f"  Method A: L1 between the two fire wordings = {l1_a:.4f}  "
          f"(0.0 expected -> counting ignores wording)")
    print(f"  Method B: L1 between the two fire wordings = {l1_b:.4f}  "
          f"(>0 -> narrative responds to wording)")


# --------------------------------------------------------------------------- #
# EXPERIMENT 3 -- sparse-cell stability (forward CPT P(outcome | cause))
# --------------------------------------------------------------------------- #
def experiment_3(outcome="fire", max_support=3, n_examples=6):
    print("\n\n" + "=" * 98)
    print("EXPERIMENT 3 -- SPARSE-CELL STABILITY  (forward CPT P(outcome | cause))")
    print("=" * 98)
    print(f"  Outcome = {outcome!r}.  We pick causes with SMALL support and compare:")
    print("    raw_count  = n(cause&outcome)/n(cause)        [Method-A-style counting; noisy]")
    print("    beta_cdf   = Zhang's parametric smoothing")
    print("    semantic   = student's neighbor smoothing (k nearest narratives had outcome?)")

    ds = main_app.refined_dataset
    count, total, n_out = sc._outcome_cause_counts(outcome, ds)
    print(f"\n  {outcome!r}: outcome accidents={n_out}, total contribution units={total}")

    # Sparse causes: small count(cause & outcome). Skip the two generic catch-alls.
    generic = zd.GENERIC_CAUSES
    sparse = sorted(
        ((c, n) for c, n in count.items()
         if 1 <= n <= max_support and normalize_label(c) not in generic and len(c) > 4),
        key=lambda x: (x[1], x[0]),
    )[:n_examples]

    if not sparse:
        print("  (no sparse causes found under the support threshold)")
        return

    print(f"\n  {'cause (support n with outcome)':<46} {'raw_count':>16} {'beta_cdf':>12} {'semantic(k=50)':>16}")
    print("  " + "-" * 92)
    for cause, n_co in sparse:
        labels = [cause]
        rc, rco, rcn = sc.raw_count_cpt(labels, outcome, ds)
        contrib, bc = sc.beta_cdf_cpt(labels, count, total)
        sm, smo, smn = sc.semantic_cpt(cause, outcome, k=50, main_app=main_app)
        nm = (cause[:42] + "\u2026") if len(cause) > 43 else cause
        print(f"  {nm:<44}({n_co:>2}) {rc:>7.3f}({rco:>2}/{rcn:>3}) "
              f"{bc:>11.3f} {sm:>9.3f}({smo:>2}/{smn:>2})")

    print("\n  Reading: raw_count jumps to 0/low or 1.0 on tiny denominators (unstable);")
    print("  semantic pools k=50 nearest narratives, so the estimate is bounded and smoother.")


def main():
    print("#" * 98)
    print("# HEAD-TO-HEAD DIAGNOSIS: Zhang counting (A) vs Narrative similarity-mass LTP (B)")
    print(f"# Active index: {main_app.ACTIVE_INDEX_LABEL if hasattr(main_app,'ACTIVE_INDEX_LABEL') else 'see config'}")
    print(f"# Dataset incidents loaded: {len(main_app.refined_dataset)}")
    print("#" * 98)

    experiment_1()
    experiment_2()
    experiment_3()

    print("\n" + "#" * 98)
    print("# DONE")
    print("#" * 98)


if __name__ == "__main__":
    main()

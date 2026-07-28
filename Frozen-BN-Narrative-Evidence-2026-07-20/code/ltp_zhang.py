"""Zhang-consistent LTP diagnosis — Maha's chain-rule decomposition with Zhang's
counting inside every cluster.

    P(cause | Q) = sum_K  P(cause | K) * P(K | Q)

Ingredients (all Zhang-consistent):
  * The partition {K}: every outcome accident (e.g. all 102 fires) is assigned to
    exactly ONE cluster via its precomputed `cluster_label` (unlabeled accidents
    form the "(unclustered)" bucket, so the partition is exhaustive + disjoint).
  * P(cause | K): Zhang's Table-7 estimator (`empirical_cause_distribution`,
    cause_factor_only=True) restricted to cluster K -- same labels, same edge
    logic, same per-accident denominator as the published table.
  * P(K | Q): the ONLY place the query enters.
      - weights="neutral":    P(K) = N_K / N_total  (no query information)
      - weights="similarity": P(K|Q) proportional to the summed retrieval
        similarity mass of K's members (query-specific tilt).

THEOREM (exact Table-7 recovery). With neutral weights the cluster structure
cancels algebraically:

    sum_K  count(cause & K)/N_K * N_K/N  =  count(cause)/N  =  Table 7 exactly.

So the LTP output equals Zhang's published Table 7 to machine precision when the
query carries no information, and the deviation under similarity weights IS the
narrative signal -- decomposed per cluster so it can be audited.

Validated by tests/ltp_zhang_recovery.py (offline, exit 0 == theorem holds).
This module IMPORTS zhang_diagnosis and never edits it.
"""
from __future__ import annotations

from collections import defaultdict

import zhang_diagnosis as zd

UNCLUSTERED = "(unclustered)"


def cluster_partition(targets: set, ds: dict) -> dict:
    """Partition ALL outcome accidents into {cluster_label: [ev_ids]}.

    Every accident with the outcome occurrence lands in exactly one bucket
    (its precomputed `cluster_label`, else UNCLUSTERED), so the buckets are
    disjoint and exhaustive over the outcome population -- the property the
    exact-recovery theorem needs."""
    part: dict[str, list] = defaultdict(list)
    for ev, inc in ds.items():
        if not zd._has_outcome(inc, targets):
            continue
        label = (inc.get("cluster_label") or "").strip() or UNCLUSTERED
        part[label].append(ev)
    return dict(part)


def cluster_weights(partition: dict, weights: str = "neutral",
                    sim_by_ev: dict | None = None) -> dict:
    """P(K | Q) over the partition.

    neutral    : N_K / N_total  (query-free -- the exact-recovery case)
    similarity : sum of retrieval similarity over K's members, renormalized.
                 Accidents absent from `sim_by_ev` contribute 0 mass (they were
                 not surfaced by the query), but every cluster keeps a tiny
                 floor so the weights remain a distribution over the partition.
    """
    if weights == "neutral":
        total = sum(len(evs) for evs in partition.values())
        return {k: len(evs) / total for k, evs in partition.items()}
    if weights != "similarity":
        raise ValueError(f"unknown weights mode: {weights!r}")
    if not sim_by_ev:
        raise ValueError("weights='similarity' requires sim_by_ev={ev_id: score}")
    mass = {k: sum(max(float(sim_by_ev.get(ev, 0.0)), 0.0) for ev in evs)
            for k, evs in partition.items()}
    total = sum(mass.values())
    if total <= 0:  # query surfaced nothing -> fall back to neutral
        return cluster_weights(partition, "neutral")
    return {k: m / total for k, m in mass.items()}


def ltp_diagnose(query: str = None, outcome: str = None, dataset: dict = None,
                 weights: str = "neutral", sim_by_ev: dict | None = None,
                 exclude_responses: bool = False, top_n: int = 30) -> dict:
    """Zhang-consistent LTP: P(cause|Q) = sum_K P(cause|K) * P(K|Q).

    Provide either `query` (outcome detected from free text) or `outcome`
    (an occurrence label, e.g. "fire"). `dataset` defaults to the engine's
    active window. Returns the ranked causes, the per-cluster decomposition
    (Maha's chain-rule breakdown), and meta describing the weight mode.
    """
    ds = dataset if dataset is not None else zd._load()

    if outcome is not None:
        name, targets = outcome, zd.OUTCOME_ALIASES.get(outcome, {outcome})
        targets = {t.lower() for t in targets}
    else:
        det = zd.detect_outcome(query, dataset=ds)
        if det is None:
            return {"error": "no recognized outcome in query", "query": query}
        name, targets = det
        targets = {t.lower() for t in targets}

    partition = cluster_partition(targets, ds)
    if not partition:
        return {"error": f"no accidents with outcome {name!r}", "outcome": name}
    w = cluster_weights(partition, weights=weights, sim_by_ev=sim_by_ev)

    # Zhang's estimator inside each cluster (same labels/edge logic/denominator
    # convention as the published Table 7).
    agg: dict[str, float] = defaultdict(float)
    n_by_cause: dict[str, int] = defaultdict(int)
    clusters_out = []
    for k, evs in sorted(partition.items(), key=lambda x: -len(x[1])):
        res = zd.empirical_cause_distribution(
            name, targets=targets, dataset=ds, restrict_ev_ids=evs,
            cause_factor_only=True,
        )
        contribs = []
        for c in res["causes"]:
            lab = c["cause"]
            if exclude_responses and lab.lower() in zd.DIAGNOSIS_RESPONSE_LABELS:
                continue
            contribution = c["probability"] * w[k]
            agg[lab] += contribution
            n_by_cause[lab] += c["n"]
            contribs.append({"cause": lab, "p_cause_given_cluster": c["probability"],
                             "n": c["n"], "contribution": contribution})
        contribs.sort(key=lambda x: -x["contribution"])
        clusters_out.append({
            "cluster": k,
            "n_accidents": len(evs),
            "weight": w[k],
            "top_contributions": contribs[:5],
        })

    causes = sorted(
        ({"cause": lab, "probability": p, "n": n_by_cause[lab]}
         for lab, p in agg.items()),
        key=lambda d: -d["probability"],
    )
    total_outcome = sum(len(evs) for evs in partition.values())
    return {
        "outcome": name,
        "outcome_labels": sorted(targets),
        "outcome_count": total_outcome,
        "weights_mode": weights,
        "n_clusters": len(partition),
        "causes": causes[:top_n],
        "clusters": clusters_out,
        "prob_semantics": (
            "P(cause|Q) = sum_K P(cause|K) * P(K|Q); P(cause|K) is Zhang Table-7 "
            "counting restricted to cluster K. weights='neutral' recovers the "
            "published Table 7 exactly (see tests/ltp_zhang_recovery.py)."
        ),
    }

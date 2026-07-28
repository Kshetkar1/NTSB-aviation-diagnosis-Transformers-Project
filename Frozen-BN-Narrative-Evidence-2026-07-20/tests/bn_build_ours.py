#!/usr/bin/env python3
"""BUILD OUR OWN Bayesian network from OUR pipeline's probabilities and run the
Table 9 queries with open-source inference (pyAgrum). ZERO inputs from Zhang's
NTSB.xdsl — his published numbers are used only as the comparison target.

Construction (Zhang's Section 4 recipe, re-implemented over OUR refined dataset):
  * graph      : prognosis.build_graph on data/processed/refined_dataset_1982_2006.json
                 (finding->occurrence, occurrence->occurrence, last->damage/injury)
  * priors     : P(root) = from-occurrence count / 184,517,128 flights (Eq. 6)
  * edge ratio : len(edge ev_ids) / len(node ev_ids), 0.95 cap on 1.0 (Eq. 8 + his code cap)
  * CPT cells  : 0 active parents -> 0 ; 1 active -> that parent's ratio ;
                 >=2 active -> Beta-CDF(sum(active)/sum(all)) with (a,b) calibrated
                 PER CHILD on its own parent ratios (his obj_beta / Nelder-Mead),
                 floored by max(active ratio)  (Eqs. 10-14, constructCPT)
  * parent cap : 12 (his maxElements), keep highest-ratio parents (deterministic
                 tie-break instead of his random jitter)
  * cycles     : self-loops dropped (as he does); other back-edges dropped weakest-
                 first until the graph is a DAG (recorded in the report)

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 tests/bn_build_ours.py
Writes outputs/bn_ours_results.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyagrum as gum
import scipy.stats
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
_SHARED = REPO_ROOT / "shared" / "code"
_FROZEN_CODE = FROZEN_DIR / "code"
for _p in (_SHARED, _FROZEN_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
ROOT = REPO_ROOT
import prognosis as pg  # noqa: E402

TOTAL_FLIGHTS = 184_517_128  # Zhang Eq. 6 denominator (BTS 1982-2006, reproduced)
MAX_PARENTS = 12             # his maxElements
OUT = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs" / "bn_ours_results.json"


# ---------------------------------------------------------------------------------
# Zhang's per-node Beta-CDF calibration (obj_beta + Nelder-Mead, x0=[2,1])
# ---------------------------------------------------------------------------------
def calibrate_beta(ratios: np.ndarray) -> tuple:
    norm = ratios / ratios.sum()

    def obj(x):
        pred = scipy.stats.beta.cdf(norm, a=x[0], b=x[1])
        return float(np.mean((pred - ratios) ** 2))

    res = minimize(obj, [2.0, 1.0], method="Nelder-Mead", tol=1e-7)
    return float(res.x[0]), float(res.x[1])


def cpt_yes(scheme: np.ndarray, ratios: np.ndarray, ab: tuple) -> float:
    """One CPT cell, Zhang's constructCPT logic."""
    active = scheme.astype(bool)
    k = int(active.sum())
    if k == 0:
        return 0.0
    if k == 1:
        return float(ratios[active][0])
    contribution = float(ratios[active].sum() / ratios.sum())
    yes = float(scipy.stats.beta.cdf(contribution, a=ab[0], b=ab[1]))
    return max(yes, float(ratios[active].max()))


# ---------------------------------------------------------------------------------
# Build the network
# ---------------------------------------------------------------------------------
def build_network(ds: dict):
    edge_events, node_events = pg.build_graph(ds)

    # edge ratios (Eq. 8 + 0.95 cap), self-loops dropped
    ratio = {}
    for (a, b), evs in edge_events.items():
        if a == b:
            continue
        denom = node_events.get(a)
        if not denom:
            continue
        r = len(evs) / len(denom)
        ratio[(a, b)] = r * 0.95 if r == 1.0 else r

    # incoming parents per child, capped at MAX_PARENTS by ratio desc, tie-broken by
    # edge support (deterministic replacement for Zhang's random jitter)
    parents_of: dict = {}
    for (a, b), r in ratio.items():
        n_evs = len(edge_events[(a, b)])
        parents_of.setdefault(b, []).append((a, r, n_evs))
    for b, plist in parents_of.items():
        plist.sort(key=lambda t: (-t[1], -t[2], t[0]))
        parents_of[b] = [(a, r) for a, r, _ in plist[:MAX_PARENTS]]

    all_nodes = set()
    for (a, b) in ratio:
        all_nodes.add(a)
        all_nodes.add(b)

    bn = gum.BayesNet("NTSB_ours")
    for n in sorted(all_nodes):
        v = gum.LabelizedVariable(n, n, 0)
        v.addLabel("Yes")
        v.addLabel("No")
        bn.add(v)

    # add arcs child by child; drop weakest arc on any cycle conflict (recorded)
    dropped = []
    kept_parents: dict = {}
    for b in sorted(parents_of, key=lambda x: len(parents_of[x])):
        kept = []
        for a, r in parents_of[b]:
            try:
                bn.addArc(a, b)
                kept.append((a, r))
            except gum.InvalidDirectedCycle:
                dropped.append((a, b, r))
        kept_parents[b] = kept

    # priors for roots: from-occurrence count / total flights (his occurrenceTimesFromID)
    from_counts = {}
    for (a, _b), evs in edge_events.items():
        from_counts.setdefault(a, set()).update(evs)
    n_roots = 0
    for n in all_nodes:
        if bn.parents(bn.idFromName(n)):
            continue
        n_roots += 1
        p = len(from_counts.get(n, ())) / TOTAL_FLIGHTS
        bn.cpt(n).fillWith([p, 1.0 - p])

    # child CPTs: iterate all parent-state combinations, Zhang's cell rule
    for b, plist in kept_parents.items():
        if not plist:
            if b in all_nodes and not bn.parents(bn.idFromName(b)):
                continue  # already filled as root
        if not plist:
            continue
        names = [a for a, _ in plist]
        ratios = np.array([r for _, r in plist], dtype=float)
        ab = calibrate_beta(ratios) if len(ratios) >= 2 else (1.0, 1.0)
        cpt = bn.cpt(b)
        n = len(names)
        for mask in range(2 ** n):
            scheme = np.array([(mask >> i) & 1 for i in range(n)], dtype=int)
            yes = cpt_yes(scheme, ratios, ab)
            # pyAgrum instantiation: state 0 = Yes, 1 = No for each parent
            idx = {names[i]: (0 if scheme[i] else 1) for i in range(n)}
            cpt[{**idx}] = [yes, 1.0 - yes]

    meta = {"nodes": bn.size(), "arcs": bn.sizeArcs(), "roots": n_roots,
            "dropped_cycle_edges": len(dropped),
            "dropped_examples": [f"{a} -> {b} (r={r:.3f})" for a, b, r in dropped[:8]]}
    return bn, meta


# ---------------------------------------------------------------------------------
# Queries (same as the parity script, but on OUR node vocabulary)
# ---------------------------------------------------------------------------------
def find_node(bn, *keywords):
    """First node whose name contains ALL keywords (case-insensitive)."""
    kws = [k.lower() for k in keywords]
    for n in bn.names():
        low = n.lower()
        if all(k in low for k in kws):
            return n
    return None


def posterior(bn, target, evidence=None, exact=True):
    ie = gum.LazyPropagation(bn) if exact else gum.LoopyBeliefPropagation(bn)
    if not exact:
        ie.setMaxIter(200)
    if evidence:
        ie.setEvidence(evidence)
    if exact:
        ie.addTarget(target)
    ie.makeInference()
    v = bn.variable(target)
    yes = [i for i in range(v.domainSize()) if v.label(i) == "Yes"][0]
    return float(ie.posterior(target)[yes])


def main():
    ds = pg.load_dataset()
    print(f"dataset: {len(ds)} incidents (1982-2006 window)")
    bn, meta = build_network(ds)
    print(f"OUR network: {meta['nodes']} nodes, {meta['arcs']} arcs, "
          f"{meta['roots']} roots, {meta['dropped_cycle_edges']} cycle edges dropped")

    results = {"meta": meta, "anchors": [], "downstream": []}

    # --- forward anchors (Zhang Table 9 fwd: 0.95 / 0.50 / 0.95) -------------------
    # The three causes feed the GENERIC 'loss of engine power' occurrence node in our
    # graph (same node Zhang's Table 9 column conditions on).
    loep = "loss of engine power"
    # Evidence nodes: the generic 'engine instrument' finding (1/1 -> 0.95 cap) is the
    # analogue of Zhang's published anchor; the EPR-gauge sibling carries a 1/2 = 0.50
    # ratio in our window (2 incidents) and is NOT the published cell.
    anchors = [
        ("engine instrument", 0.95, "engine instruments"),
        (find_node(bn, "combustion", "liner"), 0.50, "combustion liner"),
        (find_node(bn, "fluid, oil grade") or find_node(bn, "oil grade"), 0.95, "oil grade"),
    ]
    print(f"\nLOEP node: {loep!r}")
    loep_parents = [bn.variable(p).name() for p in bn.parents(bn.idFromName(loep))]
    print(f"LOEP parents kept ({len(loep_parents)}): {loep_parents}")
    print("(A) forward anchors on OUR network (exact inference):")
    for node, zhang, tag in anchors:
        if node is None:
            print(f"  {tag}: node not found in our graph")
            continue
        # find which LOEP variant this cause actually feeds
        p = posterior(bn, loep, {node: "Yes"})
        results["anchors"].append({"evidence": node, "target": loep,
                                   "ours": p, "zhang": zhang})
        print(f"  P({loep[:38]} | {tag:18}) = {p:.4f}   Zhang {zhang}")

    # --- downstream under LOEP=Yes --------------------------------------------------
    targets = [
        (find_node(bn, "forced landing"), 0.1429),
        (find_node(bn, "ditching"), 4.61e-3),
        (find_node(bn, "destroyed"), 5.59e-3),
        (find_node(bn, "serious injury"), 8.22e-3),
        (find_node(bn, "no injury"), 0.9899),
    ]
    print("\n(B) downstream on OUR network, evidence LOEP=Yes (exact where feasible):")
    for node, zhang in targets:
        if node is None:
            continue
        try:
            p = posterior(bn, node, {loep: "Yes"})
            engine = "exact"
        except Exception:
            p = posterior(bn, node, {loep: "Yes"}, exact=False)
            engine = "loopy"
        results["downstream"].append({"target": node, "ours": p, "zhang": zhang,
                                      "engine": engine})
        print(f"  P({node[:28]:28} | LOEP) = {p:.4e}   Zhang {zhang:.4g}   [{engine}]")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

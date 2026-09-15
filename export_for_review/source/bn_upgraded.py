#!/usr/bin/env python3
"""UPGRADED network — our two methodological fixes on top of the faithful
Zhang-recipe reproduction (tests/bn_build_ours.py):

  FIX 1 — person findings (human factors). Our refined dataset already carries
    `person_description` on every finding row (pilot-in-command, flightcrew,
    ATC, ...). Zhang wires person -> subject(finding); we add exactly that edge.
    This unlocks the Fig 12 pilot-error queries that were previously
    "not replicable".

  FIX 2 — multi-state severity nodes. Instead of Zhang's independent Boolean
    leaves (whose 0-active->0 CPT rule forces P(no injury) ~ 0 and cannot
    express the empirical 0.82), we model ONE `personnel injury` node
    (fatal/serious/minor/none) and ONE `aircraft damage` node
    (destroyed/substantial/minor/none) whose states sum to 1:
      * parents  : last-occurrence nodes (Zhang's own attachment semantics),
                   capped at 12 by support (deterministic)
      * 0 active : all mass on the default state ('none' / 'no injury') --
                   "nothing happened" means no injury, by definition
      * k active : support-weighted mixture of the active parents' empirical
                   severity distributions (counted from the data)
    Injury level per accident uses Zhang's own derivation recovered from his
    released code (per-person injury.xlsx, worst level, default none).

  Everything else (priors, edge ratios, Beta-CDF CPTs for event nodes) is
  unchanged from the faithful reproduction.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 tests/bn_upgraded.py
Writes outputs/bn_upgraded.json.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyagrum as gum

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import prognosis as pg  # noqa: E402
import bn_build_ours as builder  # noqa: E402
from bn_zhang_injury_derivation import zhang_injury_label  # noqa: E402

OUT = ROOT / "outputs" / "bn_upgraded.json"
TOTAL_FLIGHTS = builder.TOTAL_FLIGHTS
MAX_PARENTS = builder.MAX_PARENTS

INJ_STATES = ["fatal injury", "serious injury", "minor injury", "no injury"]
DMG_STATES = ["destroyed aircraft", "substantial damage", "minor damage", "no damage"]
DMG_BY_CODE = {"DEST": 0, "SUBS": 1, "MINR": 2, "NONE": 3, "UNK": 3}
INJ_NODE, DMG_NODE = "personnel injury", "aircraft damage"


def person_label(f: dict) -> str:
    p = str(f.get("person_description") or "").strip().lower()
    if not p or p in ("nan", "none", "0", "(unspecified person)"):
        return ""
    return f"person: {p}"


def build_edges_upgraded(inc: dict) -> set:
    """Zhang graph + person->finding edges. Severity leaves handled separately."""
    edges = set()
    occ_by_no = pg._occ_by_no(inc)
    for f in inc.get("findings", []):
        d = pg._s(f.get("finding_description")).lower()
        occ = occ_by_no.get(pg._s(f.get("Occurrence_No")))
        if d and occ and d != occ:
            edges.add((d, occ))
        per = person_label(f)
        if per and d and per != d:
            edges.add((per, d))          # FIX 1: Zhang's person -> subject edge
    descs, _ = pg._ordered_occurrences(inc)
    for i in range(len(descs) - 1):
        if descs[i] != descs[i + 1]:
            edges.add((descs[i], descs[i + 1]))
    return edges


def injury_state(inc: dict) -> int:
    return INJ_STATES.index(zhang_injury_label(inc))


def damage_state(inc: dict) -> int | None:
    code = pg._s(inc.get("damage")).upper()
    return DMG_BY_CODE.get(code)


def last_occurrence(inc: dict) -> str | None:
    descs, _ = pg._ordered_occurrences(inc)
    return descs[-1] if descs else None


def severity_parent_stats(ds: dict):
    """Per last-occurrence label: support + empirical severity distributions."""
    inj_counts: dict = defaultdict(lambda: np.zeros(4))
    dmg_counts: dict = defaultdict(lambda: np.zeros(4))
    support: Counter = Counter()
    for inc in ds.values():
        last = last_occurrence(inc)
        if not last:
            continue
        support[last] += 1
        inj_counts[last][injury_state(inc)] += 1
        d = damage_state(inc)
        if d is not None:
            dmg_counts[last][d] += 1
    return support, inj_counts, dmg_counts


def build_upgraded(ds: dict):
    # ---- event-node network (Zhang recipe, person edges added) -----------------
    orig = pg.build_edges
    pg.build_edges = build_edges_upgraded
    try:
        bn, meta = builder.build_network(ds)
    finally:
        pg.build_edges = orig

    # ---- severity nodes ---------------------------------------------------------
    support, inj_counts, dmg_counts = severity_parent_stats(ds)
    parents = [lab for lab, _ in support.most_common() if lab in bn.names()]
    parents = parents[:MAX_PARENTS]

    specs = [
        (INJ_NODE, INJ_STATES, inj_counts, 3),
        (DMG_NODE, DMG_STATES, dmg_counts, 3),
    ]
    for node_name, states, counts, default_idx in specs:
        v = gum.LabelizedVariable(node_name, node_name, 0)
        for s in states:
            v.addLabel(s)
        bn.add(v)
        for p in parents:
            bn.addArc(p, node_name)

        # per-parent empirical distributions
        dists, weights = [], []
        for p in parents:
            c = counts[p]
            tot = c.sum()
            dists.append(c / tot if tot > 0 else np.eye(4)[default_idx])
            weights.append(support[p])
        dists = np.array(dists)
        weights = np.array(weights, dtype=float)

        cpt = bn.cpt(node_name)
        n = len(parents)
        default = np.eye(4)[default_idx]
        for mask in range(2 ** n):
            active = np.array([(mask >> i) & 1 for i in range(n)], dtype=bool)
            if not active.any():
                dist = default                       # FIX 2: default state
            else:
                w = weights[active]
                dist = (dists[active] * w[:, None]).sum(axis=0) / w.sum()
                dist = dist / dist.sum()
            idx = {parents[i]: (0 if active[i] else 1) for i in range(n)}
            cpt[{**idx}] = list(dist)

    meta["severity_parents"] = parents
    meta["nodes_after"] = bn.size()
    return bn, meta


def post(bn, target, evidence, state):
    ie = gum.LazyPropagation(bn)
    if evidence:
        ie.setEvidence(evidence)
    ie.addTarget(target)
    ie.makeInference()
    v = bn.variable(target)
    i = [k for k in range(v.domainSize()) if v.label(k) == state][0]
    return float(ie.posterior(target)[i])


def empirical_severity(ds: dict, occ_targets: set):
    """Empirical injury/damage distribution among accidents containing any target occ."""
    inj = np.zeros(4); dmg = np.zeros(4); n = 0
    for inc in ds.values():
        descs, _ = pg._ordered_occurrences(inc)
        if not any(d in occ_targets for d in descs):
            continue
        n += 1
        inj[injury_state(inc)] += 1
        d = damage_state(inc)
        if d is not None:
            dmg[d] += 1
    return n, inj / max(n, 1), dmg / max(dmg.sum(), 1)


def main():
    ds = pg.load_dataset()
    bn, meta = build_upgraded(ds)
    print(f"UPGRADED network: {meta['nodes_after']} nodes, {bn.sizeArcs()} arcs")
    print(f"severity parents ({len(meta['severity_parents'])}): "
          f"{meta['severity_parents']}")

    results = {"meta": {k: v for k, v in meta.items() if k != 'dropped_examples'},
               "table9": [], "fig12": []}

    # ---------- Table 9 outcome rows, evidence LOEP=Yes --------------------------
    loep = "loss of engine power"
    loep_targets = pg.resolve_outcome_targets(loep, ds)
    n_emp, inj_emp, dmg_emp = empirical_severity(ds, loep_targets)
    print(f"\nempirical severity among {n_emp} LOEP accidents "
          f"(injury {np.round(inj_emp, 3)}, damage {np.round(dmg_emp, 3)})")

    rows = [
        (INJ_NODE, "no injury", 0.9899, inj_emp[3]),
        (INJ_NODE, "serious injury", 0.00822, inj_emp[1]),
        (DMG_NODE, "destroyed aircraft", 0.00559, dmg_emp[0]),
        (DMG_NODE, "substantial damage", 0.0166, dmg_emp[1]),
        (DMG_NODE, "minor damage", 0.00378, dmg_emp[2]),
    ]
    print(f"\nTable 9 LOEP column on UPGRADED network:")
    print(f"{'target':22} {'Zhang':>9} {'old BN':>10} {'upgraded':>10} {'empirical':>10}")
    old = {"no injury": 0.02173, "serious injury": 0.005315,
           "destroyed aircraft": 0.02564, "substantial damage": 0.007271,
           "minor damage": 0.5021}
    for node, state, zhang, emp in rows:
        p = post(bn, node, {loep: "Yes"}, state)
        results["table9"].append({"state": state, "zhang": zhang, "upgraded": p,
                                  "empirical": float(emp), "old": old[state]})
        print(f"{state:22} {zhang:9.4g} {old[state]:10.4g} {p:10.4g} {emp:10.4g}")

    prior_ni = post(bn, INJ_NODE, None, "no injury")
    print(f"\nprior P(no injury): upgraded {prior_ni:.4f}   Zhang published 0.9999   "
          f"(old Boolean net: 6.3e-7)")
    results["prior_no_injury"] = prior_ni

    # ---------- Fig 12 pilot-error queries (previously NOT REPLICABLE) -----------
    pilot = "person: pilot-in-command"
    if pilot not in bn.names():
        print(f"\n{pilot!r} not in network!")
    else:
        # empirical: among accidents with a pilot-in-command finding
        def has_pilot(inc):
            return any(person_label(f) == pilot for f in inc.get("findings", []))
        pool = [inc for inc in ds.values() if has_pilot(inc)]
        def emp_occ(lbl):
            k = sum(1 for inc in pool
                    if lbl in pg._ordered_occurrences(inc)[0])
            return k / len(pool)
        inj_p = np.zeros(4); dmg_p = np.zeros(4)
        for inc in pool:
            inj_p[injury_state(inc)] += 1
            d = damage_state(inc)
            if d is not None:
                dmg_p[d] += 1
        inj_p /= len(pool); dmg_p /= max(dmg_p.sum(), 1)

        checks = [
            ("unstabilized approach", "occ", 0.00484, emp_occ("unstabilized approach")),
            ("dragged wing, rotor, pod, float or tail/skid", "occ", 0.023,
             emp_occ("dragged wing, rotor, pod, float or tail/skid")),
            (DMG_NODE, "substantial damage", 0.0458, float(dmg_p[1])),
            (INJ_NODE, "no injury", 0.97, float(inj_p[3])),
        ]
        print(f"\nFig 12 pilot-error queries (evidence {pilot!r} = Yes; "
              f"{len(pool)} accidents have this finding):")
        print(f"{'target':46} {'Zhang':>9} {'upgraded':>10} {'empirical':>10}")
        for node, state, zhang, emp in checks:
            if state == "occ":
                if node not in bn.names():
                    print(f"{node:46} node missing"); continue
                p = post(bn, node, {pilot: "Yes"}, "Yes")
                label = node
            else:
                p = post(bn, node, {pilot: "Yes"}, state)
                label = state
            results["fig12"].append({"target": label, "zhang": zhang,
                                     "upgraded": p, "empirical": emp})
            print(f"{label:46} {zhang:9.4g} {p:10.4g} {emp:10.4g}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""EXPERIMENT: does adopting Zhang's injury derivation close the Table 9
damage/injury gaps?

Discovery (from his released main.py, calculate_injury_level): Zhang does NOT
use events.ev_highest_injury. He derives each accident's injury level from the
per-person injury.xlsx table:

    Fatal>0 -> 'Fatal injury'; elif Serious>0 -> 'Serious injury';
    elif Minor>0 -> 'Minor injury'; else -> 'No injury'
    (and if the accident has no usable injury rows at all -> 'No injury')

and his graph builder ALWAYS attaches last-occurrence -> injury edge.

Our pipeline uses ev_highest_injury and skips the edge when the field is
missing/unmapped. This script rebuilds OUR network with Zhang's derivation
(everything else identical to tests/bn_build_ours.py) and re-runs every
published Table 9 damage/injury cell + the Fig 12 no-injury prior.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 tests/bn_zhang_injury_derivation.py
Writes outputs/bn_zhang_injury_derivation.json.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
_SHARED = REPO_ROOT / "shared" / "code"
_FROZEN_CODE = FROZEN_DIR / "code"
for _p in (_SHARED, _FROZEN_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
ROOT = REPO_ROOT
import prognosis as pg  # noqa: E402
import pyagrum as gum  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
import bn_build_ours as builder  # noqa: E402

OUT = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs" / "bn_zhang_injury_derivation.json"

INJ_NODE = {"FATL": "fatal injury", "SERS": "serious injury",
            "MINR": "minor injury", "NONE": "no injury"}


def zhang_injury_label(inc: dict) -> str:
    """calculate_injury_level from Zhang's released main.py, verbatim logic."""
    tot = {"FATL": 0.0, "SERS": 0.0, "MINR": 0.0, "NONE": 0.0, "TOTL": 0.0}
    for r in inc.get("injuries") or []:
        lvl = str(r.get("injury_level") or "").strip().upper()
        if lvl in tot:
            try:
                tot[lvl] += float(r.get("inj_person_count") or 0)
            except (TypeError, ValueError):
                pass
    if tot["TOTL"] != 0:
        if tot["FATL"] > 0:
            return INJ_NODE["FATL"]
        if tot["SERS"] > 0:
            return INJ_NODE["SERS"]
        if tot["MINR"] > 0:
            return INJ_NODE["MINR"]
        return INJ_NODE["NONE"]
    return INJ_NODE["NONE"]  # his fallback: no usable rows -> No injury


ORIG_BUILD_EDGES = pg.build_edges


def build_edges_zhang(inc: dict) -> set:
    """pg.build_edges, but the injury edge uses Zhang's derivation and is ALWAYS
    attached (his builder adds an injury node for every accident)."""
    edges = set()
    occ_by_no = pg._occ_by_no(inc)
    for f in inc.get("findings", []):
        d = pg._s(f.get("finding_description")).lower()
        occ = occ_by_no.get(pg._s(f.get("Occurrence_No")))
        if d and occ and d != occ:
            edges.add((d, occ))
    descs, _ = pg._ordered_occurrences(inc)
    for i in range(len(descs) - 1):
        if descs[i] != descs[i + 1]:
            edges.add((descs[i], descs[i + 1]))
    if descs:
        last = descs[-1]
        dmg = pg.DAMAGE_LABELS.get(pg._s(inc.get("damage")).upper())
        if dmg:
            edges.add((last, dmg))
        edges.add((last, zhang_injury_label(inc)))  # <-- the change
    return edges


# Zhang's published Table 9 damage/injury rows (cols: engine instruments,
# combustion liner, improper oil, instruments+oil, LOEP) + Fig 12 prior.
PUBLISHED = {
    "destroyed aircraft":  [0.0133, 0.0023, 0.00437, 0.0135, 0.00559],
    "substantial damage":  [0.046, 0.00363, 0.00609, 0.0463, 0.0166],
    "minor damage":        [0.00934, 0.00154, 0.00292, 0.00947, 0.00378],
    "serious injury":      [0.0623, 0.000768, 0.00146, 0.0623, 0.00822],
    "no injury":           [0.9431, 0.9978, 0.9958, 0.9429, 0.9899],
}
EV_TAGS = ["engine instruments", "combustion liner", "improper oil",
           "instruments + oil", "LOEP"]


def posterior(bn, target, evidence):
    try:
        ie = gum.LazyPropagation(bn)
        if evidence:
            ie.setEvidence(evidence)
        ie.addTarget(target)
        ie.makeInference()
        engine = "exact"
    except Exception:
        ie = gum.LoopyBeliefPropagation(bn)
        ie.setMaxIter(200)
        if evidence:
            ie.setEvidence(evidence)
        ie.makeInference()
        engine = "loopy"
    v = bn.variable(target)
    yes = [i for i in range(v.domainSize()) if v.label(i) == "Yes"][0]
    return float(ie.posterior(target)[yes]), engine


def main():
    ds = pg.load_dataset()

    # ---- (1) how much do the two injury derivations disagree? -------------------
    both = Counter()
    for inc in ds.values():
        ours = pg.INJURY_LABELS.get(pg._s(inc.get("ev_highest_injury")).upper(),
                                    "(no edge)")
        zh = zhang_injury_label(inc)
        both[(ours, zh)] += 1
    n_disagree = sum(c for (a, b), c in both.items() if a != b)
    print(f"injury derivation: ev_highest_injury vs Zhang per-person rule "
          f"on {len(ds)} accidents -> {n_disagree} disagree "
          f"({100 * n_disagree / len(ds):.1f}%)")
    for (a, b), c in sorted(both.items(), key=lambda kv: -kv[1]):
        if a != b:
            print(f"   {a:16} -> {b:16}  x{c}")

    # ---- (2) rebuild the network with Zhang's derivation ------------------------
    pg.build_edges = build_edges_zhang
    try:
        bn, meta = builder.build_network(ds)
    finally:
        pg.build_edges = ORIG_BUILD_EDGES
    print(f"\nvariant network: {meta['nodes']} nodes, {meta['arcs']} arcs, "
          f"{meta['roots']} roots")

    noinj = builder.find_node(bn, "no injury")
    parents = sorted(bn.variable(p).name() for p in bn.parents(bn.idFromName(noinj)))
    print(f"'no injury' parents now ({len(parents)}): {parents}")

    # ---- (3) evidence nodes (same ones the full comparison used) ----------------
    loep = "loss of engine power"
    ev_nodes = [
        "engine instrument",
        builder.find_node(bn, "combustion", "liner"),
        builder.find_node(bn, "fluid, oil grade") or builder.find_node(bn, "oil grade"),
    ]
    evidences = [
        {ev_nodes[0]: "Yes"},
        {ev_nodes[1]: "Yes"},
        {ev_nodes[2]: "Yes"},
        {ev_nodes[0]: "Yes", ev_nodes[2]: "Yes"},
        {loep: "Yes"},
    ]

    results = {"meta": meta, "no_injury_parents": parents, "rows": []}
    print("\nTable 9 damage/injury rows, REBUILT with Zhang's injury derivation:")
    print(f"{'target':22} {'evidence':20} {'Zhang':>10} {'before':>10} "
          f"{'after':>10}  verdict")

    # 'before' values from the frozen full-comparison report
    before = {
        "destroyed aircraft": [0.004378, 0.002304, 0.004378, 0.004606, 0.02564],
        "substantial damage": [0.006907, 0.003636, 0.006907, 0.007266, 0.007271],
        "minor damage":       [0.477, 0.2511, 0.477, 0.5018, 0.5021],
        "serious injury":     [0.00146, 0.000769, 0.00146, 0.001536, 0.005315],
        "no injury":          [0.007656, 0.00403, 0.007656, 0.008054, 0.02173],
    }

    for tgt_key, pubs in PUBLISHED.items():
        node = builder.find_node(bn, *tgt_key.split()[:2])
        if node is None:
            print(f"  {tgt_key}: NODE NOT FOUND")
            continue
        for j, (pub, ev) in enumerate(zip(pubs, evidences)):
            p, engine = posterior(bn, node, ev)
            b = before[tgt_key][j]
            gap_before = abs(b - pub) / pub
            gap_after = abs(p - pub) / pub
            verdict = ("CLOSES" if gap_after < 0.25 else
                       "improves" if gap_after < 0.6 * gap_before else
                       "worsens" if gap_after > 1.5 * gap_before else "~same")
            results["rows"].append({
                "target": node, "evidence": EV_TAGS[j], "zhang": pub,
                "before": b, "after": p, "engine": engine, "verdict": verdict})
            print(f"{tgt_key:22} {EV_TAGS[j]:20} {pub:10.4g} {b:10.4g} "
                  f"{p:10.4g}  {verdict} [{engine}]")

    # ---- (4) Fig 12 prior P(no injury) ------------------------------------------
    p_prior, engine = posterior(bn, noinj, None)
    print(f"\nFig 12 prior P(no injury): Zhang 0.9999 | before 6.3e-7 | "
          f"after {p_prior:.4g} [{engine}]")
    results["fig12_prior_no_injury"] = {"zhang": 0.9999, "after": p_prior}

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

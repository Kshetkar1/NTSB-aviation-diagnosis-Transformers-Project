#!/usr/bin/env python3
"""PARITY CHECK: Zhang's published BN posteriors (Table 8 + Table 9) reproduced from
his released NTSB.xdsl with OPEN-SOURCE inference (pyAgrum) — no GeNIe, no pySMILE.

His file is used ONLY as ground truth (same role as the published Table 7 values).
The independent lane — the network built from OUR OWN CPTs — is tests/bn_build_ours.py.

What is checked
  A. Table 9 forward anchors   : P(LOEP | single cause observed)      -> 0.95 / 0.50 / 0.95
  B. Table 9 downstream        : P(x | LOEP observed)                 -> forced landing 0.1429,
                                 ditching 4.61e-3, + damage/injury rows (approx engines,
                                 flagged as unverified where exact inference is infeasible)
  C. Table 8 sensitivity sweep : SET THE PRIOR of 'Landing gear, main gear strut'
                                 (not evidence!) and read the two collapse children.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 tests/bn_posterior_parity.py
Writes outputs/bn_posterior_parity.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyagrum as gum

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
_SHARED = REPO_ROOT / "shared" / "code"
_FROZEN_CODE = FROZEN_DIR / "code"
for _p in (_SHARED, _FROZEN_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
ROOT = REPO_ROOT
XDSL = ROOT / "Zhang-Replication-Foundation-2026-06-04" / "reference" / "NTSB.xdsl"
OUT = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs" / "bn_posterior_parity.json"

LOEP = "Loss of engine power"

# (evidence node in xdsl, Zhang's published P(LOEP | evidence))
ANCHORS = [
    ("Engine instruments, exhaust pressure ratio (EPR) gauge/system", 0.95),
    ("Combustion assembly, combustion liner", 0.50),
    ("Fluid, oil grade", 0.95),
]

# Table 9 downstream targets under evidence LOEP=Yes: (node, Zhang published, heavy?)
DOWNSTREAM = [
    ("Forced landing", 0.1429, False),
    ("Ditching", 4.61e-3, False),
    ("Destroyed aircraft damage", 5.59e-3, True),
    ("Substantial aircraft damage", None, True),
    ("Serious injury", 8.22e-3, True),
    ("No injury", 0.9899, True),
]

STRUT = "Landing gear, main gear strut"
# Table 8: (prior value, Zhang main gear collapse, Zhang gear collapse)
TABLE8 = [
    (None, 1.21e-7, 9.51e-8),          # None = leave the file's prior as-is
    (6.5e-7, 2.67e-7, 2.42e-7),
    (6.5e-6, 1.73e-6, 1.70e-6),
    (6.5e-4, 1.63e-4, 1.63e-4),
    (6.5e-2, 1.62e-2, 1.62e-2),
    (0.1, 2.50e-2, 2.50e-2),
    (0.5, 12.50e-2, 12.50e-2),
    (1.0, 25.00e-2, 25.00e-2),
]


def _yes(bn, name):
    v = bn.variable(name)
    return [i for i in range(v.domainSize()) if v.label(i) == "Yes"][0]


def exact_posterior(bn, target, evidence=None):
    """Targeted exact inference (LazyPropagation prunes to the relevant subgraph)."""
    ie = gum.LazyPropagation(bn)
    if evidence:
        ie.setEvidence(evidence)
    ie.addTarget(target)
    ie.makeInference()
    return float(ie.posterior(target)[_yes(bn, target)])


def exact_fragment_posterior(bn, target, evidence_node, evidence_val="Yes"):
    """Exact inference restricted to the ancestral closure of {target, evidence}
    (barren-node removal). Makes the heavy outcome rows tractable."""
    frag = gum.BayesNetFragment(bn)
    frag.installAscendants(bn.idFromName(target))
    frag.installAscendants(bn.idFromName(evidence_node))
    ie = gum.LazyPropagation(frag)
    ie.setEvidence({evidence_node: evidence_val})
    ie.addTarget(target)
    ie.makeInference()
    return float(ie.posterior(target)[_yes(bn, target)])


def loopy_posterior(bn, target, evidence=None, max_iter=200):
    ie = gum.LoopyBeliefPropagation(bn)
    ie.setMaxIter(max_iter)
    if evidence:
        ie.setEvidence(evidence)
    ie.makeInference()
    return float(ie.posterior(target)[_yes(bn, target)])


def set_root_prior(bn, name, p_yes):
    v = bn.variable(name)
    yes = _yes(bn, name)
    vals = [0.0, 0.0]
    vals[yes] = p_yes
    vals[1 - yes] = 1.0 - p_yes
    bn.cpt(name).fillWith(vals)


def main(argv=None):
    argv = argv or sys.argv[1:]
    heavy = "--heavy" in argv  # attempt exact inference on the big outcome nodes too

    results = {"anchors": [], "downstream": [], "table8": []}
    bn = gum.loadBN(str(XDSL))
    print(f"loaded {XDSL.name}: {bn.size()} nodes, {bn.sizeArcs()} arcs (pyAgrum, open source)")

    print("\n(A) Table 9 forward anchors — evidence: cause=Yes, target LOEP")
    ok_all = True
    for node, zhang in ANCHORS:
        p = loopy_posterior(bn, LOEP, {node: "Yes"})
        ok = abs(p - zhang) < 5e-3
        ok_all &= ok
        results["anchors"].append({"evidence": node, "ours": p, "zhang": zhang, "ok": ok})
        print(f"  {'OK ' if ok else 'XX '} P(LOEP | {node[:48]:48}) = {p:.4f}  Zhang {zhang}")

    print("\n(B) Table 9 downstream — evidence: LOEP=Yes")
    for node, zhang, is_heavy in DOWNSTREAM:
        if is_heavy:
            p = exact_fragment_posterior(bn, node, LOEP)
            engine = "exact (ancestral fragment)"
        else:
            p = exact_posterior(bn, node, {LOEP: "Yes"})
            engine = "exact"
        ok = (zhang is not None and abs(p - zhang) / max(zhang, 1e-12) < 0.05
              and engine == "exact")
        zs = f"{zhang:.4g}" if zhang is not None else "  --"
        results["downstream"].append({"target": node, "ours": p, "zhang": zhang,
                                      "engine": engine, "ok": ok})
        mark = "OK " if ok else ("~~ " if "loopy" in engine else "XX ")
        print(f"  {mark} P({node[:30]:30} | LOEP) = {p:.4e}  Zhang {zs}   [{engine}]")

    print("\n(C) Table 8 — SET PRIOR of strut failure, read collapse children (exact)")
    base_yes = float(bn.cpt(STRUT)[_yes(bn, STRUT)])
    print(f"  file prior P({STRUT}) = {base_yes:.3e}   (Zhang prior row: 6.5e-8)")
    for prior, z_main, z_gear in TABLE8:
        bn2 = gum.BayesNet(bn)  # fresh copy per row
        if prior is not None:
            set_root_prior(bn2, STRUT, prior)
        p_main = exact_posterior(bn2, "Main gear collapsed")
        p_gear = exact_posterior(bn2, "Gear collapsed")
        label = f"{prior:.2g}" if prior is not None else f"file({base_yes:.2g})"
        results["table8"].append({"prior": prior, "main": p_main, "gear": p_gear,
                                  "zhang_main": z_main, "zhang_gear": z_gear})
        print(f"  prior={label:>10}: main={p_main:.3e} (Zhang {z_main:.3g})   "
              f"gear={p_gear:.3e} (Zhang {z_gear:.3g})")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print(f"anchors: {'ALL OK' if ok_all else 'MISMATCH'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

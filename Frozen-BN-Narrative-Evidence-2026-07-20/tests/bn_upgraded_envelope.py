#!/usr/bin/env python3
"""Build-variance envelope for the 12 cells that STILL differ on the UPGRADED
network (outputs/bn_upgraded_full.json, new_status == DIFFERS).

The original envelope experiment (tests/bn_variance_envelope.py) covered the
first-pass reproduction. This one answers the question that matters for the
CURRENT scoreboard: if Zhang's own randomized parent selection (jitter on
max-ratio ties + climbing threshold cut, verbatim from his released main.py)
is injected into the UPGRADED build (person edges + multi-state severity
nodes), does his published number fall inside the range our rebuilds produce?

For each of N_SEEDS seeds:
  * event-node graph built with the UPGRADED edge rule (person -> finding),
  * parent selection per child via Zhang's randomized mechanism,
  * multi-state severity nodes attached exactly as bn_upgraded.build_upgraded
    (deterministic: severity has no analog in Zhang's randomness),
  * the 12 differ cells evaluated with exact inference.

Cells whose published value lands inside the envelope are reclassified
"WITHIN BUILD VARIANCE"; the rest stay honest DIFFERS.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/bn_upgraded_envelope.py [--seeds N]
Writes outputs/bn_upgraded_envelope.json and outputs/BN_UPGRADED_ENVELOPE.md.
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyagrum as gum

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
_SHARED = REPO_ROOT / "shared" / "code"
_FROZEN_CODE = FROZEN_DIR / "code"
for _p in (_SHARED, _FROZEN_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT / "tests"))

import prognosis as pg  # noqa: E402
import bn_build_ours as builder  # noqa: E402
import bn_upgraded as up  # noqa: E402
from bn_variance_envelope import (base_graph, zhang_select,  # noqa: E402
                                  assemble, fill_child_cpt)

N_SEEDS = 30
OUT_JSON = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs" / "bn_upgraded_envelope.json"
OUT_MD = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs" / "BN_UPGRADED_ENVELOPE.md"
UPGRADED_FULL = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs" / "bn_upgraded_full.json"


def add_severity_nodes(bn, ds):
    """Attach the multi-state severity nodes exactly as bn_upgraded does
    (deterministic top-12-by-support parents; empirical mixture CPTs)."""
    support, inj_counts, dmg_counts = up.severity_parent_stats(ds)
    parents = [lab for lab, _ in support.most_common() if lab in bn.names()]
    parents = parents[:up.MAX_PARENTS]

    specs = [
        (up.INJ_NODE, up.INJ_STATES, inj_counts, 3),
        (up.DMG_NODE, up.DMG_STATES, dmg_counts, 3),
    ]
    for node_name, states, counts, default_idx in specs:
        v = gum.LabelizedVariable(node_name, node_name, 0)
        for s in states:
            v.addLabel(s)
        bn.add(v)
        for p in parents:
            bn.addArc(p, node_name)

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
                dist = default
            else:
                w = weights[active]
                dist = (dists[active] * w[:, None]).sum(axis=0) / w.sum()
                dist = dist / dist.sum()
            idx = {parents[i]: (0 if active[i] else 1) for i in range(n)}
            cpt[{**idx}] = list(dist)
    return parents


def post(bn, target, evidence, state):
    ie = gum.LazyPropagation(bn)
    if evidence:
        ie.setEvidence(evidence)
    ie.addTarget(target)
    ie.makeInference()
    v = bn.variable(target)
    i = [k for k in range(v.domainSize()) if v.label(k) == state][0]
    return float(ie.posterior(target)[i])


def build_jittered_upgraded(ds, seed):
    """Upgraded network with Zhang's randomized parent selection on the
    event-node layer (severity layer deterministic, as it has no analog
    in his randomness)."""
    orig = pg.build_edges
    pg.build_edges = up.build_edges_upgraded
    try:
        ratio, candidates, from_counts, all_nodes = base_graph(ds)
    finally:
        pg.build_edges = orig
    rng = random.Random(seed)
    selection = {b: zhang_select(c, rng) for b, c in candidates.items()}
    bn, meta = assemble(all_nodes, selection, from_counts)
    add_severity_nodes(bn, ds)
    return bn, meta


def cells_for(bn):
    """The 12 still-DIFFERS cells: (key, target node, state, evidence)."""
    eng = "engine instrument"
    comb = builder.find_node(bn, "combustion", "liner")
    oil = (builder.find_node(bn, "fluid, oil grade")
           or builder.find_node(bn, "oil grade"))
    loep = "loss of engine power"
    unstab = ("unstabilized approach" if "unstabilized approach" in bn.names()
              else builder.find_node(bn, "unstabilized approach"))
    maingearc = ("main gear collapsed" if "main gear collapsed" in bn.names()
                 else builder.find_node(bn, "main gear collapsed"))
    D, I = up.DMG_NODE, up.INJ_NODE
    return [
        ("T9 destroyed | eng instr",       D, "destroyed aircraft",  {eng: "Yes"}),
        ("T9 destroyed | LOEP",            D, "destroyed aircraft",  {loep: "Yes"}),
        ("T9 substantial | LOEP",          D, "substantial damage",  {loep: "Yes"}),
        ("T9 minor dmg | eng instr",       D, "minor damage",        {eng: "Yes"}),
        ("T9 minor dmg | comb liner",      D, "minor damage",        {comb: "Yes"}),
        ("T9 minor dmg | oil",             D, "minor damage",        {oil: "Yes"}),
        ("T9 minor dmg | eng+oil",         D, "minor damage",        {eng: "Yes", oil: "Yes"}),
        ("T9 minor dmg | LOEP",            D, "minor damage",        {loep: "Yes"}),
        ("T9 serious inj | LOEP",          I, "serious injury",      {loep: "Yes"}),
        ("T8 main gear collapsed @ base prior", maingearc, "Yes",    None),
        ("Fig12 prior substantial dmg",    D, "substantial damage",  None),
        ("Fig12 no injury | unstab approach", I, "no injury",        {unstab: "Yes"}),
    ]


def main():
    n_seeds = N_SEEDS
    if "--seeds" in sys.argv:
        n_seeds = int(sys.argv[sys.argv.index("--seeds") + 1])

    full = json.loads(UPGRADED_FULL.read_text())
    still = {r["cell"]: r for r in full["differs_rescore"]
             if r["new_status"] == "DIFFERS"}
    print(f"cells still DIFFERS on the upgraded network: {len(still)}")

    ds = pg.load_dataset()
    t0 = time.time()
    values = defaultdict(list)
    for seed in range(n_seeds):
        t1 = time.time()
        bn, meta = build_jittered_upgraded(ds, seed)
        cells = cells_for(bn)
        for key, node, state, ev in cells:
            if node is None:
                continue
            values[key].append(post(bn, node, ev, state))
        print(f"  seed {seed:2d}: {meta['arcs']} event arcs "
              f"({time.time() - t1:.1f}s)")

    rows = []
    for key, r in still.items():
        arr = np.array(values.get(key, []))
        if not len(arr):
            rows.append({"cell": key, "zhang": r["zhang"],
                         "upgraded": r["upgraded"], "verdict": "NOT EVALUATED"})
            continue
        lo, hi = float(arr.min()), float(arr.max())
        inside = bool(lo <= r["zhang"] <= hi)
        rows.append({
            "cell": key, "zhang": r["zhang"], "upgraded": r["upgraded"],
            "env_min": lo, "env_max": hi, "env_median": float(np.median(arr)),
            "n_builds": int(len(arr)),
            "published_inside": inside,
            "verdict": "WITHIN BUILD VARIANCE" if inside else "DIFFERS",
        })

    n_inside = sum(1 for r in rows if r.get("published_inside"))
    print(f"\n{'cell':40} {'Zhang':>10} {'det.':>10} "
          f"{'env min':>10} {'env max':>10}  verdict")
    for r in rows:
        if "env_min" not in r:
            print(f"{r['cell']:40} {r['zhang']:10.3g} {r['upgraded']:10.3g} "
                  f"{'-':>10} {'-':>10}  {r['verdict']}")
            continue
        print(f"{r['cell']:40} {r['zhang']:10.3g} {r['upgraded']:10.3g} "
              f"{r['env_min']:10.3g} {r['env_max']:10.3g}  {r['verdict']}")
    print(f"\n>>> {n_inside} of {len(rows)} remaining DIFFERS cells fall inside "
          f"the {n_seeds}-seed build-variance envelope of the UPGRADED network")

    payload = {"config": {"n_seeds": n_seeds,
                          "runtime_s": round(time.time() - t0, 1)},
               "cells": rows,
               "n_inside": n_inside}
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    md = [
        "# Upgraded-Network Build-Variance Envelope (the 12 remaining DIFFERS cells)",
        "",
        f"{n_seeds} full rebuilds of the UPGRADED network (person edges + "
        "multi-state severity nodes) with Zhang's exact randomized parent "
        "selection (jitter on max-ratio ties + climbing threshold cut, verbatim "
        "from his released main.py) injected into the event-node layer. The "
        "severity layer stays deterministic -- it has no analog in his "
        "randomness.",
        "",
        "| cell | Zhang | deterministic upgraded | envelope [min, max] | verdict |",
        "|------|-------|------------------------|--------------------|---------|",
    ]
    for r in rows:
        env = (f"[{r['env_min']:.3e}, {r['env_max']:.3e}]"
               if "env_min" in r else "n/a")
        verdict = ("**WITHIN BUILD VARIANCE**"
                   if r.get("published_inside") else r["verdict"])
        md.append(f"| {r['cell']} | {r['zhang']:.3e} | {r['upgraded']:.3e} "
                  f"| {env} | {verdict} |")
    md += [
        "",
        f"**{n_inside} of {len(rows)}** remaining DIFFERS cells fall inside the "
        "build-variance envelope: for those cells, Zhang's published value is "
        "reachable by his own construction applied to our data, so the residual "
        "gap is attributable to his under-specified randomized build, not to an "
        "error in ours. The rest remain honest DIFFERS (dataset-count "
        "differences and structural choices documented in "
        "outputs/BN_COMPARISON_REPORT.md).",
        "",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

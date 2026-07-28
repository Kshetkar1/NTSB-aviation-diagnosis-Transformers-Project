#!/usr/bin/env python3
"""REAL narratives, not hand-crafted sentences: show that on actual NTSB
narrative text the parsed dictionaries are messy (multiple facts, hard + soft
mixes) and the posteriors match NEITHER the prior NOR any single hand-click
column from the tables.

This is the answer to "why were narrative and direct identical in the tables?"
-- because those table sentences were WRITTEN to name one fact verbatim. Real
narratives aren't like that. Here we take held-out accidents (2007-2019, never
seen by the network or the retrieval index), parse their real factual
narratives, and print:

    dictionary        what the parser extracted (node -> confidence)
    P(injury), P(damage) posteriors under that dictionary
    vs prior          the network with no evidence
    vs single-click   the closest single-evidence column (LOEP)
    truth             what actually happened in that accident

Run (needs OPENAI_API_KEY for retrieval):
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/real_narratives_demo.py
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
_SHARED = REPO_ROOT / "shared" / "code"
_FROZEN_CODE = FROZEN_DIR / "code"
for _p in (_SHARED, _FROZEN_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
import pyagrum as gum  # noqa: E402

import prognosis as pg  # noqa: E402
import query_to_bn as qb  # noqa: E402
from bn_upgraded import (build_upgraded, INJ_NODE, DMG_NODE,  # noqa: E402
                         INJ_STATES, DMG_STATES)

FULL = ROOT / "shared" / "data" / "processed" / "refined_dataset.json"
WINDOW = ROOT / "shared" / "data" / "processed" / "refined_dataset_1982_2006.json"

N_SHOW = 5


def posteriors(bn, confidence):
    ie = gum.LazyPropagation(bn)
    if confidence:
        qb.apply_evidence(ie, bn, confidence)
    ie.addTarget(INJ_NODE)
    ie.addTarget(DMG_NODE)
    ie.makeInference()

    def dist(node, states):
        v = bn.variable(node)
        post = ie.posterior(node)
        by = {v.label(i): float(post[i]) for i in range(v.domainSize())}
        return {s: by[s] for s in states}
    return dist(INJ_NODE, INJ_STATES), dist(DMG_NODE, DMG_STATES)


def fmt_dist(d):
    return "  ".join(f"{k.split()[0]}={v:.4f}" for k, v in d.items())


def main():
    full = json.loads(FULL.read_text())
    window_ids = set(json.loads(WINDOW.read_text()).keys())
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]

    heldout = []
    for ev, inc in full.items():
        if ev in window_ids:
            continue
        narr = str(inc.get("narr_accf") or "").strip()
        if len(narr) >= 200:
            heldout.append((ev, inc, narr))
    heldout.sort(key=lambda t: t[0])
    print(f"held-out accidents with a factual narrative: {len(heldout)}; "
          f"showing the first {N_SHOW}\n")

    prior_inj, prior_dmg = posteriors(bn, {})
    loep_inj, loep_dmg = posteriors(bn, {"loss of engine power": 1.0})

    print("REFERENCE ROWS")
    print(f"  PRIOR (no evidence) : inj {fmt_dist(prior_inj)}")
    print(f"                        dmg {fmt_dist(prior_dmg)}")
    print(f"  SINGLE CLICK (LOEP) : inj {fmt_dist(loep_inj)}")
    print(f"                        dmg {fmt_dist(loep_dmg)}")

    for ev, inc, narr in heldout[:N_SHOW]:
        print("\n" + "=" * 78)
        print(f"ACCIDENT {ev}  (held-out, {str(inc.get('ev_date'))[:10]})")
        print(textwrap.fill("NARRATIVE: " + narr[:600] +
                            ("..." if len(narr) > 600 else ""), width=78))
        p = qb.parse_query_to_bn_evidence(narr, names, dataset=ds,
                                          semantic=True)
        print("\n  PARSED DICTIONARY:")
        if not p["confidence"]:
            print("    (nothing extracted)")
        for node, why in p["trace"]:
            c = p["confidence"][node]
            kind = "HARD" if c >= 0.999 else f"soft {c:.3f}"
            print(f"    [{kind:10}] {node!r}")
            print(f"                 <- {why[:80]}")
        inj, dmg = posteriors(bn, p["confidence"])
        print(f"\n  NARRATIVE posterior : inj {fmt_dist(inj)}")
        print(f"                        dmg {fmt_dist(dmg)}")
        same_prior = all(abs(inj[s] - prior_inj[s]) < 1e-12 for s in inj)
        same_loep = all(abs(inj[s] - loep_inj[s]) < 1e-12 for s in inj)
        print(f"  identical to prior? {same_prior}   identical to the LOEP "
              f"click? {same_loep}")
        truth_inj = pg.zhang_injury_code(inc)
        print(f"  TRUTH: injury={truth_inj}  damage={inc.get('damage')}")

    print("\n" + "=" * 78)
    print("POINT: real narratives produce messy dictionaries (several facts,")
    print("hard+soft mixes) and posteriors that match neither the prior nor any")
    print("single-click table column. The table identity existed only because")
    print("those sentences were written to name exactly one fact verbatim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""THE test for Maha's three questions:
  1. Where do the narratives connect to the Bayesian network?
  2. Where does the query come in?
  3. How are the narratives exploited all the way to the end?

For each narrative below we print the LADDER Maha wants to see:

  PRIOR      what the network believes knowing nothing
  NARRATIVE  what it believes after the typed sentence (parsed to evidence,
             hard when explicit / soft when vague, then propagated)
  ZHANG      his published number for the same scenario (when one exists)
  DATA       the raw empirical rate among accidents matching the evidence
             (when the evidence is hard and countable)

Reading the ladder answers everything: the sentence is the ONLY thing we
change, so any movement from PRIOR to NARRATIVE is the narrative being
exploited, all the way to the final posterior.

Run (needs OPENAI_API_KEY for retrieval on the vague sentences):
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/maha_narrative_demo.py
"""
from __future__ import annotations

import sys
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
                         INJ_STATES, DMG_STATES, DMG_BY_CODE)

DRAGGED = "dragged wing, rotor, pod, float or tail/skid"


def posterior(bn, confidence, target, state):
    ie = gum.LazyPropagation(bn)
    if confidence:
        qb.apply_evidence(ie, bn, confidence)
    ie.addTarget(target)
    ie.makeInference()
    v = bn.variable(target)
    i = [k for k in range(v.domainSize()) if v.label(k) == state][0]
    return float(ie.posterior(target)[i])


def incident_has(inc, lab):
    lab = lab.lower()
    if lab.startswith("person: "):
        want = lab[len("person: "):]
        return any(str(f.get("person_description") or "").strip().lower() == want
                   for f in inc.get("findings", []))
    descs = {str(s.get("Occurrence_Description") or "").strip().lower()
             for s in inc.get("sequence_of_events", [])}
    if lab in descs:
        return True
    return any(str(f.get("finding_description") or "").strip().lower() == lab
               for f in inc.get("findings", []))


def empirical(ds, evidence_nodes, target, state):
    """Raw rate of the outcome among accidents containing ALL evidence."""
    pool = [inc for inc in ds.values()
            if all(incident_has(inc, e) for e in evidence_nodes)]
    n = len(pool)
    if not n:
        return None, 0
    if target == INJ_NODE:
        idx = {"FATL": 0, "SERS": 1, "MINR": 2, "NONE": 3}
        want = INJ_STATES.index(state)
        hit = sum(1 for inc in pool
                  if idx[pg.zhang_injury_code(inc)] == want)
    elif target == DMG_NODE:
        want = DMG_STATES.index(state)
        hit = sum(1 for inc in pool
                  if DMG_BY_CODE.get(str(inc.get("damage") or "").upper()) == want)
    else:
        hit = sum(1 for inc in pool if incident_has(inc, target))
    return hit / n, n


# (title, narrative, [(pretty, target, state, zhang_or_None)])
SCENARIOS = [
    ("A. Zhang's Table 9 scenario, spoken as a sentence",
     "the aircraft experienced a loss of engine power", [
        ("P(no injury)",          INJ_NODE, "no injury",          0.9899),
        ("P(serious injury)",     INJ_NODE, "serious injury",     0.00822),
        ("P(substantial damage)", DMG_NODE, "substantial damage", 0.0166),
        ("P(destroyed aircraft)", DMG_NODE, "destroyed aircraft", 0.00559),
     ]),
    ("B. Zhang's Fig 12 scenario, spoken as a sentence",
     "the pilot in command was a factor in the accident", [
        ("P(no injury)",             INJ_NODE, "no injury",          0.97),
        ("P(substantial damage)",    DMG_NODE, "substantial damage", 0.0458),
        ("P(unstabilized approach)", "unstabilized approach", "Yes", 0.00484),
        ("P(dragged wing/pod/tail)", DRAGGED, "Yes",                 0.023),
     ]),
    ("C. TWO facts in one sentence (Zhang never published this - counting dies)",
     "engine lost power and the pilot in command was a factor", [
        ("P(no injury)",          INJ_NODE, "no injury",          None),
        ("P(serious injury)",     INJ_NODE, "serious injury",     None),
        ("P(substantial damage)", DMG_NODE, "substantial damage", None),
     ]),
    ("D. VAGUE sentence - no NTSB vocabulary at all (soft evidence)",
     "the engine quit on climbout", [
        ("P(no injury)",          INJ_NODE, "no injury",          None),
        ("P(serious injury)",     INJ_NODE, "serious injury",     None),
        ("P(substantial damage)", DMG_NODE, "substantial damage", None),
     ]),
    ("E. VAGUE sentence about fire (soft evidence)",
     "flames coming from the engine cowling", [
        ("P(no injury)",          INJ_NODE, "no injury",          None),
        ("P(serious injury)",     INJ_NODE, "serious injury",     None),
        ("P(substantial damage)", DMG_NODE, "substantial damage", None),
     ]),
]


def fmt(x, width=11):
    if x is None:
        return " " * (width - 2) + "--"
    return f"{x:{width}.4g}"


def main():
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]
    print(f"Network: {bn.size()} nodes, {bn.sizeArcs()} arcs "
          f"(our deterministic build of Zhang's Section 4)\n")

    for title, narrative, checks in SCENARIOS:
        print("=" * 78)
        print(title)
        print(f'NARRATIVE TYPED: "{narrative}"')
        parsed = qb.parse_query_to_bn_evidence(narrative, names, dataset=ds,
                                               semantic=True)
        if not parsed["evidence"]:
            print("  parsed: NOTHING -- narrative unusable")
            continue
        hard_ev = [e for e in parsed["evidence"]
                   if parsed["confidence"][e] >= 0.999]
        print("EVIDENCE THE PARSER EXTRACTED:")
        for node, why in parsed["trace"]:
            c = parsed["confidence"][node]
            kind = "HARD 100%" if c >= 0.999 else f"soft {c:.0%}"
            print(f"  [{kind:9}] {node}")
            print(f"              ({why})")

        countable = hard_ev and len(hard_ev) == len(parsed["evidence"])
        head = (f"  {'':26} {'PRIOR':>11} {'NARRATIVE':>11} {'ZHANG':>11} "
                f"{'DATA':>11}")
        print(head)
        n_pool = None
        for pretty, target, state, zhang in checks:
            prior = posterior(bn, {}, target, state)
            ours = posterior(bn, parsed["confidence"], target, state)
            emp, n = (empirical(ds, hard_ev, target, state)
                      if countable else (None, 0))
            n_pool = n if countable else None
            print(f"  {pretty:26} {fmt(prior)} {fmt(ours)} {fmt(zhang)} "
                  f"{fmt(emp)}")
        if n_pool is not None:
            print(f"  (DATA column counted over the {n_pool} accidents that "
                  f"contain all hard evidence)")
        print()

    print("=" * 78)
    print("HOW TO READ THIS (Maha's three questions):")
    print(" 1. WHERE do narratives connect?  The 'EVIDENCE THE PARSER "
          "EXTRACTED' block\n    -- the sentence becomes network nodes, "
          "hard when explicit, soft when vague.")
    print(" 2. WHERE does the query come in?  It is the ONLY input that "
          "changed between\n    the PRIOR column and the NARRATIVE column.")
    print(" 3. EXPLOITED to the end?  PRIOR -> NARRATIVE movement IS the "
          "narrative's\n    effect on the final posterior. A/B land on "
          "Zhang's numbers; C answers a\n    question counting cannot "
          "(4 matching accidents); D/E answer questions the\n    network "
          "alone could never see.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

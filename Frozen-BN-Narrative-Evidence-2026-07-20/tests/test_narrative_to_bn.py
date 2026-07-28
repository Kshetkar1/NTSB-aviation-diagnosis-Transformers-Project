#!/usr/bin/env python3
"""END-TO-END test of the narrative -> BN bridge against Zhang's published numbers.

For each of Zhang's published scenarios we write a plain-English narrative
(the kind Maha would type), run it through query_to_bn.parse_query_to_bn_evidence,
verify the parser lands on the intended evidence nodes, clamp them, and compare
the network's posteriors to the values printed in the paper (Table 9 LOEP
column, Fig 12 pilot queries, Fig 12 stage-2).

Verdicts use the same thresholds as the 93-item scoreboard:
  EXACT  rel err < 2%     CLOSE  rel err < 25%     DIFFERS otherwise

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 tests/test_narrative_to_bn.py
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

import pyagrum as gum  # noqa: E402

import prognosis as pg  # noqa: E402
import query_to_bn as qb  # noqa: E402
from bn_upgraded import build_upgraded, INJ_NODE, DMG_NODE  # noqa: E402


def post(bn, target, evidence, state):
    ie = gum.LazyPropagation(bn)
    if evidence:
        ie.setEvidence(evidence)
    ie.addTarget(target)
    ie.makeInference()
    v = bn.variable(target)
    i = [k for k in range(v.domainSize()) if v.label(k) == state][0]
    return float(ie.posterior(target)[i])


def verdict(zhang, val):
    rel = abs(val - zhang) / zhang if zhang else float("inf")
    return ("EXACT" if rel < 0.02 else "CLOSE" if rel < 0.25 else "DIFFERS"), rel


DRAGGED = "dragged wing, rotor, pod, float or tail/skid"

# (narrative, expected evidence nodes, [(what, target node, state, zhang value)])
SCENARIOS = [
    (
        "the aircraft experienced a loss of engine power",
        ["loss of engine power"],
        [
            ("Table 9  P(no injury)",          INJ_NODE, "no injury",           0.9899),
            ("Table 9  P(serious injury)",     INJ_NODE, "serious injury",      0.00822),
            ("Table 9  P(destroyed aircraft)", DMG_NODE, "destroyed aircraft",  0.00559),
            ("Table 9  P(substantial damage)", DMG_NODE, "substantial damage",  0.0166),
            ("Table 9  P(minor damage)",       DMG_NODE, "minor damage",        0.00378),
        ],
    ),
    (
        "the pilot in command was a factor in the accident",
        ["person: pilot-in-command"],
        [
            ("Fig 12  P(no injury)",             INJ_NODE, "no injury",          0.97),
            ("Fig 12  P(substantial damage)",    DMG_NODE, "substantial damage", 0.0458),
            ("Fig 12  P(unstabilized approach)", "unstabilized approach", "Yes", 0.00484),
            ("Fig 12  P(dragged wing/pod/tail)", DRAGGED, "Yes",                 0.023),
        ],
    ),
    (
        "the flight had an unstabilized approach",
        ["unstabilized approach"],
        [
            ("Fig 12 stage-2  P(no injury)", INJ_NODE, "no injury", 0.613),
        ],
    ),
    (
        "trouble with an engine instrument during the flight",
        ["engine instrument"],
        [
            ("Table 9  P(gear collapsed)",     "gear collapsed", "Yes",         0.0096),
            ("Table 9  P(destroyed aircraft)", DMG_NODE, "destroyed aircraft",  0.0133),
            ("Table 9  P(substantial damage)", DMG_NODE, "substantial damage",  0.046),
            ("Table 9  P(serious injury)",     INJ_NODE, "serious injury",      0.0623),
            ("Table 9  P(no injury)",          INJ_NODE, "no injury",           0.9431),
        ],
    ),
]


def post_conf(bn, target, confidence, state):
    """Posterior with hard/soft (virtual) evidence via qb.apply_evidence."""
    ie = gum.LazyPropagation(bn)
    qb.apply_evidence(ie, bn, confidence)
    ie.addTarget(target)
    ie.makeInference()
    v = bn.variable(target)
    i = [k for k in range(v.domainSize()) if v.label(k) == state][0]
    return float(ie.posterior(target)[i])


# Paraphrases of Zhang's Table 9 evidence 'loss of engine power': the words
# never say "loss of engine power", so ONLY the retrieval soft-fact pass can
# ground them. Compared against the same Table 9 column.
PARAPHRASES = [
    "the engine quit on climbout",
    "the motor suddenly stopped producing power in flight",
]
T9_CHECKS = [
    ("P(no injury)",          INJ_NODE, "no injury",           0.9899),
    ("P(serious injury)",     INJ_NODE, "serious injury",      0.00822),
    ("P(destroyed aircraft)", DMG_NODE, "destroyed aircraft",  0.00559),
    ("P(substantial damage)", DMG_NODE, "substantial damage",  0.0166),
]


def main():
    semantic = "--semantic" in sys.argv
    ds = pg.load_dataset()
    bn, meta = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]
    print(f"UPGRADED network: {bn.size()} nodes, {bn.sizeArcs()} arcs\n")

    counts = {"EXACT": 0, "CLOSE": 0, "DIFFERS": 0}
    parse_fail = False

    # ---- part A: Zhang's scenarios verbatim (hard evidence, must not drift) --
    print("PART A -- Zhang's scenarios, verbatim narrative, hard evidence")
    print("=" * 72)
    for narrative, want_ev, checks in SCENARIOS:
        parsed = qb.parse_query_to_bn_evidence(
            narrative, names, dataset=ds, semantic=semantic)
        got = parsed["evidence"]
        ok = got == want_ev
        all_hard = all(parsed["confidence"][e] >= 0.999 for e in got)
        parse_fail = parse_fail or not ok or not all_hard
        print(f'NARRATIVE: "{narrative}"')
        print(f"  parsed evidence: {got}  "
              f"{'(as intended, all HARD)' if ok and all_hard else f'!! expected {want_ev} hard'}")
        for what, target, state, zhang in checks:
            val = post_conf(bn, target, parsed["confidence"], state)
            v, rel = verdict(zhang, val)
            counts[v] += 1
            print(f"    {v:7}  {what:36}  Zhang {zhang:<9.4g}  ours {val:<9.4g}  "
                  f"(rel {rel:.1%})")
        print()

    total = sum(counts.values())
    print(f"Part A: {counts['EXACT']} exact, {counts['CLOSE']} close, "
          f"{counts['DIFFERS']} differ (of {total}) -- must match the "
          f"93-item scoreboard cell for cell.")

    # ---- part B: paraphrases -> retrieval soft evidence (needs API key) ------
    if semantic:
        print("\nPART B -- paraphrases of 'loss of engine power', SOFT evidence")
        print("=" * 72)
        # reference points for interpretation
        prior = {what: post_conf(bn, t, {}, s) for what, t, s, _ in T9_CHECKS}
        hard = {what: post_conf(bn, t, {"loss of engine power": 1.0}, s)
                for what, t, s, _ in T9_CHECKS}
        for narrative in PARAPHRASES:
            parsed = qb.parse_query_to_bn_evidence(
                narrative, names, dataset=ds, semantic=True)
            print(f'NARRATIVE: "{narrative}"')
            for n in parsed["evidence"]:
                print(f"  soft {parsed['confidence'][n]:<6} {n!r}")
            if not parsed["evidence"]:
                print("  (nothing parsed -- retrieval found no lifted facts)")
                continue
            for what, target, state, zhang in T9_CHECKS:
                val = post_conf(bn, target, parsed["confidence"], state)
                print(f"    {what:24} prior {prior[what]:<9.3g} soft {val:<9.3g} "
                      f"hard-LOEP {hard[what]:<9.3g} Zhang {zhang:<9.4g}")
            print()
        print("READ: soft posteriors must sit BETWEEN the prior and the "
              "hard-clamped answer, moving toward Zhang's column -- the "
              "narrative pushed the network the right way without asserting "
              "facts it only implied.")

    print("\n" + "=" * 72)
    print(f"parser: {'ALL verbatim narratives parsed to the intended nodes (hard)' if not parse_fail else 'SOME PARSES WRONG'}")
    return 1 if parse_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

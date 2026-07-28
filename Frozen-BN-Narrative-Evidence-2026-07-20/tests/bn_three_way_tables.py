#!/usr/bin/env python3
"""THREE-WAY comparison for every BN-side table in Zhang's paper:

    ZHANG      the number printed in the paper
    DIRECT     our network, evidence clicked in directly (the scoreboard way)
    NARRATIVE  our network, evidence obtained by PARSING a typed sentence

Covers: Table 9 (all five evidence sets), the Table 9 forward anchors,
Figure 12 (pilot queries + stage 2 + priors), and Table 8 (what-if dial --
narrative not applicable: it changes a PRIOR, no sentence can express it).

The point of the NARRATIVE column: when the sentence names the fact, parsing
must land on the identical evidence, so DIRECT == NARRATIVE to machine
precision. The script asserts the parse for every scenario.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/bn_three_way_tables.py
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
import bn_build_ours as builder  # noqa: E402
from bn_upgraded import build_upgraded, INJ_NODE, DMG_NODE  # noqa: E402

DRAGGED = "dragged wing, rotor, pod, float or tail/skid"
UNSTAB = "unstabilized approach"


def post(bn, evidence, target, state):
    """evidence: dict node->conf (qb.apply_evidence semantics) or None."""
    ie = gum.LazyPropagation(bn)
    if evidence:
        qb.apply_evidence(ie, bn, evidence)
    ie.addTarget(target)
    ie.makeInference()
    v = bn.variable(target)
    i = [k for k in range(v.domainSize()) if v.label(k) == state][0]
    return float(ie.posterior(target)[i])


def verdict(zhang, val):
    if zhang in (None, 0):
        return "  --  "
    rel = abs(val - zhang) / zhang
    return "EXACT " if rel < 0.02 else "CLOSE " if rel < 0.25 else "DIFFERS"


def fmt(x, w=10):
    return (" " * (w - 2) + "--") if x is None else f"{x:{w}.4g}"


def main():
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]

    comb = builder.find_node(bn, "combustion", "liner")
    oil = builder.find_node(bn, "fluid, oil grade") or builder.find_node(bn, "oil grade")
    loep = "loss of engine power"
    eng = "engine instrument"
    pilot = "person: pilot-in-command"

    # evidence-set key -> (narrative sentence, [expected nodes])
    SENTENCES = {
        "LOEP":      ("the aircraft experienced a loss of engine power", [loep]),
        "eng instr": ("trouble with an engine instrument during the flight", [eng]),
        "comb":      ("a failure of the combustion assembly, combustion liner", [comb]),
        "oil":       ("the wrong fluid, oil grade was used", [oil]),
        "eng+oil":   ("trouble with an engine instrument and the wrong fluid, oil grade", [eng, oil]),
        "pilot":     ("the pilot in command was a factor in the accident", [pilot]),
        "unstab":    ("the flight had an unstabilized approach", [UNSTAB]),
    }

    parsed_conf = {}
    print("PARSE CHECK -- every sentence must land exactly on the intended nodes")
    print("=" * 78)
    all_ok = True
    for key, (sentence, want) in SENTENCES.items():
        p = qb.parse_query_to_bn_evidence(sentence, names, dataset=ds,
                                          semantic=True)
        got = p["evidence"]
        ok = sorted(got) == sorted(want) and all(
            p["confidence"][e] >= 0.999 for e in got)
        all_ok = all_ok and ok
        parsed_conf[key] = p["confidence"]
        print(f"  [{'OK' if ok else 'XX'}] \"{sentence}\"")
        print(f"        -> {got}")
    print()

    def direct_conf(key):
        return {n: 1.0 for n in SENTENCES[key][1]}

    # (section, [(row label, evidence key, target, state, zhang), ...])
    TABLES = [
        ("TABLE 9 -- evidence: loss of engine power", [
            ("P(no injury)",          "LOEP", INJ_NODE, "no injury",          0.9899),
            ("P(serious injury)",     "LOEP", INJ_NODE, "serious injury",     0.00822),
            ("P(substantial damage)", "LOEP", DMG_NODE, "substantial damage", 0.0166),
            ("P(destroyed aircraft)", "LOEP", DMG_NODE, "destroyed aircraft", 0.00559),
            ("P(minor damage)",       "LOEP", DMG_NODE, "minor damage",       0.00378),
        ]),
        ("TABLE 9 -- evidence: engine instrument", [
            ("P(LOEP)",               "eng instr", loep, "Yes",                0.95),
            ("P(no injury)",          "eng instr", INJ_NODE, "no injury",      0.9431),
            ("P(serious injury)",     "eng instr", INJ_NODE, "serious injury", 0.0623),
            ("P(substantial damage)", "eng instr", DMG_NODE, "substantial damage", 0.046),
            ("P(destroyed aircraft)", "eng instr", DMG_NODE, "destroyed aircraft", 0.0133),
            ("P(minor damage)",       "eng instr", DMG_NODE, "minor damage",   0.00934),
            ("P(gear collapsed)",     "eng instr", "gear collapsed", "Yes",    0.0096),
        ]),
        ("TABLE 9 -- evidence: combustion liner", [
            ("P(LOEP)",       "comb", loep, "Yes",           0.50),
            ("P(no injury)",  "comb", INJ_NODE, "no injury", 0.9978),
            ("P(minor damage)", "comb", DMG_NODE, "minor damage", 0.00154),
        ]),
        ("TABLE 9 -- evidence: improper oil grade", [
            ("P(no injury)",    "oil", INJ_NODE, "no injury",    0.9958),
            ("P(minor damage)", "oil", DMG_NODE, "minor damage", 0.00292),
        ]),
        ("TABLE 9 -- evidence: engine instrument + oil grade (two at once)", [
            ("P(no injury)",          "eng+oil", INJ_NODE, "no injury",      0.9429),
            ("P(serious injury)",     "eng+oil", INJ_NODE, "serious injury", 0.0623),
            ("P(substantial damage)", "eng+oil", DMG_NODE, "substantial damage", 0.0463),
            ("P(destroyed aircraft)", "eng+oil", DMG_NODE, "destroyed aircraft", 0.0135),
            ("P(minor damage)",       "eng+oil", DMG_NODE, "minor damage",   0.00947),
            ("P(gear collapsed)",     "eng+oil", "gear collapsed", "Yes",    0.00982),
        ]),
        ("FIGURE 12 -- evidence: pilot-in-command", [
            ("P(no injury)",             "pilot", INJ_NODE, "no injury",      0.97),
            ("P(substantial damage)",    "pilot", DMG_NODE, "substantial damage", 0.0458),
            ("P(unstabilized approach)", "pilot", UNSTAB, "Yes",              0.00484),
            ("P(dragged wing/pod/tail)", "pilot", DRAGGED, "Yes",             0.023),
        ]),
        ("FIGURE 12 stage 2 -- evidence: unstabilized approach", [
            ("P(no injury)", "unstab", INJ_NODE, "no injury", 0.613),
        ]),
    ]

    counts = {"EXACT": 0, "CLOSE": 0, "DIFFERS": 0}
    mismatch = 0
    for section, rows in TABLES:
        print("=" * 78)
        print(section)
        print(f'  sentence: "{SENTENCES[rows[0][1]][0]}"')
        print(f"  {'':26} {'ZHANG':>10} {'DIRECT':>10} {'NARRATIVE':>10}  "
              f"{'verdict':>7}  narrative==direct?")
        for label, key, target, state, zhang in rows:
            d = post(bn, direct_conf(key), target, state)
            n = post(bn, parsed_conf[key], target, state)
            same = abs(d - n) < 1e-12
            mismatch += (not same)
            v = verdict(zhang, d).strip()
            counts[v] = counts.get(v, 0) + 1
            print(f"  {label:26} {fmt(zhang)} {fmt(d)} {fmt(n)}  {v:>7}  "
                  f"{'YES' if same else 'NO  <-- INVESTIGATE'}")
        print()

    # ---- Table 8: what-if dial (narrative NOT applicable) ----------------------
    print("=" * 78)
    print("TABLE 8 -- what-if: raise the strut-failure PRIOR, read P(gear collapse)")
    print("  narrative: NOT APPLICABLE -- this experiment changes a PRIOR, not")
    print("  evidence; no sentence can express 'imagine struts failed 100x more'.")
    strut = "landing gear, main gear strut"
    # (strut prior, Zhang's P(main gear collapsed), Zhang's P(gear collapsed))
    T8 = [
        (6.5e-7, 2.67e-7, 2.42e-7),
        (6.5e-6, 1.73e-6, 1.70e-6),
        (6.5e-4, 1.63e-4, 1.63e-4),
        (6.5e-2, 1.62e-2, 1.62e-2),
        (0.10,   2.50e-2, 2.50e-2),
        (1.00,   2.50e-1, 2.50e-1),
    ]
    print(f"  strut node used: {strut!r}")
    print(f"  {'prior / target':32} {'ZHANG':>10} {'DIRECT':>10} {'NARRATIVE':>10}")
    for prior, z_main, z_gear in T8:
        bn2 = gum.BayesNet(bn)
        v = bn2.variable(strut)
        yes = [i for i in range(v.domainSize()) if v.label(i) == "Yes"][0]
        vals = [0.0, 0.0]
        vals[yes], vals[1 - yes] = prior, 1.0 - prior
        bn2.cpt(strut).fillWith(vals)
        for tgt, z in (("main gear collapsed", z_main), ("gear collapsed", z_gear)):
            val = post(bn2, None, tgt, "Yes")
            print(f"  prior={prior:<8.2g} P({tgt:20}) {fmt(z)} {fmt(val)} "
                  f"{'--':>10}   {verdict(z, val).strip()}")
    print()

    total = sum(counts.values())
    print("=" * 78)
    print(f"SUMMARY over {total} evidence-driven cells: "
          f"{counts.get('EXACT', 0)} exact, {counts.get('CLOSE', 0)} close, "
          f"{counts.get('DIFFERS', 0)} differ (same verdicts as the scoreboard)")
    print(f"NARRATIVE vs DIRECT: "
          f"{'identical on every cell' if mismatch == 0 else f'{mismatch} cells differ'}")
    print(f"parse check: {'all sentences -> intended nodes' if all_ok else 'SOME FAILED'}")
    return 0 if (all_ok and mismatch == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())

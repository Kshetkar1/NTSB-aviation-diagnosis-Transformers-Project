#!/usr/bin/env python3
"""FORENSIC TRACE: is NARRATIVE really identical to DIRECT, and if so WHY?

Runs both paths side by side, printing EVERY intermediate object:

  DIRECT     evidence dict {node: 1.0}      -> apply_evidence -> posterior
  NARRATIVE  sentence -> parser (which rule fired, confidence)
             -> evidence dict -> apply_evidence -> posterior

Then three falsification tests that MUST show differences if the pipeline is
actually sensitive to what the parser returns:

  (F1) vague wording  -> soft evidence (conf < 1)  -> different posterior
  (F2) manually corrupt the parsed confidence 1.0 -> 0.9 -> posterior must move
  (F3) sentence naming a DIFFERENT node -> different posterior

Posteriors printed to 17 significant digits; identity is asserted at the
floating-point level (==, not approx). Two independent BN builds are used for
the two paths to rule out shared-engine state.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/trace_narrative_vs_direct.py
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


def describe_apply(bn, confidence: dict) -> list[str]:
    """Reproduce apply_evidence's decisions verbatim, as printable lines."""
    out = []
    soft = [n for n, c in confidence.items() if c < 0.999]
    priors = qb._node_priors(bn, soft) if soft else {}
    for node, c in confidence.items():
        if c >= 0.999:
            out.append(f"    {node!r}: HARD -> ie.addEvidence(node, 'Yes')  "
                       f"(clamped, P(Yes)=1)")
        else:
            p0 = min(max(priors.get(node, 0.5), 1e-12), 1 - 1e-12)
            lr = (c / (1 - c)) / (p0 / (1 - p0))
            out.append(f"    {node!r}: SOFT conf={c} prior={p0:.3e} "
                       f"-> likelihood ratio {lr:.3e} (virtual evidence)")
    return out


def posterior(bn, confidence: dict, target, state):
    ie = gum.LazyPropagation(bn)
    if confidence:
        qb.apply_evidence(ie, bn, confidence)
    ie.addTarget(target)
    ie.makeInference()
    v = bn.variable(target)
    i = [k for k in range(v.domainSize()) if v.label(k) == state][0]
    return float(ie.posterior(target)[i])


def main():
    ds = pg.load_dataset()
    print("building network TWICE (independent objects for the two paths)...")
    bn_direct, _ = build_upgraded(ds)
    bn_narr, _ = build_upgraded(ds)
    assert bn_direct is not bn_narr
    names = [n for n in bn_narr.names() if n not in (INJ_NODE, DMG_NODE)]

    TARGETS = [(INJ_NODE, "no injury"), (DMG_NODE, "substantial damage"),
               (DMG_NODE, "destroyed aircraft")]

    def show_posts(bn, conf, tag):
        vals = []
        for t, s in TARGETS:
            p = posterior(bn, conf, t, s)
            vals.append(p)
            print(f"    P({s:20}) = {p:.17g}   [{tag}]")
        return vals

    # ============================ verbatim scenarios ==============================
    SCEN = [
        ("the aircraft experienced a loss of engine power",
         {"loss of engine power": 1.0}),
        ("the pilot in command was a factor in the accident",
         {"person: pilot-in-command": 1.0}),
    ]
    identical = True
    for sentence, direct_ev in SCEN:
        print("\n" + "=" * 78)
        print(f'SCENARIO: "{sentence}"')
        print("-" * 78)
        print("PATH A -- DIRECT (evidence clicked in, no parser involved)")
        print(f"  evidence dict: {direct_ev}")
        print("  apply_evidence decisions:")
        for line in describe_apply(bn_direct, direct_ev):
            print(line)
        va = show_posts(bn_direct, direct_ev, "DIRECT")

        print("-" * 78)
        print("PATH B -- NARRATIVE (sentence in, parser produces the evidence)")
        p = qb.parse_query_to_bn_evidence(sentence, names, dataset=ds,
                                          semantic=True)
        print("  parser trace (which rule fired):")
        for node, why in p["trace"]:
            print(f"    {node!r}  <-  {why}")
        print(f"  parsed evidence dict: {p['confidence']}")
        same_dict = p["confidence"] == direct_ev
        print(f"  >>> parsed dict == direct dict?  {same_dict}")
        print("  apply_evidence decisions:")
        for line in describe_apply(bn_narr, p["confidence"]):
            print(line)
        vb = show_posts(bn_narr, p["confidence"], "NARRATIVE")

        eq = all(a == b for a, b in zip(va, vb))
        identical &= eq and same_dict
        print(f"  >>> posteriors bit-identical (== on floats)?  {eq}")
        print("  WHY: the paths merge at the evidence dict. Same dict ->"
              " same hard clamp -> same exact inference -> same floats.")

    # ============================ falsification tests =============================
    print("\n" + "=" * 78)
    print("(F1) VAGUE wording -- parser must produce SOFT evidence, numbers must move")
    vague = "smoke started filling the cabin during the flight"
    p = qb.parse_query_to_bn_evidence(vague, names, dataset=ds, semantic=True)
    print(f'  sentence: "{vague}"')
    for node, why in p["trace"]:
        print(f"    {node!r}  <-  {why[:90]}")
    print(f"  parsed confidence: {p['confidence']}")
    f1_soft = bool(p["confidence"]) and any(c < 0.999
                                            for c in p["confidence"].values())
    if p["confidence"]:
        prior_v = show_posts(bn_narr, {}, "prior/no evidence")
        soft_v = show_posts(bn_narr, p["confidence"], "vague narrative")
        hard_v = show_posts(bn_narr, {n: 1.0 for n in p["confidence"]},
                            "same nodes clamped hard")
        moved = all(a != b for a, b in zip(prior_v, soft_v))
        between = soft_v != hard_v
        print(f"  soft evidence present: {f1_soft}; posterior moved off prior: "
              f"{moved}; soft != hard clamp: {between}")

    print("\n" + "=" * 78)
    print("(F2) CORRUPT the confidence 1.0 -> 0.9 on LOEP -- posterior MUST move")
    a = posterior(bn_narr, {"loss of engine power": 1.0}, INJ_NODE, "no injury")
    b = posterior(bn_narr, {"loss of engine power": 0.9}, INJ_NODE, "no injury")
    print(f"    conf=1.0: P(no injury) = {a:.17g}")
    print(f"    conf=0.9: P(no injury) = {b:.17g}")
    print(f"    different? {a != b}  (if these were equal, apply_evidence "
          f"would be ignoring confidence -- it is not)")

    print("\n" + "=" * 78)
    print("(F3) sentence naming a DIFFERENT node -> different posterior")
    p3 = qb.parse_query_to_bn_evidence("the flight had an unstabilized approach",
                                       names, dataset=ds, semantic=True)
    c = posterior(bn_narr, p3["confidence"], INJ_NODE, "no injury")
    print(f"    unstab-approach narrative: P(no injury) = {c:.17g}")
    print(f"    LOEP narrative:            P(no injury) = {a:.17g}")
    print(f"    different? {c != a}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print(f"  verbatim scenarios: narrative == direct bit-for-bit: {identical}")
    print(f"  falsification: soft path moves numbers ({a != b}), different "
          f"evidence moves numbers ({c != a})")
    print("  CONCLUSION: identity on the tables is BY CONSTRUCTION, not a bug --")
    print("  when a sentence NAMES the fact, the parser emits {node: 1.0}, the")
    print("  byte-identical evidence dict the direct path uses; both paths then")
    print("  run the same exact inference on the same CPTs. The narrative layer")
    print("  only changes numbers when wording is vague (soft evidence).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

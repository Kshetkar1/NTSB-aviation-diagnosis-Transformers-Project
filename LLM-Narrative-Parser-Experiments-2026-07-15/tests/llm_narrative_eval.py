#!/usr/bin/env python3
"""Evaluate the LLM narrative->evidence parser (llm_evidence.py) end to end.

PART A -- Zhang verbatim scenarios: the sentence names the fact. The LLM must
    extract exactly the intended node(s) as 'asserted'; then the BN posterior
    is BY CONSTRUCTION the same validated number from the comparison tables,
    and we print it next to Zhang's published value.

PART B -- paraphrases (vague wording): what does the LLM extract, with what
    confidence, and where do the BN posteriors land vs prior / hard clamp?
    Compared side-by-side with the retrieval soft-fact parser.

PART C -- real held-out narratives (2007-2019, unseen by the network): for
    each accident we KNOW the true NTSB labels, so we score extraction
    precision (extracted facts that are in the accident's true label set)
    for the LLM parser vs our deterministic parser.

Run (needs OPENAI_API_KEY):
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/llm_narrative_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import pyagrum as gum  # noqa: E402

import prognosis as pg  # noqa: E402
import query_to_bn as qb  # noqa: E402
from llm_evidence import llm_parse_evidence  # noqa: E402
from bn_upgraded import build_upgraded, INJ_NODE, DMG_NODE  # noqa: E402

N_HELDOUT = 8


def posterior(bn, conf, target, state):
    ie = gum.LazyPropagation(bn)
    if conf:
        qb.apply_evidence(ie, bn, conf)
    ie.addTarget(target)
    ie.makeInference()
    v = bn.variable(target)
    i = [k for k in range(v.domainSize()) if v.label(k) == state][0]
    return float(ie.posterior(target)[i])


def main():
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]

    # ---------------- PART A: Zhang verbatim scenarios --------------------------
    # (sentence, expected node set, [(target, state, zhang), ...])
    SCEN = [
        ("The aircraft experienced a loss of engine power.",
         {"loss of engine power"},
         [(INJ_NODE, "no injury", 0.9899),
          (DMG_NODE, "substantial damage", 0.0166)]),
        ("The pilot reported trouble with an engine instrument during the flight.",
         {"engine instrument"},
         [("loss of engine power", "Yes", 0.95),
          (INJ_NODE, "no injury", 0.9431)]),
        ("The wrong oil grade had been used when servicing the engine.",
         {"fluid, oil grade"},
         [("loss of engine power", "Yes", 0.95),
          (INJ_NODE, "no injury", 0.9958)]),
        ("Investigators found a failure of the combustion liner in the "
         "combustion assembly.",
         {"combustion assembly, combustion liner"},
         [("loss of engine power", "Yes", 0.50)]),
        ("The pilot in command was cited as a factor in the accident.",
         {"person: pilot-in-command"},
         [(INJ_NODE, "no injury", 0.97),
          (DMG_NODE, "substantial damage", 0.0458)]),
        ("The airplane made an unstabilized approach to the runway.",
         {"unstabilized approach"},
         [(INJ_NODE, "no injury", 0.613)]),
    ]

    print("=" * 78)
    print("PART A -- Zhang verbatim scenarios: LLM extraction -> BN posterior")
    print("=" * 78)
    a_exact = a_total = 0
    for sentence, want, targets in SCEN:
        out = llm_parse_evidence(sentence, names)
        got = set(out["evidence"])
        hard_ok = got == want and all(out["confidence"][n] >= 0.999
                                      for n in got)
        a_total += 1
        a_exact += hard_ok
        print(f'\n  "{sentence}"')
        print(f"    LLM extracted: {[(n, out['confidence'][n]) for n in out['evidence']]}")
        if out["dropped"]:
            print(f"    dropped (not in vocabulary): {out['dropped']}")
        print(f"    exact intended evidence? {'YES' if hard_ok else 'NO -- ' + str(want)}")
        for t, s, z in targets:
            p = posterior(bn, out["confidence"], t, s)
            print(f"      P({s:22}) = {p:.4g}   (Zhang {z})")
    print(f"\n  PART A extraction score: {a_exact}/{a_total} scenarios "
          f"landed exactly on the intended hard evidence")

    # ---------------- PART B: paraphrases ---------------------------------------
    print("\n" + "=" * 78)
    print("PART B -- vague paraphrases: LLM vs retrieval soft facts")
    print("=" * 78)
    PARA = [
        "The plane came down hard in a field after the propeller stopped turning.",
        "Witnesses heard the motor sputtering before the crash.",
        "Smoke started filling the cabin during the flight.",
    ]
    for sentence in PARA:
        print(f'\n  "{sentence}"')
        out = llm_parse_evidence(sentence, names)
        det = qb.parse_query_to_bn_evidence(sentence, names, dataset=ds,
                                            semantic=True)
        print(f"    LLM      : {[(n, round(out['confidence'][n], 2)) for n in out['evidence']]}")
        if out["dropped"]:
            print(f"               dropped: {out['dropped']}")
        print(f"    retrieval: {[(n, round(det['confidence'][n], 2)) for n in det['evidence']]}")
        p0 = posterior(bn, {}, INJ_NODE, "no injury")
        pl = posterior(bn, out["confidence"], INJ_NODE, "no injury")
        pr = posterior(bn, det["confidence"], INJ_NODE, "no injury")
        print(f"    P(no injury): prior={p0:.5f}  LLM={pl:.5f}  retrieval={pr:.5f}")

    # ---------------- PART C: real narratives with known labels -----------------
    # NOTE: the 2007-2019 held-out incidents have EMPTY occurrence/finding lists
    # in our refined dataset (post-2008 NTSB taxonomy fields not populated), so
    # extraction cannot be scored against truth there. We score on 1982-2006
    # incidents, which have BOTH a factual narrative and coded labels. This is
    # an extraction-quality test (text -> labels), not a prediction test; the
    # parser never sees the labels.
    print("\n" + "=" * 78)
    print(f"PART C -- {N_HELDOUT} real narratives (1982-2006, labels known): "
          "extraction precision vs true NTSB labels")
    print("=" * 78)
    heldout = []
    for ev, inc in ds.items():
        narr = str(inc.get("narr_accf") or "").strip()
        if len(narr) >= 300:
            heldout.append((ev, inc, narr))
    heldout.sort(key=lambda t: t[0])
    heldout = heldout[::max(1, len(heldout) // N_HELDOUT)][:N_HELDOUT]
    name_set = set(names)

    def true_labels(inc):
        return qb._incident_bn_labels(inc) & name_set

    tot = {"llm": [0, 0, 0], "det": [0, 0, 0]}   # [correct, extracted, truth]
    for ev, inc, narr in heldout[:N_HELDOUT]:
        truth = true_labels(inc)
        if not truth:
            continue
        out = llm_parse_evidence(narr[:2400], names)
        det = qb.parse_query_to_bn_evidence(narr, names, dataset=ds,
                                            semantic=True)
        llm_hits = [n for n in out["evidence"] if n in truth]
        det_hits = [n for n in det["evidence"] if n in truth]
        for k, o in (("llm", out), ("det", det)):
            hits = llm_hits if k == "llm" else det_hits
            tot[k][0] += len(hits)
            tot[k][1] += len(o["evidence"])
            tot[k][2] += len(truth)
        print(f"\n  {ev} -- true labels in vocab ({len(truth)}): "
              f"{sorted(truth)[:6]}{' ...' if len(truth) > 6 else ''}")
        print(f"    LLM  extracted {len(out['evidence'])}, "
              f"in-truth {len(llm_hits)}: {out['evidence']}")
        if out["dropped"]:
            print(f"          dropped: {out['dropped']}")
        print(f"    DET  extracted {len(det['evidence'])}, "
              f"in-truth {len(det_hits)}: {det['evidence']}")

    def score(k):
        c, n, t = tot[k]
        p = f"{c}/{n} = {c / n:.0%}" if n else "n/a"
        r = f"{c}/{t} = {c / t:.0%}" if t else "n/a"
        return f"precision {p}   recall {r}"
    print("\n  EXTRACTION QUALITY (against the accident's coded NTSB labels):")
    print(f"    LLM parser          : {score('llm')}")
    print(f"    deterministic parser: {score('det')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Paraphrase robustness: LLM extraction on sentences the prompt was NEVER
tuned on.

The 11/11 result in docs/LLM_VS_ZHANG_TABLES.md came after a prompt rule was
added in response to failures ON THOSE SENTENCES (generic-node over-selection).
Honest validation requires fresh sentences. This suite is adversarial by
design:

  A. clean paraphrases  -- same facts as Zhang scenarios, new wording; the
                           extraction must land on the intended nodes
  B. generic/specific   -- new traps of the kind rule 1c addresses, phrased
                           differently from the diagnosed failures
  C. word-sense traps   -- narrative words that overlap an unrelated node
                           (metal fatigue vs crew fatigue, etc.); selecting
                           the wrong-sense node is a failure
  D. no-node facts      -- facts with NO vocabulary match; extracting anything
                           for them is a hallucination failure

Scoring is strict: a case passes only if required nodes are extracted,
forbidden nodes are absent, and nothing off-vocabulary survives the guard.

Run (needs OPENAI_API_KEY + network):
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/llm_paraphrase_robustness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import prognosis as pg  # noqa: E402
from llm_evidence import llm_parse_evidence  # noqa: E402
from bn_upgraded import build_upgraded, INJ_NODE, DMG_NODE  # noqa: E402

# each case: (sentence, required nodes, forbidden nodes)
CASES = {
    "A. clean paraphrases": [
        ("Halfway through the flight the engine quit producing power.",
         {"loss of engine power"}, set()),
        ("The gauge for the engine had stopped working before departure.",
         {"engine instrument"}, set()),
        ("Maintenance had serviced the engine with oil of the wrong grade.",
         {"fluid, oil grade"}, set()),
        ("The captain flying the aircraft was found to have contributed to "
         "the accident.",
         {"person: pilot-in-command"}, set()),
        ("The jet came in fast and steep, never stabilizing on final.",
         {"unstabilized approach"}, set()),
    ],
    "B. generic vs specific (fresh traps)": [
        ("The strut on the main landing gear fractured on touchdown.",
         {"landing gear, main gear strut"}, {"landing gear"}),
        ("The liner inside the combustion assembly had burned through.",
         {"combustion assembly, combustion liner"}, {"combustion assembly"}),
        ("The nose gear would not come down and lock.",
         {"landing gear, nose gear"}, {"landing gear"}),
    ],
    "C. word-sense traps": [
        ("Metallurgical examination found fatigue cracking in the "
         "attachment bolts.",
         set(), {"fatigue (flight and ground schedule)"}),
        ("The airplane was parked at the gate when the tug struck it.",
         set(), {"gate"}),
        ("Light rain was falling but visibility remained good.",
         set(), {"light condition"}),
    ],
    "D. facts with no matching node": [
        ("The flight attendants served dinner shortly before the seatbelt "
         "sign came on.",
         set(), set()),
        ("The aircraft was painted in the airline's new livery.",
         set(), set()),
    ],
}


def main():
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]

    n_pass = n_tot = 0
    hallucinated = 0
    for section, cases in CASES.items():
        print(f"\n=== {section} ===")
        for sentence, required, forbidden in cases:
            out = llm_parse_evidence(sentence, names)
            got = set(out["evidence"])
            ok_req = required <= got
            ok_forb = not (got & forbidden)
            # in section D, extracting ANY node is over-extraction
            ok_d = True
            if section.startswith("D.") and got:
                ok_d = False
            ok = ok_req and ok_forb and ok_d
            n_tot += 1
            n_pass += ok
            hallucinated += len(out["dropped"])
            flag = "PASS" if ok else "FAIL"
            print(f"  [{flag}] \"{sentence[:64]}\"")
            print(f"         got: {sorted(got) or '(nothing)'}")
            if not ok_req:
                print(f"         MISSING required: {sorted(required - got)}")
            if not ok_forb:
                print(f"         SELECTED forbidden: {sorted(got & forbidden)}")
            if not ok_d:
                print(f"         over-extraction on a no-fact sentence")
            if out["dropped"]:
                print(f"         off-vocab (guard caught): {out['dropped']}")

    print(f"\nSCORE: {n_pass}/{n_tot} strict passes; "
          f"{hallucinated} off-vocabulary names caught by the guard "
          f"(0 reached the network)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

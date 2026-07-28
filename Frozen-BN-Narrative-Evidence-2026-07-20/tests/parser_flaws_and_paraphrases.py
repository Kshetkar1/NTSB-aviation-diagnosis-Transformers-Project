#!/usr/bin/env python3
"""Text -> codes: how each code is created, where the parser breaks, and what
happens when the sentence does NOT name the node (paraphrases).

Part A  TRACE      for the 11 scoreboard sentences, show WHICH pass produced
                   each code (outcome parse / person alias / phrase mention).
Part B  FLAW HUNT  adversarial sentences aimed at known weaknesses of
                   keyword parsers: negation, hypotheticals, generic terms,
                   out-of-vocabulary facts, substring collisions.
Part C  PARAPHRASE the same 11 facts reworded so the node name is NOT spoken.
                   The deterministic passes should miss; the retrieval pass
                   should answer with SOFT evidence (confidence < 1). We
                   report what was parsed and how far the posterior lands
                   from the clicked-evidence posterior.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/parser_flaws_and_paraphrases.py
Writes outputs/parser_flaws_and_paraphrases.md.
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

OUT = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs" / "parser_flaws_and_paraphrases.md"
L: list[str] = []


def emit(s: str = "") -> None:
    print(s)
    L.append(s)


STRUT = "landing gear, main gear strut"
LOEP = "loss of engine power"
ENG = "engine instrument"
COMB = "combustion assembly, combustion liner"
OIL = "fluid, oil grade"
PILOT = "person: pilot-in-command"
UNSTAB = "unstabilized approach"

SCOREBOARD = [
    ("trouble with an engine instrument during the flight", [ENG]),
    ("a failure of the combustion assembly, combustion liner", [COMB]),
    ("the wrong fluid, oil grade was used", [OIL]),
    ("trouble with an engine instrument and the wrong fluid, oil grade",
     [ENG, OIL]),
    ("the aircraft experienced a loss of engine power", [LOEP]),
    ("the pilot in command was a factor in the accident", [PILOT]),
    ("the flight had an unstabilized approach", [UNSTAB]),
    ("a failure of the landing gear, main gear strut", [STRUT]),
]

FLAW_PROBES = [
    ("negation",
     "there was no fire on board and the engine did not lose power"),
    ("hypothetical",
     "the pilot worried about a loss of engine power but the engine ran fine"),
    ("substring safety",
     "the engine misfired repeatedly on final approach"),
    ("generic term",
     "the landing gear was damaged during the landing"),
    ("out of vocabulary",
     "the drone's parachute recovery system failed to deploy"),
    ("empty-ish",
     "routine flight"),
]

# (label, paraphrase that does NOT speak the node name, intended node,
#  target to score, target state)
PARAPHRASES = [
    ("engine instrument",
     "the cockpit gauges monitoring the engine were giving bad readings",
     ENG, LOEP, "Yes"),
    ("combustion liner",
     "a component inside the burner section of the engine cracked",
     COMB, LOEP, "Yes"),
    ("oil grade",
     "the mechanic had serviced the engine with the wrong type of oil",
     OIL, LOEP, "Yes"),
    ("loss of engine power",
     "the engine quit while the airplane was in cruise flight",
     LOEP, "forced landing", "Yes"),
    ("pilot-in-command",
     "the captain made an error during the approach",
     PILOT, DMG_NODE, "substantial damage"),
    ("unstabilized approach",
     "the airplane came in too high and too fast to land safely",
     UNSTAB, "dragged wing, rotor, pod, float or tail/skid", "Yes"),
    ("main gear strut",
     "the strut holding the main wheel collapsed under the airplane",
     STRUT, "main gear collapsed", "Yes"),
]


def post(bn, conf, target, state):
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

    emit("# Text -> codes: trace, flaw hunt, paraphrases")
    emit()

    # ---------------- Part A: which pass created each code -------------------
    emit("## Part A -- how each scoreboard code was created")
    emit()
    emit("| Sentence | Code produced | Which pass | Confidence |")
    emit("|---|---|---|---|")
    for sentence, want in SCOREBOARD:
        p = qb.parse_query_to_bn_evidence(sentence, names, dataset=ds,
                                          semantic=True)
        for node, reason in p["trace"]:
            emit(f'| "{sentence}" | `{node}` | {reason} | '
                 f'{p["confidence"][node]} |')
    emit()

    # ---------------- Part B: adversarial probes ------------------------------
    emit("## Part B -- flaw hunt (adversarial sentences)")
    emit()
    emit("| Probe | Sentence | Parsed codes (confidence) | Reading |")
    emit("|---|---|---|---|")
    for label, sentence in FLAW_PROBES:
        p = qb.parse_query_to_bn_evidence(sentence, names, dataset=ds,
                                          semantic=True)
        codes = "; ".join(f"`{n}` ({p['confidence'][n]:.2f})"
                          for n in p["evidence"]) or "(none -> prior)"
        emit(f"| {label} | \"{sentence}\" | {codes} | |")
    emit()

    # ---------------- Part C: paraphrases -------------------------------------
    emit("## Part C -- paraphrases (node name NOT spoken)")
    emit()
    emit("| Fact | Paraphrase | Parsed codes (conf) | Intended found? | "
         "P(target): prior / paraphrase / clicked |")
    emit("|---|---|---|---|---|")
    for label, sentence, intended, target, state in PARAPHRASES:
        p = qb.parse_query_to_bn_evidence(sentence, names, dataset=ds,
                                          semantic=True)
        codes = "; ".join(f"`{n}` ({p['confidence'][n]:.2f})"
                          for n in p["evidence"]) or "(none)"
        if intended in p["evidence"]:
            hit = ("HARD" if p["confidence"][intended] >= 0.999
                   else f"SOFT {p['confidence'][intended]:.2f}")
        else:
            hit = "missed"
        pr = post(bn, {}, target, state)
        pq = post(bn, p["confidence"], target, state)
        pc = post(bn, {intended: 1.0}, target, state)
        emit(f"| {label} | \"{sentence}\" | {codes} | {hit} | "
             f"{pr:.3g} / {pq:.3g} / {pc:.3g} |")
    emit()

    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

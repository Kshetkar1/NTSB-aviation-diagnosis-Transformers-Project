#!/usr/bin/env python3
"""LLM parser vs Zhang's tables -- every evidence-driven table, five ways:

    ZHANG        number printed in the paper
    UPGRADED     our upgraded network, evidence set directly (clicked)
    DET PARSE    upgraded network driven by the sentence through OUR
                 deterministic + retrieval parser (query_to_bn)
    LLM PARSE    upgraded network driven by the SAME sentence through the
                 LLM parser (llm_evidence, gpt-4o-mini) -- the LLM's own
                 evidence dict and confidences

Evidence-driven tables covered: Table 9 (all 5 evidence sets x 10 targets),
Section 5.2 cumulative evidence, Figure 12 stage 1 (pilot) and stage 2
(unstabilized approach). Table 8 and the Fig 12 priors involve NO evidence
(they edit a prior / query the empty network), so no parser -- LLM or
otherwise -- applies.

Writes docs/LLM_VS_ZHANG_TABLES.md.

Run (needs OPENAI_API_KEY + network):
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/llm_all_tables.py
"""
from __future__ import annotations

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

OUT_MD = ROOT / "docs" / "LLM_VS_ZHANG_TABLES.md"

STRUT = "landing gear, main gear strut"
LOEP = "loss of engine power"
ENG = "engine instrument"
COMB = "combustion assembly, combustion liner"
OIL = "fluid, oil grade"
PILOT = "person: pilot-in-command"
UNSTAB = "unstabilized approach"
DRAGGED = "dragged wing, rotor, pod, float or tail/skid"

lines: list[str] = []


def emit(s: str = "") -> None:
    print(s)
    lines.append(s)


def g6(x) -> str:
    if x is None:
        return "--"
    if isinstance(x, str):
        return x
    return f"{x:.6g}"


def posts(bn, evidence_conf, targets):
    ie = gum.LazyPropagation(bn)
    if evidence_conf:
        qb.apply_evidence(ie, bn, evidence_conf)
    for n, _ in targets:
        ie.addTarget(n)
    ie.makeInference()
    out = []
    for n, state in targets:
        v = bn.variable(n)
        i = [k for k in range(v.domainSize()) if v.label(k) == state][0]
        out.append(float(ie.posterior(n)[i]))
    return out


def ev_str(conf: dict) -> str:
    if not conf:
        return "(none)"
    return "; ".join(f"{n}" + ("" if c >= 0.999 else f" (soft {c:.0%})")
                     for n, c in conf.items())


# Same sentences as tests/all_tables_exact.py -- the fair comparison is
# both parsers reading identical text.
SENT = {
    "eng":      ("trouble with an engine instrument during the flight", [ENG]),
    "comb":     ("a failure of the combustion assembly, combustion liner", [COMB]),
    "oil":      ("the wrong fluid, oil grade was used", [OIL]),
    "eng+oil":  ("trouble with an engine instrument and the wrong fluid, oil grade",
                 [ENG, OIL]),
    "loep":     ("the aircraft experienced a loss of engine power", [LOEP]),
    "pilot":    ("the pilot in command was a factor in the accident", [PILOT]),
    "unstab":   ("the flight had an unstabilized approach", [UNSTAB]),
    "s1":       ("a failure of the landing gear, main gear strut", [STRUT]),
    "s2":       ("a failure of the landing gear, main gear strut and the "
                 "landing gear, emergency extension assembly",
                 [STRUT, "landing gear, emergency extension assembly"]),
    "s3":       ("a failure of the landing gear, main gear strut, the landing "
                 "gear, emergency extension assembly and the landing gear, "
                 "gear locking mechanism",
                 [STRUT, "landing gear, emergency extension assembly",
                  "landing gear, gear locking mechanism"]),
    "s4":       ("a failure of the landing gear, main gear strut, the landing "
                 "gear, emergency extension assembly, the landing gear, gear "
                 "locking mechanism and the landing gear, main gear attachment",
                 [STRUT, "landing gear, emergency extension assembly",
                  "landing gear, gear locking mechanism",
                  "landing gear, main gear attachment"]),
}


def main():
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]

    DET, LLM = {}, {}
    for key, (sentence, want) in SENT.items():
        DET[key] = qb.parse_query_to_bn_evidence(
            sentence, names, dataset=ds, semantic=True)["confidence"]
        out = llm_parse_evidence(sentence, names)
        LLM[key] = out["confidence"]
        exact = sorted(out["evidence"]) == sorted(want) and all(
            c >= 0.999 for c in out["confidence"].values())
        print(f"[parse] {key:8s} LLM -> {ev_str(LLM[key])}"
              f"{'' if exact else '   <-- DIFFERS from intended ' + str(want)}"
              + (f"   dropped={out['dropped']}" if out["dropped"] else ""))

    emit("# LLM parser vs Zhang's tables -- every evidence-driven table")
    emit()
    emit("Same upgraded network, same exact inference, same English sentences. "
         "The only thing that changes between the last two columns is WHO reads "
         "the sentence: our deterministic + retrieval parser vs the LLM "
         "(gpt-4o-mini, guard-railed to the network vocabulary).")
    emit()
    emit("Not applicable to any parser: Table 8 (edits a prior, observes "
         "nothing) and the Fig 12 priors (network before any evidence). "
         "Table 6/Eq 6 and Table 7 are the counting layer -- no evidence "
         "parsing involved; both already reproduce exactly.")
    emit()

    emit("## What each parser extracted from each sentence")
    emit()
    emit("| Scenario | Sentence | Det parse | LLM parse | Same? |")
    emit("|---|---|---|---|---|")
    for key, (sentence, want) in SENT.items():
        same = "YES" if dict(DET[key]) == dict(LLM[key]) else "**NO**"
        emit(f"| {key} | {sentence} | {ev_str(DET[key])} | "
             f"{ev_str(LLM[key])} | {same} |")
    emit()

    # =========================== Table 9: full grid ==============================
    T9_COLS = [
        ("Inoperative engine instruments", "eng", {ENG: 1.0}),
        ("Combustion liner failure", "comb", {COMB: 1.0}),
        ("Improper oil usage", "oil", {OIL: 1.0}),
        ("Engine instr + improper oil", "eng+oil", {ENG: 1.0, OIL: 1.0}),
        ("Loss of engine power", "loep", {LOEP: 1.0}),
    ]
    T9_ROWS = [
        ("Loss of engine power",        (LOEP, "Yes"),
         [0.95, 0.50, 0.95, 0.99, 1.0]),
        ("Forced landing",              ("forced landing", "Yes"),
         [0.1357, 0.0714, 0.1357, 0.1471, 0.1429]),
        ("Ditching",                    ("ditching", "Yes"),
         [4.37e-3, 2.30e-3, 4.37e-3, 4.57e-3, 4.61e-3]),
        ("Gear collapsed",              ("gear collapsed", "Yes"),
         [9.60e-3, 2.30e-3, 4.37e-3, 9.82e-3, 5.18e-3]),
        ("Other gear collapsed",        ("other gear collapsed", "Yes"),
         [4.80e-3, 2.30e-3, 4.37e-3, 5.00e-3, 4.66e-3]),
        ("Destroyed aircraft",          (DMG_NODE, "destroyed aircraft"),
         [1.33e-2, 2.30e-3, 4.37e-3, 1.35e-2, 5.59e-3]),
        ("Substantial aircraft damage", (DMG_NODE, "substantial damage"),
         [4.60e-2, 3.63e-3, 6.09e-3, 4.63e-2, 1.66e-2]),
        ("Minor aircraft damage",       (DMG_NODE, "minor damage"),
         [9.34e-3, 1.54e-3, 2.92e-3, 9.47e-3, 3.78e-3]),
        ("Serious injury",              (INJ_NODE, "serious injury"),
         [6.23e-2, 7.68e-4, 1.46e-3, 6.23e-2, 8.22e-3]),
        ("No injury",                   (INJ_NODE, "no injury"),
         [0.9431, 0.9978, 0.9958, 0.9429, 0.9899]),
    ]
    targets = [r[1] for r in T9_ROWS]
    for ci, (col_label, skey, ev) in enumerate(T9_COLS):
        direct = posts(bn, ev, targets)
        det = posts(bn, DET[skey], targets)
        llm = posts(bn, LLM[skey], targets)
        emit(f"## Table 9 -- evidence: {col_label}")
        emit()
        emit(f'Sentence: *"{SENT[skey][0]}"* -- LLM extracted: '
             f'{ev_str(LLM[skey])}')
        emit()
        emit("| Target | Zhang | Upgraded (direct) | Det parse | LLM parse |")
        emit("|---|---|---|---|---|")
        for ri, (row_label, _, zvals) in enumerate(T9_ROWS):
            emit(f"| P({row_label}) | {g6(zvals[ci])} | {g6(direct[ri])} | "
                 f"{g6(det[ri])} | {g6(llm[ri])} |")
        emit()

    # =========================== Section 5.2 cumulative ==========================
    emit("## Section 5.2 -- cumulative evidence on P(main gear collapsed)")
    emit()
    SEC52 = [
        ("s1", "strut", 0.25),
        ("s2", "strut + emergency extension", 0.682),
        ("s3", "strut + ext + gear locking", 0.777),
        ("s4", "strut + ext + lock + attachment", 0.894),
    ]
    emit("| Evidence (cumulative) | Zhang | Upgraded (direct) | Det parse | LLM parse |")
    emit("|---|---|---|---|---|")
    tgt = [("main gear collapsed", "Yes")]
    for skey, label, z in SEC52:
        ev = {n: 1.0 for n in SENT[skey][1]}
        d = posts(bn, ev, tgt)[0]
        n = posts(bn, DET[skey], tgt)[0]
        m = posts(bn, LLM[skey], tgt)[0]
        emit(f"| {label} | {g6(z)} | {g6(d)} | {g6(n)} | {g6(m)} |")
    emit()

    # =========================== Figure 12 =======================================
    emit("## Figure 12 stage 1 -- evidence: pilot-in-command")
    emit()
    emit(f'Sentence: *"{SENT["pilot"][0]}"* -- LLM extracted: '
         f'{ev_str(LLM["pilot"])}')
    emit()
    F12_PILOT = [
        ("P(unstabilized approach)", (UNSTAB, "Yes"), 4.84e-3),
        ("P(dragged wing/rotor/pod/tail)", (DRAGGED, "Yes"), 2.30e-2),
        ("P(substantial damage)", (DMG_NODE, "substantial damage"), 4.58e-2),
        ("P(no injury)", (INJ_NODE, "no injury"), 0.97),
    ]
    t = [x[1] for x in F12_PILOT]
    d = posts(bn, {PILOT: 1.0}, t)
    n = posts(bn, DET["pilot"], t)
    m = posts(bn, LLM["pilot"], t)
    emit("| Quantity | Zhang | Upgraded (direct) | Det parse | LLM parse |")
    emit("|---|---|---|---|---|")
    for (label, _, z), dv, nv, mv in zip(F12_PILOT, d, n, m):
        emit(f"| {label} | {g6(z)} | {g6(dv)} | {g6(nv)} | {g6(mv)} |")
    emit()

    emit("## Figure 12 stage 2 -- evidence: unstabilized approach")
    emit()
    emit(f'Sentence: *"{SENT["unstab"][0]}"* -- LLM extracted: '
         f'{ev_str(LLM["unstab"])}')
    emit()
    F12_S2 = [
        ("P(dragged wing/rotor/pod/tail)", (DRAGGED, "Yes"), 0.4172),
        ("P(substantial damage)", (DMG_NODE, "substantial damage"), 0.2464),
        ("P(no injury)", (INJ_NODE, "no injury"), 0.613),
    ]
    t = [x[1] for x in F12_S2]
    d = posts(bn, {UNSTAB: 1.0}, t)
    n = posts(bn, DET["unstab"], t)
    m = posts(bn, LLM["unstab"], t)
    emit("| Quantity | Zhang | Upgraded (direct) | Det parse | LLM parse |")
    emit("|---|---|---|---|---|")
    for (label, _, z), dv, nv, mv in zip(F12_S2, d, n, m):
        emit(f"| {label} | {g6(z)} | {g6(dv)} | {g6(nv)} | {g6(mv)} |")
    emit()

    # =========================== verdict =========================================
    n_same = sum(1 for k in SENT if dict(DET[k]) == dict(LLM[k]))
    emit("## Verdict")
    emit()
    emit(f"- LLM parse landed on the identical evidence dict as the "
         f"deterministic parser on **{n_same}/{len(SENT)}** scenarios; "
         "wherever the dicts match, every posterior is identical (same "
         "network, same inference).")
    emit("- Wherever they differ, the tables above show exactly how far the "
         "LLM's numbers drift from Zhang / the direct evidence.")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

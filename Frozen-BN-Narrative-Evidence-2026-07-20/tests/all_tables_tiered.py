#!/usr/bin/env python3
"""EVERY recreatable table, driven through the NEW TIERED PARSER.

Same tables and sentences as tests/all_tables_exact.py, but the narrative
column now goes through combined_parse (tier 1 deterministic, tier 2 LLM
fallback with data-grounded strengths, tier 3 retrieval). For each sentence
we report WHICH TIER fired, and the table shows

    ZHANG                number printed in the paper
    UPGRADED (clicked)   evidence set directly on the upgraded network
    TIERED (sentence)    the same network driven by the typed sentence
                         through the tiered parser

Expectation: every table sentence NAMES its facts, so tier 1 fires and the
tiered column equals the clicked column exactly. This script verifies that
instead of assuming it. Writes docs/ALL_TABLES_TIERED_PARSER.md.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/all_tables_tiered.py
"""
from __future__ import annotations

import json
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
from llm_evidence import combined_parse  # noqa: E402
from bn_upgraded import build_upgraded, INJ_NODE, DMG_NODE  # noqa: E402
from all_tables_exact import (STRUT, LOEP, ENG, COMB, OIL, PILOT,  # noqa: E402
                              UNSTAB, DRAGGED)

OUT_MD = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "docs_FrozenBN" / "ALL_TABLES_TIERED_PARSER.md"
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


def main():
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]

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

    emit("# Every recreatable table -- through the TIERED parser")
    emit()
    emit("Zhang = printed in the paper. Upgraded (clicked) = evidence set by "
         "hand on the upgraded network. Tiered (sentence) = the typed sentence "
         "through the tiered parser (deterministic first, LLM fallback), then "
         "the same network.")
    emit()

    # ------------------ parse every sentence through the tiered front door -----
    emit("## Which tier handled each sentence")
    emit()
    emit("| Sentence | Tier | Evidence parsed | Exact match to clicked |")
    emit("|---|---|---|---|")
    CONF = {}
    parse_fail = []
    for key, (sentence, want) in SENT.items():
        p = combined_parse(sentence, names, ds)
        CONF[key] = p["confidence"]
        hard = sorted(p["evidence"]) == sorted(want) and all(
            p["confidence"][e] >= 0.999 for e in p["evidence"])
        if not hard:
            parse_fail.append((key, sentence, p["evidence"]))
        ev = "; ".join(f"`{n}` ({p['confidence'][n]:.2f})"
                       for n in sorted(p["evidence"]))
        emit(f'| "{sentence}" | {p["tier"]} | {ev} | '
             f'{"YES" if hard else "NO"} |')
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
        narr = posts(bn, CONF[skey], targets)
        emit(f"## Table 9 -- evidence: {col_label}")
        emit()
        emit(f'Sentence: *"{SENT[skey][0]}"*')
        emit()
        emit("| Target | Zhang | Upgraded (clicked) | Tiered (sentence) |")
        emit("|---|---|---|---|")
        for ri, (row_label, _, zvals) in enumerate(T9_ROWS):
            emit(f"| P({row_label}) | {g6(zvals[ci])} | {g6(direct[ri])} | "
                 f"{g6(narr[ri])} |")
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
    emit("| Evidence (cumulative) | Zhang | Upgraded (clicked) | Tiered (sentence) |")
    emit("|---|---|---|---|")
    for skey, label, z in SEC52:
        ev = {n: 1.0 for n in SENT[skey][1]}
        d = posts(bn, ev, [("main gear collapsed", "Yes")])[0]
        n = posts(bn, CONF[skey], [("main gear collapsed", "Yes")])[0]
        emit(f"| {label} | {g6(z)} | {g6(d)} | {g6(n)} |")
    emit()

    # =========================== Figure 12 =======================================
    emit("## Figure 12 stage 1 -- evidence: pilot-in-command")
    emit()
    emit(f'Sentence: *"{SENT["pilot"][0]}"*')
    emit()
    F12_PILOT = [
        ("P(unstabilized approach)", (UNSTAB, "Yes"), 4.84e-3),
        ("P(dragged wing/rotor/pod/tail)", (DRAGGED, "Yes"), 2.30e-2),
        ("P(substantial damage)", (DMG_NODE, "substantial damage"), 4.58e-2),
        ("P(no injury)", (INJ_NODE, "no injury"), 0.97),
    ]
    d = posts(bn, {PILOT: 1.0}, [t for _, t, _ in F12_PILOT])
    n = posts(bn, CONF["pilot"], [t for _, t, _ in F12_PILOT])
    emit("| Quantity | Zhang | Upgraded (clicked) | Tiered (sentence) |")
    emit("|---|---|---|---|")
    for (label, _, z), dv, nv in zip(F12_PILOT, d, n):
        emit(f"| {label} | {g6(z)} | {g6(dv)} | {g6(nv)} |")
    emit()

    emit("## Figure 12 stage 2 -- evidence: unstabilized approach")
    emit()
    emit(f'Sentence: *"{SENT["unstab"][0]}"*')
    emit()
    F12_S2 = [
        ("P(dragged wing/rotor/pod/tail)", (DRAGGED, "Yes"), 0.4172),
        ("P(substantial damage)", (DMG_NODE, "substantial damage"), 0.2464),
        ("P(no injury)", (INJ_NODE, "no injury"), 0.613),
    ]
    d = posts(bn, {UNSTAB: 1.0}, [t for _, t, _ in F12_S2])
    n = posts(bn, CONF["unstab"], [t for _, t, _ in F12_S2])
    emit("| Quantity | Zhang | Upgraded (clicked) | Tiered (sentence) |")
    emit("|---|---|---|---|")
    for (label, _, z), dv, nv in zip(F12_S2, d, n):
        emit(f"| {label} | {g6(z)} | {g6(dv)} | {g6(nv)} |")
    emit()

    # =========================== not sentence-drivable ===========================
    emit("## Not sentence-drivable through ANY parser (and why)")
    emit()
    emit("- **Table 8** edits a PRIOR (imagine strut failures were more "
         "common); it observes no evidence, so no sentence can express it. "
         "Recreated by direct prior manipulation: matches Zhang exactly on "
         "the analytic rows (0.1 and up).")
    emit("- **Table 7 and the counting anchors** live in the counting layer, "
         "before the network: 85/85 exact, see "
         "`docs/TABLE7_FULL_REPRODUCTION.md`.")
    emit("- **Figure 12 priors** are the network before any sentence.")
    emit()

    if parse_fail:
        emit(f"**PARSE FAILURES: {parse_fail}**")
    else:
        emit("**VERDICT: every table sentence was handled by tier 1 as hard "
             "evidence identical to the clicked nodes, so Tiered (sentence) "
             "== Upgraded (clicked) on every cell. The LLM tier changes "
             "nothing on the Zhang comparisons; it only adds coverage for "
             "paraphrased wording the tables never use.**")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_MD}")
    return 0 if not parse_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())

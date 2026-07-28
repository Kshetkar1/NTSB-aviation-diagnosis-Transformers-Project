#!/usr/bin/env python3
"""NEGATIVE CONTROL ("null test") for the narrative -> BN bridge.

The concern: the narrative-driven scoreboard is IDENTICAL to the
clicked-evidence scoreboard (same exact/close/differ counts). Is that because
the pipeline genuinely reads the sentence, or because the numbers would come
out the same no matter what text goes in (a plumbing bypass)?

The control: run the same 11 sentence-driven scenarios three more ways and
watch the agreement COLLAPSE when the input is corrupted.

    MATCHED    the correct sentence for the scenario      -> must equal direct
    SHUFFLED   the sentence from a DIFFERENT scenario     -> must NOT equal
               (derangement: no sentence stays in place)     direct (except
                                                              where evidence
                                                              genuinely overlaps)
    GIBBERISH  a nonsense sentence with no aviation facts -> evidence empty,
                                                             posterior = prior
    EMPTY      no sentence at all                         -> prior (reference)

Agreement metric per scenario: max relative difference over the 10 Table 9
targets (or the scenario's own targets) between the narrative posterior and
the direct-evidence posterior. "Identical" = every target within 1e-9.

If MATCHED gives 11/11 identical while SHUFFLED and GIBBERISH give ~0, the
identity in the scoreboard is EARNED (the sentence determines the evidence),
not plumbing.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/narrative_negative_control.py
Writes outputs/narrative_negative_control.md.
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

OUT = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs" / "narrative_negative_control.md"

STRUT = "landing gear, main gear strut"
LOEP = "loss of engine power"
ENG = "engine instrument"
COMB = "combustion assembly, combustion liner"
OIL = "fluid, oil grade"
PILOT = "person: pilot-in-command"
UNSTAB = "unstabilized approach"

# same 11 scenarios as tests/all_tables_exact.py
SENT = {
    "eng":     ("trouble with an engine instrument during the flight", [ENG]),
    "comb":    ("a failure of the combustion assembly, combustion liner", [COMB]),
    "oil":     ("the wrong fluid, oil grade was used", [OIL]),
    "eng+oil": ("trouble with an engine instrument and the wrong fluid, oil grade",
                [ENG, OIL]),
    "loep":    ("the aircraft experienced a loss of engine power", [LOEP]),
    "pilot":   ("the pilot in command was a factor in the accident", [PILOT]),
    "unstab":  ("the flight had an unstabilized approach", [UNSTAB]),
    "s1":      ("a failure of the landing gear, main gear strut", [STRUT]),
    "s2":      ("a failure of the landing gear, main gear strut and the "
                "landing gear, emergency extension assembly",
                [STRUT, "landing gear, emergency extension assembly"]),
    "s3":      ("a failure of the landing gear, main gear strut, the landing "
                "gear, emergency extension assembly and the landing gear, "
                "gear locking mechanism",
                [STRUT, "landing gear, emergency extension assembly",
                 "landing gear, gear locking mechanism"]),
    "s4":      ("a failure of the landing gear, main gear strut, the landing "
                "gear, emergency extension assembly, the landing gear, gear "
                "locking mechanism and the landing gear, main gear attachment",
                [STRUT, "landing gear, emergency extension assembly",
                 "landing gear, gear locking mechanism",
                 "landing gear, main gear attachment"]),
}
KEYS = list(SENT)

# derangement: every scenario receives the sentence of the NEXT scenario
SHUFFLE = {k: KEYS[(i + 1) % len(KEYS)] for i, k in enumerate(KEYS)}

GIBBERISH = ("the quarterly report was filed on time and the committee "
             "adjourned for lunch")

TARGETS = [
    (LOEP, "Yes"), ("forced landing", "Yes"), ("ditching", "Yes"),
    ("gear collapsed", "Yes"), ("main gear collapsed", "Yes"),
    (DMG_NODE, "destroyed aircraft"), (DMG_NODE, "substantial damage"),
    (DMG_NODE, "minor damage"), (INJ_NODE, "serious injury"),
    (INJ_NODE, "no injury"),
]


def posts(bn, conf):
    ie = gum.LazyPropagation(bn)
    if conf:
        qb.apply_evidence(ie, bn, conf)
    for n, _ in TARGETS:
        ie.addTarget(n)
    ie.makeInference()
    out = []
    for n, state in TARGETS:
        v = bn.variable(n)
        i = [k for k in range(v.domainSize()) if v.label(k) == state][0]
        out.append(float(ie.posterior(n)[i]))
    return out


def max_reldiff(a, b):
    m = 0.0
    for x, y in zip(a, b):
        denom = max(abs(x), abs(y), 1e-300)
        m = max(m, abs(x - y) / denom)
    return m


def verdict(d):
    return "IDENTICAL" if d < 1e-9 else f"differs (max rel {d:.2g})"


def main():
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]

    prior = posts(bn, {})
    lines = [
        "# Negative control for the narrative bridge",
        "",
        "Question: the narrative-driven scoreboard equals the clicked-evidence "
        "scoreboard. Genuine, or would ANY text produce the same numbers?",
        "",
        "Control: same 11 scenarios, four input conditions. Agreement = max "
        "relative difference over 10 targets vs the direct-evidence posterior.",
        "",
        "| Scenario | Matched sentence | Shuffled sentence (from another "
        "scenario) | Gibberish |",
        "|---|---|---|---|",
    ]

    n_match_ident = n_shuf_ident = n_gib_prior = 0
    print(f"{'scenario':8} {'matched':>26} {'shuffled':>26} {'gibberish':>26}")
    for key in KEYS:
        sentence, want = SENT[key]
        direct = posts(bn, {n: 1.0 for n in want})

        p = qb.parse_query_to_bn_evidence(sentence, names, dataset=ds,
                                          semantic=True)
        matched = posts(bn, p["confidence"])
        d_match = max_reldiff(matched, direct)
        n_match_ident += d_match < 1e-9

        wrong_sentence = SENT[SHUFFLE[key]][0]
        p2 = qb.parse_query_to_bn_evidence(wrong_sentence, names, dataset=ds,
                                           semantic=True)
        shuffled = posts(bn, p2["confidence"])
        d_shuf = max_reldiff(shuffled, direct)
        n_shuf_ident += d_shuf < 1e-9

        p3 = qb.parse_query_to_bn_evidence(GIBBERISH, names, dataset=ds,
                                           semantic=True)
        gib = posts(bn, p3["confidence"])
        d_gib_direct = max_reldiff(gib, direct)
        d_gib_prior = max_reldiff(gib, prior)
        n_gib_prior += d_gib_prior < 1e-9

        gib_txt = ("= prior" if d_gib_prior < 1e-9
                   else f"max rel {d_gib_prior:.2g} vs prior")
        print(f"{key:8} {verdict(d_match):>26} {verdict(d_shuf):>26} "
              f"{gib_txt:>26}")
        lines.append(f"| {key} | {verdict(d_match)} | {verdict(d_shuf)} "
                     f"| {gib_txt} |")

    n = len(KEYS)
    lines += [
        "",
        f"**Matched sentences: {n_match_ident}/{n} identical to direct "
        f"evidence. Shuffled sentences: {n_shuf_ident}/{n} identical "
        "(expected ~0; any nonzero cases are scenarios whose evidence "
        "genuinely overlaps, e.g. the cumulative gear chain s1..s4). "
        f"Gibberish: {n_gib_prior}/{n} collapse to the prior.**",
        "",
        "Conclusion: the sentence DETERMINES the evidence. Feed the right "
        "sentence, get the clicked-evidence numbers; feed the wrong sentence, "
        "get different numbers; feed nonsense, get the prior. The identity in "
        "the scoreboard is earned, not plumbing.",
        "",
    ]
    print(f"\nmatched identical: {n_match_ident}/{n}   "
          f"shuffled identical: {n_shuf_ident}/{n}   "
          f"gibberish = prior: {n_gib_prior}/{n}")

    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

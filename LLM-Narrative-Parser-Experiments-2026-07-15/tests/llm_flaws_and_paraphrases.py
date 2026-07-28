#!/usr/bin/env python3
"""The SAME flaw probes and paraphrases, through the LLM parser.

Companion to tests/parser_flaws_and_paraphrases.py: the deterministic keyword
parser fails on negation, hypotheticals and paraphrases (by construction --
it matches words, not meaning). This script shows how much of that boundary
the LLM parser (llm_evidence.llm_parse_evidence + hybrid_confidence) closes.

For each sentence we report the extracted codes and, for paraphrases, the
posterior of the scenario's target vs the clicked-evidence posterior.

Run (needs OPENAI_API_KEY):
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/llm_flaws_and_paraphrases.py
Writes outputs/llm_flaws_and_paraphrases.md.
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
from llm_evidence import llm_parse_evidence, hybrid_confidence  # noqa: E402
from bn_upgraded import build_upgraded, INJ_NODE, DMG_NODE  # noqa: E402
from parser_flaws_and_paraphrases import (  # noqa: E402
    FLAW_PROBES, PARAPHRASES, post)

OUT = ROOT / "outputs" / "llm_flaws_and_paraphrases.md"
L: list[str] = []


def emit(s: str = "") -> None:
    print(s)
    L.append(s)


def main():
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]

    emit("# LLM parser on the flaw probes and paraphrases")
    emit()
    emit("Same sentences as `parser_flaws_and_paraphrases.py`; parser is "
         "gpt-4o-mini with vocabulary guardrails, confidences re-grounded by "
         "`hybrid_confidence` (LLM picks WHICH facts, retrieval data decides "
         "HOW STRONGLY).")
    emit()

    emit("## Flaw probes")
    emit()
    emit("| Probe | Sentence | Raw LLM codes | LLM codes (hybrid conf) | "
         "Expected |")
    emit("|---|---|---|---|---|")
    expect = {
        "negation": "nothing (both facts are negated)",
        "hypothetical": "nothing, or pilot only",
        "substring safety": "nothing hard; engine trouble soft at most",
        "generic term": "landing gear (generic, unknown component)",
        "out of vocabulary": "nothing (fact has no node)",
        "empty-ish": "nothing",
    }
    for label, sentence in FLAW_PROBES:
        raw = llm_parse_evidence(sentence, names)
        conf = hybrid_confidence(sentence, raw, ds)
        raw_codes = "; ".join(f"`{n}`" for n in raw["evidence"]) or "(none)"
        codes = "; ".join(f"`{n}` ({c:.2f})"
                          for n, c in conf.items()) or "(none -> prior)"
        emit(f"| {label} | \"{sentence}\" | {raw_codes} | {codes} | "
             f"{expect[label]} |")
    emit()

    emit("## Paraphrases (node name NOT spoken)")
    emit()
    emit("| Fact | Paraphrase | LLM codes (hybrid conf) | Intended found? | "
         "P(target): keyword parser / LLM / clicked |")
    emit("|---|---|---|---|---|")
    n_hit = 0
    for label, sentence, intended, target, state in PARAPHRASES:
        raw = llm_parse_evidence(sentence, names)
        conf = hybrid_confidence(sentence, raw, ds)
        codes = "; ".join(f"`{n}` ({c:.2f})"
                          for n, c in conf.items()) or "(none)"
        if intended in conf:
            hit = ("HARD" if conf[intended] >= 0.999
                   else f"SOFT {conf[intended]:.2f}")
            n_hit += 1
        else:
            hit = "missed"
        kw = qb.parse_query_to_bn_evidence(sentence, names, dataset=ds,
                                           semantic=True)
        p_kw = post(bn, kw["confidence"], target, state)
        p_llm = post(bn, conf, target, state)
        p_click = post(bn, {intended: 1.0}, target, state)
        emit(f"| {label} | \"{sentence}\" | {codes} | {hit} | "
             f"{p_kw:.3g} / {p_llm:.3g} / {p_click:.3g} |")
    emit()
    emit(f"**LLM found the intended node in {n_hit}/{len(PARAPHRASES)} "
         "paraphrases (keyword parser: 1/7).**")
    emit()

    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

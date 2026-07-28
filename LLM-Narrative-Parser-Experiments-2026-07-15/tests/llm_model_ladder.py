#!/usr/bin/env python3
"""Does a STRONGER LLM close the paraphrase gap?

Same flaw probes and paraphrases as tests/parser_flaws_and_paraphrases.py,
run across a ladder of models. Two scores per model:

  SAFETY    the 6 adversarial probes: negation and hypotheticals must yield
            no event evidence; out-of-vocabulary must yield nothing; the
            generic-term probe should yield exactly 'landing gear'.
  RECOVERY  the 7 paraphrases: how often the intended node is found at all
            (hard or soft, after hybrid_confidence re-grounds strengths).

Confidences are always re-grounded by hybrid_confidence (LLM picks WHICH
facts, retrieval data decides HOW STRONGLY), so the comparison isolates the
model's reading ability, not its (miscalibrated) self-confidence.

Run (needs OPENAI_API_KEY):
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/llm_model_ladder.py
Writes outputs/llm_model_ladder.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import prognosis as pg  # noqa: E402
from llm_evidence import llm_parse_evidence, hybrid_confidence  # noqa: E402
from bn_upgraded import build_upgraded, INJ_NODE, DMG_NODE  # noqa: E402
from parser_flaws_and_paraphrases import FLAW_PROBES, PARAPHRASES  # noqa: E402

# default ladder; override from the command line, e.g.
#   python3.11 tests/llm_model_ladder.py claude-sonnet-4-5 claude-opus-4-6
MODELS = sys.argv[1:] or ["gpt-4o-mini", "gpt-4.1", "gpt-5", "gpt-5.5"]

OUT = ROOT / "outputs" / "llm_model_ladder.md"
L: list[str] = []


def emit(s: str = "") -> None:
    print(s)
    L.append(s)


def probe_safe(label: str, conf: dict) -> bool:
    ev = set(conf)
    non_person = {n for n in ev if not n.startswith("person: ")}
    if label in ("negation", "hypothetical", "out of vocabulary", "empty-ish",
                 "substring safety"):
        return not non_person or (label == "substring safety" and
                                  all(conf[n] < 0.999 for n in non_person))
    if label == "generic term":
        return ev == {"landing gear"}
    return True


def main():
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]

    emit("# LLM model ladder: safety and paraphrase recovery")
    emit()
    emit(f"Models: {', '.join(MODELS)}")
    emit()
    emit("| Model | Safety (6 probes) | Paraphrase recovery (7) | "
         "Hallucinated names dropped |")
    emit("|---|---|---|---|")

    detail: list[str] = []
    for model in MODELS:
        n_safe = n_rec = n_drop = 0
        detail.append(f"\n## {model}\n")
        try:
            for label, sentence in FLAW_PROBES:
                raw = llm_parse_evidence(sentence, names, model=model)
                conf = hybrid_confidence(sentence, raw, ds)
                ok = probe_safe(label, conf)
                n_safe += ok
                n_drop += len(raw["dropped"])
                codes = "; ".join(f"`{n}` ({c:.2f})"
                                  for n, c in conf.items()) or "(none)"
                detail.append(f"- probe {label}: {codes} "
                              f"{'OK' if ok else '<-- UNSAFE'}")
            for lab, sentence, intended, _t, _s in PARAPHRASES:
                raw = llm_parse_evidence(sentence, names, model=model)
                conf = hybrid_confidence(sentence, raw, ds)
                n_drop += len(raw["dropped"])
                got = intended in conf
                n_rec += got
                codes = "; ".join(f"`{n}` ({c:.2f})"
                                  for n, c in conf.items()) or "(none)"
                detail.append(f"- paraphrase {lab}: {codes} "
                              f"{'FOUND' if got else 'missed'}")
        except Exception as e:
            emit(f"| {model} | ERROR | ERROR | {type(e).__name__}: "
                 f"{str(e)[:80]} |")
            continue
        emit(f"| {model} | {n_safe}/6 | {n_rec}/7 | {n_drop} |")

    emit()
    emit("Keyword parser reference: safety 6/6 (after this week's guards), "
         "recovery 1/7.")
    L.extend(detail)
    OUT.write_text("\n".join(L) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

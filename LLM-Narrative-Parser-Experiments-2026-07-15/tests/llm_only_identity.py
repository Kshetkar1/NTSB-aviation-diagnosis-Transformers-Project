#!/usr/bin/env python3
"""Can the LLM ALONE (no deterministic pass) reproduce the 11 scoreboard
scenarios bit-identically? Run each sentence through llm_parse_evidence +
hybrid_confidence several times to also expose run-to-run variation.

Run (needs OPENAI_API_KEY):
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/llm_only_identity.py [model] [n_repeats]
Writes outputs/llm_only_identity.md.
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
from codes_side_by_side import SCEN  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-4.1"
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OUT = ROOT / "outputs" / "llm_only_identity.md"
L: list[str] = []


def emit(s: str = "") -> None:
    print(s)
    L.append(s)


def main():
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]

    emit(f"# LLM-only identity test -- {MODEL}, {REPEATS} repeats per sentence")
    emit()
    emit("Question: with NO deterministic pass, does the LLM land on exactly "
         "the clicked evidence for the 11 table scenarios, every time?")
    emit()
    emit("| Sentence | Run | Parsed (conf) | Exact+hard match |")
    emit("|---|---|---|---|")

    n_exact = n_stable = 0
    for sentence, clicked in SCEN:
        runs = []
        for r in range(REPEATS):
            try:
                raw = llm_parse_evidence(sentence, names, model=MODEL)
                conf = hybrid_confidence(sentence, raw, ds)
            except Exception as exc:
                conf = {"<API ERROR>": 0.0}
                print(f"  API error: {exc}")
            runs.append(conf)
            ok = (sorted(conf) == sorted(clicked)
                  and all(c >= 0.999 for c in conf.values()))
            n_exact += ok
            ev = "; ".join(f"`{n}` ({c:.2f})" for n, c in conf.items()) \
                 or "(none)"
            emit(f'| "{sentence[:60]}..." | {r + 1} | {ev} | '
                 f'{"YES" if ok else "NO"} |')
        stable = all(sorted(r) == sorted(runs[0]) for r in runs[1:])
        n_stable += stable
    emit()
    emit(f"Exact+hard matches: {n_exact}/{len(SCEN) * REPEATS} runs. "
         f"Stable across repeats: {n_stable}/{len(SCEN)} sentences. "
         f"(Deterministic tier: 11/11, always, at zero cost.)")

    OUT.write_text("\n".join(L) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

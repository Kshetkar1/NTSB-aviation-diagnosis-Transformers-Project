#!/usr/bin/env python3
"""Validate the TIERED combined parser (deterministic first, LLM fallback).

Three requirements, all must hold:

  IDENTITY   the 11 scoreboard sentences must still produce exactly the
             clicked evidence codes via tier 1 (the LLM must not be called).
  SAFETY     the 6 adversarial probes must stay safe.
  RECOVERY   the 7 paraphrases should now route to tier 2 and recover more
             intended nodes than the retrieval-only fallback did.

Run (needs OPENAI_API_KEY):
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/combined_parser_validation.py [model]
Writes outputs/combined_parser_validation.md.
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

import prognosis as pg  # noqa: E402
from llm_evidence import combined_parse  # noqa: E402
from bn_upgraded import build_upgraded, INJ_NODE, DMG_NODE  # noqa: E402
from codes_side_by_side import SCEN  # noqa: E402
from parser_flaws_and_paraphrases import FLAW_PROBES, PARAPHRASES  # noqa: E402
from llm_model_ladder import probe_safe  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-4.1"
OUT = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs" / "combined_parser_validation.md"
L: list[str] = []


def emit(s: str = "") -> None:
    print(s)
    L.append(s)


def main():
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]

    emit(f"# Combined (tiered) parser validation -- LLM fallback: {MODEL}")
    emit()

    # ---------------- identity ------------------------------------------------
    n_id = n_t1 = 0
    for sentence, clicked in SCEN:
        p = combined_parse(sentence, names, ds, model=MODEL)
        same = sorted(p["evidence"]) == sorted(clicked) and all(
            p["confidence"][e] >= 0.999 for e in p["evidence"])
        n_id += same
        n_t1 += p["tier"] == 1
    emit(f"IDENTITY: {n_id}/11 scoreboard sentences -> exact clicked codes; "
         f"{n_t1}/11 stayed in tier 1 (LLM never called).")
    emit()

    # ---------------- safety --------------------------------------------------
    emit("| Probe | Tier | Parsed codes (conf) | Safe |")
    emit("|---|---|---|---|")
    n_safe = 0
    for label, sentence in FLAW_PROBES:
        p = combined_parse(sentence, names, ds, model=MODEL)
        ok = probe_safe(label, p["confidence"])
        n_safe += ok
        codes = "; ".join(f"`{n}` ({c:.2f})"
                          for n, c in p["confidence"].items()) or "(none)"
        emit(f"| {label} | {p['tier']} | {codes} | "
             f"{'yes' if ok else 'NO'} |")
    emit()
    emit(f"SAFETY: {n_safe}/6.")
    emit()

    # ---------------- recovery ------------------------------------------------
    emit("| Paraphrase | Tier | Parsed codes (conf) | Intended found |")
    emit("|---|---|---|---|")
    n_rec = 0
    for lab, sentence, intended, _t, _s in PARAPHRASES:
        p = combined_parse(sentence, names, ds, model=MODEL)
        got = intended in p["confidence"]
        n_rec += got
        codes = "; ".join(f"`{n}` ({c:.2f})"
                          for n, c in p["confidence"].items()) or "(none)"
        emit(f"| {lab} | {p['tier']} | {codes} | "
             f"{'YES' if got else 'no'} |")
    emit()
    emit(f"RECOVERY: {n_rec}/7 (retrieval-only fallback was 1/7).")

    OUT.write_text("\n".join(L) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""WHERE does paraphrase recovery die: the LLM's reading, or the data
grounding (hybrid_confidence drops facts with f_q < 0.05 among similar
accidents)?

For the four stubborn paraphrases, print the RAW LLM selection (with the
paraphrase-mapping rule and the retrieval+name-embedding shortlist) and the
measured f_q of the intended node, before and after the floor filter.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import prognosis as pg  # noqa: E402
import llm_evidence as le  # noqa: E402
from llm_evidence import llm_parse_evidence, measured_fq  # noqa: E402
from bn_upgraded import build_upgraded, INJ_NODE, DMG_NODE  # noqa: E402
from llm_paraphrase_upgrade import shortlist, name_embeddings, _PARA_RULE  # noqa: E402

CASES = [
    ("engine instrument",
     "the cockpit gauges monitoring the engine were giving bad readings",
     "engine instrument"),
    ("combustion liner",
     "a component inside the burner section of the engine cracked",
     "combustion assembly, combustion liner"),
    ("oil grade",
     "the mechanic had serviced the engine with the wrong type of oil",
     "fluid, oil grade"),
    ("unstabilized approach",
     "the airplane came in too high and too fast to land safely",
     "unstabilized approach"),
]


def main():
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]
    embs = name_embeddings(names)

    old = le._SYSTEM
    le._SYSTEM = old.replace("2. Mark each selected node:",
                             _PARA_RULE + "2. Mark each selected node:")
    try:
        for model in ["gpt-4.1", "claude-sonnet-5"]:
            print(f"\n===== {model} =====")
            for label, sentence, intended in CASES:
                cand = shortlist(sentence, names, ds, embs)
                in_short = intended in cand
                raw = llm_parse_evidence(sentence, cand, model=model)
                picked = intended in raw["evidence"]
                fq = measured_fq(sentence, [intended], ds)[intended]
                print(f"[{label}]")
                print(f"  intended node in shortlist: {in_short}")
                print(f"  LLM picked intended (raw):  {picked}   "
                      f"raw selection: {raw['evidence']}")
                print(f"  measured f_q of intended:   {fq:.4f}  "
                      f"({'SURVIVES' if fq >= 0.05 else 'DROPPED by 0.05 floor'})")
    finally:
        le._SYSTEM = old
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

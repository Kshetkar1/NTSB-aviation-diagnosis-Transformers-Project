#!/usr/bin/env python3
"""Held-out severity prediction with the LLM parser in the loop.

Same protocol as tests/heldout_narrative_bn_eval.py (network + retrieval built
on 1982-2006 only; held-out = 2007-2019 accidents with a factual narrative),
on a random-but-fixed subset (cost control: one LLM call per accident).

Predictors compared on P(injury), P(damage):
    PRIOR   network, no evidence
    DET     deterministic parse + retrieval soft facts (current pipeline)
    LLM     llm_evidence.llm_parse_evidence -> hard/soft evidence

Scores: multiclass Brier (lower better), top-1 accuracy (higher better).

Run (needs OPENAI_API_KEY):
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/llm_heldout_eval.py [--limit N]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

import prognosis as pg  # noqa: E402
import query_to_bn as qb  # noqa: E402
from llm_evidence import llm_parse_evidence, hybrid_confidence  # noqa: E402
from heldout_narrative_bn_eval import (truth_states, posteriors,  # noqa: E402
                                       brier)
from bn_upgraded import build_upgraded, INJ_NODE, DMG_NODE  # noqa: E402

LIMIT = 30


def main():
    limit = LIMIT
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    full = json.loads((ROOT / "data" / "processed" /
                       "refined_dataset.json").read_text())
    window_ids = set(json.loads((ROOT / "data" / "processed" /
                                 "refined_dataset_1982_2006.json").read_text()).keys())
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]

    heldout = []
    for ev, inc in full.items():
        if ev in window_ids:
            continue
        narr = str(inc.get("narr_accf") or "").strip()
        if len(narr) >= 200:
            heldout.append((ev, inc, narr))
    heldout.sort(key=lambda t: t[0])
    random.Random(7).shuffle(heldout)
    heldout = heldout[:limit]
    print(f"held-out subset: {len(heldout)} accidents (seed 7)")

    prior_inj, prior_dmg = posteriors(bn, {})
    S = {k: {"bri": [], "brd": [], "acci": 0, "accd": 0, "n_ev": []}
         for k in ("prior", "det", "llm", "hyb", "sup")}
    n_scored = 0
    for ev, inc, narr in heldout:
        inj_t, dmg_t = truth_states(inc)
        det = qb.parse_query_to_bn_evidence(narr, names, dataset=ds,
                                            semantic=True)
        try:
            llm = llm_parse_evidence(narr[:2400], names)
        except Exception as e:
            print(f"  {ev}: LLM call failed ({e}); skipping")
            continue
        hyb = hybrid_confidence(narr, llm, ds)
        # supplement: det parse is the base; LLM may only ADD data-calibrated
        # facts the parser missed (never override or remove det facts)
        sup = dict(hyb)
        sup.update(det["confidence"])
        n_scored += 1
        for key, conf in (("prior", {}), ("det", det["confidence"]),
                          ("llm", llm["confidence"]), ("hyb", hyb),
                          ("sup", sup)):
            pi, pd = (prior_inj, prior_dmg) if not conf else posteriors(bn, conf)
            S[key]["bri"].append(brier(pi, inj_t))
            S[key]["acci"] += int(np.argmax(pi) == inj_t)
            if dmg_t is not None:
                S[key]["brd"].append(brier(pd, dmg_t))
                S[key]["accd"] += int(np.argmax(pd) == dmg_t)
            S[key]["n_ev"].append(len(conf))

    nd = len(S["prior"]["brd"])
    print(f"\nscored: {n_scored} accidents (damage scorable: {nd})")
    print(f"{'predictor':10} {'inj Brier':>10} {'inj acc':>8} "
          f"{'dmg Brier':>10} {'dmg acc':>8} {'mean #ev':>9}")
    for key in ("prior", "det", "llm", "hyb", "sup"):
        s = S[key]
        print(f"{key:10} {np.mean(s['bri']):10.3f} "
              f"{s['acci'] / n_scored:8.0%} "
              f"{np.mean(s['brd']) if nd else float('nan'):10.3f} "
              f"{(s['accd'] / nd) if nd else float('nan'):8.0%} "
              f"{np.mean(s['n_ev']):9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

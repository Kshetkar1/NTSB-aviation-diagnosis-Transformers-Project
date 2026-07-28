#!/usr/bin/env python3
"""ONE REAL ACCIDENT, walked through the whole pipeline.

Picks two held-out accidents (2007+, never used to build anything):
  * one whose narrative NAMES facts  -> handled by tier 1 (deterministic)
  * one whose narrative DESCRIBES facts -> handled by tier 2 (LLM)
and shows, for each: the narrative text, which tier fired, the evidence
extracted (with strengths and why), the network's severity posteriors,
and the outcome that actually happened.

Run (needs OPENAI_API_KEY):
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/real_narrative_walkthrough.py
Writes outputs/real_narrative_walkthrough.md.
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

import numpy as np  # noqa: E402
import pyagrum as gum  # noqa: E402

import prognosis as pg  # noqa: E402
import query_to_bn as qb  # noqa: E402
from llm_evidence import combined_parse  # noqa: E402
from bn_upgraded import (build_upgraded, INJ_NODE, DMG_NODE,  # noqa: E402
                         INJ_STATES, DMG_STATES, DMG_BY_CODE)

FULL = ROOT / "shared" / "data" / "processed" / "refined_dataset.json"
WINDOW = ROOT / "shared" / "data" / "processed" / "refined_dataset_1982_2006.json"
OUT = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs" / "real_narrative_walkthrough.md"
L: list[str] = []


def emit(s: str = "") -> None:
    print(s)
    L.append(s)


def sev_posteriors(bn, conf, sev=None):
    ie = gum.LazyPropagation(bn)
    if conf:
        qb.apply_evidence(ie, bn, conf)
    if sev:
        if "injury" in sev:
            ie.addEvidence(INJ_NODE, list(sev["injury"]))
        if "damage" in sev:
            ie.addEvidence(DMG_NODE, list(sev["damage"]))
    ie.addTarget(INJ_NODE)
    ie.addTarget(DMG_NODE)
    ie.makeInference()

    def dist(node, states):
        v = bn.variable(node)
        post = ie.posterior(node)
        by = {v.label(i): float(post[i]) for i in range(v.domainSize())}
        return np.array([by[s] for s in states])
    return dist(INJ_NODE, INJ_STATES), dist(DMG_NODE, DMG_STATES)


def truth(inc):
    inj = pg.zhang_injury_code(inc)
    inj_i = {"FATL": 0, "SERS": 1, "MINR": 2, "NONE": 3}[inj]
    dmg_code = str(inc.get("damage") or "").upper()
    dmg_i = DMG_BY_CODE.get(dmg_code)
    return inj_i, dmg_i


def show(bn, ds, key, inc, narr, p):
    import main_app  # noqa: F401  (retrieval index)
    text = narr[:4000]
    inj_y, dmg_y = truth(inc)

    emit(f"## Accident `{key}` (held out -- never used in any build)")
    emit()
    short = narr[:900] + ("..." if len(narr) > 900 else "")
    emit(f"> {short}")
    emit()
    tier_name = {1: "tier 1, deterministic (facts NAMED in text)",
                 2: "tier 2, LLM (facts DESCRIBED, not named)",
                 3: "tier 3, retrieval fallback"}[p["tier"]]
    emit(f"**Which reader handled it:** {tier_name}")
    emit()
    emit("**Evidence extracted:**")
    emit()
    emit("| Node | Strength | Why |")
    emit("|---|---|---|")
    why = {n: r for n, r in p["trace"]}
    for n in sorted(p["confidence"], key=lambda x: -p["confidence"][x]):
        emit(f"| `{n}` | {p['confidence'][n]:.2f} | "
             f"{why.get(n, '')} |")
    emit()
    sev = qb.severity_virtual_evidence(text, pg.load_dataset())
    stated = ", ".join(sorted(sev)) if sev else "none"
    emit(f"**Severity stated outright in the narrative:** {stated}")
    emit()

    pi0, pd0 = sev_posteriors(bn, {})
    pi, pd = sev_posteriors(bn, p["confidence"], sev=sev or None)
    emit("**Network severity posteriors (prior -> with this narrative):**")
    emit()
    emit("| State | Prior | With narrative |")
    emit("|---|---|---|")
    for i, s in enumerate(INJ_STATES):
        mark = "  <-- ACTUAL" if i == inj_y else ""
        emit(f"| {s} | {pi0[i]:.4f} | {pi[i]:.4f}{mark} |")
    for i, s in enumerate(DMG_STATES):
        mark = "  <-- ACTUAL" if dmg_y is not None and i == dmg_y else ""
        emit(f"| {s} | {pd0[i]:.4f} | {pd[i]:.4f}{mark} |")
    emit()
    ok_i = int(pi.argmax()) == inj_y
    ok_d = dmg_y is not None and int(pd.argmax()) == dmg_y
    emit(f"**Top prediction correct?** injury: {'YES' if ok_i else 'no'}, "
         f"damage: {'YES' if ok_d else 'no'}")
    emit()


def main():
    full = json.loads(FULL.read_text())
    window_ids = set(json.loads(WINDOW.read_text()).keys())
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]
    import main_app  # noqa: F401

    held = []
    for k, inc in sorted(full.items()):
        if k in window_ids:
            continue
        narr = str(inc.get("narr_accf") or "").strip()
        if 300 < len(narr) < 1600:      # short enough to show on a page
            held.append((k, inc, narr))

    emit("# One real accident through the whole pipeline")
    emit()
    emit("Both accidents are from the held-out years (2007+). Nothing about "
         "them was used to build the network, the clusters, or the "
         "retrieval index.")
    emit()

    got1 = got2 = None
    for k, inc, narr in held:
        p = combined_parse(narr[:4000], names, ds)
        if p["tier"] == 1 and got1 is None and len(p["evidence"]) >= 2:
            got1 = (k, inc, narr, p)
        if p["tier"] == 2 and got2 is None and len(p["evidence"]) >= 2:
            got2 = (k, inc, narr, p)
        if got1 and got2:
            break

    for g in (got1, got2):
        if g:
            show(bn, ds, *g)

    OUT.write_text("\n".join(L) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

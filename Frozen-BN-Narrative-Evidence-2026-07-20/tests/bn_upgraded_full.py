#!/usr/bin/env python3
"""Re-score EVERY cell that DIFFERS in the reproduction scoreboard against the
UPGRADED network (person findings + multi-state severity nodes), plus the four
previously-not-replicable Fig 12 pilot queries, plus spot-checks that the
previously-EXACT anchors did not regress.

Output: per cell -> Zhang published | old (faithful reproduction) | upgraded,
with old/new status at the same thresholds the full comparison used
(EXACT < 2%, CLOSE < 25%, DIFFERS otherwise).

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 tests/bn_upgraded_full.py
Writes outputs/bn_upgraded_full.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyagrum as gum

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
import bn_build_ours as builder  # noqa: E402
from bn_upgraded import DMG_NODE, INJ_NODE, build_upgraded  # noqa: E402

OUT = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs" / "bn_upgraded_full.json"


def post(bn, target, evidence, state="Yes"):
    ie = gum.LazyPropagation(bn)
    if evidence:
        ie.setEvidence(evidence)
    ie.addTarget(target)
    ie.makeInference()
    v = bn.variable(target)
    i = [k for k in range(v.domainSize()) if v.label(k) == state][0]
    return float(ie.posterior(target)[i])


def status(zhang, val):
    if val is None:
        return "n/a"
    rel = abs(val - zhang) / zhang if zhang else float("inf")
    return "EXACT" if rel < 0.02 else "CLOSE" if rel < 0.25 else "DIFFERS"


def main():
    ds = pg.load_dataset()
    bn, meta = build_upgraded(ds)
    print(f"UPGRADED network: {bn.size()} nodes, {bn.sizeArcs()} arcs\n")

    # evidence nodes (same selections as the full comparison)
    loep = "loss of engine power"
    eng = "engine instrument"
    comb = builder.find_node(bn, "combustion", "liner")
    oil = builder.find_node(bn, "fluid, oil grade") or builder.find_node(bn, "oil grade")
    EV = {
        "eng instr": {eng: "Yes"},
        "comb liner": {comb: "Yes"},
        "oil": {oil: "Yes"},
        "eng+oil": {eng: "Yes", oil: "Yes"},
        "LOEP": {loep: "Yes"},
    }

    def exact_node(name):
        """Exact-name lookup; fall back to substring only if no exact match."""
        return name if name in bn.names() else builder.find_node(bn, name)

    gearc = exact_node("gear collapsed")
    dragged = exact_node("dragged wing, rotor, pod, float or tail/skid")
    unstab = exact_node("unstabilized approach")
    maingearc = exact_node("main gear collapsed")
    pilot = "person: pilot-in-command"

    # (label, kind, node, state, evidence, zhang, old)
    # kind: sev = multi-state severity node; occ = Boolean occurrence node
    CELLS = [
        # --- Table 9 gear collapsed (occurrence node; severity fix shouldn't move) --
        ("T9 gear collapsed | eng instr", gearc, "Yes", EV["eng instr"], 0.0096, 0.004378),
        ("T9 gear collapsed | eng+oil", gearc, "Yes", EV["eng+oil"], 0.00982, 0.004606),
        # --- Table 9 destroyed ------------------------------------------------------
        ("T9 destroyed | eng instr", DMG_NODE, "destroyed aircraft", EV["eng instr"], 0.0133, 0.004378),
        ("T9 destroyed | eng+oil", DMG_NODE, "destroyed aircraft", EV["eng+oil"], 0.0135, 0.004606),
        ("T9 destroyed | LOEP", DMG_NODE, "destroyed aircraft", EV["LOEP"], 0.00559, 0.02564),
        # --- Table 9 substantial ----------------------------------------------------
        ("T9 substantial | eng instr", DMG_NODE, "substantial damage", EV["eng instr"], 0.046, 0.006907),
        ("T9 substantial | eng+oil", DMG_NODE, "substantial damage", EV["eng+oil"], 0.0463, 0.007266),
        ("T9 substantial | LOEP", DMG_NODE, "substantial damage", EV["LOEP"], 0.0166, 0.007271),
        # --- Table 9 minor damage ---------------------------------------------------
        ("T9 minor dmg | eng instr", DMG_NODE, "minor damage", EV["eng instr"], 0.00934, 0.477),
        ("T9 minor dmg | comb liner", DMG_NODE, "minor damage", EV["comb liner"], 0.00154, 0.2511),
        ("T9 minor dmg | oil", DMG_NODE, "minor damage", EV["oil"], 0.00292, 0.477),
        ("T9 minor dmg | eng+oil", DMG_NODE, "minor damage", EV["eng+oil"], 0.00947, 0.5018),
        ("T9 minor dmg | LOEP", DMG_NODE, "minor damage", EV["LOEP"], 0.00378, 0.5021),
        # --- Table 9 serious injury -------------------------------------------------
        ("T9 serious inj | eng instr", INJ_NODE, "serious injury", EV["eng instr"], 0.0623, 0.00146),
        ("T9 serious inj | eng+oil", INJ_NODE, "serious injury", EV["eng+oil"], 0.0623, 0.001536),
        ("T9 serious inj | LOEP", INJ_NODE, "serious injury", EV["LOEP"], 0.00822, 0.005315),
        # --- Table 9 no injury ------------------------------------------------------
        ("T9 no injury | eng instr", INJ_NODE, "no injury", EV["eng instr"], 0.9431, 0.007656),
        ("T9 no injury | comb liner", INJ_NODE, "no injury", EV["comb liner"], 0.9978, 0.00403),
        ("T9 no injury | oil", INJ_NODE, "no injury", EV["oil"], 0.9958, 0.007656),
        ("T9 no injury | eng+oil", INJ_NODE, "no injury", EV["eng+oil"], 0.9429, 0.008054),
        ("T9 no injury | LOEP", INJ_NODE, "no injury", EV["LOEP"], 0.9899, 0.02173),
        # --- Table 8 base-prior row (occurrence nodes; jitter cell) ------------------
        ("T8 main gear collapsed @ base prior", maingearc, "Yes", None, 1.21e-7, 1.843e-7),
        # --- Fig 12 priors ------------------------------------------------------------
        ("Fig12 prior dragged wing", dragged, "Yes", None, 1.14e-7, 1.442e-7),
        ("Fig12 prior substantial dmg", DMG_NODE, "substantial damage", None, 2.22e-7, 5.559e-7),
        ("Fig12 prior no injury", INJ_NODE, "no injury", None, 0.9999, 6.3e-7),
        # --- Fig 12 stage 2 ------------------------------------------------------------
        ("Fig12 no injury | unstab approach", INJ_NODE, "no injury", {unstab: "Yes"}, 0.613, 0.3657),
    ]

    PILOT = [
        ("Fig12 unstab approach | pilot", unstab, "Yes", {pilot: "Yes"}, 0.00484),
        ("Fig12 dragged wing | pilot", dragged, "Yes", {pilot: "Yes"}, 0.023),
        ("Fig12 substantial dmg | pilot", DMG_NODE, "substantial damage", {pilot: "Yes"}, 0.0458),
        ("Fig12 no injury | pilot", INJ_NODE, "no injury", {pilot: "Yes"}, 0.97),
    ]

    ANCHOR_CHECKS = [
        ("T9 LOEP | eng instr", loep, "Yes", EV["eng instr"], 0.95),
        ("T9 LOEP | comb liner", loep, "Yes", EV["comb liner"], 0.50),
        ("T9 forced landing | LOEP", builder.find_node(bn, "forced landing"), "Yes", EV["LOEP"], 0.1429),
        ("T9 ditching | LOEP", builder.find_node(bn, "ditching"), "Yes", EV["LOEP"], 0.00461),
        ("T9 gear collapsed | oil", gearc, "Yes", EV["oil"], 0.00437),
    ]

    results = {"differs_rescore": [], "pilot": [], "anchors": []}

    print(f"{'cell':38} {'Zhang':>10} {'old':>10} {'upgraded':>10}  old -> new")
    n_resolved = n_improved = 0
    for label, node, state, ev, zhang, old in CELLS:
        if node is None:
            print(f"{label:38} node missing")
            continue
        val = post(bn, node, ev, state)
        s_old, s_new = status(zhang, old), status(zhang, val)
        gap_old = abs(old - zhang) / zhang
        gap_new = abs(val - zhang) / zhang
        if s_new in ("EXACT", "CLOSE"):
            n_resolved += 1
            tag = "RESOLVED"
        elif gap_new < 0.5 * gap_old:
            n_improved += 1
            tag = "improved"
        else:
            tag = "unchanged"
        results["differs_rescore"].append(
            {"cell": label, "zhang": zhang, "old": old, "upgraded": val,
             "old_status": s_old, "new_status": s_new, "tag": tag})
        print(f"{label:38} {zhang:10.4g} {old:10.4g} {val:10.4g}  "
              f"{s_old} -> {s_new}  [{tag}]")

    print(f"\n>>> of {len(CELLS)} previously-DIFFERS cells: "
          f"{n_resolved} now EXACT/CLOSE, {n_improved} improved >2x, "
          f"{len(CELLS) - n_resolved - n_improved} unchanged")

    print(f"\npreviously NOT REPLICABLE (pilot queries):")
    n_pilot_close = 0
    for label, node, state, ev, zhang in PILOT:
        val = post(bn, node, ev, state)
        s = status(zhang, val)
        n_pilot_close += s in ("EXACT", "CLOSE")
        results["pilot"].append({"cell": label, "zhang": zhang, "upgraded": val,
                                 "status": s})
        print(f"{label:38} {zhang:10.4g} {'':>10} {val:10.4g}  -> {s}")
    print(f">>> {n_pilot_close}/4 pilot queries now EXACT/CLOSE")

    print(f"\nregression spot-checks (must stay EXACT):")
    ok = True
    for label, node, state, ev, zhang in ANCHOR_CHECKS:
        val = post(bn, node, ev, state)
        s = status(zhang, val)
        ok = ok and s == "EXACT"
        results["anchors"].append({"cell": label, "zhang": zhang, "upgraded": val,
                                   "status": s})
        print(f"{label:38} {zhang:10.4g} {'':>10} {val:10.4g}  -> {s}")
    print(f">>> anchors {'ALL STILL EXACT' if ok else 'REGRESSION DETECTED'}")

    results["summary"] = {
        "differs_total": len(CELLS), "resolved": n_resolved,
        "improved": n_improved, "pilot_close": n_pilot_close,
        "anchors_ok": ok,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

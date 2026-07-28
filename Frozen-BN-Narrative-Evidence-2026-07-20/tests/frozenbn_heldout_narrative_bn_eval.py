#!/usr/bin/env python3
"""HELD-OUT evaluation of the narrative -> BN chain.

Question: does feeding the accident NARRATIVE into the network improve
out-of-sample prediction of severity (injury + damage), compared to the
network alone?

Protocol (no leakage):
  * Network + retrieval index are built on the 1982-2006 window (1,742
    accidents) -- exactly the artifact validated against Zhang's tables.
  * Held-out set: accidents in the full refined corpus but NOT in the window
    (2007-2019), keeping those with a factual narrative (narr_accf) and
    scorable outcomes. The factual narrative describes what happened, the
    same information an investigator would have.
  * For each held-out accident, four predictors of P(injury), P(damage):
      PRIOR      network, no evidence  (same for every accident)
      SOFT-ONLY  retrieval soft facts alone (no deterministic parse) --
                 isolates what the embedding layer contributes by itself
      HARD       deterministic parse of the narrative -> hard evidence
      HARD+SOFT  adds retrieval soft facts (Jeffrey conditioning) when the
                 narrative's event content is otherwise ungrounded
  * Truth: injury = Zhang's per-person derivation; damage = NTSB code.
  * Scores: multiclass Brier (lower better), log-loss (lower better),
    top-1 accuracy (higher better).

Run (needs OPENAI_API_KEY for the soft pass):
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/heldout_narrative_bn_eval.py [--limit N]
"""
from __future__ import annotations

import json
import math
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
from bn_upgraded import (build_upgraded, INJ_NODE, DMG_NODE,  # noqa: E402
                         INJ_STATES, DMG_STATES, DMG_BY_CODE)

FULL = ROOT / "shared" / "data" / "processed" / "refined_dataset.json"
WINDOW = ROOT / "shared" / "data" / "processed" / "refined_dataset_1982_2006.json"
OUT = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs" / "heldout_narrative_bn_eval.json"

INJ_IDX = {s: i for i, s in enumerate(INJ_STATES)}


def truth_states(inc):
    """(injury state idx, damage state idx or None) ground truth."""
    inj = pg.zhang_injury_code(inc)  # FATL/SERS/MINR/NONE
    inj_i = {"FATL": 0, "SERS": 1, "MINR": 2, "NONE": 3}[inj]
    dmg_code = str(inc.get("damage") or "").upper()
    dmg_i = DMG_BY_CODE.get(dmg_code) if dmg_code in DMG_BY_CODE else None
    return inj_i, dmg_i


def posteriors(bn, confidence, sev=None):
    """sev: optional {'injury': [4 likelihoods], 'damage': [4]} virtual evidence
    on the multi-state severity nodes (calibrated from the training window)."""
    ie = gum.LazyPropagation(bn)
    if confidence:
        qb.apply_evidence(ie, bn, confidence)
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


def brier(p, y):
    t = np.zeros_like(p)
    t[y] = 1.0
    return float(((p - t) ** 2).sum())


def logloss(p, y):
    return float(-math.log(max(p[y], 1e-12)))


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    # --llm [model]: add the tiered combined parser (deterministic first,
    # LLM fallback with data-grounded strengths) as predictors
    llm_model = None
    if "--llm" in sys.argv:
        i = sys.argv.index("--llm")
        llm_model = (sys.argv[i + 1] if i + 1 < len(sys.argv)
                     and not sys.argv[i + 1].startswith("--") else "gpt-4.1")

    full = json.loads(FULL.read_text())
    window_ids = set(json.loads(WINDOW.read_text()).keys())
    ds = pg.load_dataset()          # window dataset (network + retrieval pool)
    bn, _ = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]
    import main_app                 # loads window retrieval index

    held = []
    for k, inc in full.items():
        if k in window_ids:
            continue
        narr = str(inc.get("narr_accf") or "").strip()
        if len(narr) < 100:
            continue
        held.append((k, inc, narr))
    held.sort(key=lambda t: t[0])
    if limit:
        held = held[:limit]
    print(f"held-out accidents with narratives: {len(held)} "
          f"(window {len(window_ids)}, full corpus {len(full)})")

    prior_inj, prior_dmg = posteriors(bn, {})
    predictors = ["prior", "soft-only", "hard", "hard+soft", "full",
                  "soft+stated", "narr-sev"]
    if llm_model:
        predictors += ["llm-tier", "llm-tier+stated", "llm-first",
                       "llm-first+stated"]
        print(f"LLM tiered predictor enabled: {llm_model}")
        tier_counts = {1: 0, 2: 0, 3: 0}
        n_llm_first_fail = 0
    scores = {p: {"inj_brier": [], "inj_ll": [], "inj_acc": [],
                  "dmg_brier": [], "dmg_ll": [], "dmg_acc": []}
              for p in predictors}
    n_hard_ev, n_soft_ev = [], []
    n_sev_dmg = n_sev_inj = 0
    per_item = []                     # per-accident record for paired tests

    for idx, (k, inc, narr) in enumerate(held):
        text = narr[:4000]
        inj_y, dmg_y = truth_states(inc)
        try:
            hard = qb.parse_query_to_bn_evidence(text, names, dataset=ds,
                                                 semantic=False)
            soft = qb.parse_query_to_bn_evidence(
                text, names, dataset=main_app.refined_dataset, semantic=True)
            softonly = {lab: c for lab, c, _ in qb.retrieval_facts(
                text, names, main_app.refined_dataset)}
        except Exception as exc:
            print(f"  parse failed for {k}: {exc}")
            continue
        n_hard_ev.append(len(hard["evidence"]))
        n_soft_ev.append(len(softonly))

        preds = {"prior": (prior_inj, prior_dmg)}
        try:
            preds["soft-only"] = (posteriors(bn, softonly) if softonly
                                  else (prior_inj, prior_dmg))
        except Exception:
            preds["soft-only"] = (prior_inj, prior_dmg)
        try:
            preds["hard"] = posteriors(bn, hard["confidence"])
        except Exception:
            preds["hard"] = (prior_inj, prior_dmg)
        try:
            preds["hard+soft"] = posteriors(bn, soft["confidence"])
        except Exception:
            preds["hard+soft"] = preds["hard"]
        # FULL: hard+soft evidence PLUS what the narrative STATES about severity,
        # entered as virtual evidence calibrated on the training window only
        sev = qb.severity_virtual_evidence(text, ds)
        if "damage" in sev:
            n_sev_dmg += 1
        if "injury" in sev:
            n_sev_inj += 1
        try:
            preds["full"] = (posteriors(bn, soft["confidence"], sev=sev)
                             if sev else preds["hard+soft"])
        except Exception:
            preds["full"] = preds["hard+soft"]
        # the best-injury layer (retrieval soft facts) + stated severity
        try:
            preds["soft+stated"] = (posteriors(bn, softonly, sev=sev)
                                    if (softonly or sev)
                                    else (prior_inj, prior_dmg))
        except Exception:
            preds["soft+stated"] = preds["soft-only"]
        # NARRATIVE-SEVERITY predictor: retrieval severity distribution
        # (similarity-weighted injury/damage frequencies among the 100 most
        # similar accidents), sharpened by the stated-severity confusion
        # likelihood when the narrative says the level outright. This is the
        # narrative-side severity readout; no event evidence needed.
        # LLM-TIER: the combined parser (deterministic first, LLM fallback
        # for ungrounded narratives, strengths always from the data)
        if llm_model:
            try:
                from llm_evidence import combined_parse
                cp = combined_parse(text, names, main_app.refined_dataset,
                                    model=llm_model)
                tier_counts[cp["tier"]] += 1
                preds["llm-tier"] = posteriors(bn, cp["confidence"])
                preds["llm-tier+stated"] = (
                    posteriors(bn, cp["confidence"], sev=sev) if sev
                    else preds["llm-tier"])
            except Exception:
                preds["llm-tier"] = preds["hard+soft"]
                preds["llm-tier+stated"] = preds["full"]
            # LLM-FIRST: the LLM is the ONLY front door -- no deterministic
            # pass, no retrieval suggestions. Facts from the LLM, strengths
            # still measured from the data (hybrid grounding).
            try:
                from llm_evidence import llm_parse_evidence, hybrid_confidence
                raw = llm_parse_evidence(text, names, model=llm_model)
                lf = hybrid_confidence(text, raw, main_app.refined_dataset)
                preds["llm-first"] = (posteriors(bn, lf) if lf
                                      else (prior_inj, prior_dmg))
                preds["llm-first+stated"] = (posteriors(bn, lf, sev=sev)
                                             if sev else preds["llm-first"])
            except Exception:
                n_llm_first_fail += 1
                preds["llm-first"] = (prior_inj, prior_dmg)
                preds["llm-first+stated"] = preds["llm-first"]

        try:
            rdist = qb.severity_retrieval_distributions(text, ds)
            if rdist:
                ni = np.array(rdist["injury"])
                nd = np.array(rdist["damage"])
                if "injury" in sev:
                    v = ni * np.array(sev["injury"])
                    ni = v / v.sum()
                if "damage" in sev:
                    v = nd * np.array(sev["damage"])
                    nd = v / v.sum()
                preds["narr-sev"] = (ni, nd)
            else:
                preds["narr-sev"] = preds["soft+stated"]
        except Exception:
            preds["narr-sev"] = preds["soft+stated"]

        rec = {"id": k, "inj_true": inj_y, "dmg_true": dmg_y}
        for p in predictors:
            pi, pdm = preds[p]
            scores[p]["inj_brier"].append(brier(pi, inj_y))
            scores[p]["inj_ll"].append(logloss(pi, inj_y))
            scores[p]["inj_acc"].append(float(int(pi.argmax()) == inj_y))
            rec[f"{p}:inj_pred"] = int(pi.argmax())
            rec[f"{p}:inj_brier"] = brier(pi, inj_y)
            if dmg_y is not None:
                scores[p]["dmg_brier"].append(brier(pdm, dmg_y))
                scores[p]["dmg_ll"].append(logloss(pdm, dmg_y))
                scores[p]["dmg_acc"].append(float(int(pdm.argmax()) == dmg_y))
                rec[f"{p}:dmg_pred"] = int(pdm.argmax())
                rec[f"{p}:dmg_brier"] = brier(pdm, dmg_y)
        per_item.append(rec)
        if (idx + 1) % 25 == 0:
            print(f"  ... {idx + 1}/{len(held)}")

    print(f"\nevidence per narrative: hard mean {np.mean(n_hard_ev):.1f}, "
          f"retrieval soft facts mean {np.mean(n_soft_ev):.1f}")
    if llm_model:
        print(f"tier usage: tier1(deterministic) {tier_counts[1]}, "
              f"tier2(LLM) {tier_counts[2]}, tier3(fallback) {tier_counts[3]}")
        print(f"llm-first API failures (fell back to prior): "
              f"{n_llm_first_fail}")
    print(f"severity stated in narrative: damage {n_sev_dmg}, injury {n_sev_inj} "
          f"of {len(held)}")
    print(f"scored: injury n={len(scores['prior']['inj_brier'])}, "
          f"damage n={len(scores['prior']['dmg_brier'])}\n")

    header = f"{'predictor':12} {'inj Brier':>10} {'inj logloss':>12} {'inj acc':>8} {'dmg Brier':>10} {'dmg logloss':>12} {'dmg acc':>8}"
    print(header)
    print("-" * len(header))
    summary = {}
    for p in predictors:
        s = scores[p]
        row = {m: float(np.mean(v)) for m, v in s.items() if v}
        summary[p] = row
        print(f"{p:12} {row['inj_brier']:10.4f} {row['inj_ll']:12.4f} "
              f"{row['inj_acc']:8.3f} {row['dmg_brier']:10.4f} "
              f"{row['dmg_ll']:12.4f} {row['dmg_acc']:8.3f}")

    OUT.write_text(json.dumps({
        "n_heldout": len(held),
        "evidence_per_narrative": {"hard_mean": float(np.mean(n_hard_ev)),
                                   "soft_added_mean": float(np.mean(n_soft_ev))},
        "severity_stated": {"damage": n_sev_dmg, "injury": n_sev_inj},
        "summary": summary,
    }, indent=2))
    print(f"\nwrote {OUT}")
    per_out = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs" / "heldout_per_item.json"
    per_out.write_text(json.dumps({"predictors": predictors,
                                   "items": per_item}, indent=1))
    print(f"wrote {per_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

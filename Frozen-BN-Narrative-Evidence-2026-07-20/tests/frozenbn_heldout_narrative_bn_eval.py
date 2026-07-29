#!/usr/bin/env python3
"""HELD-OUT evaluation of the narrative -> BN chain.

Question: does feeding the accident NARRATIVE into the network improve
out-of-sample prediction of severity (injury + damage), compared to the
network alone?

Protocol (no train/test leakage):
  * Network + retrieval index are built on the 1982-2006 window (1,742
    accidents) -- exactly the artifact validated against Zhang's tables.
  * Held-out set: accidents in the full refined corpus but NOT in the window
    (2007-2019), keeping those with a factual narrative (narr_accf) and
    scorable outcomes.

Predictors (leak-safe by default):
  * BN-SEV (PRIMARY)   -- k-NN severity distributions entered as VIRTUAL
    EVIDENCE (Jeffrey conditioning) on the frozen BN's severity nodes,
    per-target inference. Posterior ~= k-NN readout by construction
    (self-test asserts this), so the BN mediates the narrative signal.
  * RETRIEVAL-SEV      -- raw k-NN severity readout (ablation: no BN)
  * BN-FUSED / BN-FUSED-T -- event evidence + k-NN severity in ONE BN
    inference (negative ablation: double-counts the narrative, hurts)
  * SOFT-PRIORITY, HARD+SOFT, HARD, SOFT-ONLY, PRIOR
  * NARRATIVE-EVIDENCE -- legacy alias of retrieval-sev (kept for
    comparability with earlier runs)

Stated-severity readout is OFF unless NTSB_ALLOW_STATED_SEVERITY=1 (ablation only).

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/frozenbn_heldout_narrative_bn_eval.py [--limit N] [--sev-topk K]
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

# Lock retrieval index to Zhang window (1982-2006) before main_app loads.
os.environ.pop("NTSB_FULL_CORPUS", None)
os.environ.pop("NTSB_USE_TRAIN_INDEX", None)

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
    on the multi-state severity nodes (calibrated from the training window).

    When sev is present, each severity target is inferred SEPARATELY with only
    its own virtual evidence. Entering both likelihood vectors in one joint
    inference couples them through common ancestors and distorts both
    marginals (per-target Jeffrey conditioning avoids that)."""
    def run(evid_sev):
        ie = gum.LazyPropagation(bn)
        if confidence:
            qb.apply_evidence(ie, bn, confidence)
        if evid_sev:
            for node, lik in evid_sev:
                ie.addEvidence(node, [float(x) for x in lik])
        ie.addTarget(INJ_NODE)
        ie.addTarget(DMG_NODE)
        ie.makeInference()

        def dist(node, states):
            v = bn.variable(node)
            post = ie.posterior(node)
            by = {v.label(i): float(post[i]) for i in range(v.domainSize())}
            return np.array([by[s] for s in states])
        return dist(INJ_NODE, INJ_STATES), dist(DMG_NODE, DMG_STATES)

    if not sev:
        return run(None)
    inj = run([(INJ_NODE, sev["injury"])] if "injury" in sev else None)[0]
    dmg = run([(DMG_NODE, sev["damage"])] if "damage" in sev else None)[1]
    return inj, dmg


def selftest_virtual_evidence(bn):
    """Sanity check that virtual evidence propagates as designed: a likelihood
    vector L(j) = f(j)/p0(j) on a severity node must move that node's
    posterior to exactly f. Run once before scoring; abort loudly if broken."""
    target = np.array([0.05, 0.10, 0.15, 0.70])
    for node in (INJ_NODE, DMG_NODE):
        v = bn.variable(node)
        ie = gum.LazyPropagation(bn)
        ie.addTarget(node)
        ie.makeInference()
        post = ie.posterior(node)
        p0 = np.array([float(post[i]) for i in range(v.domainSize())])
        lik = target / np.maximum(p0, 1e-12)
        lik = lik / lik.max()
        ie2 = gum.LazyPropagation(bn)
        ie2.addEvidence(node, [float(x) for x in lik])
        ie2.addTarget(node)
        ie2.makeInference()
        post2 = ie2.posterior(node)
        got = np.array([float(post2[i]) for i in range(v.domainSize())])
        err = float(np.abs(got - target).max())
        assert err < 1e-6, (
            f"virtual-evidence self-test FAILED on {node}: got {got}, "
            f"wanted {target} (max err {err:.2e})")
    print("virtual-evidence self-test passed "
          "(single-node posterior == target distribution)")


def sev_likelihoods(rdist, prior_inj, prior_dmg):
    """Virtual-evidence likelihood vectors from a k-NN severity distribution:
    L(j) ∝ f_q(j) / p0(j) (Jeffrey conditioning). Alone, the posterior equals
    f_q exactly; combined with event evidence, the BN fuses both sources."""
    def lik(fq, p0):
        v = np.maximum(np.asarray(fq, dtype=float), 1e-9) / np.maximum(p0, 1e-12)
        return list(v / v.max())
    return {"injury": lik(rdist["injury"], prior_inj),
            "damage": lik(rdist["damage"], prior_dmg)}


def selftest_virtual_evidence_e2e(bn, prior_inj, prior_dmg):
    """HARD failure if severity virtual evidence does not move the posterior
    through the SAME posteriors() path the eval uses (per-target inference).
    Guards against the silent-fallback bug: with only the injury virtual
    evidence set, the injury posterior must equal the target distribution."""
    target = np.array([0.55, 0.25, 0.15, 0.05])
    lik = sev_likelihoods({"injury": target, "damage": prior_dmg},
                          prior_inj, prior_dmg)
    got_i, _ = posteriors(bn, {}, sev={"injury": lik["injury"]})
    if not np.allclose(got_i, target, atol=1e-4):
        raise RuntimeError(
            f"virtual-evidence self-test FAILED: posterior {got_i} "
            f"!= target {target} -- severity evidence is not propagating")
    print("virtual-evidence self-test: OK (posterior tracks target exactly)")


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
    # --sev-topk N: neighbors for the k-NN severity readout (default 100;
    # 25 selected on the internal 2002-2006 validation split)
    sev_topk = 100
    if "--sev-topk" in sys.argv:
        sev_topk = int(sys.argv[sys.argv.index("--sev-topk") + 1])
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

    selftest_virtual_evidence(bn)
    if sev_topk != 100:
        print(f"k-NN severity retrieval: top_k = {sev_topk}")

    prior_inj, prior_dmg = posteriors(bn, {})
    selftest_virtual_evidence_e2e(bn, prior_inj, prior_dmg)
    predictors = ["prior", "soft-only", "hard", "hard+soft",
                  "soft-priority", "retrieval-sev", "bn-sev",
                  "bn-fused", "bn-fused-t", "narrative-evidence"]
    if os.environ.get("NTSB_ALLOW_STATED_SEVERITY", "").strip().lower() in ("1", "true", "yes"):
        predictors += ["full", "soft+stated", "narr-sev"]
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
        soft_pri = qb.merge_evidence_soft_priority(hard["confidence"], softonly)
        try:
            preds["soft-priority"] = (posteriors(bn, soft_pri) if soft_pri
                                      else (prior_inj, prior_dmg))
        except Exception:
            preds["soft-priority"] = preds["soft-only"]
        rdist = None
        try:
            rdist = qb.severity_retrieval_distributions(text, ds,
                                                        top_k=sev_topk)
            if rdist:
                ni = np.array(rdist["injury"])
                nd = np.array(rdist["damage"])
                # RETRIEVAL-SEV: raw k-NN readout, no BN (ablation arm).
                preds["retrieval-sev"] = (ni, nd)
                preds["narrative-evidence"] = (ni, nd)  # legacy alias
            else:
                preds["retrieval-sev"] = (prior_inj, prior_dmg)
                preds["narrative-evidence"] = preds["soft-priority"]
        except Exception:
            preds["retrieval-sev"] = (prior_inj, prior_dmg)
            preds["narrative-evidence"] = preds["soft-priority"]
        # BN-SEV (PRIMARY): the same k-NN severity distributions enter the
        # FROZEN BN as virtual evidence (Jeffrey conditioning, per-target
        # inference); the reported numbers are BN posteriors.
        sev_knn = {}
        try:
            if rdist:
                sev_knn = qb.retrieval_severity_virtual_evidence(
                    text, ds, bn, rdist=rdist)
            preds["bn-sev"] = (posteriors(bn, {}, sev=sev_knn)
                               if sev_knn else (prior_inj, prior_dmg))
        except Exception:
            preds["bn-sev"] = (prior_inj, prior_dmg)
        # BN-FUSED: event evidence AND k-NN severity in one inference.
        # Negative ablation -- both signals derive from the same narrative,
        # so the BN double-counts them (violated conditional independence).
        try:
            preds["bn-fused"] = (posteriors(bn, soft_pri, sev=sev_knn)
                                 if (soft_pri or sev_knn)
                                 else (prior_inj, prior_dmg))
        except Exception:
            preds["bn-fused"] = preds["bn-sev"]
        # BN-FUSED-T: same, event confidences tempered (sqrt) to test whether
        # softer event evidence rescues the fusion. (It does not.)
        try:
            temp = {n2: float(c) ** 0.5 for n2, c in soft_pri.items()}
            preds["bn-fused-t"] = (posteriors(bn, temp, sev=sev_knn)
                                   if (temp or sev_knn)
                                   else (prior_inj, prior_dmg))
        except Exception:
            preds["bn-fused-t"] = preds["bn-fused"]
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

        if "narr-sev" in predictors:
            try:
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
            rec[f"{p}:inj_probs"] = [round(float(x), 6) for x in pi]
            if dmg_y is not None:
                scores[p]["dmg_brier"].append(brier(pdm, dmg_y))
                scores[p]["dmg_ll"].append(logloss(pdm, dmg_y))
                scores[p]["dmg_acc"].append(float(int(pdm.argmax()) == dmg_y))
                rec[f"{p}:dmg_pred"] = int(pdm.argmax())
                rec[f"{p}:dmg_brier"] = brier(pdm, dmg_y)
                rec[f"{p}:dmg_probs"] = [round(float(x), 6) for x in pdm]
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

    suffix = "" if sev_topk == 100 else f"_k{sev_topk}"
    out_path = OUT if not suffix else OUT.with_name(
        OUT.stem + suffix + OUT.suffix)
    out_path.write_text(json.dumps({
        "n_heldout": len(held),
        "sev_topk": sev_topk,
        "evidence_per_narrative": {"hard_mean": float(np.mean(n_hard_ev)),
                                   "soft_added_mean": float(np.mean(n_soft_ev))},
        "severity_stated": {"damage": n_sev_dmg, "injury": n_sev_inj},
        "summary": summary,
    }, indent=2))
    print(f"\nwrote {out_path}")
    per_out = (ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs" /
               f"heldout_per_item{suffix}.json")
    per_out.write_text(json.dumps({"predictors": predictors,
                                   "items": per_item}, indent=1))
    print(f"wrote {per_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

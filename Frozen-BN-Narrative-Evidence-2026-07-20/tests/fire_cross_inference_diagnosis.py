#!/usr/bin/env python3
"""Why does BN cross-inference on `fire` fail, and is the failure the network's?

`fire_node_cross_inference.py` reports ROC AUC 0.38-0.46 for the BN against
0.96-0.98 for retrieval, and concludes the frozen network carries no usable
information about an unobserved fire node. Its own diagnostics suggest a
narrower explanation: in the frozen DAG `fire` has 13 ancestors out of 783
nodes (6 barred by the leak guard) and exactly two children, both severity
nodes -- and that experiment deliberately never observes severity, to keep the
`fire -> damage` collider closed.

Blocking a collider is how you make a node uninformative. This script asks
whether the failure is the network's or the query's, by scoring `fire` under
four evidence regimes on the same cohort:

  prior      -- no evidence (rank-uninformative by construction)
  events     -- narrative event evidence only  [reproduces the published arm]
  severity   -- k-NN severity virtual evidence only, opening the collider
  events+sev -- both

All arms use the MASKED text (fire lexicon deleted before embedding and
parsing) and the same node-level leak guard, so no arm can see a fire word.
If the severity arms rank fire well above chance, the published verdict needs
rewording: the network is not inert, it was asked a question with no open path.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/fire_cross_inference_diagnosis.py [--limit N]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.pop("NTSB_FULL_CORPUS", None)
os.environ.pop("NTSB_USE_TRAIN_INDEX", None)

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "shared" / "code", FROZEN_DIR / "code",
           FROZEN_DIR / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pyagrum as gum  # noqa: E402

import prognosis as pg  # noqa: E402
import query_to_bn as qb  # noqa: E402
from bn_upgraded import (build_upgraded, INJ_NODE, DMG_NODE,  # noqa: E402
                         INJ_STATES, DMG_STATES)
import fire_node_cross_inference as F  # noqa: E402

FULL = REPO_ROOT / "shared" / "data" / "processed" / "refined_dataset.json"
WINDOW = REPO_ROOT / "shared" / "data" / "processed" / "refined_dataset_1982_2006.json"
OUTDIR = FROZEN_DIR / "outputs"
FIRE = "fire"


def p_fire(bn, ev_conf=None, sev=None):
    ie = gum.LazyPropagation(bn)
    if ev_conf:
        qb.apply_evidence(ie, bn, ev_conf)
    if sev:
        for key, node in (("injury", INJ_NODE), ("damage", DMG_NODE)):
            if key in sev:
                ie.addEvidence(node, [float(x) for x in sev[key]])
    ie.addTarget(FIRE)
    ie.makeInference()
    v = bn.variable(FIRE)
    post = ie.posterior(FIRE)
    by = {v.label(i): float(post[i]) for i in range(v.domainSize())}
    return by.get("Yes", by.get("yes", 0.0))


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    full = json.loads(FULL.read_text())
    window_ids = set(json.loads(WINDOW.read_text()).keys())
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]
    safe_names = [n for n in names if not F.fire_leaky_node(n)]
    import main_app  # noqa: F401  (loads the window retrieval index)

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
    print(f"cohort: {len(held)}   evidence-eligible nodes after fire guard: "
          f"{len(safe_names)} of {len(names)}")

    anc = bn.ancestors(bn.idFromName(FIRE))
    anc_names = {bn.variable(i).name() for i in anc}
    enterable = sorted(n for n in anc_names if not F.fire_leaky_node(n))
    children = [bn.variable(i).name() for i in bn.children(bn.idFromName(FIRE))]
    print(f"`fire` ancestors: {len(anc_names)} ({len(enterable)} enterable)   "
          f"children: {children}")

    prior_fire = p_fire(bn)
    prior_inj, prior_dmg = None, None
    ie = gum.LazyPropagation(bn)
    ie.addTarget(INJ_NODE)
    ie.addTarget(DMG_NODE)
    ie.makeInference()
    prior_inj = np.array([float(ie.posterior(INJ_NODE)[i])
                          for i in range(bn.variable(INJ_NODE).domainSize())])
    prior_dmg = np.array([float(ie.posterior(DMG_NODE)[i])
                          for i in range(bn.variable(DMG_NODE).domainSize())])
    print(f"prior P(fire) = {prior_fire:.4e}")

    arms = {"prior": [], "events": [], "severity": [], "events+sev": [],
            "events+sev (fixed)": [], "retrieval[masked]": []}
    y_coded, y_occ, keep = [], [], []
    n_anc_hit = 0

    for i, (k, inc, narr) in enumerate(held):
        base = qb.redact_severity_phrases(narr[:4000])
        text = F.mask_fire_words(base)          # strictest variant
        try:
            softonly = {lab: c for lab, c, _ in qb.retrieval_facts(
                text, safe_names, main_app.refined_dataset)}
            hard = qb.parse_query_to_bn_evidence(text, safe_names, dataset=ds,
                                                semantic=False)
            ev = qb.merge_evidence_soft_priority(hard["confidence"], softonly)
            ev = {n: c for n, c in ev.items() if not F.fire_leaky_node(n)}
        except Exception as exc:
            print(f"  parse failed {k}: {exc}")
            continue
        if set(ev) & set(enterable):
            n_anc_hit += 1
        sev = sev_fixed = None
        try:
            rdist = qb.severity_retrieval_distributions(text, ds, top_k=100)
            if rdist:
                sev = qb.retrieval_severity_virtual_evidence(text, ds, bn,
                                                             rdist=rdist)
                # same assertion, ratio taken against the marginal that holds
                # once the event evidence is in (see qb.jeffrey_likelihood)
                sev_fixed = {
                    "injury": qb.jeffrey_likelihood(
                        bn, INJ_NODE, INJ_STATES, rdist["injury"],
                        confidence=ev),
                    "damage": qb.jeffrey_likelihood(
                        bn, DMG_NODE, DMG_STATES, rdist["damage"],
                        confidence=ev),
                }
        except Exception:
            sev = sev_fixed = None

        arms["prior"].append(prior_fire)
        arms["events"].append(p_fire(bn, ev, None))
        arms["severity"].append(p_fire(bn, None, sev) if sev else prior_fire)
        arms["events+sev"].append(p_fire(bn, ev, sev) if sev
                                  else arms["events"][-1])
        arms["events+sev (fixed)"].append(p_fire(bn, ev, sev_fixed)
                                          if sev_fixed
                                          else arms["events"][-1])
        # neighbour vote on the same masked text, as the reference
        try:
            pool = qb.severity_retrieval_distributions  # noqa: F841
            nb = qb.retrieval_facts(text, [FIRE], main_app.refined_dataset,
                                    min_fq=0.0, min_lift=0.0, top_m=10**6)
            arms["retrieval[masked]"].append(
                float(dict((l, c) for l, c, _ in nb).get(FIRE, 0.0)))
        except Exception:
            arms["retrieval[masked]"].append(0.0)

        y_coded.append(F.fire_coded_field(inc))
        y_occ.append(F.fire_occurrence(inc))
        keep.append(k)
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(held)}")

    print(f"\nevidence reached an enterable ancestor of `fire` in "
          f"{n_anc_hit}/{len(keep)} accidents")

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit()
    emit("# Fire cross-inference: is the failure the network's or the query's?")
    emit()
    emit(f"Cohort n = {len(keep)}, masked text (fire lexicon deleted before "
         "embedding and parsing), fire-lexicon nodes barred from evidence. "
         f"`fire` has {len(anc_names)} ancestors ({len(enterable)} enterable) "
         f"and children {children}.")
    emit()
    emit("The published experiment enters event evidence and never observes "
         "severity, so the only children of `fire` stay unobserved and the "
         "`fire -> severity` collider stays closed. The `severity` arms below "
         "open it with exactly the k-NN virtual evidence the main pipeline "
         "already uses.")
    emit()

    for tname, y_all in (("coded-field", y_coded), ("occurrence", y_occ)):
        idx = [j for j, v in enumerate(y_all) if v is not None]
        y = [int(y_all[j]) for j in idx]
        emit(f"## Truth: {tname} (n = {len(y)}, base rate "
             f"{100*sum(y)/len(y):.1f}%)")
        emit()
        emit("| arm | ROC AUC | 95% CI | avg precision |")
        emit("|---|---|---|---|")
        for arm, s_all in arms.items():
            s = [s_all[j] for j in idx]
            auc = F.roc_auc(y, s)
            lo, hi = F.bootstrap_auc_ci(y, s)
            ap = F.average_precision(y, s)
            emit(f"| {arm} | {auc:.3f} | [{lo:.3f}, {hi:.3f}] | {ap:.3f} |")
        emit()

    out = OUTDIR / "fire_cross_inference_diagnosis.md"
    out.write_text("\n".join(lines) + "\n")
    json.dump({"ids": keep, "y_coded": y_coded, "y_occ": y_occ,
               "arms": {k: list(map(float, v)) for k, v in arms.items()}},
              open(OUTDIR / "fire_cross_inference_diagnosis.json", "w"))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

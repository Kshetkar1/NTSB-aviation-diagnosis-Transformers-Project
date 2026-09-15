#!/usr/bin/env python3
"""Can event evidence and retrieval evidence actually be combined?

Three ways to put the k-NN severity distribution f_q and the parsed event
evidence into the frozen network, scored on the held-out cohort.

  A. jeffrey-p0   -- the shipped `bn-fused`: likelihood f_q/p0 with p0 the
                     UNCONDITIONAL severity prior, entered alongside event
                     evidence. Wrong denominator: the posterior becomes
                     f_q * (p1/p0), and on per-flight priors that factor spans
                     1e5 and drives the majority state to zero.
  B. jeffrey-p1   -- same assertion, correct denominator (qb.jeffrey_likelihood).
                     Posterior equals f_q exactly. Correct, but a no-op: once
                     you ASSERT a marginal, other evidence cannot contribute to
                     it. This is why `bn-fused` can never be a real fusion.
  C. likelihood   -- treat f_q as an OBSERVATION rather than an assertion:
                       P(sev | events, retrieval) prop p1(j) * f_q(j)/q0(j)
                     with q0 the per-ACCIDENT severity base rate in the build
                     window. This is the only arm where both sources speak.
                     The scale correction matters: f_q counts accidents, while
                     the network's p0 is per-flight (Zhang Eq. 6), so dividing
                     f_q by p0 mixes two different denominators.

If C beats retrieval alone, the network contributes information beyond the
neighbours and the paper has a positive fusion result. If it does not, the
per-flight prior is blocking evidence combination and that becomes the finding.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/severity_fusion_fix.py [--limit N]
"""
from __future__ import annotations

import json
import math
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
                         INJ_STATES, DMG_STATES, DMG_BY_CODE)

FULL = REPO_ROOT / "shared" / "data" / "processed" / "refined_dataset.json"
WINDOW = REPO_ROOT / "shared" / "data" / "processed" / "refined_dataset_1982_2006.json"
OUTDIR = FROZEN_DIR / "outputs"

INJ_LAB = ["fatal", "serious", "minor", "none"]
DMG_LAB = ["destroyed", "substantial", "minor", "none"]


def truth_states(inc):
    inj = pg.zhang_injury_code(inc)
    inj_i = {"FATL": 0, "SERS": 1, "MINR": 2, "NONE": 3}[inj]
    dmg_code = str(inc.get("damage") or "").upper()
    dmg_i = DMG_BY_CODE.get(dmg_code) if dmg_code in DMG_BY_CODE else None
    return inj_i, dmg_i


def marginals(bn, confidence=None):
    ie = gum.LazyPropagation(bn)
    if confidence:
        qb.apply_evidence(ie, bn, confidence)
    ie.addTarget(INJ_NODE)
    ie.addTarget(DMG_NODE)
    ie.makeInference()

    def dist(node, states):
        v = bn.variable(node)
        post = ie.posterior(node)
        by = {v.label(i): float(post[i]) for i in range(v.domainSize())}
        return np.array([by[s] for s in states])
    return dist(INJ_NODE, INJ_STATES), dist(DMG_NODE, DMG_STATES)


def per_accident_base(ds):
    """Empirical severity distribution over build-window ACCIDENTS (q0)."""
    inj = np.zeros(4)
    dmg = np.zeros(4)
    for inc in ds.values():
        i, d = truth_states(inc)
        if i is not None:
            inj[i] += 1
        if d is not None:
            dmg[d] += 1
    return inj / inj.sum(), dmg / dmg.sum()


def macro_f1(preds, truth, n_cls):
    f1 = []
    for c in range(n_cls):
        tp = sum(1 for p, y in zip(preds, truth) if p == c and y == c)
        fp = sum(1 for p, y in zip(preds, truth) if p == c and y != c)
        fn = sum(1 for p, y in zip(preds, truth) if p != c and y == c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(f1) / n_cls


def mcnemar(a, b):
    x = sum(1 for p, q in zip(a, b) if p and not q)
    y = sum(1 for p, q in zip(a, b) if q and not p)
    n = x + y
    if n == 0:
        return x, y, 1.0
    k = min(x, y)
    return x, y, min(1.0, 2 * sum(math.comb(n, i)
                                  for i in range(k + 1)) / 2 ** n)


def boot_ci(ok, iters=4000, seed=42):
    rng = np.random.default_rng(seed)
    a = np.asarray(ok, dtype=float)
    idx = rng.integers(0, len(a), size=(iters, len(a)))
    m = a[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    full = json.loads(FULL.read_text())
    window_ids = set(json.loads(WINDOW.read_text()).keys())
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]
    import main_app

    q0_i, q0_d = per_accident_base(ds)
    p0_i, p0_d = marginals(bn)
    print("per-ACCIDENT base rate q0  injury",
          np.array2string(q0_i, precision=3))
    print("per-FLIGHT network prior p0 injury",
          np.array2string(p0_i, formatter={'float': lambda x: f'{x:.2e}'}))

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
    print(f"cohort: {len(held)}")

    arms = ["retrieval (f_q)", "A jeffrey-p0 (shipped bn-fused)",
            "B jeffrey-p1 (fixed)", "C likelihood (q0-scaled)"]
    pred = {a: {"inj": [], "dmg": []} for a in arms}
    truth = {"inj": [], "dmg": []}

    for i, (k, inc, narr) in enumerate(held):
        t = qb.redact_severity_phrases(narr[:4000])
        yi, yd = truth_states(inc)
        try:
            hard = qb.parse_query_to_bn_evidence(t, names, dataset=ds,
                                                 semantic=False)
            softonly = {l: c for l, c, _ in qb.retrieval_facts(
                t, names, main_app.refined_dataset)}
            ev = qb.merge_evidence_soft_priority(hard["confidence"], softonly)
            rd = qb.severity_retrieval_distributions(t, ds, top_k=100)
        except Exception as exc:
            print(f"  skip {k}: {exc}")
            continue
        if not rd:
            continue
        fq_i = np.asarray(rd["injury"], dtype=float)
        fq_d = np.asarray(rd["damage"], dtype=float)
        p1_i, p1_d = marginals(bn, ev) if ev else (p0_i, p0_d)

        def norm(v):
            v = np.maximum(v, 0.0)
            s = v.sum()
            return v / s if s > 0 else v

        pred["retrieval (f_q)"]["inj"].append(int(np.argmax(fq_i)))
        pred["retrieval (f_q)"]["dmg"].append(int(np.argmax(fq_d)))
        # A: the shipped arm, f_q/p0 entered on top of event evidence
        pred["A jeffrey-p0 (shipped bn-fused)"]["inj"].append(
            int(np.argmax(norm(p1_i * (fq_i / np.maximum(p0_i, 1e-12))))))
        pred["A jeffrey-p0 (shipped bn-fused)"]["dmg"].append(
            int(np.argmax(norm(p1_d * (fq_d / np.maximum(p0_d, 1e-12))))))
        # B: correct denominator -> posterior == f_q
        pred["B jeffrey-p1 (fixed)"]["inj"].append(
            int(np.argmax(norm(p1_i * (fq_i / np.maximum(p1_i, 1e-12))))))
        pred["B jeffrey-p1 (fixed)"]["dmg"].append(
            int(np.argmax(norm(p1_d * (fq_d / np.maximum(p1_d, 1e-12))))))
        # C: f_q as a likelihood against the per-ACCIDENT base rate
        pred["C likelihood (q0-scaled)"]["inj"].append(
            int(np.argmax(norm(p1_i * (fq_i / np.maximum(q0_i, 1e-12))))))
        pred["C likelihood (q0-scaled)"]["dmg"].append(
            int(np.argmax(norm(p1_d * (fq_d / np.maximum(q0_d, 1e-12))))))
        truth["inj"].append(yi)
        truth["dmg"].append(yd)
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(held)}")

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit()
    emit("# Combining event evidence with retrieval evidence")
    emit()
    emit(f"n = {len(truth['inj'])} held-out accidents. All arms use the same "
         "parsed event evidence and the same k = 100 severity distribution; "
         "they differ only in how the two are combined.")
    emit()
    for tgt, states, lab in (("inj", INJ_STATES, INJ_LAB),
                             ("dmg", DMG_STATES, DMG_LAB)):
        y = truth[tgt]
        emit(f"## {'Injury' if tgt == 'inj' else 'Damage'}")
        emit()
        emit("| combination | Accuracy | 95% CI | Macro-F1 | "
             + " | ".join(f"{s} recall" for s in lab) + " |")
        emit("|---|---|---|---|" + "---|" * len(lab))
        for a in arms:
            p = pred[a][tgt]
            ok = [1 if x == t else 0 for x, t in zip(p, y)]
            lo, hi = boot_ci(ok)
            rec = []
            for c in range(len(lab)):
                n = sum(1 for t in y if t == c)
                h = sum(1 for x, t in zip(p, y) if x == c and t == c)
                rec.append(f"{h}/{n}" if n else "-")
            emit(f"| {a} | {100*sum(ok)/len(ok):.1f}% | "
                 f"[{100*lo:.1f}%, {100*hi:.1f}%] | "
                 f"{macro_f1(p, y, len(lab)):.3f} | " + " | ".join(rec) + " |")
        emit()
        base = [x == t for x, t in zip(pred["retrieval (f_q)"][tgt], y)]
        emit("Paired exact McNemar against retrieval alone:")
        emit()
        emit("| arm | arm only right | retrieval only right | p |")
        emit("|---|---|---|---|")
        for a in arms[1:]:
            cur = [x == t for x, t in zip(pred[a][tgt], y)]
            x, z, p = mcnemar(cur, base)
            emit(f"| {a} | {x} | {z} | {p:.4f}{' *' if p < 0.05 else ''} |")
        emit()

    out = OUTDIR / "severity_fusion_fix.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

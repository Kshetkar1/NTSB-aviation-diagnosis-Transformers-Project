#!/usr/bin/env python3
"""A12 audit + D7 fix: does soft evidence land where the method says it does?

`apply_evidence` documents Jeffrey conditioning: a fact entered at confidence
c should end up believed at ~c. The likelihood ratio is built against the
node's UNCONDITIONAL prior p0:

    LR = [c/(1-c)] / [p0/(1-p0)]

That is exact when the soft fact is the only evidence. It is not exact when
hard event evidence is also present, because the node's prior at inference
time is p1 = P(node | hard evidence), not p0. On per-flight priors
(p0 ~ 1e-5) the LR reaches ~1e4-1e5 and saturates the node.

This script measures the gap on the real held-out cohort and compares three
mechanisms:

  current  - LR against unconditional p0 (what ships today)
  capped   - same, LR clipped to a maximum (the D7 proposal)
  cond     - LR against p1 = P(node | hard evidence) (the principled fix,
             i.e. the BII-9 correction applied to event nodes)

Reports, for each: the distribution of |posterior - intended c|, the
saturation rate (posterior > 0.999 when c < 0.5), and whether severity
accuracy changes.

Run:
    python3 tests/a12_soft_evidence_audit.py [--limit N]
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT.parent / "shared" / "code"))

import pyagrum as gum
import prognosis as pg
import query_to_bn as qb
from bn_upgraded import (build_upgraded, INJ_NODE, DMG_NODE,
                         INJ_STATES, DMG_STATES, injury_state, damage_state)

OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
PROC = ROOT.parent / "shared" / "data" / "processed"

LR_CAP = 100.0   # D7 proposal: clip the likelihood ratio


def set_soft(ie, bn, node, c, p_ref):
    """Enter one soft fact with LR taken against reference prior p_ref."""
    v = bn.variable(node)
    p_ref = min(max(p_ref, 1e-12), 1 - 1e-12)
    lr = (c / (1.0 - c)) / (p_ref / (1.0 - p_ref))
    return lr, v


def apply_variant(bn, hard: dict, soft: dict, mode: str):
    """Build an inference engine with hard evidence + soft facts under `mode`.
    Returns (engine, {node: lr_used})."""
    ie = gum.LazyPropagation(bn)
    for n in hard:
        ie.addEvidence(n, "Yes")

    # reference priors
    if mode == "cond":
        # p1 = P(node | hard evidence): one inference, then reuse
        ref_ie = gum.LazyPropagation(bn)
        for n in hard:
            ref_ie.addEvidence(n, "Yes")
        for n in soft:
            ref_ie.addTarget(n)
        ref_ie.makeInference()
        refs = {}
        for n in soft:
            v = bn.variable(n)
            yes = [i for i in range(v.domainSize()) if v.label(i) == "Yes"][0]
            refs[n] = float(ref_ie.posterior(n)[yes])
        del ref_ie
        gc.collect()
    else:
        refs = qb._node_priors(bn, list(soft.keys()))

    lrs = {}
    for node, c in soft.items():
        v = bn.variable(node)
        p_ref = min(max(refs.get(node, 0.5), 1e-12), 1 - 1e-12)
        lr = (c / (1.0 - c)) / (p_ref / (1.0 - p_ref))
        if mode == "capped":
            lr = min(lr, LR_CAP) if lr >= 1 else max(lr, 1.0 / LR_CAP)
        lrs[node] = lr
        l_yes, l_no = (1.0, 1.0 / lr) if lr >= 1.0 else (lr, 1.0)
        lik = [l_yes if v.label(i) == "Yes" else l_no
               for i in range(v.domainSize())]
        ie.addEvidence(node, lik)
    return ie, lrs


def _checkpoint(modes, gaps, sat, lr_used, acc, n_scored, n_soft):
    """Write partial results so an OOM kill does not lose the whole run."""
    (OUT_DIR / "a12_soft_evidence_audit.json").write_text(json.dumps({
        "n_accidents": n_scored, "n_soft_facts": n_soft, "lr_cap": LR_CAP,
        "complete": False,
        "by_mechanism": {m: {
            "saturation_rate": sat[m] / max(n_soft, 1),
            "mean_abs_gap": float(np.mean(gaps[m])) if gaps[m] else None,
            "median_abs_gap": float(np.median(gaps[m])) if gaps[m] else None,
            "max_lr": float(max(lr_used[m])) if lr_used[m] else None,
            "injury_acc": acc[m]["inj"] / max(n_scored, 1),
            "damage_acc": acc[m]["dmg"] / max(n_scored, 1),
        } for m in modes},
    }, indent=2))


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    print("Building network ...", flush=True)
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]
    print(f"  {bn.size()} nodes, {bn.sizeArcs()} arcs")

    import main_app  # loads window retrieval index

    full = json.loads((PROC / "refined_dataset.json").read_text())
    window_ids = set(json.loads(
        (PROC / "refined_dataset_1982_2006.json").read_text()).keys())
    held = [(k, v) for k, v in full.items()
            if k not in window_ids
            and len(str(v.get("narr_accf") or "").strip()) >= 100]
    held.sort(key=lambda t: t[0])
    if limit:
        held = held[:limit]
    print(f"  held-out cohort: {len(held)}\n")

    modes = ["current", "capped", "cond"]
    gaps = {m: [] for m in modes}
    sat = {m: 0 for m in modes}
    n_soft_total = 0
    lr_used = {m: [] for m in modes}
    acc = {m: {"inj": 0, "dmg": 0} for m in modes}
    n_scored = 0

    for idx, (ev_id, inc) in enumerate(held):
        narr = str(inc.get("narr_accf") or "")
        red = qb.redact_severity_phrases(narr[:4000])

        parsed = qb.parse_query_to_bn_evidence(red, names, dataset=ds,
                                               semantic=False)
        hard = {n: 1.0 for n, c in parsed.get("confidence", {}).items()
                if c >= 0.999 and n in bn.names()}
        soft = {lab: c for lab, c, _ in
                qb.retrieval_facts(red, names, main_app.refined_dataset)
                if lab in bn.names() and lab not in hard}

        if not soft:
            continue
        n_soft_total += len(soft)
        n_scored += 1

        inj_true = injury_state(inc)
        dmg_true = damage_state(inc)

        for m in modes:
            ie, lrs = apply_variant(bn, hard, soft, m)
            for n in soft:
                ie.addTarget(n)
            ie.addTarget(INJ_NODE)
            ie.addTarget(DMG_NODE)
            ie.makeInference()

            for node, c in soft.items():
                v = bn.variable(node)
                yes = [i for i in range(v.domainSize())
                       if v.label(i) == "Yes"][0]
                post = float(ie.posterior(node)[yes])
                gaps[m].append(abs(post - c))
                lr_used[m].append(lrs[node])
                if c < 0.5 and post > 0.999:
                    sat[m] += 1

            pi = ie.posterior(INJ_NODE)
            pd_ = ie.posterior(DMG_NODE)
            if int(np.argmax([float(pi[i]) for i in range(4)])) == inj_true:
                acc[m]["inj"] += 1
            if dmg_true is not None and int(
                    np.argmax([float(pd_[i]) for i in range(4)])) == dmg_true:
                acc[m]["dmg"] += 1

            del ie

        gc.collect()
        if (idx + 1) % 25 == 0:
            print(f"  [{idx+1}/{len(held)}] soft facts so far: {n_soft_total}",
                  flush=True)
            _checkpoint(modes, gaps, sat, lr_used, acc, n_scored, n_soft_total)

    # ─── report ────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("A12 AUDIT: does soft evidence land at its stated confidence?")
    print("=" * 72)
    print(f"\n  {n_scored} accidents contributed {n_soft_total} soft facts\n")

    print(f"  {'mechanism':10} {'saturated':>12} {'mean |post-c|':>14} "
          f"{'median':>9} {'max LR':>12}")
    print(f"  {'-'*10} {'-'*12} {'-'*14} {'-'*9} {'-'*12}")
    for m in modes:
        g = np.array(gaps[m])
        s = sat[m] / max(n_soft_total, 1)
        print(f"  {m:10} {s:11.1%} {g.mean():14.4f} "
              f"{np.median(g):9.4f} {max(lr_used[m]):12.1f}")

    print(f"\n  'saturated' = fact entered at c<0.5 but ends above 0.999")
    print(f"  D7 cap = {LR_CAP:g}")

    print("\n" + "=" * 72)
    print("ACCURACY IMPACT (same cohort, severity via event+soft evidence)")
    print("=" * 72)
    print(f"\n  {'mechanism':10} {'injury':>10} {'damage':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10}")
    for m in modes:
        print(f"  {m:10} {acc[m]['inj']/max(n_scored,1):10.1%} "
              f"{acc[m]['dmg']/max(n_scored,1):10.1%}")

    base = acc["current"]
    print(f"\n  vs current:")
    for m in ["capped", "cond"]:
        di = (acc[m]["inj"] - base["inj"]) / max(n_scored, 1) * 100
        dd = (acc[m]["dmg"] - base["dmg"]) / max(n_scored, 1) * 100
        print(f"    {m:8} injury {di:+.1f} pp   damage {dd:+.1f} pp")

    out = OUT_DIR / "a12_soft_evidence_audit.json"
    out.write_text(json.dumps({
        "n_accidents": n_scored,
        "n_soft_facts": n_soft_total,
        "lr_cap": LR_CAP,
        "complete": True,
        "by_mechanism": {
            m: {
                "saturation_rate": sat[m] / max(n_soft_total, 1),
                "mean_abs_gap": float(np.mean(gaps[m])),
                "median_abs_gap": float(np.median(gaps[m])),
                "max_lr": float(max(lr_used[m])),
                "injury_acc": acc[m]["inj"] / max(n_scored, 1),
                "damage_acc": acc[m]["dmg"] / max(n_scored, 1),
            } for m in modes
        },
    }, indent=2))
    print(f"\n  wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())

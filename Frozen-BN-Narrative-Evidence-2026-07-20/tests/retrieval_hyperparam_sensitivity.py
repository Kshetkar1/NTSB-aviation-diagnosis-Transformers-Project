#!/usr/bin/env python3
"""Hyperparameter sensitivity on an INTERNAL split -- Jesse's "is that weight
fit or assumed?" answered with evidence.

The retrieval knobs (top_k neighbors; min_fq / min_lift soft-fact gates) were
hand-set. This script measures how results move across a grid, using only the
training window: queries = 2002-2006 accidents, retrieval pool = 1982-2001
accidents (the 2007-2019 test set is never touched).

Two sweeps:
  1. k-NN severity readout: top-1 injury/damage accuracy vs top_k.
  2. Soft-fact gates: number of facts emitted + their empirical precision
     (fraction of suggested facts actually coded in the query accident)
     vs (min_fq, min_lift).

Writes outputs/hyperparam_sensitivity.md. Embeddings cached in
outputs/emb_cache_redacted.npz (shared with embedding_lr_baseline.py).

Run:
  export PYTHONPATH="shared/code:Frozen-BN-Narrative-Evidence-2026-07-20/code"
  python Frozen-BN-Narrative-Evidence-2026-07-20/tests/retrieval_hyperparam_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT / "shared" / "code", FROZEN_DIR / "code"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

import prognosis as pg  # noqa: E402
import query_to_bn as qb  # noqa: E402

WINDOW = REPO_ROOT / "shared" / "data" / "processed" / "refined_dataset_1982_2006.json"
OUT = FROZEN_DIR / "outputs" / "hyperparam_sensitivity.md"

TOP_KS = [25, 50, 100, 200]
FQ_GRID = [0.10, 0.15, 0.25]
LIFT_GRID = [2.0, 3.0, 5.0]
VAL_SPLIT_YEAR = 2002          # queries >= this year; pool < this year


def year_of(inc) -> int:
    return int(str(inc.get("ev_date") or "0")[:4])


def main() -> int:
    from embedding_lr_baseline import embed_all  # cached, redacted embeddings
    import main_app

    ds = json.loads(WINDOW.read_text())
    pool_ids, val = [], []
    for k, inc in ds.items():
        narr = str(inc.get("narr_accf") or "").strip()
        y = year_of(inc)
        if y < VAL_SPLIT_YEAR:
            pool_ids.append(k)
        elif len(narr) >= 100:
            val.append((k, inc, qb.redact_severity_phrases(narr[:4000])))
    val.sort(key=lambda t: t[0])
    pool_set = set(pool_ids)
    print(f"validation queries (>= {VAL_SPLIT_YEAR}): {len(val)}   "
          f"pool (< {VAL_SPLIT_YEAR}): {len(pool_ids)}")

    Xq = embed_all([(k, t) for k, _, t in val])

    inj = {"FATL": 0, "SERS": 1, "MINR": 2, "NONE": 3}
    dmg = {"DEST": 0, "SUBS": 1, "MINR": 2, "NONE": 3}
    support = qb._label_support(ds)
    n_total = len(ds)

    # rank the whole window index once per query, then slice per config
    sev_hits = {kk: {"inj": [], "dmg": []} for kk in TOP_KS}
    fact_counts = {g: [] for g in
                   [(fq, lf) for fq in FQ_GRID for lf in LIFT_GRID]}
    fact_prec = {g: [] for g in fact_counts}

    for i, (k, inc, _) in enumerate(val):
        scores, matches = main_app.find_top_matches(np.asarray(Xq[i]))
        ranked, seen = [], set()
        for s, m in zip(scores, matches):
            if m.get("source") != "incident":
                continue
            ev = m.get("ev_id")
            if not ev or ev in seen or ev not in pool_set or ev == k:
                continue
            seen.add(ev)
            ranked.append((float(s), ds[ev]))
            if len(ranked) >= max(TOP_KS):
                break

        yi = inj[pg.zhang_injury_code(inc)]
        yd = dmg.get(str(inc.get("damage") or "").upper(), -1)
        true_labels = qb._incident_bn_labels(inc)

        for kk in TOP_KS:
            pool = ranked[:kk]
            if not pool:
                continue
            iv, dv = [0.5] * 4, [0.5] * 4
            for w, p_inc in pool:
                iv[inj[pg.zhang_injury_code(p_inc)]] += w
                d = str(p_inc.get("damage") or "").upper()
                if d in dmg:
                    dv[dmg[d]] += w
            sev_hits[kk]["inj"].append(float(int(np.argmax(iv)) == yi))
            if yd >= 0:
                sev_hits[kk]["dmg"].append(float(int(np.argmax(dv)) == yd))

        # soft-fact gate sweep on the top-100 pool
        pool = ranked[:100]
        wsum = sum(w for w, _ in pool) or 1.0
        mass = defaultdict(float)
        for w, p_inc in pool:
            for lab in qb._incident_bn_labels(p_inc):
                if not lab.startswith("person: "):
                    mass[lab] += w
        for (fq_min, lift_min) in fact_counts:
            kept = []
            for lab, m in mass.items():
                f_q = m / wsum
                f_0 = support.get(lab, 0) / n_total
                if f_q < fq_min or not f_0 or f_q >= 1.0:
                    continue
                lift = (f_q / (1 - f_q)) / (f_0 / (1 - f_0))
                if lift >= lift_min:
                    kept.append((f_q, lab))
            kept.sort(reverse=True)
            kept = kept[:3]
            fact_counts[(fq_min, lift_min)].append(len(kept))
            if kept:
                fact_prec[(fq_min, lift_min)].append(
                    float(np.mean([lab in true_labels for _, lab in kept])))
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(val)}")

    lines = ["# Retrieval hyperparameter sensitivity (internal split)", "",
             f"Queries: {len(val)} window accidents {VAL_SPLIT_YEAR}-2006; "
             f"pool: 1982-{VAL_SPLIT_YEAR - 1} only; 2007-2019 test never touched.",
             "", "## k-NN severity readout vs top_k", "",
             "| top_k | injury top-1 | damage top-1 |", "|---|---|---|"]
    for kk in TOP_KS:
        s = sev_hits[kk]
        mark = " (production)" if kk == 100 else ""
        lines.append(f"| {kk}{mark} | {np.mean(s['inj']):.1%} "
                     f"| {np.mean(s['dmg']):.1%} |")
    lines += ["", "## Soft-fact gates (top-100 pool, max 3 facts)", "",
              "| min_fq | min_lift | facts/query | fact precision |",
              "|---|---|---|---|"]
    for (fq_min, lift_min), counts in fact_counts.items():
        prec = fact_prec[(fq_min, lift_min)]
        mark = " (production)" if (fq_min, lift_min) == (0.15, 3.0) else ""
        lines.append(f"| {fq_min}{mark} | {lift_min} | {np.mean(counts):.2f} "
                     f"| {np.mean(prec):.1%} |" if prec else
                     f"| {fq_min}{mark} | {lift_min} | {np.mean(counts):.2f} | - |")
    lines += ["", "Fact precision = fraction of suggested soft facts that are "
              "actually coded (occurrence/finding) in the query accident."]

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")
    for kk in TOP_KS:
        s = sev_hits[kk]
        print(f"top_k={kk}: inj {np.mean(s['inj']):.1%}  dmg {np.mean(s['dmg']):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""THEOREM CHECK: Zhang-consistent LTP recovers Table 7 EXACTLY with neutral weights.

    P(cause|Q) = sum_K P(cause|K) * P(K|Q)

With P(cause|K) = Zhang counting restricted to cluster K and P(K) = N_K/N,
the cluster decomposition cancels and the LTP output must equal the flat
Table 7 distribution to machine precision -- for EVERY cause, not just anchors.

Also demonstrates the similarity-weighted deviation: tilting the weights toward
one cluster moves the distribution AWAY from Table 7 (that shift is the
narrative signal, decomposed per cluster).

Offline (no network). Exit 0 == theorem holds.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 tests/ltp_zhang_recovery.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ltp_zhang  # noqa: E402
import zhang_diagnosis as zd  # noqa: E402

DATASET = ROOT / "data" / "processed" / "refined_dataset_1982_2006.json"

# Published Table 7 anchors (Zhang p.12) the recovered distribution must hit.
PUBLISHED_ANCHORS = {
    "airframe/component/system failure/malfunction": 0.31372,
    "electrical system, electric wiring": 0.08823,
    "fluid, fuel": 0.05882,
    "auxiliary power unit (apu)": 0.04901,
}

OUTCOMES = ["fire", "loss of engine power (total) - mechanical failure/malfunction"]


def dist_map(causes):
    return {c["cause"].lower(): c["probability"] for c in causes}


def l1(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def main():
    ds = json.loads(DATASET.read_text(encoding="utf-8"))
    ok = True

    for outcome in OUTCOMES:
        targets = {outcome}
        flat = zd.empirical_cause_distribution(
            outcome, targets=targets, dataset=ds, cause_factor_only=True)
        flat_map = dist_map(flat["causes"])

        r = ltp_zhang.ltp_diagnose(outcome=outcome, dataset=ds, weights="neutral",
                                   top_n=10_000)
        ltp_map = dist_map(r["causes"])

        d = l1(flat_map, ltp_map)
        exact = d < 1e-9
        ok = ok and exact
        print(f"\nOUTCOME {outcome!r}: denom={r['outcome_count']} "
              f"clusters={r['n_clusters']}")
        print(f"  L1(LTP-neutral, Table 7 flat) = {d:.2e}  "
              f"{'EXACT' if exact else 'MISMATCH'}")

        if outcome == "fire":
            if r["outcome_count"] != 102:
                print(f"  FAIL: fire denominator {r['outcome_count']} != 102")
                ok = False
            for lab, pub in PUBLISHED_ANCHORS.items():
                got = ltp_map.get(lab)
                good = got is not None and abs(got - pub) < 6e-4
                ok = ok and good
                print(f"  {'OK ' if good else 'XX '} {lab[:52]:52} "
                      f"ltp={got:.5f}  published={pub:.5f}")

    # ---- Deviation demo: tilt weights toward the wiring-heaviest cluster -----
    # (What retrieval does for a wiring query: up-weight clusters where the
    # query-relevant cause is over-represented.)
    targets = {"fire"}
    part = ltp_zhang.cluster_partition(targets, ds)
    wiring = "electrical system, electric wiring"
    rate_by_cluster = {}
    for k, evs in part.items():
        res = zd.empirical_cause_distribution(
            "fire", targets=targets, dataset=ds, restrict_ev_ids=evs,
            cause_factor_only=True)
        rate = dist_map(res["causes"]).get(wiring, 0.0)
        if len(evs) >= 5:  # ignore single-accident buckets
            rate_by_cluster[k] = rate
    best_cluster = max(rate_by_cluster, key=rate_by_cluster.get)
    print(f"\nSIMILARITY-TILT DEMO (fire): weight {best_cluster!r} 5x "
          f"(highest P(wiring|K) = {rate_by_cluster[best_cluster]:.3f})")
    sim = {}
    for k, evs in part.items():
        boost = 5.0 if k == best_cluster else 1.0
        for ev in evs:
            sim[ev] = boost
    r_tilt = ltp_zhang.ltp_diagnose(outcome="fire", dataset=ds,
                                    weights="similarity", sim_by_ev=sim,
                                    top_n=10_000)
    tilt_map = dist_map(r_tilt["causes"])
    flat_fire = dist_map(zd.empirical_cause_distribution(
        "fire", targets=targets, dataset=ds, cause_factor_only=True)["causes"])
    d_tilt = l1(flat_fire, tilt_map)
    moved_up = tilt_map.get(wiring, 0) > flat_fire.get(wiring, 0)
    ok = ok and moved_up and d_tilt > 1e-3
    print(f"  P(wiring | fire): neutral {flat_fire.get(wiring, 0):.4f} -> "
          f"tilted {tilt_map.get(wiring, 0):.4f}  "
          f"({'UP as expected' if moved_up else 'DID NOT MOVE -- FAIL'})")
    print(f"  L1 to Table 7 after tilt = {d_tilt:.4f}  (deviation IS the signal)")

    print(f"\nRESULT: {'PASS -- theorem holds, tilt behaves' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

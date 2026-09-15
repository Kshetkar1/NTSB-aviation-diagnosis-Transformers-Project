#!/usr/bin/env python3
"""Offline parity check for apps-parity demo — compares ours vs Zhang published."""
from __future__ import annotations

import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
FROZEN_DIR = APP_DIR.parent
REPO_ROOT = FROZEN_DIR.parent
for p in (REPO_ROOT / "shared" / "code", FROZEN_DIR / "code", FROZEN_DIR / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import os

os.environ.setdefault("NTSB_USE_TRAIN_INDEX", "1")

import main_app  # noqa: E402
import pyagrum as gum  # noqa: E402
import query_to_bn as qb  # noqa: E402
import zhang_diagnosis as zd  # noqa: E402
import zhang_reference as zr  # noqa: E402
from bn_upgraded import build_upgraded  # noqa: E402

CLOSE_ABS = 0.02  # within 2 pp absolute
CLOSE_REL = 0.15  # or within 15% relative for small probs


def _close(ours: float, zhang: float) -> bool:
    if zhang is None:
        return False
    if abs(zhang) < 1e-9:
        return abs(ours - zhang) < 1e-6
    if abs(ours - zhang) <= CLOSE_ABS:
        return True
    return abs(ours - zhang) / abs(zhang) <= CLOSE_REL


def bn_posterior(bn, bu, evidence: dict[str, float], target: str, state: str) -> float:
    ie = gum.LazyPropagation(bn)
    qb.apply_evidence(ie, bn, evidence)
    ie.addTarget(target)
    ie.makeInference()
    post = ie.posterior(target)
    v = bn.variable(target)
    idx = [i for i in range(v.domainSize()) if v.label(i) == state]
    if not idx:
        return 0.0
    return float(post[idx[0]])


def p_event_yes(bn, bu, evidence: dict[str, float], node: str) -> float:
    if node not in bn.names():
        return 0.0
    return bn_posterior(bn, bu, evidence, node, "Yes")


def test_table7_fire():
    ds = main_app.refined_dataset
    res = zd.empirical_cause_distribution("fire", dataset=ds, cause_factor_only=True)
    csv_path = FROZEN_DIR / "docs" / "table7_full_reproduction.csv"
    rows = []
    if csv_path.is_file():
        import pandas as pd
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            cause = str(row.get("cause") or row.get("Cause") or "")
            zhang = row.get("zhang") or row.get("Zhang")
            ours_row = next((c for c in res["causes"] if c["cause"].lower() == cause.lower()), None)
            if ours_row is None or zhang is None:
                continue
            ours = float(ours_row["probability"])
            zhang_f = float(zhang)
            rows.append({
                "section": "Table 7 fire",
                "target": cause[:50],
                "ours": ours,
                "zhang": zhang_f,
                "close": _close(ours, zhang_f),
            })
    return rows


def test_table9():
    if not main_app.DATA_LOADED:
        return []
    bn, _meta = build_upgraded(main_app.refined_dataset)
    bu_inj = bn.names()
    from bn_upgraded import INJ_NODE, DMG_NODE, INJ_STATES, DMG_STATES

    rows = []
    for col_idx, col in enumerate(zr.TABLE9_COLUMNS):
        ev = {n: 1.0 for n in col["nodes"]}
        for disp, node, kind, zvals in zr.TABLE9_ROWS:
            zhang = zvals[col_idx]
            if kind == "event":
                ours = p_event_yes(bn, None, ev, node)
            elif kind == "injury":
                ours = bn_posterior(bn, None, ev, INJ_NODE, node)
            else:
                ours = bn_posterior(bn, None, ev, DMG_NODE, node)
            rows.append({
                "section": f"Table 9 col {col_idx+1}",
                "target": disp,
                "ours": ours,
                "zhang": zhang,
                "close": _close(ours, zhang),
            })
    return rows


def test_narrative_queries():
    if not main_app.DATA_LOADED:
        return []
    bn, _ = build_upgraded(main_app.refined_dataset)
    from bn_upgraded import INJ_NODE, DMG_NODE

    queries = [
        ("fire diagnosis", "engine caught fire during takeoff", None),
        ("engine instrument", "trouble with an engine instrument during the flight", ["engine instrument"]),
    ]
    rows = []
    all_nodes = sorted(bn.names())
    for label, q, forced_ev in queries:
        parsed = qb.parse_query_to_bn_evidence(q, all_nodes, dataset=main_app.refined_dataset, semantic=True)
        ev_list = forced_ev or parsed["evidence"]
        ev = {e: parsed["confidence"].get(e, 1.0) for e in ev_list}
        det = zd.detect_outcome(q, dataset=main_app.refined_dataset)
        if det:
            name, targets = det
            res = zd.empirical_cause_distribution(name, targets=targets, dataset=main_app.refined_dataset, cause_factor_only=True)
            top = res["causes"][0] if res["causes"] else None
            if top:
                rows.append({
                    "section": f"narrative:{label}",
                    "target": f"top cause P(c|{name})",
                    "ours": top["probability"],
                    "zhang": None,
                    "close": None,
                    "note": top["cause"][:40],
                })
        hard = set(e.lower() for e in ev_list)
        col_idx, col_label = zr.match_table9_column(hard)
        if col_idx is not None:
            for disp, node, kind, zvals in zr.TABLE9_ROWS[:3]:
                zhang = zvals[col_idx]
                if kind == "event":
                    ours = p_event_yes(bn, None, ev, node)
                else:
                    continue
                rows.append({
                    "section": f"narrative:{label}",
                    "target": disp,
                    "ours": ours,
                    "zhang": zhang,
                    "close": _close(ours, zhang),
                })
    return rows


def summarize(rows):
    with_zhang = [r for r in rows if r.get("zhang") is not None]
    close = [r for r in with_zhang if r.get("close")]
    far = [r for r in with_zhang if not r.get("close")]
    return len(with_zhang), len(close), len(far), far


def main():
    if not main_app.DATA_LOADED:
        print("ERROR: train index not loaded")
        sys.exit(1)

    all_rows = []
    all_rows.extend(test_table7_fire())
    all_rows.extend(test_table9())
    all_rows.extend(test_narrative_queries())

    n, close, far, far_rows = summarize(all_rows)
    print("=" * 72)
    print("ZHANG PARITY SUMMARY (apps-parity thresholds: ±0.02 abs or ±15% rel)")
    print("=" * 72)
    print(f"Cells with Zhang reference: {n}")
    print(f"Close:  {close}")
    print(f"Not close: {far}")
    if n:
        print(f"Close rate: {100*close/n:.1f}%")

    print("\nNOT CLOSE (show Maha these):")
    for r in far_rows[:25]:
        d = r["ours"] - r["zhang"]
        print(f"  [{r['section']}] {r['target'][:40]:40s} ours={r['ours']:.4f} zhang={r['zhang']:.4f} Δ={d:+.4f}")

    out = APP_DIR / "zhang_parity_test_results.json"
    out.write_text(json.dumps({"summary": {"total": n, "close": close, "not_close": far}, "rows": all_rows}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

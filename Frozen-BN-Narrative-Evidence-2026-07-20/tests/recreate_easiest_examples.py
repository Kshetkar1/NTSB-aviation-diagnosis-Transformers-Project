#!/usr/bin/env python3
"""Scratch runner (read-only validation): recreate the SINGLE EASIEST diagnosis and
prognosis examples from Zhang (2021) using OUR current tree code, and print a
side-by-side vs the paper. Writes nothing except stdout; the report file is authored
separately. Does NOT edit trees.py / zhang_diagnosis.py / prognosis.py.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 tests/recreate_easiest_examples.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
_SHARED = REPO_ROOT / "shared" / "code"
_FROZEN_CODE = FROZEN_DIR / "code"
for _p in (_SHARED, _FROZEN_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
ROOT = REPO_ROOT
def _imp():
    """Import the (possibly concurrently-edited) modules with a small retry."""
    last = None
    for _ in range(6):
        try:
            import importlib
            import trees, zhang_diagnosis, prognosis, main_app, config
            importlib.reload(zhang_diagnosis)
            importlib.reload(prognosis)
            importlib.reload(trees)
            return trees, zhang_diagnosis, prognosis, main_app, config
        except Exception as exc:  # other agent may be mid-save
            last = exc
            print(f"[import retry] {type(exc).__name__}: {exc}")
            time.sleep(4)
    raise last


def banner(t):
    print("\n" + "=" * 88)
    print(t)
    print("=" * 88)


def main():
    trees, zd, pg, main_app, config = _imp()
    print(f"ACTIVE INDEX: {getattr(config, 'ACTIVE_INDEX_LABEL', '?')}")
    print(f"DATASET PATH: {getattr(config, 'ACTIVE_INCIDENT_DATA_PATH', '?')}")

    # ============================================================== DIAGNOSIS
    banner("DIAGNOSIS  --  Zhang Table 7  P(cause | fire)   (denominator = 102 fire accidents)")

    # (A) Clean cross-check: faithful Table-7 reproduction (Cause/Factor filter).
    faithful = zd.empirical_cause_distribution("fire", cause_factor_only=True)
    print(f"\n[A] empirical_cause_distribution('fire', cause_factor_only=True)")
    print(f"    fire accidents (denominator) = {faithful['outcome_count']}  | n causes = {len(faithful['causes'])}")
    fmap = {c['cause'].lower(): c for c in faithful['causes']}

    # (B) Default counting (all findings on the fire occurrence; what the TREE uses).
    default = zd.empirical_cause_distribution("fire")
    print(f"[B] empirical_cause_distribution('fire')  (default, all findings -- the tree's counting)")
    print(f"    fire accidents (denominator) = {default['outcome_count']}  | n causes = {len(default['causes'])}")
    dmap = {c['cause'].lower(): c for c in default['causes']}

    anchors = [
        ("airframe/component/system failure/malfunction", 0.31372, 32),
        ("loss of engine power (total) - mechanical failure/malfunction", 0.08823, 9),
        ("electrical system, electric wiring", 0.08823, 9),
        ("fluid, fuel", 0.05882, 6),
        ("auxiliary power unit (apu)", 0.04901, 5),
    ]
    print(f"\n{'cause':62} {'ZhangP':>8} {'Zn':>3} | {'faithfulP':>9} {'fn':>3} | {'defaultP':>9} {'dn':>3}")
    print("-" * 110)
    for lab, zp, zn in anchors:
        f = fmap.get(lab, {})
        d = dmap.get(lab, {})
        print(f"{lab[:62]:62} {zp:8.5f} {zn:3d} | "
              f"{f.get('probability', float('nan')):9.5f} {f.get('n', 0):3d} | "
              f"{d.get('probability', float('nan')):9.5f} {d.get('n', 0):3d}")

    # (C) The actual diagnosis TREE (query-first retrieval pool) level-1 edges.
    q_diag = "engine caught fire during takeoff"
    print(f"\n[C] trees.build_diagnosis_tree({q_diag!r})  -- level-1 edges = P(cause | fire) over the retrieved pool")
    diag = trees.build_diagnosis_tree(q_diag, top_n_incidents=300, branching=8,
                                      depth=1, min_prob=0.0, drop_generic=False)
    meta = diag.get("meta", {})
    print(f"    outcome={meta.get('outcome')!r}  fire-in-pool={meta.get('outcome_count_in_pool')}  "
          f"retrieved={meta.get('retrieved_incidents')}")
    tree = diag.get("tree")
    if tree:
        lvl1 = {c['label'].lower(): c for c in tree['children']}
        print(f"\n    {'cause':58} {'treeP':>8} {'n':>4} {'denom':>6}")
        print("    " + "-" * 80)
        for lab, _zp, _zn in anchors:
            c = lvl1.get(lab)
            if c:
                print(f"    {lab[:58]:58} {c['edge_prob']:8.5f} {c['n']:>4} {c['denom']:>6}")
            else:
                print(f"    {lab[:58]:58} {'--':>8}  (not in top branches)")
        # full top-6 for context
        print("\n    Top-6 level-1 causes from the tree (pool-restricted):")
        for c in tree['children'][:6]:
            print(f"      P={c['edge_prob']:.5f}  n={c['n']}/{c['denom']}  {c['label']}")

    # ============================================================== PROGNOSIS
    banner("PROGNOSIS  --  Zhang Table 9  single hop  P(forced landing | loss of engine power)")
    print("Zhang Table 9 (p.17), column 'Loss of engine power': Forced landing = 14.29e-2 = 0.1429")
    print("Text (p.16, sec 5.4): 'the loss of engine power leads to forced landing'.")

    q_prog = "loss of engine power"
    for qr in (True, False):
        prog = trees.build_prognosis_tree(q_prog, top_n_incidents=300, branching=8,
                                          depth=1, min_prob=0.0, min_n=1,
                                          query_relevant=qr, drop_self_loops=True)
        m = prog.get("meta", {})
        lbl = "query-relevant pool" if qr else "global dataset"
        print(f"\n[{lbl}] seed={m.get('seed_event')!r} family={m.get('seed_family')!r} "
              f"incidents={m.get('incidents_in_population')}")
        t = prog.get("tree")
        if not t:
            print(f"    ERROR: {m.get('error')}")
            continue
        # find a 'forced landing' child
        fl = [c for c in t['children'] if 'forced landing' in c['label'].lower()]
        print(f"    seed outgoing transitions (denom from seed) = {t.get('denom')}")
        if fl:
            c = fl[0]
            print(f"    --> P(forced landing | {m.get('seed_event')}) = {c['edge_prob']:.5f}  "
                  f"(n={c['n']}/{c['denom']}, incidents={c.get('n_incidents')})")
        else:
            print("    --> 'forced landing' NOT a direct next-event of the seed in this population")
        print("    Top next-events from seed:")
        for c in t['children'][:8]:
            print(f"      P={c['edge_prob']:.5f}  n={c['n']}/{c['denom']}  {c['label']}")

    # Also: the Zhang single-parent cause->LOEP cell (engine instruments) for context.
    banner("PROGNOSIS context -- Zhang single-parent P(loss of engine power | inoperative engine instruments)=0.95")
    ds = pg.load_dataset()
    ee, ne = pg.build_graph(ds)
    loep_targets = pg.resolve_outcome_targets("loss of engine power", ds)
    cell = pg.zhang_baseline_cpt("engine instruments", loep_targets, ee, ne)
    print(f"    zhang_baseline_cpt('engine instruments' -> LOEP) = {cell}")
    honest = pg.honest_forward_cpt({"engine instruments"}, loep_targets, ds)
    print(f"    honest_forward_cpt(engine instruments -> LOEP)   = {honest}")

    print("\nDONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

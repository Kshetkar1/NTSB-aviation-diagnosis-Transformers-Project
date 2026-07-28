"""SUCCESS TEST: reproduce / compare Zhang (2026) Table 9 (LOEP column) on our data.

For every Table-9 forward cell we compute THREE estimators (see prognosis.py) and print:

    cell | Zhang published | Zhang-method-on-our-data (N) | honest ratio (N) | semantic (K) | note

The single-evidence LOEP cells are the first numeric targets; we then attempt the
2-parent (instruments AND oil) Beta-CDF cell and the downstream/leaf cells.

HONESTY: Zhang's published 0.95 values are 1/1 ratios capped by his hardcoded *0.95.
We surface the underlying N on our data and whether our ratio is also 1.0 (per-node
breakdown printed below the table). Downstream cells are true BN posteriors in Zhang;
our forward/transition machinery gives an empirical *reachability* conditional, not a
full BN posterior, so the Zhang-method column for those reads 'requires BN propagation'.

Run (framework python + network needed for the semantic estimator):
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 tests/prognosis_table9.py
"""
from __future__ import annotations

import re
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
import prognosis as pg  # noqa: E402


# --------------------------------------------------------------------------------------
# Cell definitions. Each evidence concept maps to a FAMILY of finding-subject labels in
# our data (keyword match). The merged family is treated as one logical node for the
# Zhang-baseline column; a per-node breakdown (below the table) exposes the sparse 1/1
# sub-node that Zhang's capped 0.95 corresponds to.
# --------------------------------------------------------------------------------------
EVIDENCE = {
    "inoperative engine instruments": {
        "keyword": "engine instrument",
        "semantic_query": "inoperative engine instruments engine gauge failure",
    },
    "improper oil usage": {
        "keyword": "oil",
        "semantic_query": "improper oil usage oil starvation contamination engine",
    },
    "combustion liner failure": {
        "keyword": "combustion liner",
        "semantic_query": "combustion liner failure combustion assembly engine",
    },
}

OUTCOME = "loss of engine power"

# Published Table-9 (LOEP column) single-evidence + 2-parent targets.
PUB_SINGLE = {
    "inoperative engine instruments": 0.95,
    "improper oil usage": 0.95,
    "combustion liner failure": 0.50,
}
PUB_TWO_PARENT = 0.99  # P(LOEP | engine instruments AND improper oil)

# Downstream / leaf cells (given engine-instruments evidence) -- BN posteriors in Zhang.
DOWNSTREAM = [
    ("forced landing", 0.1357, {"kind": "occurrence", "targets": {"forced landing"}}),
    ("ditching", 4.37e-3, {"kind": "occurrence", "targets": {"ditching"}}),
    ("destroyed aircraft", 1.33e-2, {"kind": "damage", "code": "DEST"}),
    ("substantial damage", 0.046, {"kind": "damage", "code": "SUBS"}),
    ("serious injury", 0.0623, {"kind": "injury", "code": "SERS"}),
    ("no injury", 0.9431, {"kind": "injury", "code": "NONE"}),
]

K_SEMANTIC = 50


def _family_labels(ds, keyword):
    """All finding-subject labels whose text contains the keyword at a word boundary
    (leading boundary only, so 'oil' matches 'oil line' but NOT 'spoiler', and
    'engine instrument' still matches the plural 'engine instruments, ...')."""
    pat = re.compile(rf"\b{re.escape(keyword.lower())}")
    labs = set()
    for inc in ds.values():
        for f in inc.get("findings", []):
            d = pg._s(f.get("finding_description")).lower()
            if pat.search(d):
                labs.add(d)
    return sorted(labs)


def _merged_zhang(family, outcome_targets, edge_events, node_events):
    """Zhang ratio treating the whole family as ONE node: joint = events with any
    family-label -> outcome edge; denom = events where any family-label appears."""
    joint, denom = set(), set()
    for lab in family:
        denom |= node_events.get(lab, set())
        for (a, b), evs in edge_events.items():
            if a == lab and b in outcome_targets:
                joint |= evs
    if not denom:
        return {"value": None, "joint_n": 0, "denom_n": 0, "raw_ratio": None, "capped": False}
    raw = len(joint) / len(denom)
    val, capped = raw, False
    if raw == 1.0:
        val, capped = raw * 0.95, True
    return {"value": val, "joint_n": len(joint), "denom_n": len(denom),
            "raw_ratio": raw, "capped": capped}


def _fmt(v, nd=4):
    return "  n/a " if v is None else f"{v:.{nd}f}"


def main():
    ds = pg.load_dataset()
    edge_events, node_events = pg.build_graph(ds)
    outcome_targets = pg.resolve_outcome_targets(OUTCOME, ds)

    print("=" * 110)
    print(f"PROGNOSIS vs Zhang Table 9  --  outcome node = {OUTCOME!r}")
    print(f"  (outcome family = {len(outcome_targets)} coded labels; "
          f"dataset = {len(ds)} incidents 1982-2006)")
    print("=" * 110)

    # ---- import semantic lane lazily (needs embeddings + network) -----------------
    semantic_ok = True
    try:
        import main_app  # noqa: F401
        if not getattr(main_app, "DATA_LOADED", False):
            semantic_ok = False
    except Exception as e:  # pragma: no cover
        print(f"[warn] semantic lane unavailable ({e}); semantic column = n/a")
        semantic_ok = False

    # =================================================================================
    # SINGLE-EVIDENCE LOEP CELLS
    # =================================================================================
    hdr = (f"{'cell':46} {'Zhang':>7} {'Zhang/ours(N)':>17} "
           f"{'honest(N)':>15} {'semantic(K)':>14}  note")
    print("\n--- SINGLE-EVIDENCE  P(LOEP | cause) ---")
    print(hdr)
    print("-" * 130)

    parents = pg.parent_ratios(outcome_targets, edge_events, node_events)
    per_node = {}
    zhang_nodes = {}
    for ev_name, cfg in EVIDENCE.items():
        family = _family_labels(ds, cfg["keyword"])
        # Zhang-method column: the family's representative coded sub-node = the LOEP-parent
        # node with the highest Zhang ratio (the strong predictor Zhang's Table 9 picks).
        cand = [l for l in family if l in parents]
        node = max(cand, key=lambda l: parents[l], default=None)
        zhang_nodes[ev_name] = node
        z = (pg.zhang_baseline_cpt(node, outcome_targets, edge_events, node_events)
             if node else {"value": None, "joint_n": 0, "denom_n": 0,
                           "raw_ratio": None, "capped": False})
        # honest column: whole family, uncapped, N exposed.
        h = pg.honest_forward_cpt(family, outcome_targets, ds, require_all=False)
        if semantic_ok:
            s = pg.semantic_forward_cpt(cfg["semantic_query"], outcome_targets,
                                        k=K_SEMANTIC)
        else:
            s = {"value": None, "k": 0}

        pub = PUB_SINGLE[ev_name]
        z_str = f"{_fmt(z['value'])}({z['joint_n']}/{z['denom_n']})"
        h_str = f"{_fmt(h['value'])}({h['n_cause']})"
        s_str = f"{_fmt(s['value'])}({s.get('k', 0)})"

        if z["capped"]:
            note = "ratio=1/1 -> *0.95 cap (reproduces Zhang)"
        elif z["raw_ratio"] is not None and abs(z["value"] - pub) < 5e-3:
            note = "reproduces Zhang ratio"
        else:
            note = "differs (sparsity); see per-node"
        print(f"{ev_name:46} {pub:7.4f} {z_str:>17} {h_str:>15} {s_str:>14}  {note}")

        rows = []
        for lab in family:
            zb = pg.zhang_baseline_cpt(lab, outcome_targets, edge_events, node_events)
            if zb["denom_n"]:
                rows.append((lab, zb))
        per_node[ev_name] = rows
    print("  (Zhang/ours column shows: value(jointN/denomN) for the representative node:")
    for ev_name, node in zhang_nodes.items():
        print(f"     {ev_name!r:46} -> node {node!r}")

    # =================================================================================
    # 2-PARENT CELL via Beta-CDF
    # =================================================================================
    print("\n--- 2-PARENT  P(LOEP | engine instruments AND improper oil)  [Beta-CDF path] ---")
    # representative single parent nodes (must each have an edge into LOEP)
    ei_family = _family_labels(ds, "engine instrument")
    oil_family = _family_labels(ds, "oil")
    parents = pg.parent_ratios(outcome_targets, edge_events, node_events)
    ei_node = max((l for l in ei_family if l in parents), key=lambda l: parents[l],
                  default=None)
    oil_node = max((l for l in oil_family if l in parents), key=lambda l: parents[l],
                   default=None)
    print(f"  chosen parent nodes: engine-instr -> {ei_node!r}  oil -> {oil_node!r}")
    if ei_node and oil_node:
        mp = pg.zhang_baseline_multiparent([ei_node, oil_node], outcome_targets,
                                           edge_events, node_events)
        print(f"  Zhang published                 : {PUB_TWO_PARENT:.4f}")
        print(f"  Zhang-method-on-our-data (Beta) : {_fmt(mp['value'])}  "
              f"(contribution={_fmt(mp['contribution'],5)}, "
              f"floor=max active ratio={_fmt(mp.get('floor'))}, "
              f"#parents of LOEP node={mp['n_parents']})")
    else:
        print("  Zhang-method-on-our-data (Beta) :  n/a  (a parent node missing edge to LOEP)")
    h2 = pg.honest_forward_cpt(ei_family + oil_family, outcome_targets, ds,
                               require_all=False)
    h2_both = pg.honest_forward_cpt([ei_node, oil_node] if ei_node and oil_node else [],
                                    outcome_targets, ds, require_all=True)
    print(f"  honest (any instr OR oil)       : {_fmt(h2['value'])} (N={h2['n_cause']})")
    print(f"  honest (BOTH instr AND oil)     : {_fmt(h2_both['value'])} (N={h2_both['n_cause']})")
    if semantic_ok:
        s2 = pg.semantic_forward_cpt(
            "inoperative engine instruments and improper oil usage engine power loss",
            outcome_targets, k=K_SEMANTIC)
        print(f"  semantic (K)                    : {_fmt(s2['value'])} (K={s2['k']})")

    # =================================================================================
    # DOWNSTREAM / LEAF CELLS  (evidence = engine instruments)
    # =================================================================================
    print("\n--- DOWNSTREAM (given inoperative engine instruments) ---")
    print(f"{'cell':22} {'Zhang':>8} {'Zhang/ours':>22} {'honest(N)':>15} {'semantic(K)':>14}")
    print("-" * 95)
    ei_family = _family_labels(ds, "engine instrument")
    down_query = "inoperative engine instruments loss of engine power escalation"
    for name, pub, spec in DOWNSTREAM:
        pred = pg.make_outcome_predicate(spec)
        hd = pg.honest_downstream(ei_family, pred, ds, require_all=False)
        if semantic_ok:
            sd = pg.semantic_forward_cpt(down_query, set(), k=K_SEMANTIC,
                                         outcome_predicate=pred)
        else:
            sd = {"value": None, "k": 0}
        h_str = f"{_fmt(hd['value'])}({hd['n_cause']})"
        s_str = f"{_fmt(sd['value'])}({sd.get('k', 0)})"
        print(f"{name:22} {pub:8.4f} {'requires BN propagation':>22} "
              f"{h_str:>15} {s_str:>14}")

    # =================================================================================
    # PER-NODE ZHANG BREAKDOWN (transparency for the capped 1/1 nodes)
    # =================================================================================
    print("\n--- PER-NODE Zhang-baseline breakdown (which sub-node carries the cap) ---")
    for ev_name, rows in per_node.items():
        print(f"\n  evidence concept: {ev_name!r}")
        if not rows:
            print("    (no family node appears in the graph)")
            continue
        for lab, zb in sorted(rows, key=lambda r: -(r[1]["raw_ratio"] or 0)):
            flag = "  <-- 1/1 capped *0.95" if zb["raw_ratio"] == 1.0 else ""
            print(f"    {lab[:60]:60} joint={zb['joint_n']:>2} "
                  f"denom={zb['denom_n']:>3} ratio={_fmt(zb['raw_ratio'])} "
                  f"-> cell={_fmt(zb['value'])}{flag}")

    print("\n" + "=" * 110)
    print("DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

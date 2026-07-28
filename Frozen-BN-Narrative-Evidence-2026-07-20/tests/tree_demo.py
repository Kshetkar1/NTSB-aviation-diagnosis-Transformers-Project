#!/usr/bin/env python3
"""DEMO + VALIDATION: branching DIAGNOSIS + PROGNOSIS trees, query-first.

The advisor clarified that diagnosis and prognosis are TREES, not linear chains:
from one event MANY outcomes branch, each with its own probability, and each branch
expands further. This demo builds both trees STARTING FROM a free-text narrative
query (embed -> retrieve similar incidents -> build tree), prints them, AND validates
them against Zhang et al. (RESS 2021):

  1. DIAGNOSIS TREE  -- root = observed outcome -> candidate causes P(cause|outcome).
     Validated against Zhang's Table 7 (faithful cause/factor mode reproduces the
     PUBLISHED fire-cause cells exactly) + spot-checked on non-fire outcomes.
  2. PROGNOSIS TREE  -- root = initial event -> branching forward escalation
     P(next|current), now with: terminal damage/injury LEAF outcomes, deep-chain
     GLOBAL backoff, and optional Zhang Beta-CDF multiparent BN posteriors.
     Validated against Zhang's Table 9 forward cells.

Both trees are also exported to JSON (outputs/) for the Streamlit visualization.

Run (framework Python 3.11 + network for the query-first embedding retrieval):
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 tests/tree_demo.py
  ... --query "what caused the landing gear to collapse?"   # try another outcome
  ... --no-prognosis        # diagnosis tree only
  ... --semantic            # add semantic smoothing to sparse prognosis hops

Exit code 0 only if every sanity + Zhang-validation gate passes.
"""
from __future__ import annotations

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
import trees  # noqa: E402
import config  # noqa: E402
import zhang_diagnosis as zd  # noqa: E402
import prognosis as pg  # noqa: E402


def _get_arg(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# ===================================================================================
# Zhang reference values (paper page numbers in the comments)
# ===================================================================================
# Table 7 (p.12) -- published P(cause | fire) contributory-factor cells. Reproduced
# EXACTLY by the faithful cause/factor counting mode (denominator = 102 fire accidents).
TABLE7_PUBLISHED = {
    "airframe/component/system failure/malfunction": 0.31372,
    "electrical system, electric wiring": 0.08823,
    "loss of engine power (total) - mechanical failure/malfunction": 0.08823,
    "fluid, fuel": 0.05882,
    "auxiliary power unit (apu)": 0.04901,
    "procedure inadequate": 0.03921,
    "maintenance, installation": 0.03921,
    "loss of engine power (partial) - mechanical failure/malfunction": 0.03921,
    "engine compartment": 0.02941,
    "maintenance, service bulletin/letter": 0.02941,
    "cargo/baggage": 0.02941,
    "fuel system, drain": 0.01960,
    "fuel system, nozzle": 0.01960,
    "fuel system, fuel control": 0.01960,    # Eq.9 anchor (p.8): fuel control = 2/102
    "emergency procedure": 0.00980,
    "evacuation": 0.00980,
}
# Table 9 (p.17) / Eq.9 (p.8) -- forward escalation anchors as Zhang's single-parent
# edge ratio count(cause->outcome)/count(cause appears).  (cause, outcome, expected).
TABLE9_ANCHORS = [
    ("fluid, oil grade", "loss of engine power", 0.95),   # 1/1 capped (Zhang's 0.95 flagship)
    ("combustion assembly, combustion liner", "loss of engine power", 0.50),  # 1/2
    ("loss of engine power", "forced landing", 0.1429),   # 2/14 == Table 9 forced landing
]


def validate_table7(ds):
    """Faithful Table-7 reproduction: P(cause|fire) cause/factor mode == published."""
    res = zd.empirical_cause_distribution("fire", targets={"fire"}, dataset=ds,
                                          cause_factor_only=True)
    by = {c["cause"].lower(): c for c in res["causes"]}
    print(f"  fire denominator = {res['outcome_count']}  (Zhang: 102)")
    ok = (res["outcome_count"] == 102)
    miss = 0
    for lab, pub in sorted(TABLE7_PUBLISHED.items(), key=lambda x: -x[1]):
        got = by.get(lab, {}).get("probability")
        good = got is not None and abs(got - pub) < 0.0006
        miss += (not good)
        if not good:
            print(f"    MISMATCH {lab!r}: published {pub:.5f}, ours {got}")
    print(f"  Table 7 cells matched: {len(TABLE7_PUBLISHED) - miss}/{len(TABLE7_PUBLISHED)}")
    return ok and miss == 0


def validate_table9(ds):
    """Zhang forward edge-ratio anchors (Table 9 / Eq.9) reproduced exactly."""
    edge_events, node_events = pg.build_graph(ds)
    ok = True
    for cause, outcome, exp in TABLE9_ANCHORS:
        targets = pg.resolve_outcome_targets(outcome, ds)
        r = pg.zhang_baseline_cpt(cause, targets, edge_events, node_events)
        got = r["value"]
        good = got is not None and abs(got - exp) < 0.005
        ok = ok and good
        print(f"  {'OK ' if good else 'XX '} P({outcome} | {cause[:34]:34}) = "
              f"{got if got is None else round(got, 4)}  (Zhang {exp})  "
              f"[{r['joint_n']}/{r['denom_n']}{', capped' if r['capped'] else ''}]")
    return ok


def validate_diag_tree_matches_table7(ds):
    """The diagnosis tree's level-1 edges (faithful mode, full population) ARE the
    Table 7 distribution -- confirm via _causes_list over the whole dataset."""
    dist, denom = trees._causes_list("fire", {"fire"}, ds, list(ds), set(), True,
                                     cause_factor_only=True)
    by = {c["label"].lower(): c for c in dist}
    anchors = ["electrical system, electric wiring",
               "loss of engine power (total) - mechanical failure/malfunction",
               "fluid, fuel", "auxiliary power unit (apu)"]
    ok = (denom == 102)
    for a in anchors:
        c = by.get(a)
        good = c and abs(c["prob"] - TABLE7_PUBLISHED[a]) < 0.0006
        ok = ok and bool(good)
        print(f"    L1 edge {a[:48]:48} p={c['prob']:.5f} n={c['n']}" if c
              else f"    L1 edge {a!r} MISSING")
    return ok


def main():
    query = _get_arg("--query", "engine caught fire during takeoff")
    do_prognosis = "--no-prognosis" not in sys.argv
    semantic = "--semantic" in sys.argv
    out_dir = config.OUTPUT_DIR

    section(f"QUERY-FIRST INPUT:  {query!r}")
    print("Every tree below is built by embedding this narrative query, retrieving the\n"
          "most similar historical incidents, then growing the tree over that pool.")

    import main_app  # noqa: E402  (loads the embedding index; needs network)
    ds = main_app.refined_dataset

    # ----------------------------------------------------------------------------- #
    section("1. DIAGNOSIS TREE  (root = observed outcome -> branching causes)")
    diag = trees.build_diagnosis_tree(
        query, top_n_incidents=300, branching=4, depth=2, min_prob=0.03,
    )
    print(trees.render_tree(diag))
    diag_path = trees.export_json(diag, out_dir / "diagnosis_tree.json")
    print(f"\n[JSON exported -> {diag_path}]")

    # Faithful Table-7 variant (cause/factor mode), same query.
    section("1b. DIAGNOSIS TREE  (faithful Zhang Table-7 cause/factor mode)")
    diag_faithful = trees.build_diagnosis_tree(
        query, top_n_incidents=300, branching=5, depth=1, min_prob=0.0,
        cause_factor_only=True,
    )
    print(trees.render_tree(diag_faithful))

    # ----------------------------------------------------------------------------- #
    prog = None
    if do_prognosis:
        section("2. PROGNOSIS TREE  (query-relevant + leaf outcomes + deep backoff)")
        prog = trees.build_prognosis_tree(
            query, top_n_incidents=300, branching=3, depth=3, min_prob=0.05,
            min_n=2, query_relevant=True, semantic_smoothing=semantic,
            add_outcome_leaves=True, deep_backoff=True,
        )
        print(trees.render_tree(prog))
        prog_path = trees.export_json(prog, out_dir / "prognosis_tree.json")
        print(f"\n[JSON exported -> {prog_path}]")

        # Global-population variant with Beta-CDF multiparent BN posteriors enabled.
        section("2b. PROGNOSIS TREE  (global pop + Zhang Beta-CDF BN posteriors)")
        prog_g = trees.build_prognosis_tree(
            query, branching=3, depth=3, min_prob=0.05, min_n=3,
            query_relevant=False, add_outcome_leaves=True, deep_backoff=True,
            bn_posteriors=True, drop_generic=True,
        )
        print(trees.render_tree(prog_g))

    # ----------------------------------------------------------------------------- #
    section("3. ZHANG VALIDATION  (diagnosis Table 7 + prognosis Table 9)")
    print("Table 7 -- P(cause | fire), faithful cause/factor mode vs PUBLISHED:")
    t7 = validate_table7(ds)
    print("\nDiagnosis tree level-1 edges (full population) == Table 7:")
    t7tree = validate_diag_tree_matches_table7(ds)
    print("\nTable 9 / Eq.9 -- forward escalation edge ratios vs Zhang:")
    t9 = validate_table9(ds)

    # ----------------------------------------------------------------------------- #
    section("4. SANITY CHECKS")
    ok = _sanity(diag["tree"]) if diag["tree"] else True
    ok = (_sanity(diag_faithful["tree"]) and ok) if diag_faithful["tree"] else ok
    if do_prognosis and prog and prog["tree"]:
        ok = _sanity(prog["tree"]) and ok
    print("  edge_prob in [0,1], path_prob == product down path, n<=denom: "
          + ("PASS" if ok else "FAIL"))
    print(f"  Table 7 faithful reproduction:        {'PASS' if t7 else 'FAIL'}")
    print(f"  Diagnosis tree L1 == Table 7:         {'PASS' if t7tree else 'FAIL'}")
    print(f"  Table 9 forward-edge reproduction:    {'PASS' if t9 else 'FAIL'}")

    all_ok = ok and t7 and t7tree and t9
    print("\nDONE." if all_ok else "\nDONE (with FAILURES).")
    return 0 if all_ok else 2


def _sanity(node, parent_path=1.0):
    ok = True
    expected = parent_path * node["edge_prob"]
    if abs(expected - node["path_prob"]) > 1e-9:
        print(f"  FAIL path_prob: {node['label']!r} {node['path_prob']} != {expected}")
        ok = False
    if not (0.0 <= node["edge_prob"] <= 1.0 + 1e-9):
        print(f"  FAIL edge_prob out of range: {node['label']!r} {node['edge_prob']}")
        ok = False
    if node.get("n") is not None and node.get("denom"):
        if node["n"] > node["denom"]:
            print(f"  FAIL n>denom: {node['label']!r} {node['n']}>{node['denom']}")
            ok = False
    for c in node["children"]:
        ok = _sanity(c, node["path_prob"]) and ok
    return ok


if __name__ == "__main__":
    raise SystemExit(main())

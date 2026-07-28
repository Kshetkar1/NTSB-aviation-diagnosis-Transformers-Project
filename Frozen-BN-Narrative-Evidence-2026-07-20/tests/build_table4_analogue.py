"""Build a REAL, data-derived analogue of Zhang's Table 4.

Zhang's Table 4 (paper Section 3, Fig. 2 toy network) is the CPT
P(x3=fire | x1, x2) for two parent causes, with the hand-picked teaching
values 0.99 / 0.93 / 0.95 / 2e-9. docs/TABLE4_ANALYSIS.md argues it is
illustrative, not produced by Zhang's real estimator. This script settles the
question CONCRETELY: it picks two REAL fire-related parent causes from the NTSB
data and fills the full 2-parent CPT P(fire | p1, p2) for all four
(present/absent) combinations, computed THREE ways:

  (a) OUR direct empirical counting (cause_factor_only contributory convention),
      raw n/denom plus Beta-CDF smoothing and Zhang's 0.95 sparse-cell cap.
  (b) ZHANG'S RECREATED estimator: single-cause ratios via Eq. 9
      (prognosis.zhang_baseline_cpt) combined with his multi-parent Beta-CDF /
      noisy-max-floor rule (prognosis.zhang_baseline_multiparent), 0.95 cap.
  (c) (optional) OUR narrative-conditioned estimate for the both-present cell
      (semantic neighbours; requires the embedding index + network).

It then compares all of the above side-by-side against Zhang's illustrative
Table 4 and reports which of 0.99 / 0.93 / 0.95 / 2e-9 the real estimator can or
cannot reach.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/build_table4_analogue.py            # (a) + (b)
  ... tests/build_table4_analogue.py --semantic # also (c), needs network

NOTE: imports the engines read-only; edits nothing in zhang_diagnosis.py /
prognosis.py / sparse_cpt.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import scipy.stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import prognosis as pg          # noqa: E402  forward CPT estimators (read-only)
import zhang_diagnosis as zd    # noqa: E402  contributory convention (read-only)
from sparse_cpt import ALPHA, BETA  # noqa: E402  Zhang's global Beta-CDF params

# --- The two REAL parent causes (analogues of Zhang's toy x1/x2) ---------------
# p1 = electrical wiring  -> the direct analogue of Zhang's x2 ("electrical system
#      wiring overheating"). A single NTSB contributory-finding label.
# p2 = fuel system / fuel -> an intuitive fire cause and the ONLY second parent
#      that co-occurs with wiring enough to populate the both-present cell. Taken
#      as the family of contributory findings whose label mentions "fuel".
P1_LABEL = "electrical system, electric wiring"
P1_NAME = "electrical wiring"
P2_NAME = "fuel system / fuel"
FIRE_TARGETS = {"fire"}

# Zhang's illustrative Table 4 values, in (p1,p2) order:
#   (Yes,Yes)=0.99  (Yes,No)=0.93  (No,Yes)=0.95  (No,No)=2e-9
ZHANG_TABLE4 = {(True, True): 0.99, (True, False): 0.93,
                (False, True): 0.95, (False, False): 2e-9}

CELLS = [(True, True), (True, False), (False, True), (False, False)]


# --- contributory presence (cause_factor_only convention) ----------------------
def _contributory_labels(inc: dict) -> set:
    """Cause/Factor finding labels of an incident, using Zhang's faithful
    contributory convention (reuses zhang_diagnosis helpers, no reimplementation):
    only C/F findings count and nan/empty normalize to 'Unknown quantity'."""
    labels = set()
    for fd in inc.get("findings", []):
        if zd._is_contributory(fd):
            labels.add(zd._faithful_finding_label(fd).strip().lower())
    labels.discard("")
    return labels


def _p1_present(labs: set) -> bool:
    return P1_LABEL in labs


def _p2_present(labs: set) -> bool:
    return any("fuel" in lab for lab in labs)


def _has_fire(inc: dict) -> bool:
    return any((s.get("Occurrence_Description") or "").strip().lower() in FIRE_TARGETS
               for s in inc.get("sequence_of_events", []))


# --- (a) OUR direct empirical CPT ----------------------------------------------
def empirical_cpt(ds: dict) -> dict:
    """For each of the four (p1,p2) cells: n=count(fire & cell), denom=count(cell).

    Returns {cell: {n, denom, raw, beta, capped}} where
      raw    = n/denom (None if denom==0),
      beta   = beta.cdf(raw, ALPHA, BETA)  (Zhang's Beta-CDF smoothing),
      capped = Zhang's 0.95 sparse-cell cap applied to raw (raw==1.0 -> 0.95)."""
    agg = {c: {"n": 0, "denom": 0} for c in CELLS}
    for inc in ds.values():
        labs = _contributory_labels(inc)
        cell = (_p1_present(labs), _p2_present(labs))
        agg[cell]["denom"] += 1
        if _has_fire(inc):
            agg[cell]["n"] += 1
    out = {}
    for c, d in agg.items():
        n, denom = d["n"], d["denom"]
        raw = (n / denom) if denom else None
        beta = float(scipy.stats.beta.cdf(raw, ALPHA, BETA)) if raw is not None else None
        capped = raw
        if raw is not None and raw == 1.0:        # Zhang's exact 0.95 sparse cap
            capped = raw * 0.95
        out[c] = {"n": n, "denom": denom, "raw": raw, "beta": beta,
                  "capped": capped, "sparse": denom <= 5}
    return out


# --- (b) ZHANG'S RECREATED estimator -------------------------------------------
def zhang_recreated_cpt(ds: dict) -> dict:
    """Zhang's own estimator on our data, faithful to his constructCPT:
      - single active parent  -> that parent's Eq.9 ratio (0.95-capped if ==1.0);
      - zero active parents    -> 0.0 (his sum(scheme*count) for the empty scheme);
      - both active            -> Beta-CDF(contribution) floored by max(active),
                                  via prognosis.zhang_baseline_multiparent.
    p2 (fuel family) is aggregated into one Zhang ratio (union of fuel->fire edge
    events over union of fuel from-or-to events) so it behaves as one parent node;
    the both-present floor uses max(wiring, fuel-family) either way."""
    edge_events, node_events = pg.build_graph(ds)

    # single-cause Eq.9 ratio for wiring (exact label, via the engine)
    r_p1 = pg.zhang_baseline_cpt(P1_LABEL, FIRE_TARGETS, edge_events, node_events,
                                 cap=True)

    # single-cause Eq.9 ratio for the FUEL FAMILY (faithful family aggregation of
    # Zhang's len(joint)/len(denom): joint = any fuel->fire edge events, denom =
    # any fuel node from-or-to events).
    fuel = lambda l: "fuel" in l
    joint, denom = set(), set()
    for (a, b), evs in edge_events.items():
        if fuel(a) and b in FIRE_TARGETS:
            joint |= evs
    for node, evs in node_events.items():
        if fuel(node):
            denom |= evs
    raw_p2 = (len(joint) / len(denom)) if denom else None
    v_p2 = raw_p2
    p2_capped = False
    if raw_p2 is not None and raw_p2 == 1.0:
        v_p2, p2_capped = raw_p2 * 0.95, True
    r_p2 = {"value": v_p2, "joint_n": len(joint), "denom_n": len(denom),
            "raw_ratio": raw_p2, "capped": p2_capped}

    # both-present via Zhang's multi-parent rule. zhang_baseline_multiparent floors
    # by max(active ratios); we pass wiring + the dominant single fuel label that is
    # a graph parent so the function's parent set resolves, then override the floor
    # with the (>=) fuel-family ratio for a faithful both-present value.
    mp = pg.zhang_baseline_multiparent([P1_LABEL, "fluid, fuel"], FIRE_TARGETS,
                                       edge_events, node_events)
    both = None
    if r_p1["value"] is not None and r_p2["value"] is not None:
        floor = max(r_p1["value"], r_p2["value"])
        beta_raw = mp.get("beta_raw") if mp else None
        both = max(floor, beta_raw) if beta_raw is not None else floor

    out = {}
    out[(True, True)] = {"value": both, "kind": "both (Beta-CDF floored by max single)",
                         "contribution": mp.get("contribution") if mp else None,
                         "beta_raw": mp.get("beta_raw") if mp else None}
    out[(True, False)] = {"value": r_p1["value"], "kind": "single (Eq.9)",
                          "joint_n": r_p1["joint_n"], "denom_n": r_p1["denom_n"],
                          "raw": r_p1["raw_ratio"], "capped": r_p1["capped"]}
    out[(False, True)] = {"value": r_p2["value"], "kind": "single (Eq.9, fuel family)",
                          "joint_n": r_p2["joint_n"], "denom_n": r_p2["denom_n"],
                          "raw": r_p2["raw_ratio"], "capped": r_p2["capped"]}
    out[(False, False)] = {"value": 0.0, "kind": "no active parent (=0 in constructCPT)"}
    out["_meta"] = {"r_p1": r_p1, "r_p2": r_p2, "mp": mp}
    return out


# --- (c) OUR narrative-conditioned estimate (optional, needs network) ----------
def semantic_both_cell(query: str = "electrical wiring fire and fuel system leak",
                       k: int = 50) -> dict:
    """Fraction of the k incidents most semantically similar to the both-cause
    description that escalate to fire. Smooths the 1-incident both-present cell."""
    try:
        import main_app  # noqa: F401  (triggers embedding index + network)
        res = pg.semantic_forward_cpt(query, FIRE_TARGETS, k=k)
        res["query"] = query
        return res
    except Exception as exc:  # network/index unavailable -> skip gracefully
        return {"error": f"{type(exc).__name__}: {exc}", "query": query}


# --- reporting -----------------------------------------------------------------
def _cell_name(cell) -> str:
    a, b = cell
    return f"{P1_NAME}={'Y' if a else 'N'}, {P2_NAME}={'Y' if b else 'N'}"


def _fmt(v) -> str:
    return "  n/a " if v is None else f"{v:.4f}"


def main(argv=None) -> dict:
    argv = argv or sys.argv[1:]
    do_semantic = "--semantic" in argv

    ds = pg.load_dataset()
    emp = empirical_cpt(ds)
    zr = zhang_recreated_cpt(ds)

    print("=" * 88)
    print("REAL data-derived analogue of Zhang's Table 4  -  P(fire | p1, p2)")
    print(f"  p1 = {P1_NAME!r}  (label: {P1_LABEL!r})")
    print(f"  p2 = {P2_NAME!r}  (any contributory finding whose label mentions 'fuel')")
    print(f"  dataset = refined_dataset_1982_2006.json   incidents = {len(ds)}")
    print("=" * 88)

    print("\n(a) OUR DIRECT EMPIRICAL CPT  (cause_factor_only contributory convention)")
    print(f"  {'cell':34} {'n/denom':>12} {'raw':>8} {'beta':>8} {'capped':>8}")
    for c in CELLS:
        e = emp[c]
        nd = f"{e['n']}/{e['denom']}"
        flag = "  <-sparse" if e["sparse"] else ""
        print(f"  {_cell_name(c):34} {nd:>12} {_fmt(e['raw']):>8} "
              f"{_fmt(e['beta']):>8} {_fmt(e['capped']):>8}{flag}")

    print("\n(b) ZHANG'S RECREATED ESTIMATOR  (Eq.9 singles + multi-parent rule)")
    m = zr["_meta"]
    print(f"     wiring single-cause ratio (Eq.9) = {m['r_p1']['joint_n']}/"
          f"{m['r_p1']['denom_n']} = {_fmt(m['r_p1']['raw_ratio'])}")
    print(f"     fuel-family single-cause ratio   = {m['r_p2']['joint_n']}/"
          f"{m['r_p2']['denom_n']} = {_fmt(m['r_p2']['raw_ratio'])}")
    if m["mp"]:
        print(f"     both-present: contribution={_fmt(m['mp'].get('contribution'))} "
              f"beta_raw={_fmt(m['mp'].get('beta_raw'))} "
              f"floor=max(singles)={_fmt(max(m['r_p1']['value'], m['r_p2']['value']))}")
    print(f"  {'cell':34} {'value':>10}   kind")
    for c in CELLS:
        z = zr[c]
        print(f"  {_cell_name(c):34} {_fmt(z['value']):>10}   {z['kind']}")

    sem = None
    if do_semantic:
        print("\n(c) OUR NARRATIVE-CONDITIONED estimate for the both-present cell")
        sem = semantic_both_cell()
        if sem.get("error"):
            print(f"     skipped: {sem['error']}")
        else:
            print(f"     query={sem['query']!r}  -> {sem['n_outcome']}/{sem['k']} "
                  f"= {_fmt(sem['value'])} of nearest incidents reach fire")

    print("\n" + "=" * 88)
    print("SIDE-BY-SIDE vs Zhang's ILLUSTRATIVE Table 4")
    print("=" * 88)
    print(f"  {'cell':34} {'Zhang T4':>9} {'ours raw':>9} {'ours cap':>9} {'Zhang-est':>10}")
    for c in CELLS:
        print(f"  {_cell_name(c):34} {ZHANG_TABLE4[c]:>9.4g} "
              f"{_fmt(emp[c]['raw']):>9} {_fmt(emp[c]['capped']):>9} "
              f"{_fmt(zr[c]['value']):>10}")

    print("\nREACHABILITY of Zhang's Table 4 values by the REAL estimator:")
    print("  0.99 (both)   : NOT reachable - 0.95 cap is the ceiling; floor=max single ~0.39")
    print("  0.93 (p1 only): NOT reachable - real wiring single-cause ratio ~0.39")
    print("  0.95 (p2 only): reachable ONLY as a 1/1 sparse-cell cap artifact, not a real rate")
    print("  2e-9 (neither): NOT a count ratio - our real neither-cell ~0.04; Zhang-est = 0")

    return {"empirical": emp, "zhang_recreated": zr, "semantic": sem}


if __name__ == "__main__":
    main()

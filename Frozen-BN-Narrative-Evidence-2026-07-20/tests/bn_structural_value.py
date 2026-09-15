#!/usr/bin/env python3
"""What does the frozen network add that retrieval cannot? Two testable claims.

This replaces the counterfactual/attribution demos in bn_value_demos.py, which
do not survive scrutiny: the severity CPT is a support-weighted mixture of the
active parents' empirical marginals, so adding or removing a parent shifts the
average mechanically rather than isolating a causal effect (see D11).

The two claims tested here do NOT depend on the CPT behaving causally.

CLAIM 1 - Compositional coverage.
  Retrieval can only answer by copying outcomes from accidents that exist.
  The network composes an answer for any evidence pattern, including
  combinations never observed. Measured: how much of the 2^12 severity-parent
  pattern space the corpus actually covers.

CLAIM 2 - Joint over injury and damage.
  Injury and damage are conditionally independent GIVEN all 12 parents (no
  edge between them), so any dependence in the posterior comes from shared
  uncertainty over unobserved parents. Measured: how large that dependence
  actually gets, so we know whether the claim is worth making at all.

Run:
    python3 tests/bn_structural_value.py
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT.parent / "shared" / "code"))

import pyagrum as gum
import prognosis as pg
from bn_upgraded import (build_upgraded, INJ_NODE, DMG_NODE,
                         INJ_STATES, DMG_STATES, last_occurrence)

OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def active_parents(inc: dict, parents: list, parent_set: set) -> frozenset:
    """Which of the 12 severity parents this accident activates, per coded data."""
    act = set()
    for occ in inc.get("sequence_of_events", []) or []:
        d = str(occ.get("Occurrence_Description", "")).strip().lower()
        if d in parent_set:
            act.add(d)
    for f in inc.get("findings", []) or []:
        d = str(f.get("finding_description", "")).strip().lower()
        if d in parent_set:
            act.add(d)
    last = last_occurrence(inc)
    if last in parent_set:
        act.add(last)
    return frozenset(act)


print("Building network ...", flush=True)
ds = pg.load_dataset()
bn, meta = build_upgraded(ds)
parents = meta["severity_parents"]
pset = set(parents)
n_par = len(parents)
print(f"  {bn.size()} nodes, {bn.sizeArcs()} arcs")
print(f"  {n_par} severity parents\n")

report = {}

# ===========================================================================
# CLAIM 1 - Compositional coverage
# ===========================================================================
print("=" * 70)
print("CLAIM 1: Compositional coverage of the evidence space")
print("=" * 70)

patterns = Counter()
for inc in ds.values():
    patterns[active_parents(inc, parents, pset)] += 1

total_space = 2 ** n_par
observed = len(patterns)
print(f"\n  Severity-parent pattern space : 2^{n_par} = {total_space:,} combinations")
print(f"  Distinct patterns observed    : {observed:,}")
print(f"  Coverage                      : {observed / total_space:.4%}")
print(f"  Unobserved                    : {total_space - observed:,} "
      f"({1 - observed / total_space:.2%})")

sizes = Counter(len(p) for p in patterns)
print(f"\n  Observed patterns by number of active parents:")
for k in sorted(sizes):
    n_possible = len(list(itertools.combinations(range(n_par), k)))
    print(f"    {k} active : {sizes[k]:4d} distinct observed "
          f"of {n_possible:5d} possible  ({sizes[k]/n_possible:6.2%})")

singles = sum(c for p, c in patterns.items() if len(p) <= 1)
print(f"\n  Accidents whose pattern has <=1 active parent: {singles:,} "
      f"of {len(ds):,} ({singles/len(ds):.1%})")

# How many observed patterns are supported by a single accident?
rare = sum(1 for p, c in patterns.items() if c == 1)
print(f"  Observed patterns seen exactly once: {rare:,} of {observed:,} "
      f"({rare/observed:.1%})")

print(f"\n  -> Retrieval answers by copying outcomes from accidents that exist.")
print(f"     {1 - observed/total_space:.2%} of the pattern space has zero examples,")
print(f"     and {rare/observed:.0%} of the patterns that DO occur are supported by a")
print(f"     single accident. The network returns a coherent posterior for every")
print(f"     one of the {total_space:,} combinations by composing parent contributions.")

report["coverage"] = {
    "n_parents": n_par,
    "pattern_space": total_space,
    "observed_patterns": observed,
    "coverage_fraction": observed / total_space,
    "patterns_seen_once": rare,
    "accidents_with_le1_parent": singles,
    "n_accidents": len(ds),
}

# ===========================================================================
# CLAIM 2 - How much does the joint actually add?
# ===========================================================================
print()
print("=" * 70)
print("CLAIM 2: Dependence between injury and damage in the posterior")
print("=" * 70)
print("\n  Injury and damage share all 12 parents and have no edge between them,")
print("  so they are conditionally independent GIVEN the full parent set. Any")
print("  dependence in the posterior comes from uncertainty over UNOBSERVED")
print("  parents. Measuring how large it gets:\n")


def joint_vs_product(evidence: dict):
    """Return (max_ratio, total_variation) between the true joint and the
    product of the marginals, for P(injury, damage | evidence)."""
    ie = gum.LazyPropagation(bn)
    if evidence:
        ie.setEvidence(evidence)
    ie.addJointTarget({INJ_NODE, DMG_NODE})
    ie.makeInference()
    joint = ie.jointPosterior({INJ_NODE, DMG_NODE})
    pi = ie.posterior(INJ_NODE)
    pd = ie.posterior(DMG_NODE)

    max_ratio, tv = 1.0, 0.0
    for i in range(len(INJ_STATES)):
        for d in range(len(DMG_STATES)):
            pj = float(joint[{INJ_NODE: i, DMG_NODE: d}])
            pp = float(pi[i]) * float(pd[d])
            tv += abs(pj - pp)
            if pp > 1e-9:
                max_ratio = max(max_ratio, pj / pp)
    return max_ratio, tv / 2.0


# test across the most common observed patterns
common = [p for p, _ in patterns.most_common(25) if len(p) > 0]
rows = []
for pat in common:
    ev = {p: "Yes" for p in pat}
    mr, tv = joint_vs_product(ev)
    rows.append((sorted(pat), patterns[pat], mr, tv))

rows.sort(key=lambda r: -r[3])
print(f"  {'n active':>9} {'support':>8} {'max ratio':>10} {'total var':>10}   pattern")
print(f"  {'-'*9} {'-'*8} {'-'*10} {'-'*10}   {'-'*30}")
for pat, sup, mr, tv in rows[:12]:
    label = ", ".join(s[:28] for s in pat[:2])
    if len(pat) > 2:
        label += f", +{len(pat)-2} more"
    print(f"  {len(pat):9d} {sup:8d} {mr:10.4f} {tv:10.5f}   {label}")

tvs = [r[3] for r in rows]
mrs = [r[2] for r in rows]
print(f"\n  Across {len(rows)} common patterns:")
print(f"    total variation from independence : max {max(tvs):.5f}, "
      f"mean {np.mean(tvs):.5f}")
print(f"    max joint/product ratio           : max {max(mrs):.4f}")

report["joint_dependence"] = {
    "n_patterns_tested": len(rows),
    "max_total_variation": float(max(tvs)),
    "mean_total_variation": float(np.mean(tvs)),
    "max_ratio": float(max(mrs)),
}

if max(tvs) < 0.02:
    print(f"\n  -> VERDICT: the dependence is negligible (max TV {max(tvs):.4f}).")
    print(f"     Do NOT claim the joint distribution as a contribution. Injury and")
    print(f"     damage are conditionally independent by construction and the")
    print(f"     residual dependence from unobserved parents is too small to matter.")
else:
    print(f"\n  -> VERDICT: dependence is measurable (max TV {max(tvs):.4f}).")
    print(f"     The joint is defensible, but report the magnitude honestly.")

# ===========================================================================
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
  CLAIM 1 (coverage)  : {observed:,} of {total_space:,} patterns observed
                        ({observed/total_space:.3%}). {rare/observed:.0%} of observed patterns
                        rest on a single accident. DEFENSIBLE.

  CLAIM 2 (joint)     : max total variation from independence
                        {max(tvs):.5f}. {'TOO SMALL - DROP IT.' if max(tvs) < 0.02 else 'Defensible, report magnitude.'}

  Plus, from the held-out run (not measured here): 0 of 296 post-2007
  accidents carry a label in the network's vocabulary, so the frozen
  network + LLM mapper is the only route that spans the 2007 CICTT
  recoding. That is a capability claim, not an accuracy claim, and it
  does not depend on the CPT at all.
""")

out = OUT_DIR / "bn_structural_value.json"
out.write_text(json.dumps(report, indent=2))
print(f"  wrote {out.relative_to(ROOT)}")

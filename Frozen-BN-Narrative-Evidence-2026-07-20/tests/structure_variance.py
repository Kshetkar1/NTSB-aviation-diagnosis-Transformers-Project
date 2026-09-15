#!/usr/bin/env python3
"""Structure variance: what happens when the graph is learned instead of fixed?

This answers the question Maha asked directly and the project never tested.
Everything so far varies the EVIDENCE against one frozen graph. This varies
the GRAPH.

Three questions, in increasing order of how much they hurt:

  Q1  Does data-driven structure learning, run on the same 1,742 accidents
      Zhang's network was built from, recover Zhang's 12 severity parents?

  Q2  How stable is any learned edge? The network has 785 nodes and 1,742
      accidents -- roughly one accident per variable. Bootstrap-resample the
      accidents, relearn, and count how often each edge reappears. An edge
      that survives 95% of resamples is real; one that survives 40% is noise
      dressed as a causal claim.

  Q3  Does severity prediction actually change when the structure changes?
      Refit CPTs on each learned structure and re-predict. If predictions are
      stable across structures, the frozen choice is defensible even if
      individual edges are not. If they swing, the frozen graph is doing
      unearned work.

Scope: structure learning over the full 785 nodes at n = 1,742 is not
identifiable, and pretending otherwise would be the exact error the sparsity
critique points at. We learn over a reduced set -- the two severity nodes,
their 12 frozen parents, and the next N most frequent nodes -- and say so.
That is a local structure claim, which is the only honest one available here.

Run:
    python3 tests/structure_variance.py [--boot B] [--extra N]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyagrum as gum

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT.parent / "export_for_review" / "accident_variable_matrix.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

INJ, DMG = "personnel injury", "aircraft damage"

ZHANG_PARENTS = [
    "loss of engine power (total) - mechanical failure/malfunction",
    "hard landing",
    "in flight collision with terrain/water",
    "in flight encounter with weather",
    "in flight collision with object",
    "near collision between aircraft",
    "fire",
    "on ground/water collision with object",
    "on ground/water collision with terrain/water",
    "airframe/component/system failure/malfunction",
    "dragged wing, rotor, pod, float or tail/skid",
    "miscellaneous/other",
]


MAX_INDEGREE = 4   # Zhang's severity nodes take 12 parents; see MAX_PARENTS
                   # note in the report -- 12 binary parents is 4,096 cells
                   # against 1,742 accidents, so learning is capped lower.


def learn(df: pd.DataFrame, algo: str = "hc"):
    """Learn a BN structure from a binary accident x variable frame.

    A smoothing prior is required, not optional: without it pyAgrum aborts on
    conditioning sets that never occur, which at 1,742 accidents over this
    many variables happens immediately. That failure is itself the sparsity
    finding, so it is reported rather than hidden.
    """
    learner = gum.BNLearner(df)
    learner.useSmoothingPrior(1.0)
    learner.useScoreBIC()
    learner.setMaxIndegree(MAX_INDEGREE)
    if algo == "miic":
        learner.useMIIC()
    else:
        learner.useGreedyHillClimbing()
    return learner.learnBN()


def edge_set(bn):
    return {(bn.variable(a).name(), bn.variable(b).name())
            for a, b in bn.arcs()}


def skeleton(bn):
    """Undirected edges -- direction is often unidentifiable from data."""
    return {frozenset(e) for e in edge_set(bn)}


def main():
    boot = 200
    extra = 60
    if "--boot" in sys.argv:
        boot = int(sys.argv[sys.argv.index("--boot") + 1])
    if "--extra" in sys.argv:
        extra = int(sys.argv[sys.argv.index("--extra") + 1])

    print("Loading accident x variable matrix ...", flush=True)
    df = pd.read_csv(MATRIX)
    df = df.drop(columns=["ev_id"])
    print(f"  {df.shape[0]} accidents x {df.shape[1]} variables")

    # ── reduced variable set ───────────────────────────────────────────────
    event_cols = [c for c in df.columns if c not in (INJ, DMG)]
    freq = df[event_cols].sum().sort_values(ascending=False)
    keep_extra = [c for c in freq.index if c not in ZHANG_PARENTS][:extra]
    cols = [INJ, DMG] + ZHANG_PARENTS + keep_extra
    sub = df[cols].copy()
    # drop variables with no variation -- they carry no structural signal
    sub = sub.loc[:, sub.nunique() > 1]
    for c in sub.columns:
        sub[c] = sub[c].astype(str)
    print(f"  reduced set: {sub.shape[1]} variables "
          f"({len(ZHANG_PARENTS)} Zhang severity parents + "
          f"{sub.shape[1]-len(ZHANG_PARENTS)-2} frequent others + 2 severity)")
    print(f"  severity states: injury {df[INJ].nunique()}, "
          f"damage {df[DMG].nunique()}")

    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    # ── Q1: does learning recover Zhang's parents? ─────────────────────────
    say("\n" + "=" * 74)
    say("Q1 - DOES DATA-DRIVEN LEARNING RECOVER ZHANG'S SEVERITY PARENTS?")
    say("=" * 74)

    results = {}
    for algo in ("hc", "miic"):
        try:
            bn = learn(sub, algo)
        except Exception as e:
            say(f"\n  {algo}: FAILED ({e.__class__.__name__}: {e})")
            continue
        say(f"\n  {algo.upper()}  ({bn.sizeArcs()} arcs over "
            f"{bn.size()} variables)")
        for node in (INJ, DMG):
            nid = bn.idFromName(node)
            learned = {bn.variable(p).name() for p in bn.parents(nid)}
            kids = {bn.variable(c).name() for c in bn.children(nid)}
            # direction is often arbitrary; count adjacency
            adj = learned | kids
            hit = adj & set(ZHANG_PARENTS)
            say(f"    {node}:")
            say(f"      adjacent to {len(adj)} variables; "
                f"{len(hit)} of Zhang's 12 parents recovered")
            missed = [p for p in ZHANG_PARENTS if p not in adj]
            if missed:
                say(f"      NOT recovered ({len(missed)}): "
                    + "; ".join(m[:44] for m in missed[:6])
                    + ("; ..." if len(missed) > 6 else ""))
            novel = sorted(adj - set(ZHANG_PARENTS))
            if novel:
                say(f"      edges Zhang does NOT have ({len(novel)}): "
                    + "; ".join(n[:44] for n in novel[:6])
                    + ("; ..." if len(novel) > 6 else ""))
            results.setdefault(algo, {})[node] = {
                "n_adjacent": len(adj), "n_zhang_recovered": len(hit),
                "recovered": sorted(hit), "missed": missed, "novel": novel,
            }

    # ── Q2: bootstrap edge stability ───────────────────────────────────────
    say("\n" + "=" * 74)
    say(f"Q2 - EDGE STABILITY UNDER RESAMPLING ({boot} bootstrap replicates)")
    say("=" * 74)

    counts, sev_adj = Counter(), Counter()
    rng = np.random.default_rng(0)
    n = len(sub)
    ok = 0
    for b in range(boot):
        idx = rng.integers(0, n, n)
        rs = sub.iloc[idx].reset_index(drop=True)
        keep = rs.loc[:, rs.nunique() > 1]
        try:
            bb = learn(keep, "hc")
        except Exception:
            continue
        ok += 1
        for e in skeleton(bb):
            counts[e] += 1
        for node in (INJ, DMG):
            if node not in keep.columns:
                continue
            nid = bb.idFromName(node)
            for p in list(bb.parents(nid)) + list(bb.children(nid)):
                sev_adj[(node, bb.variable(p).name())] += 1
        if (b + 1) % 25 == 0:
            print(f"    [{b+1}/{boot}] replicates", flush=True)

    say(f"\n  {ok}/{boot} replicates learned successfully")
    if ok:
        f = np.array([c / ok for c in counts.values()])
        say(f"  {len(counts)} distinct undirected edges appeared at least once")
        say(f"  edges present in >=95% of replicates: {int((f>=.95).sum())}")
        say(f"  edges present in >=50% of replicates: {int((f>=.50).sum())}")
        say(f"  edges present in < 50% of replicates: {int((f<.50).sum())} "
            f"({(f<.50).mean():.0%} of everything ever learned)")

        say(f"\n  Severity adjacency stability "
            f"(how often each variable is linked to the outcome):")
        for node in (INJ, DMG):
            rows = sorted(((v / ok, p) for (nd, p), v in sev_adj.items()
                           if nd == node), reverse=True)[:12]
            say(f"\n    {node}")
            say(f"      {'freq':>6}  {'in Zhang 12?':<13} variable")
            for fr, p in rows:
                mark = "yes" if p in ZHANG_PARENTS else "NO"
                say(f"      {fr:6.0%}  {mark:<13} {p[:46]}")
            zh = [(sev_adj.get((node, p), 0) / ok, p) for p in ZHANG_PARENTS]
            stable = sum(1 for fr, _ in zh if fr >= 0.5)
            say(f"      -> {stable} of Zhang's 12 parents appear in >=50% "
                f"of resamples")
            results.setdefault("bootstrap", {})[node] = {
                "zhang_parent_stability": {p: fr for fr, p in zh},
                "n_stable_of_12": stable,
            }

        results["bootstrap_summary"] = {
            "replicates_ok": ok, "n_edges_seen": len(counts),
            "frac_edges_below_50pct": float((f < .50).mean()),
            "n_edges_ge_95pct": int((f >= .95).sum()),
            "n_edges_ge_50pct": int((f >= .50).sum()),
        }

    say("\n" + "=" * 74)
    say("HOW TO READ THIS")
    say("=" * 74)
    say("""
  If learning recovers most of Zhang's 12 and the bootstrap keeps those
  edges above 95%, the frozen structure is corroborated by the data and you
  can say so -- that is a genuine defence of reusing it.

  If learning recovers few of them, or the stable-edge count is low, then
  the honest statement is that structure is NOT identifiable at 1,742
  accidents over 785 variables, and the frozen graph should be presented as
  an EXPERT-SPECIFIED prior (Zhang's taxonomy), not as a learned model. That
  is still defensible -- it is how most safety BNs are built -- but it must
  be claimed as expert structure rather than discovered structure.

  Either way the question is now answered with evidence instead of left open.
""")

    (OUT / "structure_variance.md").write_text(
        "# Structure variance\n\n```\n" + "\n".join(lines) + "\n```\n")
    (OUT / "structure_variance.json").write_text(json.dumps(
        results, indent=2, default=str))
    print(f"  wrote outputs/structure_variance.{{md,json}}")


if __name__ == "__main__":
    raise SystemExit(main())

"""D2 ablation: how are the severity nodes' parents chosen, and does it matter?

The shipped upgraded build makes two choices at once, both by raw frequency:

    parents = support.most_common()[:12]          # WHICH labels are parents
    dist    = sum(dists[active] * support) / ...  # HOW their distributions mix

so injury and damage receive the SAME twelve parents, and when several parents
are active the most COMMON one dominates the mixture regardless of how much it
says about the target. Frequency is not informativeness, and injury and damage
are not the same question.

This script rebuilds the severity layer three ways and re-scores the held-out
cohort. Selection statistics come from the 1982-2006 build window only; the
2007-2019 cohort is used for scoring and nothing else.

    A  freq parents  + freq weights   (shipped)
    B  MI parents    + freq weights   (isolates parent SELECTION)
    C  MI parents    + MI weights     (selection + mixing)

Deviating from the frozen build is reported as an ablation, never folded into
the headline numbers.
"""
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyagrum as gum

os.environ.pop("NTSB_FULL_CORPUS", None)
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "shared/code", ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20/code",
          Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

import prognosis as pg                                          # noqa: E402
import query_to_bn as qb                                        # noqa: E402
import bn_build_ours as builder                                 # noqa: E402
from bn_upgraded import (build_edges_upgraded, severity_parent_stats,  # noqa: E402
                         last_occurrence, injury_state, damage_state,
                         INJ_NODE, DMG_NODE, INJ_STATES, DMG_STATES,
                         MAX_PARENTS)

PROC = ROOT / "shared/data/processed"


def mutual_information(ds, labels, target_state, n_states):
    """I(X ; Y) in bits, X = [last occurrence is this label], Y = severity state.

    Same empirical basis the shipped CPTs use: one last-occurrence label and one
    severity state per accident.
    """
    rows = []
    for inc in ds.values():
        last = last_occurrence(inc)
        if not last:
            continue
        y = target_state(inc)
        if y is None:
            continue
        rows.append((last, y))
    n = len(rows)
    py = np.zeros(n_states)
    for _, y in rows:
        py[y] += 1
    py /= n

    joint = defaultdict(lambda: np.zeros(n_states))
    seen = Counter()
    for last, y in rows:
        joint[last][y] += 1
        seen[last] += 1

    out = {}
    for lab in labels:
        c1 = joint[lab]
        px1 = seen[lab] / n
        if px1 <= 0 or px1 >= 1:
            out[lab] = 0.0
            continue
        p1 = c1 / n                                  # P(X=1, Y=y)
        p0 = py - p1                                 # P(X=0, Y=y)
        mi = 0.0
        for p, px in ((p1, px1), (p0, 1 - px1)):
            for y in range(n_states):
                if p[y] > 0 and py[y] > 0:
                    mi += p[y] * np.log2(p[y] / (px * py[y]))
        out[lab] = float(mi)
    return out


def add_severity(bn, ds, mode, only=None):
    """Attach the severity node(s) under one of the three schemes.

    `only` restricts the build to a single target. That is not cosmetic: when
    injury and damage are given DIFFERENT parent sets, moralization links two
    disjoint sets of twelve nodes and the junction-tree clique exceeds memory,
    so exact inference dies before the first query. Scoring one target per
    network keeps every scheme at the same clique size as the shipped build,
    and matches `posteriors`, which already infers each target separately.
    """
    support, inj_counts, dmg_counts = severity_parent_stats(ds)
    cand = [lab for lab, _ in support.most_common() if lab in bn.names()]
    freq_parents = cand[:MAX_PARENTS]

    mi_inj = mutual_information(ds, cand, injury_state, 4)
    mi_dmg = mutual_information(ds, cand, damage_state, 4)

    specs = [
        (INJ_NODE, INJ_STATES, inj_counts, 3, mi_inj),
        (DMG_NODE, DMG_STATES, dmg_counts, 3, mi_dmg),
    ]
    if only:
        specs = [s for s in specs if s[0] == only]
    chosen = {}
    for node_name, states, counts, default_idx, mi in specs:
        if mode == "A":
            parents = freq_parents
        else:
            parents = sorted(cand, key=lambda l: -mi[l])[:MAX_PARENTS]
        chosen[node_name] = parents

        v = gum.LabelizedVariable(node_name, node_name, 0)
        for s in states:
            v.addLabel(s)
        bn.add(v)
        for p in parents:
            bn.addArc(p, node_name)

        dists, weights = [], []
        for p in parents:
            c = counts[p]
            tot = c.sum()
            dists.append(c / tot if tot > 0 else np.eye(4)[default_idx])
            weights.append(support[p] if mode in ("A", "B")
                           else max(mi[p], 1e-9) * support[p])
        dists = np.array(dists)
        weights = np.array(weights, dtype=float)

        cpt = bn.cpt(node_name)
        n = len(parents)
        default = np.eye(4)[default_idx]
        for mask in range(2 ** n):
            active = np.array([(mask >> i) & 1 for i in range(n)], dtype=bool)
            if not active.any():
                dist = default
            else:
                w = weights[active]
                dist = (dists[active] * w[:, None]).sum(axis=0) / w.sum()
                dist = dist / dist.sum()
            idx = {parents[i]: (0 if active[i] else 1) for i in range(n)}
            cpt[{**idx}] = list(dist)
    return chosen


def build(ds, mode, only=None):
    orig = pg.build_edges
    pg.build_edges = build_edges_upgraded
    try:
        bn, _ = builder.build_network(ds)
    finally:
        pg.build_edges = orig
    chosen = add_severity(bn, ds, mode, only=only)
    return bn, chosen


def posterior_one(bn, confidence, node, states):
    ie = gum.LazyPropagation(bn)
    if confidence:
        qb.apply_evidence(ie, bn, confidence)
    ie.addTarget(node)
    ie.makeInference()
    v = bn.variable(node)
    post = ie.posterior(node)
    by = {v.label(i): float(post[i]) for i in range(v.domainSize())}
    return np.array([by[s] for s in states])


def mcnemar(a, b):
    """Exact two-sided McNemar on paired correctness vectors."""
    from math import comb
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0:
        return 1.0, n10, n01
    k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return p, n10, n01


def main():
    ds = pg.load_dataset()
    full = json.loads((PROC / "refined_dataset.json").read_text())
    window = set(json.loads((PROC / "refined_dataset_1982_2006.json").read_text()))
    held = sorted((k, v) for k, v in full.items()
                  if k not in window
                  and len(str(v.get("narr_accf") or "").strip()) >= 100)
    print(f"held-out cohort: {len(held)} accidents\n")

    # Networks are held ONE AT A TIME. Three 785-node networks with 12-parent
    # CPTs resident together exhausts memory and the process is killed before
    # the first inference, so evidence is computed once up front and each
    # scheme is built, scored, and discarded in turn.
    import main_app
    import gc
    import time

    probe, _ = build(ds, "A")
    names = set(n for n in probe.names() if n not in (INJ_NODE, DMG_NODE))
    del probe
    gc.collect()

    print("pass 1 — parsing narratives and retrieving evidence", flush=True)
    t0 = time.time()
    cases = []
    for i, (k, inc) in enumerate(held, 1):
        if i % 25 == 0:
            el = time.time() - t0
            print(f"  {i:4d}/{len(held)}  elapsed {el/60:5.1f}m  "
                  f"eta {el/i*(len(held)-i)/60:5.1f}m", flush=True)
        text = qb.redact_severity_phrases(str(inc["narr_accf"])[:4000])
        hard = qb.parse_query_to_bn_evidence(text, names, dataset=ds,
                                             semantic=False)["confidence"]
        soft = {l: c for l, c, _ in
                qb.retrieval_facts(text, names, main_app.refined_dataset)}
        cases.append({"hard+soft": qb.merge_evidence_soft_priority(hard, soft),
                      "soft-only": soft,
                      "inj": injury_state(inc), "dmg": damage_state(inc)})
    print(f"pass 1 done in {(time.time()-t0)/60:.1f}m\n", flush=True)

    # Scheme C (MI parents + MI mixture weights) is dropped. Its damage network
    # thrashed memory on this machine -- 5.7 min per 100 accidents degrading to
    # 113 min per 100 -- and had to be killed. A and B answer the question.
    SCHEMES = ("A", "B")
    arms = ("hard+soft", "soft-only")
    correct = {(m, a, t): [] for m in SCHEMES for a in arms
               for t in ("injury", "damage")}
    dest = Path(__file__).resolve().parents[1] / "outputs/severity_parent_ablation.md"

    def flush_results(done):
        """Write partial results after every completed scheme/target.

        The first version of this script only wrote at the end, so two crashes
        and a kill destroyed every completed measurement. Never again.
        """
        lines = ["# D2 ablation — severity parent selection\n",
                 f"Cohort: {len(held)} held-out accidents (2007-2019). Selection "
                 "statistics from the 1982-2006 build window only.\n",
                 f"\nCompleted so far: {', '.join(done) if done else 'nothing'}\n",
                 "\n| scheme | arm | injury | damage |", "|---|---|---|---|"]
        for m in SCHEMES:
            for a in arms:
                vi, vd = correct[(m, a, "injury")], correct[(m, a, "damage")]
                fi = f"{100*np.mean(vi):.1f}%" if vi else "—"
                fd = f"{100*np.mean(vd):.1f}%" if vd else "—"
                lines.append(f"| {m} | {a} | {fi} | {fd} |")
        if all(correct[("A", a, t)] and correct[("B", a, t)]
               for a in arms for t in ("injury", "damage")):
            lines += ["\n## Exact McNemar, B vs shipped A\n",
                      "| arm | target | A only right | B only right | p |",
                      "|---|---|---|---|---|"]
            for a in arms:
                for t in ("injury", "damage"):
                    p, w, l = mcnemar(correct[("A", a, t)], correct[("B", a, t)])
                    lines.append(f"| {a} | {t} | {w} | {l} | {p:.4f}"
                                 f"{' *' if p < 0.05 else ''} |")
        lines.append("\nReference: retrieval severity bypass = 90.9% injury / "
                     "77.4% damage.\n")
        lines.append("This is an ABLATION. It deviates from the frozen build and "
                     "must not be folded into the headline numbers.\n")
        lines.append("Evidence for every scheme was parsed once, before scoring, "
                     "so all schemes see identical inputs.\n")
        dest.write_text("\n".join(lines))

    targets = [("injury", "inj", INJ_NODE, INJ_STATES),
               ("damage", "dmg", DMG_NODE, DMG_STATES)]
    done = []
    for mode in SCHEMES:
        tag = {"A": "SHIPPED freq parents + freq weights",
               "B": "MI parents + freq weights"}[mode]
        print(f"--- {mode}: {tag} ---", flush=True)
        for tname, key, node, states in targets:
            bn, chosen = build(ds, mode, only=node)
            print(f"  {tname:6s} parents: {', '.join(chosen[node])}", flush=True)
            t1 = time.time()
            for i, c in enumerate(cases, 1):
                if i % 100 == 0:
                    el = time.time() - t1
                    print(f"    {i:4d}/{len(cases)}  {el/60:4.1f}m  "
                          f"eta {el/i*(len(cases)-i)/60:4.1f}m", flush=True)
                if c[key] is None:
                    continue
                for a in arms:
                    p = posterior_one(bn, c[a], node, states)
                    correct[(mode, a, tname)].append(int(p.argmax()) == c[key])
            print(f"    {tname} scored in {(time.time()-t1)/60:.1f}m", flush=True)
            del bn
            gc.collect()
            done.append(f"{mode}/{tname}")
            flush_results(done)
            print(f"    partial results written to {dest.name}", flush=True)
        print("", flush=True)

    flush_results(done)
    print("\n" + dest.read_text())
    print(f"written to {dest}")


if __name__ == "__main__":
    main()

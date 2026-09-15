#!/usr/bin/env python3
"""Does the network earn its place on the accidents retrieval cannot reach?

Retrieval wins on average -- 90.9% / 77.4% against the BN-routed 89.2% / 55.4%.
But retrieval works by finding 100 similar past accidents and letting them
vote, so its accuracy is a function of whether similar accidents EXIST. The
network composes severity from named parts and needs no neighbours at all.

So the average is the wrong place to look. The question is whether the two
methods separate on the tail: accidents whose nearest neighbours are not
actually near. If the network closes the gap, or overtakes, where retrieval
runs out of neighbours, then it has a real and statable job -- and in aviation
safety the tail is the operating regime that matters.

If retrieval wins in every stratum, the network has no measurable value on
this data and the paper should say exactly that.

Method: score each held-out accident by the quality of its retrieval pool
(mean cosine similarity of the top-k neighbours), cut into quartiles, and
compare the already-saved predictions of the BN-routed arms against the
retrieval arm inside each quartile. No inference is re-run.

Run:
    python3 tests/neighbour_quality_strata.py [--limit N]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT.parent / "shared" / "code"))

import query_to_bn as qb

OUT = ROOT / "outputs"
PROC = ROOT.parent / "shared" / "data" / "processed"

RETRIEVAL_ARM = "retrieval-sev"
BN_ARMS = ["soft-only", "soft-priority", "hard+soft"]
TOP_K = 100


def neighbour_quality(text, dataset):
    """Mean / top-10 cosine similarity of the retrieval pool for one narrative."""
    import main_app
    emb = main_app.get_embedding(qb.inference_query(text, leak_safe=True))
    scores, matches = main_app.find_top_matches(emb)
    pool, seen = [], set()
    for s, m in zip(scores, matches):
        if m.get("source") != "incident":
            continue
        ev = m.get("ev_id")
        if not ev or ev in seen or ev not in dataset:
            continue
        seen.add(ev)
        pool.append(float(s))
        if len(pool) >= TOP_K:
            break
    if not pool:
        return None
    a = np.array(pool)
    return {"mean": float(a.mean()), "top1": float(a.max()),
            "top10": float(np.sort(a)[-10:].mean()), "n": len(a)}


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    import prognosis as pg
    ds = pg.load_dataset()
    import main_app  # noqa: F401  (loads the window retrieval index)

    per_item = json.loads((OUT / "heldout_per_item.json").read_text())
    items = {r["id"]: r for r in per_item["items"]}

    full = json.loads((PROC / "refined_dataset.json").read_text())
    window_ids = set(json.loads(
        (PROC / "refined_dataset_1982_2006.json").read_text()).keys())
    held = [(k, v) for k, v in full.items()
            if k not in window_ids
            and len(str(v.get("narr_accf") or "").strip()) >= 100]
    held.sort(key=lambda t: t[0])
    if limit:
        held = held[:limit]

    print(f"Scoring retrieval-pool quality for {len(held)} accidents ...",
          flush=True)
    qual = {}
    for i, (ev_id, inc) in enumerate(held):
        if ev_id not in items:
            continue
        narr = str(inc.get("narr_accf") or "")[:4000]
        q = neighbour_quality(qb.redact_severity_phrases(narr), ds)
        if q:
            qual[ev_id] = q
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(held)}]", flush=True)

    ids = sorted(qual)
    mq = np.array([qual[i]["mean"] for i in ids])
    print(f"\n  scored {len(ids)} accidents; "
          f"pool similarity mean {mq.mean():.3f}, "
          f"range [{mq.min():.3f}, {mq.max():.3f}]")

    cuts = np.quantile(mq, [0.25, 0.50, 0.75])
    strata = np.digitize(mq, cuts)  # 0 = worst neighbours, 3 = best

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say("\n" + "=" * 78)
    say("DOES THE NETWORK CLOSE THE GAP WHERE RETRIEVAL RUNS OUT OF NEIGHBOURS?")
    say("=" * 78)
    say(f"\n  Strata cut on mean cosine similarity of the top-{TOP_K} pool.")
    say(f"  Q1 = worst neighbours (retrieval should be weakest here).\n")

    results = {}
    for task, name in (("inj", "INJURY"), ("dmg", "DAMAGE")):
        say(f"  {name}")
        say(f"    {'stratum':<22} {'n':>4} {'sim':>7} "
            + " ".join(f"{a:>14}" for a in [RETRIEVAL_ARM] + BN_ARMS))
        say(f"    {'-'*22} {'-'*4} {'-'*7} "
            + " ".join("-" * 14 for _ in [RETRIEVAL_ARM] + BN_ARMS))

        task_rows = {}
        for s in range(4):
            sel = [ids[j] for j in range(len(ids)) if strata[j] == s]
            sel = [i for i in sel if items[i][f"{task}_true"] >= 0]
            if not sel:
                continue
            simv = np.mean([qual[i]["mean"] for i in sel])
            accs = {}
            for arm in [RETRIEVAL_ARM] + BN_ARMS:
                k = sum(1 for i in sel
                        if items[i][f"{arm}:{task}_pred"] == items[i][f"{task}_true"])
                accs[arm] = k / len(sel)
            label = f"Q{s+1}" + (" (worst nbrs)" if s == 0 else
                                 " (best nbrs)" if s == 3 else "")
            say(f"    {label:<22} {len(sel):4d} {simv:7.3f} "
                + " ".join(f"{accs[a]:13.1%}" for a in [RETRIEVAL_ARM] + BN_ARMS))
            task_rows[f"Q{s+1}"] = {"n": len(sel), "mean_sim": float(simv),
                                    "acc": accs}

        # gap between retrieval and the best BN arm, worst vs best stratum
        say("")
        for arm in BN_ARMS:
            g1 = task_rows["Q1"]["acc"][arm] - task_rows["Q1"]["acc"][RETRIEVAL_ARM]
            g4 = task_rows["Q4"]["acc"][arm] - task_rows["Q4"]["acc"][RETRIEVAL_ARM]
            say(f"    {arm:>14} minus retrieval:  "
                f"Q1 {g1*100:+6.1f} pp    Q4 {g4*100:+6.1f} pp    "
                f"(shift {(g1-g4)*100:+.1f} pp toward Q1)")

        # McNemar inside the worst stratum only
        sel = [ids[j] for j in range(len(ids)) if strata[j] == 0
               and items[ids[j]][f"{task}_true"] >= 0]
        say("")
        for arm in BN_ARMS:
            a = np.array([items[i][f"{arm}:{task}_pred"] == items[i][f"{task}_true"]
                          for i in sel])
            b = np.array([items[i][f"{RETRIEVAL_ARM}:{task}_pred"] == items[i][f"{task}_true"]
                          for i in sel])
            n01 = int((a & ~b).sum())
            n10 = int((~a & b).sum())
            p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            say(f"    Q1 McNemar {arm:>14} vs retrieval: "
                f"{n01} vs {n10} discordant, p = {p:.4f}")
        say("")
        results[task] = task_rows

    say("=" * 78)
    say("VERDICT")
    say("=" * 78)
    verdict = []
    for task, name in (("inj", "injury"), ("dmg", "damage")):
        best = max(BN_ARMS,
                   key=lambda a: results[task]["Q1"]["acc"][a]
                   - results[task]["Q1"]["acc"][RETRIEVAL_ARM])
        g1 = results[task]["Q1"]["acc"][best] - results[task]["Q1"]["acc"][RETRIEVAL_ARM]
        g4 = results[task]["Q4"]["acc"][best] - results[task]["Q4"]["acc"][RETRIEVAL_ARM]
        if g1 > 0:
            verdict.append(f"  {name}: the network OVERTAKES retrieval on the "
                           f"worst-neighbour quartile ({g1*100:+.1f} pp via "
                           f"{best}). This is a real job for the BN -- claim it.")
        elif g1 - g4 > 0.05:
            verdict.append(f"  {name}: the network does not overtake, but the "
                           f"gap narrows by {(g1-g4)*100:.1f} pp from best to "
                           f"worst neighbours. Directional support for the "
                           f"compositional argument; too weak to lead with.")
        else:
            verdict.append(f"  {name}: retrieval wins in every stratum "
                           f"(Q1 {g1*100:+.1f} pp, Q4 {g4*100:+.1f} pp). No "
                           f"measurable job for the BN on this task.")
    for v in verdict:
        say(v)
    say("")

    (OUT / "neighbour_quality_strata.md").write_text(
        "# Neighbour-quality strata\n\n```\n" + "\n".join(lines) + "\n```\n")
    (OUT / "neighbour_quality_strata.json").write_text(json.dumps(
        {"top_k": TOP_K, "n": len(ids), "quartile_cuts": cuts.tolist(),
         "results": results}, indent=2))
    print("  wrote outputs/neighbour_quality_strata.{md,json}")


if __name__ == "__main__":
    raise SystemExit(main())

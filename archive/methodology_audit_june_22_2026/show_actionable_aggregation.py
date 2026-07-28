#!/usr/bin/env python3
"""Actionability demo: per-sentence diagnosis (~3%) -> decision-level categories.

Runs the REAL diagnosis pipeline leak-free/offline on the held-out test incidents
and contrasts, for each:
  * BEFORE: the top raw per-cause probability (the "3% won't make anyone act" number)
  * AFTER : main_app.aggregate_causes_to_categories(...) top-category probability

Writes outputs/actionable_aggregation.{json,md} with corpus means + worked examples.
Fully offline: query vectors come from the cached full index; retrieval is the
train-only index + exclude-self.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

os.environ["NTSB_USE_TRAIN_INDEX"] = "1"
os.environ.setdefault("OPENAI_API_KEY", "offline-dummy-key")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PROC = REPO / "data" / "processed"
SPLITS = REPO / "data" / "Testing_Data_Metrics" / "splits"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import main_app  # noqa: E402


class _OfflineClient:
    """Stand-in OpenAI client so the LLM cluster-fallback never hits the network.

    All 177 train incidents already carry a cluster_label, so this should never be
    exercised; but if a retrieved row lacks one, classify_incident catches the
    raised error and falls back to 'uncategorized' instantly (offline-safe)."""

    def __getattr__(self, _name):
        raise RuntimeError("offline: no OpenAI calls in the aggregation demo")


main_app.get_client = lambda: _OfflineClient()  # hard offline guard
from run_leakfree_eval import (  # noqa: E402
    load_full_index,
    query_vectors_from_full,
    truth_categories,
)

MAX_INCIDENTS = int(os.environ.get("MAX_INCIDENTS", "77"))


def main() -> None:
    test_ids = [t for t in (SPLITS / "test_ev_ids.txt").read_text().split() if t]
    full = json.loads((PROC / "refined_dataset.json").read_text())
    emb, fmap = load_full_index()
    qv = query_vectors_from_full(emb, fmap)

    rows = []
    examples = []
    done = 0
    for ev in test_ids:
        if done >= MAX_INCIDENTS:
            break
        inc = full.get(ev)
        q = qv.get(ev)
        if inc is None or q is None:
            continue
        truth1 = truth_categories(inc, 1)
        if not truth1:
            continue

        with contextlib.redirect_stdout(io.StringIO()):
            ts, tm = main_app.find_top_matches(q, exclude_ev_ids={ev})
            cl = main_app.cluster_incidents_by_type(ts, tm, 50)
            if not cl:
                continue
            an = main_app.calculate_cause_probabilities_per_cluster(cl)
            res = main_app.calculate_chain_rule_diagnosis(cl, an)
        wc = res.get("weighted_causes", [])
        if not wc:
            continue

        before_top = max((float(c.get("probability", 0.0)) for c in wc), default=0.0)
        agg = main_app.aggregate_causes_to_categories(wc, level=1)
        after_top = agg["top_probability"]
        top_cat = agg["top_category"]
        hit = top_cat in truth1

        rows.append({
            "ev_id": ev,
            "before_top_cause_p": before_top,
            "after_top_category_p": after_top,
            "top_category": top_cat,
            "truth": sorted(truth1),
            "hit": hit,
            "n_raw_causes": len(wc),
        })
        done += 1
        print(f"[{done}] {ev}  before={before_top*100:.1f}%  after={after_top*100:.0f}% "
              f"({top_cat}) hit={hit}", flush=True)

        if hit and len(examples) < 3:
            examples.append({
                "ev_id": ev,
                "truth": sorted(truth1),
                "before": [
                    {"cause": c["cause"][:90], "p": float(c["probability"])}
                    for c in wc[:5]
                ],
                "after": agg["categories"][:4],
            })

    before_mean = statistics.mean(r["before_top_cause_p"] for r in rows)
    after_mean = statistics.mean(r["after_top_category_p"] for r in rows)
    after_max = max(r["after_top_category_p"] for r in rows)

    summary = {
        "n": len(rows),
        "before_top_cause_p_mean": before_mean,
        "after_top_category_p_mean": after_mean,
        "after_top_category_p_max": after_max,
        "examples": examples,
        "rows": rows,
    }
    (OUT / "actionable_aggregation.json").write_text(json.dumps(summary, indent=2))

    L = [
        "# Actionability: from ~3% per-cause to decision-level categories",
        "",
        f"Held-out incidents: **{len(rows)}**, leak-free (train-only + exclude-self), offline.",
        "",
        "| | top raw cause | top aggregated category |",
        "|---|---|---|",
        f"| **mean** | {before_mean*100:.1f}% | **{after_mean*100:.1f}%** |",
        f"| **max**  | — | {after_max*100:.0f}% |",
        "",
        f"The same evidence, reported at the decision level, is **~{after_mean/before_mean:.0f}x** "
        "more concentrated — turning an un-actionable 3% into a number a safety analyst can act on.",
        "",
        "## Worked examples (before -> after)",
    ]
    for ex in examples:
        L += [
            "",
            f"### Incident {ex['ev_id']} — true cause: {', '.join(ex['truth'])}",
            "",
            "**Before (raw per-cause):**",
        ]
        for b in ex["before"]:
            L.append(f"- {b['p']*100:.1f}%  {b['cause']}")
        L.append("")
        L.append("**After (aggregated to NTSB category):**")
        for a in ex["after"]:
            L.append(f"- {a['probability']*100:.0f}%  {a['category']}  (cumulative {a['cumulative']*100:.0f}%)")
    (OUT / "actionable_aggregation.md").write_text("\n".join(L) + "\n")

    print("\n=== ACTIONABLE AGGREGATION DONE ===")
    print(f"n={len(rows)}  before mean={before_mean*100:.1f}%  after mean={after_mean*100:.1f}%  after max={after_max*100:.0f}%")
    print("→ outputs/actionable_aggregation.md")


if __name__ == "__main__":
    main()

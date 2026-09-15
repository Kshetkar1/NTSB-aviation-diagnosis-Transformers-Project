#!/usr/bin/env python3
"""Two audit items, both answerable from already-saved per-item predictions.

ITEM 1 (test-window reuse).  The 296 held-out accidents have been scored many
times while the method was being developed, so they are no longer a clean
confirmatory set. A true confirmatory run on 2020-2024 is impossible: the
corpus ends in 2019 (last year with data: 2018). The partial answer available
is a temporal split inside the held-out window -- 2007-2012 against
2013-2019 -- which at least shows whether performance is stable as the
accidents drift further from the 1982-2006 build window.

ITEM 2 (A5).  "Ties strong supervised baselines" is true on injury top-1
accuracy but not on injury Macro-F1, where tfidf-lr leads with no interval
attached. This computes a paired bootstrap CI on the Macro-F1 difference so
the claim can be stated with its uncertainty or dropped.

Both use saved predictions only; no inference is re-run.

Run:
    python3 tests/heldout_temporal_split_and_ci.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

HEADLINE = "narrative-evidence"
SPLIT_YEAR = 2013
N_BOOT = 10000
RNG = np.random.default_rng(0)

INJ_STATES = ["none", "minor", "serious", "fatal"]
DMG_STATES = ["none", "minor", "substantial", "destroyed"]


def macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        if tp == 0 and (fp == 0 or fn == 0) and np.sum(y_true == c) == 0:
            continue  # class absent from truth: undefined, skip
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    items = json.loads((OUT / "heldout_per_item.json").read_text())["items"]
    tf = {r["id"]: r for r in
          json.loads((OUT / "tfidf_lr_per_item.json").read_text())["items"]}

    ids = np.array([r["id"] for r in items])
    years = np.array([int(r["id"][:4]) for r in items])
    early = years < SPLIT_YEAR

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    # ══ ITEM 1: temporal split ════════════════════════════════════════════
    say("=" * 74)
    say("ITEM 1 - TEMPORAL SPLIT OF THE HELD-OUT WINDOW")
    say("=" * 74)
    say()
    say(f"  Corpus ends in 2018; a 2020-2024 confirmatory set does not exist.")
    say(f"  Split at {SPLIT_YEAR}: "
        f"{int(early.sum())} early (2007-{SPLIT_YEAR-1}), "
        f"{int((~early).sum())} late ({SPLIT_YEAR}-2018).")
    say()
    say("  Distance from the 1982-2006 build window grows left to right, so a")
    say("  large early-to-late drop would mean the frozen CPTs are going stale.")
    say()

    split_out = {}
    for task, states in (("inj", INJ_STATES), ("dmg", DMG_STATES)):
        yt = np.array([r[f"{task}_true"] for r in items])
        yp = np.array([r[f"{HEADLINE}:{task}_pred"] for r in items])
        valid = yt >= 0

        say(f"  {task.upper()}  ({HEADLINE})")
        say(f"    {'cohort':22} {'n':>5} {'top-1 acc':>11} {'95% CI':>16} "
            f"{'Macro-F1':>10}")
        say(f"    {'-'*22} {'-'*5} {'-'*11} {'-'*16} {'-'*10}")

        rows = {}
        for name, mask in (("2007-2012 (early)", early & valid),
                           (f"{SPLIT_YEAR}-2018 (late)", ~early & valid),
                           ("all held-out", valid)):
            n = int(mask.sum())
            k = int(np.sum(yt[mask] == yp[mask]))
            acc = k / n if n else 0.0
            lo, hi = wilson(k, n)
            f1 = macro_f1(yt[mask], yp[mask], len(states))
            say(f"    {name:22} {n:5d} {acc:10.1%} "
                f"  [{lo:.3f},{hi:.3f}] {f1:10.3f}")
            rows[name] = {"n": n, "acc": acc, "ci": [lo, hi], "macro_f1": f1}

        e = rows["2007-2012 (early)"]
        l = rows[f"{SPLIT_YEAR}-2018 (late)"]
        delta = (l["acc"] - e["acc"]) * 100
        overlap = not (e["ci"][1] < l["ci"][0] or l["ci"][1] < e["ci"][0])
        say()
        say(f"    late - early: {delta:+.1f} pp   "
            f"intervals {'OVERLAP (no detectable drift)' if overlap else 'DISJOINT (drift)'}")
        say()
        split_out[task] = rows

    # ══ ITEM 2: A5 paired bootstrap on injury Macro-F1 ════════════════════
    say("=" * 74)
    say("ITEM 2 (A5) - PAIRED BOOTSTRAP CI ON MACRO-F1 vs tfidf-lr")
    say("=" * 74)
    say()

    paired = [i for i, r in enumerate(items) if r["id"] in tf]
    say(f"  {len(paired)} accidents scored by both systems.")
    say()

    a5_out = {}
    for task, states in (("inj", INJ_STATES), ("dmg", DMG_STATES)):
        yt = np.array([items[i][f"{task}_true"] for i in paired])
        ours = np.array([items[i][f"{HEADLINE}:{task}_pred"] for i in paired])
        base = np.array([tf[items[i]["id"]][f"tfidf-lr:{task}_pred"]
                         for i in paired])
        ok = yt >= 0
        yt, ours, base = yt[ok], ours[ok], base[ok]
        n = len(yt)

        f1_o = macro_f1(yt, ours, len(states))
        f1_b = macro_f1(yt, base, len(states))
        acc_o = float(np.mean(yt == ours))
        acc_b = float(np.mean(yt == base))

        diffs = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = RNG.integers(0, n, n)
            diffs[b] = (macro_f1(yt[idx], ours[idx], len(states))
                        - macro_f1(yt[idx], base[idx], len(states)))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        obs = f1_o - f1_b
        # The percentile interval can touch zero at its boundary while almost
        # every resample points one way, so report the tail mass directly.
        frac_worse = float(np.mean(diffs < 0))
        frac_better = float(np.mean(diffs > 0))
        p_one_sided = min(frac_worse, frac_better)

        say(f"  {task.upper()}  (n = {n}, {N_BOOT} paired bootstrap resamples)")
        say(f"    top-1 accuracy   ours {acc_o:.3f}   tfidf-lr {acc_b:.3f}   "
            f"diff {acc_o-acc_b:+.3f}")
        say(f"    Macro-F1         ours {f1_o:.3f}   tfidf-lr {f1_b:.3f}   "
            f"diff {obs:+.3f}")
        say(f"    95% CI on the Macro-F1 difference: [{lo:+.3f}, {hi:+.3f}]")
        say(f"    resamples favouring tfidf-lr: {frac_worse:.1%}   "
            f"favouring ours: {frac_better:.1%}")
        say(f"    one-sided bootstrap p = {p_one_sided:.4f}")
        if p_one_sided >= 0.05:
            say(f"    -> not separable. Neither system can be claimed ahead "
                f"on this metric.")
        elif obs > 0:
            say(f"    -> real LEAD for the narrative pipeline.")
        else:
            say(f"    -> real DEFICIT. tfidf-lr leads on this metric; say so "
                f"plainly rather than letting 'ties' carry over from accuracy.")
        say()
        a5_out[task] = {
            "n": n, "ours_macro_f1": f1_o, "tfidf_macro_f1": f1_b,
            "diff": obs, "ci95": [float(lo), float(hi)],
            "frac_resamples_favouring_tfidf": frac_worse,
            "frac_resamples_favouring_ours": frac_better,
            "p_one_sided": p_one_sided,
            "separable_at_05": bool(p_one_sided < 0.05),
            "ours_acc": acc_o, "tfidf_acc": acc_b,
        }

    say("=" * 74)
    say("WHAT THIS LICENSES YOU TO WRITE")
    say("=" * 74)
    say()
    inj, dmg = a5_out["inj"], a5_out["dmg"]

    if inj["separable_at_05"]:
        say(f"  INJURY Macro-F1 - the deficit is real "
            f"(p = {inj['p_one_sided']:.3f}).")
        say(f"    Report it: {inj['ours_macro_f1']:.3f} against "
            f"{inj['tfidf_macro_f1']:.3f}, CI "
            f"[{inj['ci95'][0]:+.3f}, {inj['ci95'][1]:+.3f}].")
        say("    Do not let 'ties strong baselines' carry from accuracy to")
        say("    Macro-F1. The explanation is not a hedge: a discriminative")
        say("    model fit on this distribution should beat a frozen")
        say("    generative one on rare-class recall. That gap IS the price of")
        say("    the freeze, and the freeze is what the paper is defending.")
    else:
        say(f"  INJURY Macro-F1 - inside noise (p = {inj['p_one_sided']:.3f}); "
            f"a 'comparable' claim survives if the interval is reported.")
    say()

    if dmg["separable_at_05"]:
        say(f"  DAMAGE Macro-F1 - the lead is real "
            f"(p = {dmg['p_one_sided']:.3f}). Claim it with the interval.")
    else:
        say(f"  DAMAGE Macro-F1 - NOT separable (p = {dmg['p_one_sided']:.3f}, "
            f"CI [{dmg['ci95'][0]:+.3f}, {dmg['ci95'][1]:+.3f}]).")
        say(f"    {dmg['ours_macro_f1']:.3f} against "
            f"{dmg['tfidf_macro_f1']:.3f} looks like a lead but the interval")
        say("    is far too wide to claim one. Any existing wording calling")
        say("    damage 'a clean lead on both metrics' overstates the evidence")
        say("    and must be softened to top-1 accuracy only.")
    say()

    (OUT / "heldout_temporal_split_and_ci.md").write_text(
        "# Held-out temporal split + A5 interval\n\n```\n"
        + "\n".join(lines) + "\n```\n")
    (OUT / "heldout_temporal_split_and_ci.json").write_text(json.dumps(
        {"split_year": SPLIT_YEAR, "temporal_split": split_out,
         "a5_bootstrap": a5_out, "n_boot": N_BOOT}, indent=2))
    print(f"\n  wrote outputs/heldout_temporal_split_and_ci.{{md,json}}")


if __name__ == "__main__":
    raise SystemExit(main())

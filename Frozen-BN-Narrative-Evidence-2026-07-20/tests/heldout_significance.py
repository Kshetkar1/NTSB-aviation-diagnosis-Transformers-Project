#!/usr/bin/env python3
"""Statistical significance for the held-out comparison table.

Inputs (produced by heldout_narrative_bn_eval.py and lr_baseline_heldout.py
with per-item logging enabled):
    outputs/heldout_per_item.json
    outputs/lr_per_item.json

For every predictor and both targets (injury, damage):
    accuracy with a 95% bootstrap CI (10,000 resamples, seed 42)

For key predictor pairs, on the accidents BOTH predictors scored:
    McNemar exact test on top-1 correctness (two-sided binomial on the
    discordant pairs), plus a paired bootstrap 95% CI on the Brier
    difference.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/heldout_significance.py
Writes outputs/heldout_significance.md.
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
_SHARED = REPO_ROOT / "shared" / "code"
_FROZEN_CODE = FROZEN_DIR / "code"
for _p in (_SHARED, _FROZEN_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
ROOT = REPO_ROOT
OUT = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs" / "heldout_significance.md"
RNG = np.random.default_rng(42)
N_BOOT = 10_000

L: list[str] = []


def emit(s: str = "") -> None:
    print(s)
    L.append(s)


def boot_ci(vals: np.ndarray) -> tuple[float, float]:
    n = len(vals)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    means = vals[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar_exact(a_ok: np.ndarray, b_ok: np.ndarray) -> tuple[int, int, float]:
    """b = A right & B wrong, c = A wrong & B right, exact two-sided p."""
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return b, c, min(1.0, p)


def main():
    per = json.loads((ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs" / "heldout_per_item.json").read_text())
    predictors = list(per["predictors"])
    items = {r["id"]: r for r in per["items"]}

    lr_path = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs" / "lr_per_item.json"
    if lr_path.exists():
        for r in json.loads(lr_path.read_text())["items"]:
            if r["id"] in items:
                items[r["id"]].update(
                    {k: v for k, v in r.items() if k.startswith("lr:")})
        predictors.append("lr")

    rows = list(items.values())
    emit("# Held-out significance report")
    emit()
    emit(f"n = {len(rows)} held-out accidents (2007-2019). "
         f"95% CIs: bootstrap, {N_BOOT:,} resamples, seed 42. "
         "Pairwise: McNemar exact test on top-1 correctness.")
    emit()

    # ---- accuracy + CI per predictor --------------------------------------
    for tgt in ("inj", "dmg"):
        label = "Injury" if tgt == "inj" else "Damage"
        emit(f"## {label} accuracy with 95% CI")
        emit()
        emit("| Predictor | Accuracy | 95% CI | n |")
        emit("|---|---|---|---|")
        for p in predictors:
            ok = np.array([r[f"{p}:{tgt}_pred"] == r[f"{tgt}_true"]
                           for r in rows
                           if f"{p}:{tgt}_pred" in r
                           and r.get(f"{tgt}_true") is not None])
            if len(ok) == 0:
                continue
            lo, hi = boot_ci(ok.astype(float))
            emit(f"| {p} | {ok.mean():.1%} | [{lo:.1%}, {hi:.1%}] "
                 f"| {len(ok)} |")
        emit()

    # ---- paired tests ------------------------------------------------------
    pairs = [("full", "prior"), ("full", "hard"), ("soft+stated", "hard"),
             ("narr-sev", "soft+stated"), ("soft+stated", "lr"),
             ("narr-sev", "lr")]
    if "llm-tier" in predictors:
        pairs += [("llm-tier+stated", "soft+stated"),
                  ("llm-first", "llm-tier")]
    pairs = [(a, b) for a, b in pairs
             if a in predictors and b in predictors]

    for tgt in ("inj", "dmg"):
        label = "Injury" if tgt == "inj" else "Damage"
        emit(f"## {label}: paired comparisons (McNemar exact + Brier delta)")
        emit()
        emit("| A vs B | A only right | B only right | McNemar p | "
             "Brier delta (A-B) 95% CI | n |")
        emit("|---|---|---|---|---|---|")
        for a, b in pairs:
            sub = [r for r in rows
                   if f"{a}:{tgt}_pred" in r and f"{b}:{tgt}_pred" in r
                   and r.get(f"{tgt}_true") is not None]
            if not sub:
                continue
            a_ok = np.array([r[f"{a}:{tgt}_pred"] == r[f"{tgt}_true"]
                             for r in sub])
            b_ok = np.array([r[f"{b}:{tgt}_pred"] == r[f"{tgt}_true"]
                             for r in sub])
            nb, nc, p = mcnemar_exact(a_ok, b_ok)
            d = np.array([r[f"{a}:{tgt}_brier"] - r[f"{b}:{tgt}_brier"]
                          for r in sub])
            lo, hi = boot_ci(d)
            star = " *" if p < 0.05 else ""
            emit(f"| {a} vs {b} | {nb} | {nc} | {p:.4f}{star} | "
                 f"[{lo:+.3f}, {hi:+.3f}] | {len(sub)} |")
        emit()

    emit("`*` = significant at p < 0.05. Negative Brier delta favors A "
         "(lower Brier is better).")
    OUT.write_text("\n".join(L) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

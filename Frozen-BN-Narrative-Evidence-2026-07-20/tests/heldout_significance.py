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

Also reports Macro-F1, per-class recall, confusion matrices for the key
predictors, and a binary severe-outcome screen (FATL+SERS / DEST+SUBS)
with sensitivity + specificity -- accuracy alone is misleading under
class imbalance (Jesse's point).

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/heldout_significance.py [--per-item PATH] [--out PATH]
Writes outputs/heldout_significance.md by default.
"""
from __future__ import annotations

import json
import sys
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


INJ_LABELS = ["FATL", "SERS", "MINR", "NONE"]
DMG_LABELS = ["DEST", "SUBS", "MINR", "NONE"]


def _macro_f1(true, pred, n_classes=4):
    f1s = []
    true = np.asarray(true)
    pred = np.asarray(pred)
    for c in range(n_classes):
        tp = int(np.sum((pred == c) & (true == c)))
        fp = int(np.sum((pred == c) & (true != c)))
        fn = int(np.sum((pred != c) & (true == c)))
        if tp + fp + fn == 0:
            continue  # class absent and never predicted
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def main():
    global OUT
    per_path = (ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" /
                "outputs" / "heldout_per_item.json")
    if "--per-item" in sys.argv:
        per_path = Path(sys.argv[sys.argv.index("--per-item") + 1])
    if "--out" in sys.argv:
        OUT = Path(sys.argv[sys.argv.index("--out") + 1])
    per = json.loads(per_path.read_text())
    predictors = list(per["predictors"])
    items = {r["id"]: r for r in per["items"]}

    lr_path = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs" / "lr_per_item.json"
    if lr_path.exists():
        for r in json.loads(lr_path.read_text())["items"]:
            if r["id"] in items:
                items[r["id"]].update(
                    {k: v for k, v in r.items() if k.startswith("lr:")})
        predictors.append("lr")

    emb_path = (ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" /
                "outputs" / "emb_lr_per_item.json")
    if emb_path.exists():
        for r in json.loads(emb_path.read_text())["items"]:
            if r["id"] in items:
                items[r["id"]].update(
                    {k: v for k, v in r.items() if k.startswith("emb-lr:")})
        predictors.append("emb-lr")

    rows = list(items.values())
    emit("# Held-out significance report")
    emit()
    emit(f"n = {len(rows)} held-out accidents (2007-2019). "
         f"Leak-safe eval (outcome phrases stripped before retrieval). "
         f"95% CIs: bootstrap, {N_BOOT:,} resamples, seed 42. "
         "Pairwise: McNemar exact test on top-1 correctness.")
    emit()
    emit("**Metric definitions.** Accuracy = fraction of accidents whose "
         "top-1 predicted class (argmax of the 4-state distribution) equals "
         "the NTSB-coded class. Macro-F1 = unweighted mean of per-class F1 "
         "(robust to class imbalance). Brier = squared error of the full "
         "distribution (lower better).")
    emit()

    # ---- accuracy + CI per predictor --------------------------------------
    for tgt in ("inj", "dmg"):
        label = "Injury" if tgt == "inj" else "Damage"
        emit(f"## {label} accuracy with 95% CI")
        emit()
        true_all = np.array([r[f"{tgt}_true"] for r in rows
                             if r.get(f"{tgt}_true") is not None])
        maj = int(np.bincount(true_all, minlength=4).argmax())
        maj_lbl = (INJ_LABELS if tgt == "inj" else DMG_LABELS)[maj]
        emit(f"Majority-class baseline: always predict {maj_lbl} = "
             f"{float(np.mean(true_all == maj)):.1%} accuracy "
             "(any useful model must beat this AND have higher Macro-F1).")
        emit()
        emit("| Predictor | Accuracy | 95% CI | Macro-F1 | n |")
        emit("|---|---|---|---|---|")
        for p in predictors:
            sub = [r for r in rows
                   if f"{p}:{tgt}_pred" in r
                   and r.get(f"{tgt}_true") is not None]
            if not sub:
                continue
            true = np.array([r[f"{tgt}_true"] for r in sub])
            pred = np.array([r[f"{p}:{tgt}_pred"] for r in sub])
            ok = (pred == true)
            lo, hi = boot_ci(ok.astype(float))
            emit(f"| {p} | {ok.mean():.1%} | [{lo:.1%}, {hi:.1%}] "
                 f"| {_macro_f1(true, pred):.3f} | {len(sub)} |")
        emit()

    # ---- per-class recall + confusion matrices ----------------------------
    key_preds = [p for p in ("bn-sev", "retrieval-sev", "emb-lr", "lr",
                             "soft-priority", "prior")
                 if p in predictors]
    for tgt, labels in (("inj", INJ_LABELS), ("dmg", DMG_LABELS)):
        label = "Injury" if tgt == "inj" else "Damage"
        emit(f"## {label}: per-class recall")
        emit()
        hdr = "| Predictor | " + " | ".join(
            f"{c} recall" for c in labels) + " |"
        emit(hdr)
        emit("|---" * (len(labels) + 1) + "|")
        for p in key_preds:
            sub = [r for r in rows
                   if f"{p}:{tgt}_pred" in r
                   and r.get(f"{tgt}_true") is not None]
            if not sub:
                continue
            true = np.array([r[f"{tgt}_true"] for r in sub])
            pred = np.array([r[f"{p}:{tgt}_pred"] for r in sub])
            cells = []
            for c in range(len(labels)):
                n_c = int(np.sum(true == c))
                if n_c == 0:
                    cells.append("--")
                else:
                    hit = int(np.sum((true == c) & (pred == c)))
                    cells.append(f"{hit}/{n_c}")
            emit(f"| {p} | " + " | ".join(cells) + " |")
        emit()

    for tgt, labels in (("inj", INJ_LABELS), ("dmg", DMG_LABELS)):
        label = "Injury" if tgt == "inj" else "Damage"
        for p in key_preds[:3]:
            sub = [r for r in rows
                   if f"{p}:{tgt}_pred" in r
                   and r.get(f"{tgt}_true") is not None]
            if not sub:
                continue
            true = np.array([r[f"{tgt}_true"] for r in sub])
            pred = np.array([r[f"{p}:{tgt}_pred"] for r in sub])
            emit(f"### {label} confusion matrix: {p} "
                 "(rows = truth, cols = predicted)")
            emit()
            emit("| truth \\ pred | " + " | ".join(labels) + " |")
            emit("|---" * (len(labels) + 1) + "|")
            for i, c in enumerate(labels):
                counts = [int(np.sum((true == i) & (pred == j)))
                          for j in range(len(labels))]
                emit(f"| **{c}** | " + " | ".join(map(str, counts)) + " |")
            emit()

    # ---- binary severe-outcome screen --------------------------------------
    emit("## Binary severe-outcome screen")
    emit()
    emit("Severe injury = FATL or SERS; severe damage = DEST or SUBS. "
         "Sensitivity = severe accidents flagged severe; specificity = "
         "non-severe accidents not flagged. This is the operating view for "
         "triage use.")
    emit()
    emit("| Predictor | Target | Sensitivity | Specificity | n severe / n |")
    emit("|---|---|---|---|---|")
    for tgt, label in (("inj", "injury"), ("dmg", "damage")):
        for p in key_preds:
            sub = [r for r in rows
                   if f"{p}:{tgt}_pred" in r
                   and r.get(f"{tgt}_true") is not None]
            if not sub:
                continue
            true = np.array([r[f"{tgt}_true"] for r in sub])
            pred = np.array([r[f"{p}:{tgt}_pred"] for r in sub])
            t_sev = true <= 1   # states 0,1 = FATL/SERS or DEST/SUBS
            p_sev = pred <= 1
            n_sev = int(t_sev.sum())
            sens = (float(np.sum(t_sev & p_sev)) / n_sev) if n_sev else float("nan")
            n_neg = int((~t_sev).sum())
            spec = (float(np.sum(~t_sev & ~p_sev)) / n_neg) if n_neg else float("nan")
            emit(f"| {p} | {label} | {sens:.1%} ({int(np.sum(t_sev & p_sev))}"
                 f"/{n_sev}) | {spec:.1%} | {n_sev} / {len(sub)} |")
    emit()

    # ---- paired tests ------------------------------------------------------
    pairs = [("bn-sev", "prior"), ("bn-sev", "lr"), ("bn-sev", "emb-lr"),
             ("bn-sev", "retrieval-sev"), ("bn-sev", "bn-fused"),
             ("bn-sev", "hard+soft"), ("bn-sev", "soft-priority"),
             ("retrieval-sev", "lr"), ("retrieval-sev", "emb-lr"),
             ("emb-lr", "lr"),
             ("narrative-evidence", "prior"), ("narrative-evidence", "hard+soft"),
             ("narrative-evidence", "soft-only"), ("narrative-evidence", "lr"),
             ("soft-only", "lr"),
             ("hard+soft", "prior"), ("hard+soft", "soft-only"),
             ("hard+soft", "lr")]
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

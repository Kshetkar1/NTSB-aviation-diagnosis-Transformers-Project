#!/usr/bin/env python3
"""SPARSE-CELL ROBUSTNESS VALIDATION
=====================================
Does SEMANTIC-NEIGHBOUR SMOOTHING produce more accurate / more stable conditional
probability estimates than Zhang's sparse-cell machinery (raw count, the N=1 *0.95 cap,
and the Beta-CDF smoother) when a CPT cell has very few observations?

For a genuinely sparse cell you do not know the ground truth, so we use a
SUBSAMPLE-TO-RECOVER protocol on cells that are *well populated* in the full data:

  1. GOLD: pick incident-level conditional cells  p* = P(outcome o present | cause c present)
     whose cause c appears in >= D_MIN incidents, so the full-data empirical fraction is a
     reliable "truth" for that cell. (We additionally keep mid-range cells, 0.05<=p*<=0.95
     with >=3 positives, so the estimation problem is non-trivial.)
  2. SIMULATE SPARSITY: for each cell, repeatedly subsample its incident population down to
     small n in {1,2,3,5,10} (hypergeometric / without replacement) and from each small
     sample produce an estimate with EACH method:
        (a) raw      -- MLE k/n                                  [sparse_cpt.raw_count_cpt]
        (b) cap      -- Zhang's single-parent rule: k/n, ->0.95 when the ratio == 1.0
        (c) beta_cdf -- Zhang's parametric smoother beta.cdf(k/n, ALPHA, BETA)
                        (his calibrated ALPHA/BETA from sparse_cpt.py)
        (d) semantic -- prognosis.semantic_forward_cpt: fraction of the K embedding-nearest
                        incidents to the cause text whose sequence contains o. (This estimate
                        is n-INDEPENDENT: it borrows strength from the whole corpus, which is
                        exactly the pooling a sparse-cell smoother is meant to do.)
  3. SCORE each estimate vs the cell's gold p* with MAE, Brier (squared error) and a proper
     cross-entropy / log-loss of the probability estimate against p*. Average across many
     cells and many draws; report mean +/- 95% CI and the across-draw STD (stability).
  4. STATS: paired Wilcoxon signed-rank + paired bootstrap across cells, per n, for
     semantic-vs-beta_cdf, semantic-vs-raw, semantic-vs-cap.
  5. LEAKAGE-CONTROLLED COMPLEMENT (held-out / leave-one-incident-out): for every incident i
     in a cell's population predict y_i = [o in i] with each method trained on the OTHER
     incidents (semantic excludes i from its neighbour list via the existing exclude guard),
     and score Brier / log-loss. This removes the full-corpus advantage semantic has in the
     subsample experiment and is the honest predictive check.

Honesty: semantic's subsample numbers are an OPTIMISTIC bound (its neighbour pool includes
the cell's own incidents); the LOO eval is the leakage-free verdict. We report both plainly.

Run (framework python 3.11 + network for the semantic lane):
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/sparse_robustness_validation.py
  ... --no-semantic     # counts-only, no network (semantic columns = n/a)
Results -> docs/sparse_robustness_results.json ; figure -> docs/figures/sparse_robustness.png ;
semantic retrieval cache -> docs/sparse_robustness_semantic_cache.json (re-used across runs).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.stats

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
_SHARED = REPO_ROOT / "shared" / "code"
_FROZEN_CODE = FROZEN_DIR / "code"
for _p in (_SHARED, _FROZEN_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
ROOT = REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# matplotlib config dir is often unwritable in the sandbox -> redirect before import.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs" / ".mplcache"))
(ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs" / ".mplcache").mkdir(parents=True, exist_ok=True)

import prognosis as pg  # noqa: E402
from sparse_cpt import ALPHA, BETA  # noqa: E402

DOCS = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "docs_FrozenBN"
FIG_DIR = DOCS / "figures"
RESULTS_JSON = DOCS / "sparse_robustness_results.json"
SEM_CACHE = DOCS / "sparse_robustness_semantic_cache.json"

EPS = 1e-6  # log-loss clip


# --------------------------------------------------------------------------------------
# Estimators (point estimate of a probability from a small sample of k positives in n).
# --------------------------------------------------------------------------------------
def est_raw(k: int, n: int) -> float:
    return k / n


def est_cap(k: int, n: int) -> float:
    """Zhang's single-parent rule: raw ratio, but a 1.0 ratio is capped to 0.95
    (his hardcoded N=1 / saturated-cell guard)."""
    r = k / n
    return 0.95 if r == 1.0 else r


def est_beta(k: int, n: int) -> float:
    """Zhang's parametric smoother applied to the cell ratio: beta.cdf(k/n, ALPHA, BETA)."""
    return float(scipy.stats.beta.cdf(k / n, a=ALPHA, b=BETA))


def _clip(p: float) -> float:
    return min(1.0 - EPS, max(EPS, p))


def cross_entropy(p_true: float, p_est: float) -> float:
    """Cross-entropy of a probability estimate against the (Bernoulli) gold probability."""
    pe = _clip(p_est)
    return -(p_true * np.log(pe) + (1.0 - p_true) * np.log(1.0 - pe))


def logloss_label(y: int, p_est: float) -> float:
    pe = _clip(p_est)
    return -(np.log(pe) if y == 1 else np.log(1.0 - pe))


# --------------------------------------------------------------------------------------
# Cell construction (incident-level conditionals).
# --------------------------------------------------------------------------------------
def build_label_maps(ds: dict):
    """cause_ev[label] = set(ev with label as finding/occurrence node);
    occ_ev[label]   = set(ev whose ordered occurrence sequence contains label);
    occ_set[ev]     = set(occurrence labels in ev)  (for fast outcome lookup)."""
    cause_ev = defaultdict(set)
    occ_ev = defaultdict(set)
    occ_set = {}
    for ev, inc in ds.items():
        labs = pg._incident_node_labels(inc)
        for l in labs:
            cause_ev[l].add(ev)
        descs, _ = pg._ordered_occurrences(inc)
        s = set(descs)
        occ_set[ev] = s
        for d in s:
            occ_ev[d].add(ev)
    return cause_ev, occ_ev, occ_set


def select_cells(cause_ev, occ_ev, *, d_min, out_min, lo, hi, min_pos,
                 max_causes, max_cells_per_cause, max_cells):
    """Deterministic gold-cell selection (sorted), API-budget aware.

    A cell = (cause c, outcome occurrence o) with |incidents(c)| >= d_min, c != o,
    gold p* = |c & o| / |c| in [lo, hi], and |c & o| >= min_pos. We cap the number of
    distinct causes (richest first) and cells/cause to bound embedding calls, then cap
    total cells. Everything is sorted so runs are reproducible."""
    outcomes = sorted((o for o, evs in occ_ev.items() if len(evs) >= out_min),
                      key=lambda o: (-len(occ_ev[o]), o))
    # candidate causes sorted by support (richest first)
    causes = sorted((c for c, evs in cause_ev.items() if len(evs) >= d_min),
                    key=lambda c: (-len(cause_ev[c]), c))
    cells = []
    used_causes = []
    for c in causes:
        if len(used_causes) >= max_causes:
            break
        cevs = cause_ev[c]
        per = []
        for o in outcomes:
            if o == c:
                continue
            oevs = occ_ev[o]
            pos = len(cevs & oevs)
            p = pos / len(cevs)
            if pos >= min_pos and lo <= p <= hi:
                per.append((c, o, p, pos, len(cevs)))
        if not per:
            continue
        # keep the most informative cells for this cause (gold nearest 0.5 first)
        per.sort(key=lambda t: (abs(t[2] - 0.5), -t[3]))
        per = per[:max_cells_per_cause]
        used_causes.append(c)
        cells.extend(per)
    # global cap: spread across causes (already grouped), keep stable order
    cells = cells[:max_cells]
    return cells


# --------------------------------------------------------------------------------------
# Semantic retrieval (one embedding per cause; ranked incident list cached for re-use
# and for the leave-one-incident-out evaluation).
# --------------------------------------------------------------------------------------
def semantic_ranked_evids(cause_text: str, main_app, max_n: int = 400):
    """Ranked list of incident ev_ids most semantically similar to `cause_text`."""
    q = main_app.get_embedding(cause_text)
    _, matches = main_app.find_top_matches(q)
    seen, ev_ids = set(), []
    for m in matches:
        if m.get("source") != "incident":
            continue
        ev = m.get("ev_id")
        if ev and ev not in seen:
            seen.add(ev)
            ev_ids.append(ev)
        if len(ev_ids) >= max_n:
            break
    return ev_ids


def semantic_value(ranked, occ_set, outcome, k, exclude=None):
    """Fraction of the top-k ranked incidents (optionally excluding one) whose occurrence
    set contains `outcome`."""
    picked = []
    for ev in ranked:
        if exclude is not None and ev == exclude:
            continue
        picked.append(ev)
        if len(picked) >= k:
            break
    if not picked:
        return None
    hit = sum(1 for ev in picked if outcome in occ_set.get(ev, ()))
    return hit / len(picked)


# --------------------------------------------------------------------------------------
# Paired stats.
# --------------------------------------------------------------------------------------
def paired_bootstrap_diff(a, b, n_boot=5000, seed=0):
    """Mean(a-b) with 95% bootstrap CI over paired cells. Negative => a < b (a better
    when the metric is an error)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_safe(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if np.allclose(a, b):
        return float("nan"), 1.0
    try:
        res = scipy.stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except ValueError:
        return float("nan"), 1.0


def mean_ci(x, n_boot=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    bs = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


METHODS = ["raw", "cap", "beta_cdf", "semantic"]
EST_FN = {"raw": est_raw, "cap": est_cap, "beta_cdf": est_beta}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d-min", type=int, default=30, help="min incidents for cause c (gold reliability)")
    ap.add_argument("--out-min", type=int, default=20, help="min incidents for an outcome label to qualify")
    ap.add_argument("--lo", type=float, default=0.05)
    ap.add_argument("--hi", type=float, default=0.95)
    ap.add_argument("--min-pos", type=int, default=3)
    ap.add_argument("--max-causes", type=int, default=60, help="cap distinct cause embeddings (API budget)")
    ap.add_argument("--max-cells-per-cause", type=int, default=4)
    ap.add_argument("--max-cells", type=int, default=160)
    ap.add_argument("--k", type=int, default=50, help="semantic neighbours")
    ap.add_argument("--n-grid", default="1,2,3,5,10")
    ap.add_argument("--draws", type=int, default=400, help="subsample draws per (cell,n)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no-semantic", action="store_true")
    ap.add_argument("--refresh-cache", action="store_true", help="ignore semantic cache")
    args = ap.parse_args()

    n_grid = [int(x) for x in args.n_grid.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    print("Loading window dataset ...", flush=True)
    ds = pg.load_dataset()
    cause_ev, occ_ev, occ_set = build_label_maps(ds)
    print(f"  incidents={len(ds)}  distinct cause-labels={len(cause_ev)}  "
          f"distinct occurrence-labels={len(occ_ev)}")

    cells = select_cells(cause_ev, occ_ev, d_min=args.d_min, out_min=args.out_min,
                         lo=args.lo, hi=args.hi, min_pos=args.min_pos,
                         max_causes=args.max_causes,
                         max_cells_per_cause=args.max_cells_per_cause,
                         max_cells=args.max_cells)
    distinct_causes = sorted({c for (c, *_rest) in cells})
    print(f"  GOLD cells selected={len(cells)} across {len(distinct_causes)} distinct causes "
          f"(d_min={args.d_min}, gold in [{args.lo},{args.hi}], >= {args.min_pos} positives)")
    if not cells:
        print("No gold cells; relax thresholds.")
        return 1

    # ---- semantic lane (network) -----------------------------------------------------
    main_app = None
    sem_cache = {}
    if not args.no_semantic:
        if SEM_CACHE.is_file() and not args.refresh_cache:
            sem_cache = json.loads(SEM_CACHE.read_text())
        try:
            import main_app as _m
            if getattr(_m, "DATA_LOADED", False):
                main_app = _m
                print("[semantic lane ENABLED -- embedding index loaded]")
            else:
                print("[semantic lane SKIPPED -- knowledge base not loaded]")
        except Exception as exc:
            print(f"[semantic lane SKIPPED -- import failed: {exc}]")

    # Build / fetch ranked neighbour lists per distinct cause (one embedding each).
    ranked_by_cause = {}
    if main_app is not None:
        for i, c in enumerate(distinct_causes, 1):
            key = f"ranked:::{c}"
            if key in sem_cache and not args.refresh_cache:
                ranked_by_cause[c] = sem_cache[key]
                continue
            try:
                ranked_by_cause[c] = semantic_ranked_evids(c, main_app)
                sem_cache[key] = ranked_by_cause[c]
                print(f"  [{i}/{len(distinct_causes)}] embedded cause: {c[:60]}", flush=True)
            except Exception as exc:
                print(f"  [warn] embedding failed for {c[:50]!r}: {exc}")
                ranked_by_cause[c] = None
        SEM_CACHE.write_text(json.dumps(sem_cache))

    # ==================================================================================
    # PRIMARY: subsample-to-recover.
    # ==================================================================================
    # per_cell_err[metric][method][n] -> list over cells of that cell's MEAN error across draws
    # per_cell_std[method][n]         -> list over cells of across-draw STD of the estimate
    metrics = ["mae", "brier", "logloss"]
    per_cell_err = {mt: {m: {n: [] for n in n_grid} for m in METHODS} for mt in metrics}
    per_cell_std = {m: {n: [] for n in n_grid} for m in METHODS}

    cell_records = []
    for (c, o, gold, pos, ncause) in cells:
        pop = sorted(cause_ev[c])  # incidents containing cause c
        y = np.array([1 if o in occ_set.get(ev, ()) else 0 for ev in pop])
        # semantic point estimate (n-independent) for this cell
        sem_val = None
        if main_app is not None and ranked_by_cause.get(c):
            ck = f"semval:::{c}:::{o}:::{args.k}"
            if ck in sem_cache and not args.refresh_cache:
                sem_val = sem_cache[ck]
            else:
                sem_val = semantic_value(ranked_by_cause[c], occ_set, o, args.k)
                sem_cache[ck] = sem_val
        cell_records.append({"cause": c, "outcome": o, "gold": gold, "pos": pos,
                              "n_cause": ncause, "semantic": sem_val})

        for n in n_grid:
            if n > len(pop):
                for mt in metrics:
                    for m in METHODS:
                        per_cell_err[mt][m][n].append(np.nan)
                for m in METHODS:
                    per_cell_std[m][n].append(np.nan)
                continue
            # collect estimates per method across draws
            ests = {m: [] for m in METHODS}
            for _ in range(args.draws):
                idx = rng.choice(len(pop), size=n, replace=False)
                k = int(y[idx].sum())
                for m in ("raw", "cap", "beta_cdf"):
                    ests[m].append(EST_FN[m](k, n))
                ests["semantic"].append(sem_val if sem_val is not None else np.nan)
            for m in METHODS:
                arr = np.array(ests[m], float)
                per_cell_std[m][n].append(float(np.nanstd(arr)))
                mae = np.nanmean(np.abs(arr - gold))
                brier = np.nanmean((arr - gold) ** 2)
                ll = np.nanmean([cross_entropy(gold, v) for v in arr]) if not np.isnan(arr).all() else np.nan
                per_cell_err["mae"][m][n].append(float(mae))
                per_cell_err["brier"][m][n].append(float(brier))
                per_cell_err["logloss"][m][n].append(float(ll))

    # ---- aggregate (mean +/- CI across cells, dropping nan cells per n) --------------
    summary = {mt: {m: {} for m in METHODS} for mt in metrics}
    stability = {m: {} for m in METHODS}
    for n in n_grid:
        for m in METHODS:
            for mt in metrics:
                vals = np.array(per_cell_err[mt][m][n], float)
                vals = vals[~np.isnan(vals)]
                if len(vals):
                    mean, lo, hi = mean_ci(vals, seed=args.seed)
                    summary[mt][m][n] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n_cells": int(len(vals))}
                else:
                    summary[mt][m][n] = {"mean": None, "ci_lo": None, "ci_hi": None, "n_cells": 0}
            svals = np.array(per_cell_std[m][n], float)
            svals = svals[~np.isnan(svals)]
            stability[m][n] = float(np.mean(svals)) if len(svals) else None

    # ---- paired tests (semantic vs each baseline), per n, on MAE --------------------
    sig = {"mae": {}, "brier": {}}
    for mt in ("mae", "brier"):
        for n in n_grid:
            sig[mt][n] = {}
            sem = np.array(per_cell_err[mt]["semantic"][n], float)
            for base in ("raw", "cap", "beta_cdf"):
                ba = np.array(per_cell_err[mt][base][n], float)
                mask = ~(np.isnan(sem) | np.isnan(ba))
                a, b = sem[mask], ba[mask]
                if len(a) < 3:
                    sig[mt][n][base] = None
                    continue
                stat, p = wilcoxon_safe(a, b)
                md, clo, chi = paired_bootstrap_diff(a, b, seed=args.seed)
                sig[mt][n][base] = {
                    "n_pairs": int(len(a)),
                    "mean_diff_semantic_minus_base": md,
                    "boot_ci_lo": clo, "boot_ci_hi": chi,
                    "wilcoxon_stat": stat, "wilcoxon_p": p,
                    "semantic_better": bool(md < 0 and p < 0.05),
                }

    # ==================================================================================
    # SECONDARY: leave-one-incident-out held-out predictive eval (leakage-controlled).
    # ==================================================================================
    loo = None
    if main_app is not None:
        loo_brier = {m: [] for m in METHODS}   # per-cell mean Brier over its held-out incidents
        loo_ll = {m: [] for m in METHODS}
        for rec in cell_records:
            c, o = rec["cause"], rec["outcome"]
            pop = sorted(cause_ev[c])
            ranked = ranked_by_cause.get(c)
            if rec["semantic"] is None or not ranked:
                continue
            y = np.array([1 if o in occ_set.get(ev, ()) else 0 for ev in pop])
            tot = int(y.sum())
            N = len(pop)
            if N < 2:
                continue
            bs = {m: [] for m in METHODS}
            ls = {m: [] for m in METHODS}
            for i, ev in enumerate(pop):
                yi = int(y[i])
                others = tot - yi
                r = others / (N - 1)
                preds = {"raw": r, "cap": (0.95 if r == 1.0 else r),
                         "beta_cdf": est_beta(others, N - 1) if (N - 1) > 0 else 0.0,
                         "semantic": semantic_value(ranked, occ_set, o, args.k, exclude=ev)}
                for m in METHODS:
                    p = preds[m]
                    if p is None:
                        continue
                    bs[m].append((p - yi) ** 2)
                    ls[m].append(logloss_label(yi, p))
            for m in METHODS:
                if bs[m]:
                    loo_brier[m].append(float(np.mean(bs[m])))
                    loo_ll[m].append(float(np.mean(ls[m])))
        loo = {"brier": {}, "logloss": {}, "sig": {}}
        for m in METHODS:
            if loo_brier[m]:
                mb, blo, bhi = mean_ci(np.array(loo_brier[m]), seed=args.seed)
                ml, llo, lhi = mean_ci(np.array(loo_ll[m]), seed=args.seed)
                loo["brier"][m] = {"mean": mb, "ci_lo": blo, "ci_hi": bhi, "n_cells": len(loo_brier[m])}
                loo["logloss"][m] = {"mean": ml, "ci_lo": llo, "ci_hi": lhi, "n_cells": len(loo_ll[m])}
        # paired semantic vs baselines on per-cell Brier
        sem = np.array(loo_brier["semantic"], float)
        for base in ("raw", "cap", "beta_cdf"):
            ba = np.array(loo_brier[base], float)
            L = min(len(sem), len(ba))
            if L >= 3:
                a, b = sem[:L], ba[:L]
                stat, p = wilcoxon_safe(a, b)
                md, clo, chi = paired_bootstrap_diff(a, b, seed=args.seed)
                loo["sig"][base] = {"n_pairs": L, "mean_diff_semantic_minus_base": md,
                                    "boot_ci_lo": clo, "boot_ci_hi": chi,
                                    "wilcoxon_stat": stat, "wilcoxon_p": p,
                                    "semantic_better": bool(md < 0 and p < 0.05)}
        if main_app is not None:
            SEM_CACHE.write_text(json.dumps(sem_cache))

    # ==================================================================================
    # REPORT (console).
    # ==================================================================================
    def fmt(x, nd=4):
        return " n/a " if x is None else f"{x:.{nd}f}"

    print("\n" + "=" * 96)
    print("SUBSAMPLE-TO-RECOVER  --  mean error vs n  (averaged over cells x draws)")
    print(f"  cells={len(cells)}  draws/cell/n={args.draws}  k(semantic)={args.k}")
    print("=" * 96)
    for mt in metrics:
        print(f"\n[{mt.upper()}]  (lower is better)   mean [95% CI]")
        head = f"{'n':>4} | " + " | ".join(f"{m:>22}" for m in METHODS)
        print(head)
        print("-" * len(head))
        for n in n_grid:
            row = f"{n:>4} | "
            cellstrs = []
            for m in METHODS:
                s = summary[mt][m][n]
                if s["mean"] is None:
                    cellstrs.append(f"{'n/a':>22}")
                else:
                    cellstrs.append(f"{fmt(s['mean'])} [{fmt(s['ci_lo'],3)},{fmt(s['ci_hi'],3)}]".rjust(22))
            print(row + " | ".join(cellstrs))

    print("\n[STABILITY]  mean across-draw STD of the estimate (lower = more stable)")
    head = f"{'n':>4} | " + " | ".join(f"{m:>12}" for m in METHODS)
    print(head)
    for n in n_grid:
        print(f"{n:>4} | " + " | ".join(f"{fmt(stability[m][n]):>12}" for m in METHODS))

    print("\n" + "=" * 96)
    print("PAIRED SIGNIFICANCE on MAE  (semantic - baseline; negative => semantic better)")
    print("=" * 96)
    for n in n_grid:
        print(f"\n n = {n}")
        for base in ("raw", "cap", "beta_cdf"):
            r = sig["mae"][n].get(base)
            if r is None:
                print(f"   semantic vs {base:9}: n/a")
                continue
            verdict = ("semantic BETTER" if r["semantic_better"]
                       else ("baseline better" if r["mean_diff_semantic_minus_base"] > 0 and r["wilcoxon_p"] < 0.05
                             else "no sig. diff"))
            print(f"   semantic vs {base:9}: dMAE={fmt(r['mean_diff_semantic_minus_base'])} "
                  f"[{fmt(r['boot_ci_lo'])},{fmt(r['boot_ci_hi'])}]  "
                  f"Wilcoxon p={fmt(r['wilcoxon_p'])}  -> {verdict}  (pairs={r['n_pairs']})")

    if loo is not None:
        print("\n" + "=" * 96)
        print("LEAVE-ONE-INCIDENT-OUT held-out prediction (leakage-controlled)  -- lower better")
        print("=" * 96)
        print(f"{'method':>12} | {'Brier [95% CI]':>30} | {'log-loss [95% CI]':>30}")
        for m in METHODS:
            b = loo["brier"].get(m); l = loo["logloss"].get(m)
            bstr = "n/a" if not b else f"{fmt(b['mean'])} [{fmt(b['ci_lo'],3)},{fmt(b['ci_hi'],3)}]"
            lstr = "n/a" if not l else f"{fmt(l['mean'])} [{fmt(l['ci_lo'],3)},{fmt(l['ci_hi'],3)}]"
            print(f"{m:>12} | {bstr:>30} | {lstr:>30}")
        print("\n paired (semantic - baseline) on per-cell Brier:")
        for base in ("raw", "cap", "beta_cdf"):
            r = loo["sig"].get(base)
            if not r:
                print(f"   semantic vs {base:9}: n/a")
                continue
            verdict = ("semantic BETTER" if r["semantic_better"]
                       else ("baseline better" if r["mean_diff_semantic_minus_base"] > 0 and r["wilcoxon_p"] < 0.05
                             else "no sig. diff"))
            print(f"   semantic vs {base:9}: dBrier={fmt(r['mean_diff_semantic_minus_base'])} "
                  f"[{fmt(r['boot_ci_lo'])},{fmt(r['boot_ci_hi'])}]  "
                  f"Wilcoxon p={fmt(r['wilcoxon_p'])}  -> {verdict}")

    # ==================================================================================
    # FIGURE.
    # ==================================================================================
    fig_path = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        colors = {"raw": "#888888", "cap": "#d62728", "beta_cdf": "#1f77b4", "semantic": "#2ca02c"}
        labels = {"raw": "raw / MLE", "cap": "Zhang 1.0->0.95 cap",
                  "beta_cdf": "Zhang Beta-CDF", "semantic": "semantic (this work)"}
        ax = axes[0]
        for m in METHODS:
            ys = [summary["mae"][m][n]["mean"] for n in n_grid]
            lo = [summary["mae"][m][n]["ci_lo"] for n in n_grid]
            hi = [summary["mae"][m][n]["ci_hi"] for n in n_grid]
            if all(v is None for v in ys):
                continue
            ax.plot(n_grid, ys, "-o", color=colors[m], label=labels[m])
            ax.fill_between(n_grid, lo, hi, color=colors[m], alpha=0.15)
        ax.set_xlabel("subsample size  n  (observations available)")
        ax.set_ylabel("MAE vs full-data gold  (lower better)")
        ax.set_title("Accuracy vs sparsity")
        ax.set_xticks(n_grid)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax2 = axes[1]
        for m in METHODS:
            ys = [stability[m][n] for n in n_grid]
            if all(v is None for v in ys):
                continue
            ax2.plot(n_grid, ys, "-o", color=colors[m], label=labels[m])
        ax2.set_xlabel("subsample size  n")
        ax2.set_ylabel("across-draw STD of estimate (lower = more stable)")
        ax2.set_title("Stability vs sparsity")
        ax2.set_xticks(n_grid)
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        fig.suptitle("Sparse-cell robustness: semantic smoothing vs Zhang (Beta-CDF / cap / raw)")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig_path = FIG_DIR / "sparse_robustness.png"
        fig.savefig(fig_path, dpi=140)
        print(f"\nFigure -> {fig_path}")
    except Exception as exc:
        print(f"[figure skipped: {exc}]")

    # ==================================================================================
    # PERSIST machine-readable results.
    # ==================================================================================
    out = {
        "params": vars(args),
        "n_grid": n_grid,
        "n_cells": len(cells),
        "n_distinct_causes": len(distinct_causes),
        "summary_error": summary,
        "stability_std": stability,
        "significance_subsample": sig,
        "loo": loo,
        "cells": cell_records,
        "figure": str(fig_path) if fig_path else None,
        "alpha_beta": [ALPHA, BETA],
    }
    RESULTS_JSON.write_text(json.dumps(out, indent=2))
    print(f"Results -> {RESULTS_JSON}")
    print("\nDONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

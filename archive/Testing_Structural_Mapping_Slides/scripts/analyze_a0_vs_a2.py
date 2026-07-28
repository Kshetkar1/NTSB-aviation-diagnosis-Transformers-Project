"""
Compare A0 (embeddings only) vs A2 (causal-chain reranking) on diagnosis M1.

Inputs (default):
  - outputs/full_diagnosis_A0.csv
  - outputs/eval_diagnosis_A2.csv

Outputs (default: outputs/analysis/):
  - bar_with_ci.png          (1) headline bar chart with 95% Wilson CIs
  - paired_delta.png         (2) per-incident A2 - A0 match% delta
  - match_pct_distribution.png (3) histogram + box plot of match% per condition
  - mcnemar_table.png        (4) 2x2 flip table heatmap
  - rank_distribution.png    (5) where the first hit ranks by condition
  - summary.json             machine-readable summary
  - summary.txt              human-readable summary

Stats:
  - Wilson 95% CI for binomial proportions (top-1, R@5)
  - McNemar exact test for paired binary outcomes (m1_hit)
  - Bootstrap 95% CI for paired difference (top-1, R@5, MRR, avg match%)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Force matplotlib to use a writable cache dir (avoids the home-dir warning).
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache_a0_a2")

import numpy as np
import pandas as pd
from scipy.stats import binomtest, beta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_A0 = REPO_ROOT / "Testing_Structural_Mapping" / "outputs" / "full_diagnosis_A0.csv"
DEFAULT_A2 = REPO_ROOT / "Testing_Structural_Mapping" / "outputs" / "eval_diagnosis_A2.csv"
DEFAULT_OUT = REPO_ROOT / "Testing_Structural_Mapping" / "outputs" / "analysis"

A0_COLOR = "#4C78A8"
A2_COLOR = "#F58518"


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_ci_normal(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Normal-approx CI for the mean of a continuous quantity (used for avg match%)."""
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
    from scipy.stats import t

    tcrit = t.ppf(1 - alpha / 2, df=max(n - 1, 1))
    half = tcrit * sd / math.sqrt(n)
    return (mean - half, mean + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar two-sided p-value via binomial test on the off-diagonal."""
    n_disc = b + c
    if n_disc == 0:
        return 1.0
    return binomtest(min(b, c), n_disc, p=0.5, alternative="two-sided").pvalue


def bootstrap_paired_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Paired bootstrap CI for mean(b) - mean(a). Returns (mean_diff, lo, hi)."""
    assert len(a) == len(b)
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = b - a
    boot = np.empty(n_boot, dtype=float)
    idx_pool = np.arange(n)
    for i in range(n_boot):
        idx = rng.choice(idx_pool, size=n, replace=True)
        boot[i] = float(np.mean(diffs[idx]))
    lo = float(np.percentile(boot, 100 * (alpha / 2)))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return float(np.mean(diffs)), lo, hi


def load_and_join(a0_path: Path, a2_path: Path) -> pd.DataFrame:
    a0 = pd.read_csv(a0_path, low_memory=False)
    a2 = pd.read_csv(a2_path, low_memory=False)

    keep = ["ev_id", "m1_truth_full", "m1_match_pct", "m1_hit", "m1_recall5", "mrr_m1"]
    a0 = a0[keep].copy()
    a2 = a2[keep].copy()
    a0.columns = ["ev_id"] + [f"{c}_A0" for c in keep[1:]]
    a2.columns = ["ev_id"] + [f"{c}_A2" for c in keep[1:]]

    df = a0.merge(a2, on="ev_id", how="inner")

    for col in df.columns:
        if col == "ev_id":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce") if col != "m1_truth_full_A0" and col != "m1_truth_full_A2" else df[col]

    df = df[df["m1_truth_full_A0"].astype(str).str.strip().ne("")]
    df = df[df["m1_truth_full_A0"].astype(str).str.upper().ne("N/A")]
    return df.reset_index(drop=True)


def compute_summary(df: pd.DataFrame) -> dict:
    n = len(df)
    out: dict = {"n": n}

    for cond in ("A0", "A2"):
        hits = df[f"m1_hit_{cond}"].fillna(0).astype(int).to_numpy()
        r5 = df[f"m1_recall5_{cond}"].fillna(0).astype(int).to_numpy()
        mrr = df[f"mrr_m1_{cond}"].fillna(0.0).astype(float).to_numpy()
        pct = df[f"m1_match_pct_{cond}"].fillna(0.0).astype(float).to_numpy()

        top1 = float(hits.mean()) if n else float("nan")
        r5m = float(r5.mean()) if n else float("nan")
        mrr_m = float(mrr.mean()) if n else float("nan")
        pct_m = float(pct.mean()) if n else float("nan")

        top1_lo, top1_hi = wilson_ci(int(hits.sum()), n)
        r5_lo, r5_hi = wilson_ci(int(r5.sum()), n)
        mrr_lo, mrr_hi = mean_ci_normal(mrr)
        pct_lo, pct_hi = mean_ci_normal(pct)

        out[cond] = {
            "n_hits": int(hits.sum()),
            "n_r5_hits": int(r5.sum()),
            "top1": top1,
            "top1_ci95": [top1_lo, top1_hi],
            "recall5": r5m,
            "recall5_ci95": [r5_lo, r5_hi],
            "mrr": mrr_m,
            "mrr_ci95": [mrr_lo, mrr_hi],
            "avg_match_pct": pct_m,
            "avg_match_pct_ci95": [pct_lo, pct_hi],
        }

    a0h = df["m1_hit_A0"].fillna(0).astype(int).to_numpy()
    a2h = df["m1_hit_A2"].fillna(0).astype(int).to_numpy()
    both_pass = int(((a0h == 1) & (a2h == 1)).sum())
    a0_only = int(((a0h == 1) & (a2h == 0)).sum())
    a2_only = int(((a0h == 0) & (a2h == 1)).sum())
    both_fail = int(((a0h == 0) & (a2h == 0)).sum())

    out["mcnemar"] = {
        "table": {
            "both_pass": both_pass,
            "A0_only_pass": a0_only,
            "A2_only_pass": a2_only,
            "both_fail": both_fail,
        },
        "p_value_two_sided": mcnemar_exact(a0_only, a2_only),
        "interpretation": (
            f"Of disagreements: A2-only wins {a2_only}, A0-only wins {a0_only}; "
            f"net = +{a2_only - a0_only} for A2."
        ),
    }

    diff_top1, lo, hi = bootstrap_paired_diff(a0h.astype(float), a2h.astype(float))
    out["delta_top1"] = {"mean": diff_top1, "ci95": [lo, hi]}

    a0r = df["m1_recall5_A0"].fillna(0).astype(int).to_numpy().astype(float)
    a2r = df["m1_recall5_A2"].fillna(0).astype(int).to_numpy().astype(float)
    diff_r5, lo, hi = bootstrap_paired_diff(a0r, a2r, seed=1)
    out["delta_recall5"] = {"mean": diff_r5, "ci95": [lo, hi]}

    a0m = df["mrr_m1_A0"].fillna(0).astype(float).to_numpy()
    a2m = df["mrr_m1_A2"].fillna(0).astype(float).to_numpy()
    diff_mrr, lo, hi = bootstrap_paired_diff(a0m, a2m, seed=2)
    out["delta_mrr"] = {"mean": diff_mrr, "ci95": [lo, hi]}

    a0p = df["m1_match_pct_A0"].fillna(0).astype(float).to_numpy()
    a2p = df["m1_match_pct_A2"].fillna(0).astype(float).to_numpy()
    diff_pct, lo, hi = bootstrap_paired_diff(a0p, a2p, seed=3)
    out["delta_avg_match_pct"] = {"mean": diff_pct, "ci95": [lo, hi]}

    return out


def fmt_pp(x: float) -> str:
    return f"{x*100:+.2f} pp"


def fmt_ci(lo: float, hi: float, scale: float = 100.0, suffix: str = "%") -> str:
    return f"[{lo*scale:.1f}{suffix}, {hi*scale:.1f}{suffix}]"


def write_text_summary(summary: dict, path: Path) -> None:
    n = summary["n"]
    a0 = summary["A0"]
    a2 = summary["A2"]
    mc = summary["mcnemar"]
    lines = []
    lines.append(f"A0 vs A2 — Diagnosis M1 (Cause 'C' findings)\n")
    lines.append(f"Test set n = {n} incidents (paired comparison on the same ev_ids)\n")
    lines.append("=" * 70)
    lines.append(f"{'Metric':<24}{'A0':>20}{'A2':>20}{'Δ (A2 − A0)':>20}")
    lines.append("-" * 70)
    lines.append(
        f"{'Top-1 accuracy':<24}"
        f"{a0['top1']*100:>18.1f}% "
        f"{a2['top1']*100:>18.1f}% "
        f"{(a2['top1']-a0['top1'])*100:>+17.1f} pp"
    )
    lines.append(
        f"{'  95% Wilson CI':<24}"
        f"{fmt_ci(*a0['top1_ci95']):>20}"
        f"{fmt_ci(*a2['top1_ci95']):>20}"
        f"{fmt_ci(*summary['delta_top1']['ci95'], scale=100, suffix='pp'):>20}  (bootstrap)"
    )
    lines.append("")
    lines.append(
        f"{'Recall @ 5':<24}"
        f"{a0['recall5']*100:>18.1f}% "
        f"{a2['recall5']*100:>18.1f}% "
        f"{(a2['recall5']-a0['recall5'])*100:>+17.1f} pp"
    )
    lines.append(
        f"{'  95% Wilson CI':<24}"
        f"{fmt_ci(*a0['recall5_ci95']):>20}"
        f"{fmt_ci(*a2['recall5_ci95']):>20}"
        f"{fmt_ci(*summary['delta_recall5']['ci95'], scale=100, suffix='pp'):>20}  (bootstrap)"
    )
    lines.append("")
    lines.append(
        f"{'MRR':<24}"
        f"{a0['mrr']:>19.3f} "
        f"{a2['mrr']:>19.3f} "
        f"{(a2['mrr']-a0['mrr']):>+18.3f}"
    )
    lines.append(
        f"{'  95% bootstrap CI':<24}"
        f"{'':>20}"
        f"{'':>20}"
        f"[{summary['delta_mrr']['ci95'][0]:+.3f}, {summary['delta_mrr']['ci95'][1]:+.3f}]"
    )
    lines.append("")
    lines.append(
        f"{'Avg match %':<24}"
        f"{a0['avg_match_pct']:>18.2f}% "
        f"{a2['avg_match_pct']:>18.2f}% "
        f"{(a2['avg_match_pct']-a0['avg_match_pct']):>+17.2f} pp"
    )
    lines.append(
        f"{'  95% bootstrap CI':<24}"
        f"{'':>20}"
        f"{'':>20}"
        f"[{summary['delta_avg_match_pct']['ci95'][0]:+.2f}, {summary['delta_avg_match_pct']['ci95'][1]:+.2f}] pp"
    )
    lines.append("")
    lines.append("=" * 70)
    lines.append("McNemar paired flip table (top-1 m1_hit):")
    t = mc["table"]
    lines.append(f"  Both pass     : {t['both_pass']}")
    lines.append(f"  A0 only pass  : {t['A0_only_pass']}")
    lines.append(f"  A2 only pass  : {t['A2_only_pass']}")
    lines.append(f"  Both fail     : {t['both_fail']}")
    lines.append(f"  → {mc['interpretation']}")
    lines.append(f"  → McNemar exact two-sided p-value = {mc['p_value_two_sided']:.4f}")
    p = mc["p_value_two_sided"]
    if p < 0.05:
        verdict = "STATISTICALLY SIGNIFICANT at p < 0.05"
    elif p < 0.10:
        verdict = "Marginal (p < 0.10)"
    else:
        verdict = "Not significant at p < 0.05 (consistent direction, but small n)"
    lines.append(f"  → Verdict: {verdict}")
    lines.append("=" * 70)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_bar_with_ci(summary: dict, out_path: Path) -> None:
    metrics = [
        ("Top-1", "top1", "top1_ci95", 100.0, "%"),
        ("Recall @ 5", "recall5", "recall5_ci95", 100.0, "%"),
        ("Avg match %", "avg_match_pct", "avg_match_pct_ci95", 1.0, "%"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = np.arange(len(metrics))
    w = 0.36
    a0_vals, a2_vals = [], []
    a0_err_lo, a0_err_hi, a2_err_lo, a2_err_hi = [], [], [], []
    for _, k, kci, scale, _ in metrics:
        v0 = summary["A0"][k] * scale
        v2 = summary["A2"][k] * scale
        a0_vals.append(v0)
        a2_vals.append(v2)
        a0_err_lo.append(v0 - summary["A0"][kci][0] * scale)
        a0_err_hi.append(summary["A0"][kci][1] * scale - v0)
        a2_err_lo.append(v2 - summary["A2"][kci][0] * scale)
        a2_err_hi.append(summary["A2"][kci][1] * scale - v2)

    ax.bar(x - w / 2, a0_vals, w, yerr=[a0_err_lo, a0_err_hi], capsize=6,
           label="A0 (embeddings only)", color=A0_COLOR, edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, a2_vals, w, yerr=[a2_err_lo, a2_err_hi], capsize=6,
           label="A2 (causal-chain reranking)", color=A2_COLOR, edgecolor="black", linewidth=0.5)

    for xi, v in zip(x - w / 2, a0_vals):
        ax.text(xi, v + 1.0, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    for xi, v in zip(x + w / 2, a2_vals):
        ax.text(xi, v + 1.0, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, max(max(a0_vals), max(a2_vals)) + 20)
    n = summary["n"]
    p = summary["mcnemar"]["p_value_two_sided"]
    ax.set_title(f"Diagnosis M1 (Cause 'C' findings) — A0 vs A2  (n={n}, McNemar p={p:.3f})")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_paired_delta(df: pd.DataFrame, out_path: Path) -> None:
    delta = (df["m1_match_pct_A2"].fillna(0) - df["m1_match_pct_A0"].fillna(0)).to_numpy()
    order = np.argsort(delta)
    sorted_delta = delta[order]
    colors = ["#2CA02C" if d >= 0 else "#D62728" for d in sorted_delta]

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.bar(range(len(sorted_delta)), sorted_delta, color=colors, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.8)
    mean_d = float(np.mean(delta))
    ax.axhline(mean_d, color="#1F77B4", linestyle="--", linewidth=1.2,
               label=f"Mean Δ = {mean_d:+.2f} pp")
    ax.set_xlabel("Test incidents (sorted by Δ match%)")
    ax.set_ylabel("A2 match%  −  A0 match%")
    n_pos = int((delta > 0).sum())
    n_neg = int((delta < 0).sum())
    n_zero = int((delta == 0).sum())
    ax.set_title(
        f"Per-incident Δ match%  (A2 − A0)   "
        f"green = A2 better ({n_pos}), red = A0 better ({n_neg}), tie ({n_zero})"
    )
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_match_pct_distribution(df: pd.DataFrame, out_path: Path) -> None:
    a0 = df["m1_match_pct_A0"].fillna(0).to_numpy()
    a2 = df["m1_match_pct_A2"].fillna(0).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    ax = axes[0]
    bins = np.linspace(0, 100, 21)
    ax.hist(a0, bins=bins, alpha=0.55, label=f"A0  (mean {a0.mean():.1f}%)", color=A0_COLOR, edgecolor="black", linewidth=0.4)
    ax.hist(a2, bins=bins, alpha=0.55, label=f"A2  (mean {a2.mean():.1f}%)", color=A2_COLOR, edgecolor="black", linewidth=0.4)
    ax.axvline(75, color="black", linestyle="--", linewidth=1.0, label="Threshold = 75 (cosine 0.75)")
    ax.set_xlabel("m1_match_pct (cosine × 100)")
    ax.set_ylabel("Number of incidents")
    ax.set_title("Distribution of top-1 match%")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    bp = ax.boxplot([a0, a2], labels=["A0", "A2"], patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], [A0_COLOR, A2_COLOR]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    for med in bp["medians"]:
        med.set_color("black")
    ax.axhline(75, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylabel("m1_match_pct")
    ax.set_title("Box plot")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Match% distribution — does A2 shift the whole curve up?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_mcnemar_table(summary: dict, out_path: Path) -> None:
    t = summary["mcnemar"]["table"]
    grid = np.array([
        [t["both_pass"], t["A0_only_pass"]],
        [t["A2_only_pass"], t["both_fail"]],
    ])
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(grid, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(grid[i, j]), ha="center", va="center",
                    color="white" if grid[i, j] > grid.max() / 2 else "black",
                    fontsize=18, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["A2 = pass", "A2 = fail"])
    ax.set_yticklabels(["A0 = pass", "A0 = fail"])
    p = summary["mcnemar"]["p_value_two_sided"]
    ax.set_title(f"Paired flip table (m1_hit)\nMcNemar exact p = {p:.4f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_rank_distribution(df: pd.DataFrame, out_path: Path) -> None:
    """Bucket each incident by where its first hit appears (1, 2-5 inferred from MRR, or none)."""
    def bucket(mrr_val: float, hit: int, r5: int) -> str:
        if hit == 1:
            return "rank 1"
        if r5 == 1 and mrr_val > 0:
            r = round(1.0 / mrr_val)
            if 2 <= r <= 5:
                return f"rank {r}"
            return "rank 2-5"
        return "no hit in top 5"

    cats = ["rank 1", "rank 2", "rank 3", "rank 4", "rank 5", "no hit in top 5"]
    counts_a0 = {c: 0 for c in cats}
    counts_a2 = {c: 0 for c in cats}
    for _, row in df.iterrows():
        b0 = bucket(float(row["mrr_m1_A0"] or 0), int(row["m1_hit_A0"] or 0), int(row["m1_recall5_A0"] or 0))
        b2 = bucket(float(row["mrr_m1_A2"] or 0), int(row["m1_hit_A2"] or 0), int(row["m1_recall5_A2"] or 0))
        if b0 not in counts_a0:
            b0 = "rank 2-5" if b0.startswith("rank") else "no hit in top 5"
            cats_extra = "rank 2-5"
            counts_a0.setdefault(cats_extra, 0)
        if b2 not in counts_a2:
            b2 = "rank 2-5" if b2.startswith("rank") else "no hit in top 5"
            counts_a2.setdefault("rank 2-5", 0)
        counts_a0[b0] += 1
        counts_a2[b2] += 1

    keep_cats = [c for c in cats if counts_a0.get(c, 0) + counts_a2.get(c, 0) > 0]
    a0_vals = [counts_a0.get(c, 0) for c in keep_cats]
    a2_vals = [counts_a2.get(c, 0) for c in keep_cats]
    x = np.arange(len(keep_cats))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(x - w / 2, a0_vals, w, label="A0", color=A0_COLOR, edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, a2_vals, w, label="A2", color=A2_COLOR, edgecolor="black", linewidth=0.5)
    for xi, v in zip(x - w / 2, a0_vals):
        ax.text(xi, v + 0.4, str(v), ha="center", va="bottom", fontsize=9)
    for xi, v in zip(x + w / 2, a2_vals):
        ax.text(xi, v + 0.4, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(keep_cats, rotation=10)
    ax.set_ylabel("Number of incidents")
    ax.set_title("Where does the first 'good enough' answer rank?")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a0", type=Path, default=DEFAULT_A0)
    ap.add_argument("--a2", type=Path, default=DEFAULT_A2)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.a0.is_file():
        sys.exit(f"A0 CSV not found: {args.a0}")
    if not args.a2.is_file():
        sys.exit(f"A2 CSV not found: {args.a2}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_join(args.a0, args.a2)
    if df.empty:
        sys.exit("Joined dataframe is empty (no overlapping ev_ids with M1 truth).")

    summary = compute_summary(df)

    write_text_summary(summary, args.out_dir / "summary.txt")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_bar_with_ci(summary, args.out_dir / "bar_with_ci.png")
    plot_paired_delta(df, args.out_dir / "paired_delta.png")
    plot_match_pct_distribution(df, args.out_dir / "match_pct_distribution.png")
    plot_mcnemar_table(summary, args.out_dir / "mcnemar_table.png")
    plot_rank_distribution(df, args.out_dir / "rank_distribution.png")

    paired_path = args.out_dir / "paired_per_incident.csv"
    df.to_csv(paired_path, index=False)

    print((args.out_dir / "summary.txt").read_text())
    print(f"\nWrote analysis artifacts to: {args.out_dir}")


if __name__ == "__main__":
    main()

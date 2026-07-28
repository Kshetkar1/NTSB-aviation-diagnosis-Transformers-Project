"""
Compare A2 results from the original v2 scoring (factor overlap + pattern bonus,
weights 0.45/0.35/0.20, denom = avg chain length) against the slide-formula
scoring (clean F1-F3, weights 0.50/0.30/0.20, denom = total positions, unmapped
penalty 0.10, no factor / pattern terms), with A0 (embeddings only) as the
baseline.

Reads the actual CSV columns produced by eval_diagnosis_structural.py:
    m1_hit, m1_recall5, mrr_m1, m1_match_pct  (M1 = "Cause C findings")

Usage:
    .venv/bin/python Testing_Structural_Mapping_Slides/scripts/compare_v2_vs_slides.py
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
A0_CSV = REPO_ROOT / "Testing_Structural_Mapping" / "outputs" / "full_diagnosis_A0.csv"
A2_V2_CSV = REPO_ROOT / "Testing_Structural_Mapping" / "outputs" / "eval_diagnosis_A2.csv"
A2_SLIDES_CSV = (
    REPO_ROOT / "Testing_Structural_Mapping_Slides" / "outputs" / "eval_diagnosis_A2.csv"
)
TEST_IDS_PATH = REPO_ROOT / "data" / "Testing_Data_Metrics" / "splits" / "test_ev_ids.txt"
OUT_DIR = REPO_ROOT / "Testing_Structural_Mapping_Slides" / "outputs" / "analysis"


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _load_test_ids() -> set[str]:
    if not TEST_IDS_PATH.exists():
        return set()
    return {line.strip() for line in TEST_IDS_PATH.read_text().splitlines() if line.strip()}


def _to_float(x: str | None) -> float:
    try:
        return float(x) if x not in (None, "") else 0.0
    except ValueError:
        return 0.0


def _to_int01(x: str | None) -> int:
    """Parse a 0/1 cell (also accepts True/False, '1.0', '0.0')."""
    if x in (None, ""):
        return 0
    s = str(x).strip().lower()
    if s in ("1", "1.0", "true", "yes"):
        return 1
    if s in ("0", "0.0", "false", "no"):
        return 0
    try:
        return 1 if float(s) >= 0.5 else 0
    except ValueError:
        return 0


def _index_by_eid(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Pick one row per ev_id, preferring the M1/A2 evaluation row."""
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        eid = (r.get("ev_id") or "").strip()
        if not eid:
            continue
        out[eid] = r  # last write wins (these CSVs are 1 row per ev_id)
    return out


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _bootstrap_ci(
    deltas: list[float],
    n_boot: int = 5000,
    seed: int = 12345,
) -> tuple[float, float, float]:
    if not deltas:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(deltas)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return mean(deltas), lo, hi


def _mcnemar_pvalue(b: int, c: int) -> float:
    """Exact two-sided binomial McNemar p-value on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cdf = 0.0
    for i in range(k + 1):
        cdf += math.comb(n, i) * 0.5 ** n
    p = 2.0 * cdf
    return min(1.0, p)


def _metrics_for(rows: list[dict[str, str]]) -> dict[str, float]:
    n = len(rows)
    top1 = sum(_to_int01(r.get("m1_hit")) for r in rows)
    rec5 = sum(_to_int01(r.get("m1_recall5")) for r in rows)
    mrr = mean(_to_float(r.get("mrr_m1")) for r in rows) if n else 0.0
    avg_match = mean(_to_float(r.get("m1_match_pct")) for r in rows) if n else 0.0
    return {
        "n": n,
        "top1_rate": top1 / max(1, n),
        "top1_k": top1,
        "rec5_rate": rec5 / max(1, n),
        "rec5_k": rec5,
        "mrr": mrr,
        "avg_match_pct": avg_match,
    }


def _print_block(title: str, m: dict[str, float]) -> None:
    n = int(m["n"])
    t1_lo, t1_hi = _wilson_ci(int(m["top1_k"]), n)
    r5_lo, r5_hi = _wilson_ci(int(m["rec5_k"]), n)
    print(f"\n{title}  (n = {n})")
    print(
        f"  Top-1   = {m['top1_rate'] * 100:5.1f}%   "
        f"95% CI [{t1_lo * 100:5.1f}%, {t1_hi * 100:5.1f}%]   "
        f"({int(m['top1_k'])}/{n} hits)"
    )
    print(
        f"  Rec@5   = {m['rec5_rate'] * 100:5.1f}%   "
        f"95% CI [{r5_lo * 100:5.1f}%, {r5_hi * 100:5.1f}%]   "
        f"({int(m['rec5_k'])}/{n} hits)"
    )
    print(f"  MRR     = {m['mrr']:.4f}")
    print(f"  AvgMtch = {m['avg_match_pct']:.4f}%")


def _paired_eids(
    a0: dict[str, dict[str, str]],
    a2_v2: dict[str, dict[str, str]],
    a2_sl: dict[str, dict[str, str]],
    test_ids: set[str],
) -> list[str]:
    eids = set(a2_v2) & set(a2_sl) & set(a0)
    if test_ids:
        eids &= test_ids
    return sorted(eids)


def _paired_deltas(
    rows_x: list[dict[str, str]], rows_y: list[dict[str, str]], field: str
) -> list[float]:
    return [_to_float(y.get(field)) - _to_float(x.get(field)) for x, y in zip(rows_x, rows_y)]


def _paired_int01_deltas(
    rows_x: list[dict[str, str]], rows_y: list[dict[str, str]], field: str
) -> list[float]:
    return [
        float(_to_int01(y.get(field)) - _to_int01(x.get(field)))
        for x, y in zip(rows_x, rows_y)
    ]


def _mcnemar_counts(
    rows_x: list[dict[str, str]], rows_y: list[dict[str, str]], field: str
) -> tuple[int, int]:
    b = c = 0
    for x, y in zip(rows_x, rows_y):
        hx = _to_int01(x.get(field))
        hy = _to_int01(y.get(field))
        if hx == 0 and hy == 1:
            b += 1
        elif hx == 1 and hy == 0:
            c += 1
    return b, c


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not A2_SLIDES_CSV.exists():
        print(f"ERROR: slide-formula results not found at {A2_SLIDES_CSV}")
        return

    a0_rows = _index_by_eid(_load_csv(A0_CSV))
    v2_rows = _index_by_eid(_load_csv(A2_V2_CSV))
    sl_rows = _index_by_eid(_load_csv(A2_SLIDES_CSV))
    test_ids = _load_test_ids()

    eids = _paired_eids(a0_rows, v2_rows, sl_rows, test_ids)
    if not eids:
        print("ERROR: no overlapping ev_ids between A0, A2-v2, and A2-slides.")
        return

    a0 = [a0_rows[e] for e in eids]
    v2 = [v2_rows[e] for e in eids]
    sl = [sl_rows[e] for e in eids]

    m_a0 = _metrics_for(a0)
    m_v2 = _metrics_for(v2)
    m_sl = _metrics_for(sl)

    print("=" * 78)
    print("A0 vs A2-v2 (original code) vs A2-slides (slide formulas)")
    print(f"Paired comparison on {len(eids)} test incidents (M1 = Cause C findings)")
    print("=" * 78)
    _print_block("A0     (embeddings only)        ", m_a0)
    _print_block("A2-v2  (current code, in paper) ", m_v2)
    _print_block("A2-sl  (exact slide formulas)   ", m_sl)

    print("\n" + "-" * 78)
    print("Paired deltas (vs A0): bootstrap mean and 95% CI  (* = CI excludes 0)")
    print("-" * 78)
    for label, rows in (("A2-v2 - A0", v2), ("A2-sl - A0", sl)):
        d_t1 = _paired_int01_deltas(a0, rows, "m1_hit")
        d_r5 = _paired_int01_deltas(a0, rows, "m1_recall5")
        d_mrr = _paired_deltas(a0, rows, "mrr_m1")
        d_match = _paired_deltas(a0, rows, "m1_match_pct")
        for name, deltas in (
            ("Top-1     ", d_t1),
            ("Recall@5  ", d_r5),
            ("MRR       ", d_mrr),
            ("AvgMatch% ", d_match),
        ):
            mu, lo, hi = _bootstrap_ci(deltas)
            sig = "*" if (lo > 0 or hi < 0) else " "
            print(f"  {label:<13} {name} mean={mu:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}] {sig}")
        print()

    # Also: direct slides-vs-v2 comparison (each is reranking the same A0 result set)
    print("-" * 78)
    print("Direct paired delta: A2-slides - A2-v2")
    print("-" * 78)
    d_t1 = _paired_int01_deltas(v2, sl, "m1_hit")
    d_r5 = _paired_int01_deltas(v2, sl, "m1_recall5")
    d_mrr = _paired_deltas(v2, sl, "mrr_m1")
    d_match = _paired_deltas(v2, sl, "m1_match_pct")
    for name, deltas in (
        ("Top-1     ", d_t1),
        ("Recall@5  ", d_r5),
        ("MRR       ", d_mrr),
        ("AvgMatch% ", d_match),
    ):
        mu, lo, hi = _bootstrap_ci(deltas)
        sig = "*" if (lo > 0 or hi < 0) else " "
        print(f"  sl - v2       {name} mean={mu:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}] {sig}")
    print()

    print("-" * 78)
    print("McNemar (paired binary outcomes), exact two-sided p-values")
    print("-" * 78)
    for label, field in (("Top-1 (m1_hit)    ", "m1_hit"), ("Rec@5 (m1_recall5)", "m1_recall5")):
        b_v2, c_v2 = _mcnemar_counts(a0, v2, field)
        b_sl, c_sl = _mcnemar_counts(a0, sl, field)
        b_dir, c_dir = _mcnemar_counts(v2, sl, field)
        print(
            f"  {label}  v2 vs A0  : A2 wins={b_v2:>2}  A0 wins={c_v2:>2}  "
            f"p={_mcnemar_pvalue(b_v2, c_v2):.4f}"
        )
        print(
            f"  {label}  sl vs A0  : A2 wins={b_sl:>2}  A0 wins={c_sl:>2}  "
            f"p={_mcnemar_pvalue(b_sl, c_sl):.4f}"
        )
        print(
            f"  {label}  sl vs v2  : sl wins={b_dir:>2}  v2 wins={c_dir:>2}  "
            f"p={_mcnemar_pvalue(b_dir, c_dir):.4f}"
        )
        print()

    print("-" * 78)
    print("Recommendation")
    print("-" * 78)
    score_v2 = (
        (m_v2["top1_rate"] - m_a0["top1_rate"])
        + (m_v2["rec5_rate"] - m_a0["rec5_rate"])
        + (m_v2["mrr"] - m_a0["mrr"])
    )
    score_sl = (
        (m_sl["top1_rate"] - m_a0["top1_rate"])
        + (m_sl["rec5_rate"] - m_a0["rec5_rate"])
        + (m_sl["mrr"] - m_a0["mrr"])
    )
    if abs(score_v2 - score_sl) < 1e-6:
        winner = "TIE"
    else:
        winner = "A2-v2 (current code)" if score_v2 > score_sl else "A2-slides (clean formulas)"
    print(f"  Sum-of-deltas (Top-1 + Rec@5 + MRR) vs A0:")
    print(f"    A2-v2     = {score_v2:+.4f}")
    print(f"    A2-slides = {score_sl:+.4f}")
    print(f"  Better overall on this scoreboard: {winner}")
    print(
        "  Also weigh: paired-CI significance (* above), McNemar p-values,\n"
        "  and AvgMatch% (the embedding similarity to the true cause)."
    )

    summary = {
        "n_paired": len(eids),
        "metrics": {"A0": m_a0, "A2_v2": m_v2, "A2_slides": m_sl},
        "test_ids_used": bool(test_ids),
        "winner_rank_score": winner,
    }
    out_path = OUT_DIR / "v2_vs_slides_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

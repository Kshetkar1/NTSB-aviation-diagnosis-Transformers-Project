"""
Read aggregate_results.jsonl produced by aggregate_eval.py and compute the
defensible summary statistics that go into worked_examples_v2.html and the
paper.

The same module is imported by generate_html_v2.py to render the
"Aggregate Validation" section and used as a CLI to print stats:

    python Worked_Examples/aggregate_summary.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_PATH = _HERE / "data" / "aggregate_results.jsonl"

THRESHOLDS = (0.30, 0.40, 0.50)  # cosine-sim cutoffs we report


def load_results(path: Path = DEFAULT_RESULTS_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _mean(xs: list[float]) -> float:
    return float(statistics.fmean(xs)) if xs else 0.0


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _frac_above(xs: list[float], t: float) -> float:
    return (sum(1 for x in xs if x > t) / len(xs)) if xs else 0.0


def compute_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    s_cluster_a0 = [float(r.get("sim_cluster_a0", 0.0)) for r in rows]
    s_cluster_a2 = [float(r.get("sim_cluster_a2", 0.0)) for r in rows]
    s_cause_a0 = [float(r.get("sim_cause_a0", 0.0)) for r in rows]
    s_cause_a2 = [float(r.get("sim_cause_a2", 0.0)) for r in rows]
    p_a0 = [float(r.get("a0_top_p_k", 0.0)) for r in rows]
    p_a2 = [float(r.get("a2_top_p_k", 0.0)) for r in rows]

    def _block(s_a0: list[float], s_a2: list[float]) -> dict[str, Any]:
        win = sum(1 for a, b in zip(s_a0, s_a2) if b > a + 1e-9)
        tie = sum(1 for a, b in zip(s_a0, s_a2) if abs(b - a) <= 1e-9)
        loss = sum(1 for a, b in zip(s_a0, s_a2) if b < a - 1e-9)
        return {
            "n": n,
            "mean_a0": _mean(s_a0),
            "mean_a2": _mean(s_a2),
            "median_a0": _median(s_a0),
            "median_a2": _median(s_a2),
            "thresholds": {
                f"{t:.2f}": {
                    "frac_a0": _frac_above(s_a0, t),
                    "frac_a2": _frac_above(s_a2, t),
                    "n_a0": sum(1 for x in s_a0 if x > t),
                    "n_a2": sum(1 for x in s_a2 if x > t),
                }
                for t in THRESHOLDS
            },
            "a2_vs_a0": {"win": win, "tie": tie, "loss": loss},
        }

    return {
        "n": n,
        "cluster": _block(s_cluster_a0, s_cluster_a2),
        "cause": _block(s_cause_a0, s_cause_a2),
        "p_top_cluster": {
            "mean_a0": _mean(p_a0),
            "mean_a2": _mean(p_a2),
            "median_a0": _median(p_a0),
            "median_a2": _median(p_a2),
        },
    }


def best_and_worst(rows: list[dict[str, Any]], k: int = 5,
                   metric: str = "sim_cluster_a2") -> tuple[list[dict], list[dict]]:
    if not rows:
        return [], []
    sortable = [r for r in rows if metric in r]
    sortable.sort(key=lambda r: float(r.get(metric, 0.0)), reverse=True)
    return sortable[:k], sortable[-k:][::-1]


def print_cli_summary(rows: list[dict[str, Any]]) -> None:
    s = compute_summary(rows)
    n = s["n"]
    print("=" * 78)
    print(f"AGGREGATE VALIDATION (n={n} held-out incidents)")
    print("=" * 78)

    def _show(name: str, blk: dict[str, Any]) -> None:
        print(f"\n  {name}")
        print(f"    Mean cosine(prediction, NTSB cause): A0={blk['mean_a0']:.4f}  A2={blk['mean_a2']:.4f}")
        print(f"    Median                              : A0={blk['median_a0']:.4f}  A2={blk['median_a2']:.4f}")
        for t_str, t_block in blk["thresholds"].items():
            print(
                f"    > {t_str}: A0 {t_block['n_a0']:>2}/{n} ({t_block['frac_a0']*100:5.1f}%)"
                f" | A2 {t_block['n_a2']:>2}/{n} ({t_block['frac_a2']*100:5.1f}%)"
            )
        wtl = blk["a2_vs_a0"]
        print(f"    A2 vs A0 (per-incident):  WIN {wtl['win']:>2}  TIE {wtl['tie']:>2}  LOSS {wtl['loss']:>2}")

    _show("Top cluster label vs NTSB cause text", s["cluster"])
    _show("Top free-text cause vs NTSB cause text", s["cause"])

    pk = s["p_top_cluster"]
    print(f"\n  P(top cluster | Q):  A0 mean={pk['mean_a0']*100:.1f}%  A2 mean={pk['mean_a2']*100:.1f}%")

    best, worst = best_and_worst(rows, k=5, metric="sim_cluster_a2")
    print("\n  Top-5 best A2 cluster matches:")
    for r in best:
        print(f"    {r['ev_id']}  sim={r['sim_cluster_a2']:.3f}  cluster='{r['a2_top_cluster']}'")
        print(f"        truth: {(r.get('truth_text') or '')[:140]}")

    print("\n  Bottom-5 (worst A2 cluster matches):")
    for r in worst:
        print(f"    {r['ev_id']}  sim={r['sim_cluster_a2']:.3f}  cluster='{r['a2_top_cluster']}'")
        print(f"        truth: {(r.get('truth_text') or '')[:140]}")


if __name__ == "__main__":
    rows = load_results()
    if not rows:
        raise SystemExit(f"No results found at {DEFAULT_RESULTS_PATH}")
    print_cli_summary(rows)

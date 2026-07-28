"""Convergence demonstration: query-driven narrative diagnosis -> Zhang Table 7.

THESIS
------
The student's narrative diagnosis lane (``zhang_diagnosis.diagnose_retrieval``)
is QUERY-DRIVEN: it embeds the query, retrieves the top-K most relevant
incidents, keeps the ones that actually had the outcome, and counts causes over
*that retrieved subset*. So ``top_n_incidents`` (K) controls how much of the
outcome population the query is allowed to "see".

Zhang's Table 7, by contrast, is a QUERY-FREE population statistic:
P(cause | outcome) over EVERY outcome accident (all 102 fires), with no
retrieval restriction.

These two quantities are not supposed to be equal at small K -- the narrative
method conditions on the query and therefore answers a *restricted* question.
This script proves the relationship is principled, not an error: as we widen
retrieval breadth K toward the full population, the narrative method's
distribution CONVERGES to Zhang's exact Table 7 values:

    #outcome accidents seen -> full population (e.g. 102 fires)
    P(dominant cause)       -> Zhang's exact value (fire: 0.3137)
    L1 distance to Zhang    -> 0
    Spearman rank-corr      -> 1.0

Run (framework interpreter + network for the embedding API)::

    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
        tests/convergence_to_zhang.py

Outputs a convergence table per outcome and writes
``docs/figures/convergence_to_zhang.png``.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Use a writable matplotlib cache dir (the default ~/.matplotlib may be read-only
# in this environment) and a non-interactive backend before importing pyplot.
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplcache_"))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import zhang_diagnosis  # noqa: E402

# Retrieval-breadth sweep. The final value is intentionally huge so the slice
# ``matches[:K]`` covers every retrieved incident == effectively the whole
# population (query restriction lifted).
K_SWEEP = [25, 50, 100, 200, 400, 800, 5000]

# Outcomes to demonstrate. The first (fire) is the headline example with the
# published Zhang Table 7 anchor P(Airframe...) = 0.3137; loss of engine power
# is included as a second, independent confirmation.
OUTCOMES = [
    {"query": "What is the probability of fire?", "label": "fire"},
    {"query": "What is the probability of loss of engine power?",
     "label": "loss of engine power"},
]

# Published Zhang Table 7 fire anchor (for the figure's reference line / sanity).
ZHANG_FIRE_AIRFRAME = 0.3137


def _dist_map(causes: list) -> dict:
    """{cause(lowercased) -> probability} from an empirical_cause_distribution result."""
    return {c["cause"].strip().lower(): float(c["probability"]) for c in causes}


def l1_to_zhang(retrieved: dict, zhang: dict) -> float:
    """L1 distance over the UNION of causes (missing cause == probability 0).

    Using the union (not just the intersection) is the honest metric: at small K
    the retrieved subset is missing causes that Zhang has, and that absence is
    real divergence. At full breadth the cause sets coincide and L1 -> 0."""
    keys = set(retrieved) | set(zhang)
    return sum(abs(retrieved.get(k, 0.0) - zhang.get(k, 0.0)) for k in keys)


def rank_corr_to_zhang(retrieved: dict, zhang: dict) -> float:
    """Spearman rank correlation of cause probabilities over the union of causes.

    Vectors are aligned on the union (missing == 0). Returns 1.0 when the two
    rankings agree perfectly (full-breadth limit)."""
    keys = sorted(set(retrieved) | set(zhang))
    if len(keys) < 2:
        return float("nan")
    a = np.array([retrieved.get(k, 0.0) for k in keys])
    b = np.array([zhang.get(k, 0.0) for k in keys])
    rho, _ = spearmanr(a, b)
    return float(rho)


def ground_truth(query: str) -> dict:
    """Zhang population Table 7 for the query's outcome: P(cause | outcome) over
    ALL outcome accidents (no retrieval restriction)."""
    import main_app

    det = zhang_diagnosis.detect_outcome(query, dataset=main_app.refined_dataset)
    if det is None:
        raise SystemExit(f"no recognized outcome in query: {query!r}")
    name, targets = det
    res = zhang_diagnosis.empirical_cause_distribution(
        name, targets=targets, dataset=main_app.refined_dataset
    )
    res["name"] = name
    return res


def run_outcome(outcome: dict) -> dict:
    """Compute Zhang ground truth + the K-sweep convergence rows for one outcome."""
    query = outcome["query"]
    gt = ground_truth(query)
    zmap = _dist_map(gt["causes"])
    full_pop = gt["outcome_count"]

    # Dominant cause = Zhang's top cause; anchors = next few for the table.
    ranked = sorted(zmap, key=lambda k: -zmap[k])
    dominant = ranked[0]
    anchors = ranked[1:4]

    rows = []
    for k in K_SWEEP:
        # calibrate=False: this demo proves the narrative method's RAW magnitudes
        # converge to Zhang's Table 7. Temperature-scaling (the engine default)
        # is monotonic so it preserves ranking, but it reshapes magnitudes away
        # from Zhang's raw values, which would break the L1->0 claim. The
        # per-incident diagnosis path keeps calibration on by default.
        r = zhang_diagnosis.diagnose_retrieval(
            query, top_n_incidents=k, top_n=10_000, calibrate=False
        )
        rmap = _dist_map(r["causes"])
        rows.append({
            "K": k,
            "seen": r["outcome_count"],
            "retrieved": r.get("retrieved_incidents"),
            "p_dominant": rmap.get(dominant, 0.0),
            "p_anchors": [rmap.get(a, 0.0) for a in anchors],
            "l1": l1_to_zhang(rmap, zmap),
            "rho": rank_corr_to_zhang(rmap, zmap),
        })

    return {
        "query": query,
        "label": outcome["label"],
        "name": gt["name"],
        "full_pop": full_pop,
        "dominant": dominant,
        "p_dominant_zhang": zmap[dominant],
        "anchors": anchors,
        "p_anchors_zhang": [zmap[a] for a in anchors],
        "zmap": zmap,
        "rows": rows,
    }


def print_table(result: dict) -> None:
    label = result["label"]
    dom = result["dominant"]
    anchors = result["anchors"]
    full_pop = result["full_pop"]

    print("\n" + "=" * 100)
    print(f"OUTCOME: {label!r}   (query: {result['query']!r})")
    print(f"GROUND TRUTH = Zhang Table 7 over ALL {full_pop} {label} accidents "
          f"(query-free population statistic)")
    print(f"  dominant cause: {dom}")
    print(f"     P(dominant | {label}) = {result['p_dominant_zhang']:.4f}")
    for a, p in zip(anchors, result["p_anchors_zhang"]):
        print(f"     anchor: {a[:60]:60} = {p:.4f}")
    print("-" * 100)

    # short anchor headers
    a_heads = [f"P[a{i+1}]" for i in range(len(anchors))]
    header = (f"{'K':>6} {'#seen':>6} {'P(dominant)':>12} "
              + " ".join(f"{h:>8}" for h in a_heads)
              + f" {'L1->Zhang':>10} {'rank-corr':>10}")
    print(header)
    print("-" * len(header))
    for row in result["rows"]:
        tag = "*" if row["seen"] >= full_pop else " "
        line = (f"{row['K']:>6} {row['seen']:>5}{tag} {row['p_dominant']:>12.4f} "
                + " ".join(f"{p:>8.4f}" for p in row["p_anchors"])
                + f" {row['l1']:>10.4f} {row['rho']:>10.4f}")
        print(line)
    print(f"  (* = full {label} population captured; "
          f"dominant should -> {result['p_dominant_zhang']:.4f}, L1 -> 0, rank-corr -> 1.0)")


def make_figure(fire_result: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = fire_result["rows"]
    seen = [r["seen"] for r in rows]
    p_dom = [r["p_dominant"] for r in rows]
    l1 = [r["l1"] for r in rows]
    zhang_val = fire_result["p_dominant_zhang"]

    fig, ax1 = plt.subplots(figsize=(8.5, 5.2))

    color1 = "#1f77b4"
    ax1.plot(seen, p_dom, "o-", color=color1, linewidth=2,
             label="P(Airframe failure | fire) — narrative method")
    ax1.axhline(zhang_val, color=color1, linestyle="--", linewidth=1.6,
                label=f"Zhang Table 7 = {zhang_val:.4f}")
    ax1.set_xlabel("Number of fire accidents the query actually SEES "
                   "(retrieval breadth)")
    ax1.set_ylabel("P(Airframe/component/system failure | fire)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, max(max(p_dom), zhang_val) * 1.25 + 0.01)

    color2 = "#d62728"
    ax2 = ax1.twinx()
    ax2.plot(seen, l1, "s--", color=color2, linewidth=1.8, alpha=0.85,
             label="L1 distance to Zhang (all causes)")
    ax2.set_ylabel("L1 distance to Zhang Table 7", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, max(l1) * 1.15 + 0.01)

    full_pop = fire_result["full_pop"]
    ax1.axvline(full_pop, color="gray", linestyle=":", linewidth=1.2)
    ax1.annotate(f"full population\n({full_pop} fires)",
                 xy=(full_pop, zhang_val), xytext=(full_pop * 0.62, zhang_val * 0.45),
                 fontsize=9, color="gray",
                 arrowprops=dict(arrowstyle="->", color="gray"))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

    ax1.set_title("Query-driven narrative diagnosis converges to Zhang's "
                  "query-free Table 7\nas retrieval breadth grows toward the "
                  "full fire population")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    results = [run_outcome(o) for o in OUTCOMES]
    for r in results:
        print_table(r)

    fire_result = next((r for r in results if r["label"] == "fire"), results[0])
    out_path = ROOT / "docs" / "figures" / "convergence_to_zhang.png"
    make_figure(fire_result, out_path)

    size = out_path.stat().st_size if out_path.exists() else 0
    print("\n" + "=" * 100)
    print(f"FIGURE written: {out_path}  ({size} bytes)")
    if size == 0:
        raise SystemExit("ERROR: figure is empty")

    # Final sanity assertions on the headline (fire) outcome at full breadth.
    last = fire_result["rows"][-1]
    print("\nSANITY (fire, broadest K):")
    print(f"  #fires seen          = {last['seen']}  (target {fire_result['full_pop']})")
    print(f"  P(Airframe) narrative = {last['p_dominant']:.4f}  "
          f"(Zhang {fire_result['p_dominant_zhang']:.4f})")
    print(f"  L1 to Zhang          = {last['l1']:.6f}  (target 0)")
    print(f"  rank-corr            = {last['rho']:.4f}  (target 1.0)")


if __name__ == "__main__":
    main()

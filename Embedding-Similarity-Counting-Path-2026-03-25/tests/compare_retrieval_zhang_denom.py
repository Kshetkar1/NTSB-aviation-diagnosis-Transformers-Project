"""Retrieval lane with Zhang's denominator vs recreated Zhang Table 7.

Shows that when the query 'What is the probability of fire?' retrieves the fire
accidents and we divide by their count (Zhang's method), the MAGNITUDES match --
not just the ranking."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUERY = "What is the probability of fire?"


def main() -> None:
    import zhang_diagnosis

    zhang = zhang_diagnosis.empirical_cause_distribution("fire")
    zmap = {c["cause"].strip().lower(): c["probability"] for c in zhang["causes"]}
    top = sorted(zmap, key=lambda k: -zmap[k])[:10]
    print(f"Zhang/recreated: {zhang['outcome_count']} fire accidents\n")

    import main_app
    n_total = sum(1 for c in main_app.embeddings_map if c.get("source") == "incident")

    results = {}
    for tn in (50, 200, n_total):
        # calibrate=False: this demo checks RAW magnitudes against Zhang's Table 7.
        # The engine default (temperature scaling) is monotonic and preserves
        # ranking but changes magnitudes, so exact-magnitude matching needs the
        # uncalibrated output. Per-incident diagnosis stays calibrated by default.
        r = zhang_diagnosis.diagnose_retrieval(
            QUERY, top_n_incidents=tn, top_n=500, calibrate=False
        )
        rmap = {c["cause"].strip().lower(): c["probability"] for c in r["causes"]}
        results[tn] = (r["outcome_count"], r["retrieved_incidents"], rmap)

    cols = list(results.keys())
    print(f"{'cause':50} {'Zhang':>7} " + " ".join(f"{('tn='+str(c)):>9}" for c in cols))
    print(f"{'':50} {'':>7} " + " ".join(f"{('fire='+str(results[c][0])):>9}" for c in cols))
    print("-" * (60 + 10 * len(cols)))
    for cause in top:
        row = f"{cause[:50]:50} {zmap[cause]:>7.4f} "
        row += " ".join(f"{results[c][2].get(cause, 0.0):>9.4f}" for c in cols)
        print(row)

    print("\nfire accidents captured by retrieval (of 102):")
    for c in cols:
        tag = " (ALL)" if c == n_total else ""
        print(f"  top_n_incidents={c}{tag}: {results[c][0]} fire accidents")

    # exactness at the broadest setting
    broad = results[n_total][2]
    exact = sum(1 for cause in zmap if abs(broad.get(cause, 0) - zmap[cause]) < 1e-9)
    print(f"\nAt ALL incidents: {exact}/{len(zmap)} causes match Zhang EXACTLY")


if __name__ == "__main__":
    main()

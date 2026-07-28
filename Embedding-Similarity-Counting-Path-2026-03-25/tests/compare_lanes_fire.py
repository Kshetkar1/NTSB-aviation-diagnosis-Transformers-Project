"""Head-to-head on Zhang's fire-cause example (Table 7), 1982-2006 window.

  Zhang / recreated  = empirical P(cause | fire) = count(cause & fire) / 102
  Your engine        = retrieval P(cause | query)

After vocab re-alignment, engine labels match Zhang's dictionary labels exactly,
so we match by exact (lowercased) label. Runs at several top_n_incidents to
answer "what if we take all related incidents instead of top 50?".
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUERY = "What is the probability of fire?"


def run_engine(main_app, top_n_incidents):
    diag = main_app.diagnose_with_conditional_probabilities(
        QUERY, top_n=400, top_n_incidents=top_n_incidents
    )
    agg = {}
    for it in diag.get("weighted_causes", []):
        k = str(it.get("cause", "")).strip().lower()
        agg[k] = agg.get(k, 0.0) + float(it.get("probability", 0.0))
    return agg


def main() -> None:
    import zhang_diagnosis

    emp = zhang_diagnosis.empirical_cause_distribution("fire")
    emp_map = {c["cause"].strip().lower(): c["probability"] for c in emp["causes"]}
    top_zhang = [c for c, _ in sorted(emp_map.items(), key=lambda kv: -kv[1])[:10]]
    print(f"Recreated Zhang: {emp['outcome_count']} fire accidents (target 102)\n")

    import main_app
    n_total = sum(1 for c in main_app.embeddings_map if c.get("source") == "incident")

    for tn in (50, n_total):
        agg = run_engine(main_app, tn)
        rows = [(c, emp_map[c], agg.get(c, 0.0)) for c in
                sorted(emp_map, key=lambda k: -emp_map[k])[:12]]
        label = f"top_n_incidents = {tn}" + (" (ALL)" if tn == n_total else "")
        print("=" * 78)
        print(label)
        print("=" * 78)
        print(f"{'cause':50} {'Zhang':>9} {'engine':>9}")
        print("-" * 72)
        for c, ep, rp in rows:
            flag = "" if rp > 0 else "  (missing)"
            print(f"{c[:50]:50} {ep:>9.4f} {rp:>9.4f}{flag}")
        overlap = len(set(top_zhang) & set(sorted(agg, key=lambda k: -agg[k])[:10]))
        captured = sum(1 for c in top_zhang if agg.get(c, 0) > 0)
        print(f"\nZhang top-10 causes CAPTURED (nonzero): {captured}/10")
        print(f"Zhang top-10 in engine top-10 (overlap): {overlap}/10")
        try:
            from scipy.stats import spearmanr
            shared = [(emp_map[c], agg[c]) for c in emp_map if agg.get(c, 0) > 0]
            if len(shared) >= 3:
                rho, p = spearmanr([s[0] for s in shared], [s[1] for s in shared])
                print(f"Spearman rho (shared causes, n={len(shared)}): {rho:.3f}")
        except Exception as e:
            print(f"(spearman unavailable: {e})")
        print()


if __name__ == "__main__":
    main()

"""Close the P(loss of engine power | evidence) gap (semantic 0.60 vs Zhang 0.95).
Test: varying K (uniform fraction) and similarity-weighted neighbors (with a
temperature/sharpening knob) to see if either concentrates toward Zhang's BN."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main_app  # noqa: E402

ZHANG = {
    "Inoperative engine instruments": 0.95,
    "Combustion liner failure": 0.50,
    "Improper oil usage": 0.95,
}
QUERIES = {
    "Inoperative engine instruments":
        "The aircraft experienced inoperative engine instruments during flight. "
        "The engine instruments failed and were not providing accurate readings.",
    "Combustion liner failure":
        "The aircraft experienced a combustion liner failure during flight. "
        "The combustion assembly liner in the engine cracked and failed.",
    "Improper oil usage":
        "The aircraft had improper oil usage. The engine oil was of improper "
        "grade or contaminated, affecting engine performance.",
}


def has_loep(inc):
    return any("loss of engine power" in (s.get("Occurrence_Description") or "").lower()
               for s in inc.get("sequence_of_events", []))


def ranked_incidents(query, n=200):
    """(ev_id, score) deduped to best score per incident, ranked desc."""
    q = main_app.get_embedding(query)
    scores, matches = main_app.find_top_matches(q)
    best = {}
    for s, m in zip(scores, matches):
        if m.get("source") != "incident":
            continue
        ev = m.get("ev_id")
        if not ev:
            continue
        if ev not in best or s > best[ev]:
            best[ev] = float(s)
    ranked = sorted(best.items(), key=lambda x: -x[1])[:n]
    return ranked


def main() -> None:
    ds = main_app.refined_dataset
    Ks = [5, 10, 15, 20, 30, 50]
    temps = [1.0, 0.05, 0.02, 0.01]  # softmax temperature; lower = sharper

    for ev_label, query in QUERIES.items():
        ranked = ranked_incidents(query)
        flags = [(ev, sc, has_loep(ds.get(ev, {}))) for ev, sc in ranked]
        z = ZHANG[ev_label]
        print("=" * 78)
        print(f"EVIDENCE: {ev_label}    Zhang BN = {z}")
        print("=" * 78)

        print("  uniform fraction by K:")
        for K in Ks:
            sub = flags[:K]
            p = sum(1 for _, _, f in sub if f) / len(sub)
            print(f"    K={K:>3}: {p:.3f}")

        print("  similarity-softmax weighted (K=50) by temperature:")
        sub = flags[:50]
        for t in temps:
            ws = [math.exp(sc / t) for _, sc, _ in sub]
            Z = sum(ws) or 1.0
            p = sum(w for w, (_, _, f) in zip(ws, sub) if f) / Z
            print(f"    T={t:>5}: {p:.3f}")
        print()


if __name__ == "__main__":
    main()

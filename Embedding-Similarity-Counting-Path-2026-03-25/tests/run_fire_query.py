"""Fire the query 'What is the probability of fire?' through YOUR engine
(main_app) and print what it returns — diagnosis (causes) and prognosis
(future events), highlighting anything fire-related."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main_app  # noqa: E402

QUERY = "What is the probability of fire?"


def show(label, items, key_label, key_prob, n=12):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    for i, it in enumerate(items[:n], 1):
        lab = str(it.get(key_label, ""))
        p = float(it.get(key_prob, 0.0))
        mark = "  <== FIRE" if "fire" in lab.lower() else ""
        print(f"  {i:2}. {p*100:6.2f}%  {lab[:64]}{mark}")
    # also surface any fire entry even if outside top-n
    fire = [it for it in items if "fire" in str(it.get(key_label, "")).lower()]
    if fire:
        print("  --- all fire-related entries ---")
        for it in fire:
            print(f"      {float(it.get(key_prob,0.0))*100:6.2f}%  {str(it.get(key_label,''))[:64]}")


def main() -> None:
    print(f"DATA_LOADED = {main_app.DATA_LOADED} | backend = {main_app.active_backend}")
    print(f"QUERY: {QUERY!r}")

    diag = main_app.diagnose_with_conditional_probabilities(QUERY, top_n=10, top_n_incidents=50)
    show("DIAGNOSIS  P(cause | query)", diag.get("weighted_causes") or [],
         "cause", "probability")

    prog = main_app.predict_future_events(QUERY, top_n_incidents=50, max_chain_steps=1)
    show("PROGNOSIS  P(next event | query)", prog.get("future_events") or [],
         "event", "probability")


if __name__ == "__main__":
    main()

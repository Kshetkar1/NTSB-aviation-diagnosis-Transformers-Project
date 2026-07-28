#!/usr/bin/env python3
"""Quick Zhang vs our-system probability comparison (A0 embedding-only)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["NTSB_USE_TRAIN_INDEX"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
_SHARED = REPO_ROOT / "shared" / "code"
_FROZEN_CODE = FROZEN_DIR / "code"
for _p in (_SHARED, _FROZEN_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
ROOT = REPO_ROOT
SLIDES = ROOT / "Testing_Structural_Mapping_Slides" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SLIDES))

import main_app  # noqa: E402
from cause_code_mapper import (  # noqa: E402
    build_code_embeddings,
    map_distribution_to_codes,
    match_events_to_targets,
    match_causes_to_targets,
)

# --- Table 9 (Zhang §5.4) ---
TABLE_9_EVIDENCE = [
    "Inoperative engine instruments",
    "Combustion liner failure",
    "Improper oil usage",
    "Inop. instruments + Improper oil",
    "All three causes",
]
ZHANG_T9 = {
    "Loss of engine power": [0.95, 0.50, 0.95, 0.99, 1.0],
    "Forced landing": [0.1357, 0.0714, 0.1357, 0.1471, 0.1429],
    "No injury": [0.9431, 0.9978, 0.9958, 0.9429, 0.9899],
}
QUERIES_T9 = [
    "The aircraft experienced inoperative engine instruments during flight. "
    "The engine instruments failed and were not providing accurate readings.",
    "The aircraft experienced a combustion liner failure during flight. "
    "The combustion assembly liner in the engine cracked and failed.",
    "The aircraft had improper oil usage. The engine oil was of improper "
    "grade or contaminated, affecting engine performance.",
    "The aircraft experienced both inoperative engine instruments and "
    "improper oil usage. The engine instruments failed while the engine "
    "oil was also contaminated or of improper grade.",
    "The aircraft experienced loss of engine power due to multiple causes: "
    "inoperative engine instruments, combustion liner failure, and "
    "improper oil usage all contributed to the engine failure.",
]

# --- Fig 12 pilot error (Zhang §5.3.1) ---
FIG12_NODES = {
    "Hard landing": (2.02e-7, 5.99e-2, 5.99e-2),
    "No injury": (99.99e-2, 97.0e-2, 61.30e-2),
}
QUERY_PE = (
    "The pilot in command made an error during the approach phase of flight. "
    "The pilot's actions were erroneous and contributed to the incident."
)
QUERY_PE_UA = (
    "The pilot in command made an error during the approach phase, resulting "
    "in an unstabilized approach. The approach was not maintained stable in "
    "terms of speed, descent rate, and flight path, leading to a dangerous situation."
)

# --- Maha Table 4 style: multi-evidence → fire (illustrative BN scenario) ---
FIRE_SCENARIOS = [
    (
        "worn_yes_overheat_yes",
        "Landing gear normal brake system is worn out and the electrical system is overheating. "
        "I observe fire on the aircraft.",
        0.99,
    ),
    (
        "worn_yes_overheat_no",
        "Landing gear normal brake system is worn out but the electrical system is not overheating. "
        "I observe fire on the aircraft.",
        0.93,
    ),
    (
        "worn_no_overheat_yes",
        "Landing gear normal brake system is not worn out but the electrical system is overheating. "
        "I observe fire on the aircraft.",
        0.95,
    ),
    (
        "fire_only_diagnosis",
        "I observe fire on the aircraft during flight. What caused the fire?",
        None,
    ),
]


def find_code_prob(coded_dist, target_label: str) -> float:
    import numpy as np

    target_emb = np.array(main_app.get_embedding(target_label))
    best_prob, best_sim = 0.0, -1.0
    for cd in coded_dist:
        label = cd.get("label", "")
        if not label:
            continue
        label_emb = np.array(main_app.get_embedding(label))
        norm = np.linalg.norm(target_emb) * np.linalg.norm(label_emb)
        sim = float(np.dot(target_emb, label_emb) / norm) if norm > 0 else 0.0
        if sim > best_sim:
            best_sim = sim
            best_prob = float(cd.get("probability", 0.0))
    return best_prob if best_sim >= 0.5 else 0.0


def run():
    if not main_app.DATA_LOADED:
        raise RuntimeError("Train index not loaded.")

    codes, labels, code_embs = build_code_embeddings()

    print("=" * 72)
    print("ZHANG TABLE 9 — Loss of Engine Power (A0 embedding-only)")
    print("=" * 72)
    print(f"{'Evidence':<32} {'Outcome':<22} {'Zhang':>8} {'Ours':>8} {'Ratio':>8}")
    print("-" * 72)

    for i, (ev, query) in enumerate(zip(TABLE_9_EVIDENCE, QUERIES_T9)):
        diag = main_app.diagnose_with_conditional_probabilities(
            query, top_n=10, top_n_incidents=50
        )
        dist = [{"cause": c["cause"], "probability": c["probability"]} for c in diag.get("weighted_causes") or []]
        coded, _ = map_distribution_to_codes(dist, codes, labels, code_embs)
        prog = main_app.predict_future_events(query, top_n_incidents=50, max_chain_steps=1)
        prog_matched = match_events_to_targets(prog.get("future_events") or [], list(ZHANG_T9.keys()))

        for outcome in ZHANG_T9:
            z = ZHANG_T9[outcome][i]
            if outcome == "Loss of engine power":
                ours = find_code_prob(coded, outcome)
            else:
                ours = prog_matched.get(outcome, 0.0)
            ratio = ours / z if z > 0 else float("nan")
            print(f"{ev[:32]:<32} {outcome:<22} {z:8.4f} {ours:8.4f} {ratio:8.2f}x")

        if i == 0:
            print("\n  Top-5 diagnosis causes (evidence 1):")
            for j, c in enumerate((diag.get("weighted_causes") or [])[:5], 1):
                print(f"    {j}. {c['probability']*100:5.2f}% — {c['cause'][:70]}")
        print()

    print("=" * 72)
    print("ZHANG FIG 12 — Pilot error propagation (A0)")
    print("=" * 72)
    for cond_name, query, zhang_col in [
        ("After pilot error", QUERY_PE, 1),
        ("After PE + unstable approach", QUERY_PE_UA, 2),
    ]:
        print(f"\n--- {cond_name} ---")
        diag = main_app.diagnose_with_conditional_probabilities(query, top_n=10, top_n_incidents=50)
        prog = main_app.predict_future_events(query, top_n_incidents=50, max_chain_steps=1)
        diag_m = match_causes_to_targets(diag.get("weighted_causes") or [], list(FIG12_NODES.keys()))
        prog_m = match_events_to_targets(prog.get("future_events") or [], list(FIG12_NODES.keys()))
        for node, (_, post_pe, post_ua) in FIG12_NODES.items():
            z = post_pe if zhang_col == 1 else post_ua
            ours = max(diag_m.get(node, 0.0), prog_m.get(node, 0.0))
            print(f"  {node:<18} Zhang={z:8.4f}  Ours={ours:8.4f}  ({ours/z:.2f}x)" if z else "")

    print("\n" + "=" * 72)
    print("TABLE 4 STYLE — Multi-evidence fire scenarios (A0)")
    print("  (Zhang refs are illustrative CPT values from meeting, not Table 9)")
    print("=" * 72)
    for key, query, zhang_ref in FIRE_SCENARIOS:
        diag = main_app.diagnose_with_conditional_probabilities(query, top_n=10, top_n_incidents=50)
        prog = main_app.predict_future_events(query, top_n_incidents=50, max_chain_steps=1)
        fire_diag = match_causes_to_targets(diag.get("weighted_causes") or [], ["Fire", "Fire/explosion"])
        fire_prog = match_events_to_targets(prog.get("future_events") or [], ["Fire", "Fire/explosion"])
        ours_fire = max(fire_diag.get("Fire", 0.0), fire_diag.get("Fire/explosion", 0.0),
                        fire_prog.get("Fire", 0.0), fire_prog.get("Fire/explosion", 0.0))
        ref_s = f"{zhang_ref:.2f}" if zhang_ref is not None else "n/a"
        print(f"\n[{key}] Zhang ref P(fire)≈{ref_s}")
        print(f"  Query: {query[:90]}...")
        print(f"  Our matched P(fire): {ours_fire:.4f}")
        print("  Top-3 diagnosis:")
        for j, c in enumerate((diag.get("weighted_causes") or [])[:3], 1):
            print(f"    {j}. {c['probability']*100:5.2f}% — {c['cause'][:65]}")
        print("  Top-3 prognosis:")
        for j, c in enumerate((prog.get("future_events") or [])[:3], 1):
            print(f"    {j}. {c['probability']*100:5.2f}% — {c['event'][:65]}")


if __name__ == "__main__":
    run()

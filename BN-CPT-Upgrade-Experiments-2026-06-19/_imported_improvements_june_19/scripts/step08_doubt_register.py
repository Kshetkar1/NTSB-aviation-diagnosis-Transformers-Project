#!/usr/bin/env python3
"""Step 8 — Doubt register with computed evidence.

For each doubt Maha could raise, attach a status and (where possible) a number
computed from vendored data — so the plan is bulletproof by evidence, not assertion.

Notably tackles doubt #5 (absence states): Lane 1 (structured-field counting)
computes P(no injury) fine; only Lane 2 (narrative retrieval) is blind to it.
Emits outputs/step08_doubt_register.{json,md}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpt_config import ENGINE_VS_ZHANG_PATH, OUTPUT_DIR, REFINED_DATA_PATH  # noqa: E402
from stats import empirical_p, wilson_ci  # noqa: E402


def _eventsoe(inc: dict) -> set[int]:
    out = set()
    for e in inc.get("sequence_of_events") or []:
        if isinstance(e, dict) and e.get("eventsoe_no") is not None:
            try:
                out.add(int(e["eventsoe_no"]))
            except (TypeError, ValueError):
                pass
    return out


def main() -> None:
    data = json.loads(REFINED_DATA_PATH.read_text(encoding="utf-8"))
    incs = list(data.values())
    n = len(incs)

    # --- Doubt #5: absence states are computable in Lane 1 ---
    no_injury = sum(1 for i in incs if str(i.get("ev_highest_injury") or "").upper() == "NONE")
    p_ni = empirical_p(no_injury, n)
    lo_ni, hi_ni = wilson_ci(no_injury, n)

    # --- Doubt #7: example data-rich conditional with CI (severe damage | fire occ) ---
    FIRE = {150, 170}
    fire_inc = [i for i in incs if _eventsoe(i) & FIRE]
    severe = sum(1 for i in fire_inc if str(i.get("damage") or "").upper() in ("SUBS", "DEST"))
    p_sf = empirical_p(severe, len(fire_inc))
    lo_sf, hi_sf = wilson_ci(severe, len(fire_inc))

    # --- Doubt #4: A0 vs A2 from vendored numbers ---
    eng = json.loads(ENGINE_VS_ZHANG_PATH.read_text(encoding="utf-8"))
    a = eng.get("a0_vs_a2", {})

    register = [
        {"id": 1, "doubt": "0.09 vs 0.95 compares different quantities",
         "severity": "fatal", "status": "ADDRESSED",
         "evidence": "Two-lane framing: Lane 1 counts P(outcome|evidence); Lane 2 retrieval conditions on narrative. Each P defined explicitly."},
        {"id": 2, "doubt": "Can counting reproduce Zhang Table 9?",
         "severity": "high", "status": "HONEST-LIMIT",
         "evidence": "Band, not exact: coding era (eADMS vs legacy) + Beta-CDF smoothing + sparsity. Table 5: 0.73 vs 0.92 (within 0.20)."},
        {"id": 3, "doubt": "Free-text -> 54-code mapping is guessing",
         "severity": "med", "status": "PENDING (Lane 2 / needs embeddings)",
         "evidence": "cause_code_mapper uses embedding NN @ threshold 0.40; audit on 30-50 causes scheduled (requires OpenAI embeddings)."},
        {"id": 4, "doubt": "Structural mapping does nothing (A0 ~ A2)",
         "severity": "thesis-risk", "status": "STATED",
         "evidence": f"A2 changed top-1 in {a.get('a2_changed_top1_incidents')}/{a.get('total_incidents')}; McNemar p={a.get('mcnemar_p')}. Validates, doesn't boost at code level. Value is label-quality + top-1 within clusters."},
        {"id": 5, "doubt": "P(no injury)=0 vs 0.94",
         "severity": "med", "status": "RESOLVED for Lane 1",
         "evidence": f"Lane 1 computes P(no injury) = {no_injury}/{n} = {p_ni:.3f} (95% CI [{lo_ni:.3f}, {hi_ni:.3f}]). Only Lane 2 (narrative retrieval) is blind to absence; fix = complement."},
        {"id": 6, "doubt": "Are these calibrated probabilities?",
         "severity": "med", "status": "PARTIAL",
         "evidence": "Lane 1 reports Wilson CIs (uncertainty). Full calibration (Brier/reliability) pending a held-out prediction task."},
        {"id": 7, "doubt": "n=1-4 isn't trustable",
         "severity": "med", "status": "ADDRESSED",
         "evidence": f"Magnitude claimed only at n>=10 with Wilson CIs. Example data-rich cell: P(severe damage|fire occ) = {severe}/{len(fire_inc)} = {p_sf:.3f} CI [{lo_sf:.3f}, {hi_sf:.3f}]."},
        {"id": 8, "doubt": "Did you leak test into train?",
         "severity": "med", "status": "PROTOCOL",
         "evidence": "Frozen test ids, train-only index (NTSB_USE_TRAIN_INDEX), exclude-self retrieval available in the engine harness."},
        {"id": 9, "doubt": "Table 4 is what I asked for",
         "severity": "low", "status": "REFRAMED",
         "evidence": "Verified in NTSB.xdsl: Fire's only parent is anti-ice (0.95). Table 4 is the Fig-2 pedagogical CPT; confirm origin with Maha."},
        {"id": 10, "doubt": "Is your Zhang replication stable?",
         "severity": "low", "status": "VALIDATED",
         "evidence": "Seed-band + 99M sample runs; 88% of 86 published cells reproduce within tolerance."},
    ]

    summary = {
        "n_incidents": n,
        "p_no_injury": {"k": no_injury, "n": n, "p": p_ni, "ci": [lo_ni, hi_ni]},
        "p_severe_given_fire_occ": {"k": severe, "n": len(fire_inc), "p": p_sf, "ci": [lo_sf, hi_sf]},
        "register": register,
    }
    (OUTPUT_DIR / "step08_doubt_register.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Step 8 — Doubt register (with computed evidence)",
        "",
        f"Corpus: {n} incidents. Absence-state check: P(no injury) = "
        f"{p_ni:.3f} (CI [{lo_ni:.3f}, {hi_ni:.3f}]) — **Lane 1 is not blind to absence states.**",
        "",
        "| # | Doubt | Severity | Status | Evidence |",
        "|---|-------|----------|--------|----------|",
    ]
    for r in register:
        lines.append(
            f"| {r['id']} | {r['doubt']} | {r['severity']} | {r['status']} | {r['evidence']} |"
        )
    (OUTPUT_DIR / "step08_doubt_register.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Step 8 → outputs/step08_doubt_register.md")
    print(f"  P(no injury) Lane 1 = {p_ni:.3f}  (resolves doubt #5 for Lane 1)")
    print(f"  P(severe|fire occ) = {p_sf:.3f} CI [{lo_sf:.3f},{hi_sf:.3f}]")


if __name__ == "__main__":
    main()

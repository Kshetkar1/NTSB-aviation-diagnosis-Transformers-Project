"""Phase 3: run YOUR engine's diagnosis on the corrected 1982-2006 data and
compare the fire-cause distribution to Zhang's Table 7.

Builds an in-memory retrieval index for the 1982-2006 window by REUSING the
existing narrative vectors (no new embedding calls) and RE-INJECTING the
corrected causes (findings restored in Phase 2). Then overrides main_app's
module globals and runs four Zhang-style queries.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DP = ROOT / "data" / "processed"

ZHANG_T7 = {
    "airframe/component/system failure/malfunction": 0.3137,
    "loss of engine power (total) - mechanical failure/malfunction": 0.0882,
    "electrical system, electric wiring": 0.0882,
    "fluid, fuel": 0.0588,
    "auxiliary power unit (apu)": 0.0490,
    "maintenance, installation": 0.0392,
    "brakes (normal)": 0.0196,
}

QUERIES = [
    "What is the probability of fire?",
    "The brake system is worn out and the electrical system is overheating. What is the probability of fire?",
    "The brake system is worn out and the electrical system is not overheating. What is the probability of fire?",
    "The brake system is not worn out and the electrical system is overheating. What is the probability of fire?",
]


def subject_key(label: str) -> str:
    """Collapse engine cause (subject, sub, modifier) to Zhang subject level
    by dropping the trailing modifier component."""
    parts = [p.strip() for p in label.split(",")]
    if len(parts) >= 3:
        return ", ".join(parts[:2])
    return label.strip()


def aggregate(causes):
    agg = {}
    for it in causes:
        k = subject_key(str(it.get("cause", "")))
        agg[k] = agg.get(k, 0.0) + float(it.get("probability", 0.0))
    return sorted(agg.items(), key=lambda kv: -kv[1])


def build_window_index():
    ds = json.loads((DP / "refined_dataset_1982_2006.json").read_text())
    window = set(ds.keys())
    emb = np.load(DP / "embeddings.npy")
    emap = json.loads((DP / "embeddings_map.json").read_text())

    sel_idx, new_map = [], []
    for i, chunk in enumerate(emap):
        ev_id = chunk.get("ev_id")
        if ev_id not in window:
            continue
        inc = ds[ev_id]
        old_bd = chunk.get("bayesian_data") or chunk.get("diagnostic_data") or {}
        narr_causes = list(old_bd.get("narrative_causes", []))
        finding_strs, causal = [], []
        for fd in inc.get("findings", []):
            s = fd.get("finding_description", "").strip()
            if not s:
                continue
            mod = (fd.get("modifier_description") or "").strip()
            label = f"{s}, {mod}" if mod else s
            finding_strs.append(label)
            if str(fd.get("Cause_Factor") or "").strip().upper().startswith("C"):
                causal.append(label)
        all_causes = []
        for c in finding_strs + narr_causes:
            if c and c not in all_causes:
                all_causes.append(c)
        bd = {
            "findings": finding_strs,
            "causal_findings": causal or finding_strs,
            "narrative_causes": narr_causes,
            "all_causes": all_causes,
            "narr_cause": old_bd.get("narr_cause"),
            "has_diagnostic_data": bool(all_causes),
        }
        nc = dict(chunk)
        nc["bayesian_data"] = bd
        nc["diagnostic_data"] = bd
        sel_idx.append(i)
        new_map.append(nc)

    return ds, emb[sel_idx], new_map


def main() -> None:
    ds, emb, emap = build_window_index()
    print(f"window index: {len(emap)} narrative vectors, {len(ds)} incidents")

    import main_app
    main_app.refined_dataset = ds
    main_app.embeddings = emb
    main_app.embeddings_map = emap
    main_app.DATA_LOADED = True

    for q in QUERIES:
        print("\n" + "#" * 80)
        print(f"QUERY: {q}")
        print("#" * 80)
        diag = main_app.diagnose_with_conditional_probabilities(
            q, top_n=15, top_n_incidents=50
        )
        causes = diag.get("weighted_causes") or []
        print(f"\nTop causes  (incidents analyzed: {diag.get('total_incidents_analyzed','?')})")
        print(f"{'cause':62} {'P(cause|q)':>10} {'Zhang T7':>9}")
        print("-" * 84)
        for it in causes[:15]:
            lab = str(it.get("cause", ""))
            p = float(it.get("probability", 0.0))
            z = ZHANG_T7.get(lab.strip().lower())
            zs = f"{z:.4f}" if z is not None else ""
            mark = " <FIRE" if "fire" in lab.lower() else ""
            print(f"{lab[:62]:62} {p:>10.4f} {zs:>9}{mark}")

        print(f"\nSUBJECT-LEVEL aggregation (modifier collapsed)  [matches Zhang labels]")
        print(f"{'cause (subject)':62} {'engine':>10} {'Zhang T7':>9}")
        print("-" * 84)
        for k, p in aggregate(causes)[:15]:
            z = ZHANG_T7.get(k.strip().lower())
            zs = f"{z:.4f}" if z is not None else ""
            mark = " <-- in T7" if z is not None else ""
            print(f"{k[:62]:62} {p:>10.4f} {zs:>9}{mark}")


if __name__ == "__main__":
    main()

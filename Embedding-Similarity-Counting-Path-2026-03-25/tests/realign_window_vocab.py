"""Re-align the 1982-2006 window index to Zhang's cause vocabulary.

For each incident chunk, set all_causes = BOTH layers, using the exact
dictionary labels (metaData 'meaning'), with NO modifier appended:
  - findings  (finding_description)  -> subject-level causes
  - occurrences (Occurrence_Description) -> occurrence-level causes
This matches deriveNamebyCode + buildOneGraphRep in Zhang's main.py.
"""
from __future__ import annotations

import json
from pathlib import Path

DP = Path(__file__).resolve().parents[1] / "data" / "processed"
DS = DP / "refined_dataset_1982_2006.json"
EMAP = DP / "embeddings_map_1982_2006.json"


def main() -> None:
    ds = json.loads(DS.read_text(encoding="utf-8"))
    emap = json.loads(EMAP.read_text(encoding="utf-8"))

    n = 0
    for chunk in emap:
        ev_id = chunk.get("ev_id")
        inc = ds.get(ev_id)
        if not inc:
            continue
        findings = [
            (fd.get("finding_description") or "").strip()
            for fd in inc.get("findings", [])
        ]
        findings = [f for f in findings if f]
        causal = [
            (fd.get("finding_description") or "").strip()
            for fd in inc.get("findings", [])
            if str(fd.get("Cause_Factor") or "").strip().upper().startswith("C")
        ]
        causal = [c for c in causal if c]
        occurrences = [
            (s.get("Occurrence_Description") or "").strip()
            for s in inc.get("sequence_of_events", [])
        ]
        occurrences = [o for o in occurrences if o]

        all_causes = []
        for c in findings + occurrences:
            if c not in all_causes:
                all_causes.append(c)

        old_bd = chunk.get("bayesian_data") or {}
        bd = {
            "findings": findings,
            "causal_findings": causal or findings,
            "occurrences": occurrences,
            "narrative_causes": list(old_bd.get("narrative_causes", [])),
            "all_causes": all_causes,
            "narr_cause": old_bd.get("narr_cause"),
            "has_diagnostic_data": bool(all_causes),
        }
        chunk["bayesian_data"] = bd
        chunk["diagnostic_data"] = bd
        n += 1

    EMAP.write_text(json.dumps(emap), encoding="utf-8")
    print(f"re-aligned {n} window chunks to Zhang vocabulary (findings + occurrences, dict labels, no modifier)")


if __name__ == "__main__":
    main()

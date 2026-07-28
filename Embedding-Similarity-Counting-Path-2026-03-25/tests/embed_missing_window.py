"""Embed window incidents that have no narrative vector, using synthetic text
built from their structured content (occurrence sequence + findings).

These are old (pre-2007) records with structured NTSB coding but no narrative
text anywhere, so retrieval could not surface them. After this, the window
index covers them and the retrieval+Zhang-denominator lane can reach all 102
fire accidents.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DP = ROOT / "data" / "processed"
EMB = DP / "embeddings_1982_2006.npy"
EMAP = DP / "embeddings_map_1982_2006.json"
DS = DP / "refined_dataset_1982_2006.json"


def synth_text(inc: dict) -> str:
    occ = [
        (s.get("Occurrence_Description") or "").strip()
        for s in inc.get("sequence_of_events", [])
    ]
    occ = [o for o in occ if o]
    find = [
        (f.get("finding_description") or "").strip()
        for f in inc.get("findings", [])
    ]
    find = [f for f in find if f]
    parts = []
    if occ:
        parts.append("Sequence of events: " + "; ".join(occ))
    if find:
        parts.append("Findings: " + "; ".join(find))
    return ". ".join(parts)


def bayesian_data(inc: dict) -> dict:
    findings = [(f.get("finding_description") or "").strip() for f in inc.get("findings", [])]
    findings = [f for f in findings if f]
    causal = [
        (f.get("finding_description") or "").strip()
        for f in inc.get("findings", [])
        if str(f.get("Cause_Factor") or "").strip().upper().startswith("C")
    ]
    causal = [c for c in causal if c]
    occurrences = [(s.get("Occurrence_Description") or "").strip() for s in inc.get("sequence_of_events", [])]
    occurrences = [o for o in occurrences if o]
    all_causes = []
    for c in findings + occurrences:
        if c not in all_causes:
            all_causes.append(c)
    return {
        "findings": findings,
        "causal_findings": causal or findings,
        "occurrences": occurrences,
        "narrative_causes": [],
        "all_causes": all_causes,
        "narr_cause": None,
        "has_diagnostic_data": bool(all_causes),
    }


def main() -> None:
    ds = json.loads(DS.read_text(encoding="utf-8"))
    emb = np.load(EMB)
    emap = json.loads(EMAP.read_text(encoding="utf-8"))
    indexed = {c.get("ev_id") for c in emap}

    todo = []
    for ev_id, inc in ds.items():
        if ev_id in indexed:
            continue
        text = synth_text(inc)
        if len(text) > 20:
            todo.append((ev_id, text))
    print(f"missing window incidents with structured text to embed: {len(todo)}")
    if not todo:
        return

    import main_app
    from config import EMBEDDING_MODEL

    client = main_app.get_client()
    new_vecs, new_chunks = [], []
    B = 100
    for i in range(0, len(todo), B):
        batch = todo[i:i + B]
        resp = client.embeddings.create(input=[t for _, t in batch], model=EMBEDDING_MODEL)
        for (ev_id, text), d in zip(batch, resp.data):
            new_vecs.append(d.embedding)
            new_chunks.append({
                "source": "incident",
                "ev_id": ev_id,
                "type": "narrative",
                "synthetic_text": True,
                "bayesian_data": bayesian_data(ds[ev_id]),
                "diagnostic_data": bayesian_data(ds[ev_id]),
            })
        print(f"  embedded {min(i + B, len(todo))}/{len(todo)}")

    new_arr = np.asarray(new_vecs, dtype=emb.dtype)
    merged = np.vstack([emb, new_arr])
    np.save(EMB, merged)
    (EMAP).write_text(json.dumps(emap + new_chunks))
    print(f"index grew {emb.shape[0]} -> {merged.shape[0]} vectors")

    # verify fire coverage
    fire = [k for k, v in ds.items() if any(
        (s.get("Occurrence_Description") or "").strip().lower() == "fire"
        for s in v.get("sequence_of_events", []))]
    now_indexed = {c.get("ev_id") for c in (emap + new_chunks)}
    print(f"fire accidents in index now: {sum(1 for k in fire if k in now_indexed)}/{len(fire)}")


if __name__ == "__main__":
    main()

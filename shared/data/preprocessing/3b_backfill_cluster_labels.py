#!/usr/bin/env python3
"""Backfill cluster_label for incidents the original clustering skipped.

WHY they were skipped: step 2 (embedding) only embedded incidents with narrative
text; 379/1742 window incidents (incl. 19/102 fires) are legacy records with NO
narrative, so step 3's k-means never saw them and left cluster_label unset.

FIX: those incidents still have structured data (sequence_of_events + findings).
We synthesize a pseudo-narrative from the structured fields, embed it with the
SAME model as the corpus, and assign the nearest existing cluster centroid
(cosine). Centroids = mean of the already-labeled member embeddings, so no
re-clustering happens and every previously-assigned label is untouched.

Writes cluster_label + cluster_label_source="structured_backfill" (auditable)
back into the active dataset JSON.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      data/preprocessing/3b_backfill_cluster_labels.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def pseudo_narrative(inc: dict) -> str:
    """Readable text from structured fields, mimicking narrative content."""
    parts = []
    for s in inc.get("sequence_of_events", []):
        occ = (s.get("Occurrence_Description") or "").strip()
        ph = (s.get("phase_description") or "").strip()
        if occ:
            parts.append(f"{occ} during {ph}" if ph else occ)
    for f in inc.get("findings", []):
        desc = (f.get("finding_description") or "").strip()
        mod = (f.get("modifier_description") or "").strip()
        who = (f.get("person_description") or "").strip()
        if desc:
            seg = f"{desc} - {mod}" if mod else desc
            if who:
                seg += f" ({who})"
            parts.append(seg)
    return ". ".join(parts)


def main():
    ds_path = Path(config.ACTIVE_INCIDENT_DATA_PATH)
    ds = json.loads(ds_path.read_text(encoding="utf-8"))
    emb = np.load(config.EMBEDDINGS_PATH)
    emap = json.loads(Path(config.EMBEDDINGS_MAP_PATH).read_text(encoding="utf-8"))

    # First incident embedding per ev_id (same dedup rule as the clustering step).
    first_idx: dict[str, int] = {}
    for i, m in enumerate(emap):
        ev = m.get("ev_id")
        if m.get("source") == "incident" and ev and ev not in first_idx:
            first_idx[ev] = i

    # Cluster centroids from already-labeled incidents.
    members = defaultdict(list)
    for ev, inc in ds.items():
        lab = (inc.get("cluster_label") or "").strip()
        if lab and ev in first_idx:
            members[lab].append(first_idx[ev])
    labels = sorted(members)
    cents = np.stack([emb[members[k]].mean(axis=0) for k in labels])
    cents = cents / np.linalg.norm(cents, axis=1, keepdims=True)
    print(f"{len(labels)} cluster centroids built from labeled incidents.")

    todo = []
    for ev, inc in ds.items():
        if (inc.get("cluster_label") or "").strip():
            continue
        text = pseudo_narrative(inc)
        if text.strip():
            todo.append((ev, text))
    print(f"{len(todo)} unlabeled incidents with structured data to backfill.")
    if not todo:
        return 0

    import main_app  # noqa: E402  (imports OpenAI client lazily)
    client = main_app.get_client()

    assigned = 0
    B = 100
    for start in range(0, len(todo), B):
        batch = todo[start:start + B]
        resp = client.embeddings.create(
            input=[t[:6000] for _, t in batch],
            model=main_app.EMBEDDING_MODEL,
        )
        vecs = np.array([d.embedding for d in resp.data])
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        sims = vecs @ cents.T
        for (ev, _), row in zip(batch, sims):
            k = int(np.argmax(row))
            ds[ev]["cluster_label"] = labels[k]
            ds[ev]["cluster_label_source"] = "structured_backfill"
            assigned += 1
        print(f"  ...{min(start + B, len(todo))}/{len(todo)} embedded+assigned")

    ds_path.write_text(json.dumps(ds, indent=4), encoding="utf-8")
    print(f"Assigned {assigned}; saved to {ds_path}")

    # Report the fire backfills specifically (the ones Maha would ask about).
    import zhang_diagnosis as zd  # noqa: E402
    print("\nBackfilled FIRE accidents:")
    for ev, inc in ds.items():
        if inc.get("cluster_label_source") == "structured_backfill" and \
                zd._has_outcome(inc, {"fire"}):
            print(f"  {ev}  ->  {inc['cluster_label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

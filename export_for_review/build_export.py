#!/usr/bin/env python3
"""Build the self-contained export bundle.

Constructs the frozen upgraded BN from the 1982-2006 data, serializes it to
a portable JSON and BIF, picks 10 representative narratives, extracts their
coded fields, and writes nodes.json.

Run from the repo root:
    python3 export_for_review/build_export.py
"""
import csv
import json
import os
import random
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "Frozen-BN-Narrative-Evidence-2026-07-20"
EXPORT = REPO / "export_for_review"

os.environ.pop("NTSB_FULL_CORPUS", None)
for p in (REPO / "shared/code", FROZEN / "code", FROZEN / "tests"):
    sys.path.insert(0, str(p))

import numpy as np
import pyagrum as gum
import prognosis as pg
from bn_upgraded import (build_upgraded, INJ_NODE, DMG_NODE,
                         INJ_STATES, DMG_STATES, injury_state, damage_state)

PROC = REPO / "shared/data/processed"

# ──────────────────────────────────────────────────────────────────────────────
# 1. BUILD AND SERIALIZE THE FROZEN NETWORK
# ──────────────────────────────────────────────────────────────────────────────
print("building network …", flush=True)
ds = pg.load_dataset()
bn, meta = build_upgraded(ds)
print(f"  {bn.size()} nodes, {bn.sizeArcs()} arcs")

# 1a. pyAgrum BIF — requires BIF-safe node names (no leading digits, no
#     special chars). Some NTSB labels like "1 engine" violate this.
bif_path = EXPORT / "frozen_bn.bif"
try:
    gum.saveBN(bn, str(bif_path))
    print(f"  wrote {bif_path.name}  ({bif_path.stat().st_size / 1e6:.1f} MB)")
except Exception as e:
    print(f"  BIF export failed ({e.__class__.__name__}: {e})")
    print("  → will rely on JSON + pyAgrum .o3prm/.bifxml instead")
    bif_path = None
    # try BIFXML which is more permissive
    bifxml_path = EXPORT / "frozen_bn.bifxml"
    try:
        bn.saveBIFXML(str(bifxml_path))
        bif_path = bifxml_path
        print(f"  wrote {bifxml_path.name}  ({bifxml_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as e2:
        print(f"  BIFXML also failed ({e2.__class__.__name__}: {e2})")
        # last resort: pyAgrum's native o3prm
        o3_path = EXPORT / "frozen_bn.o3prm"
        try:
            bn.saveO3PRM(str(o3_path))
            print(f"  wrote {o3_path.name}")
        except Exception:
            print("  all native formats failed — JSON export is the only portable copy")

# 1b. JSON with full CPTs
bn_json = {"nodes": [], "arcs": []}
for nid in bn.nodes():
    v = bn.variable(nid)
    states = [v.label(i) for i in range(v.domainSize())]
    parent_ids = list(bn.parents(nid))
    parent_names = [bn.variable(p).name() for p in parent_ids]
    cpt = bn.cpt(nid)
    cpt_arr = np.asarray(cpt.toarray(), dtype=float)
    bn_json["nodes"].append({
        "name": v.name(),
        "states": states,
        "parents": parent_names,
        "cpt_flat": cpt_arr.ravel().tolist(),
        "cpt_shape": list(cpt_arr.shape),
    })
for a in bn.arcs():
    bn_json["arcs"].append({
        "parent": bn.variable(a[0]).name(),
        "child": bn.variable(a[1]).name(),
    })
bn_json["meta"] = {
    "total_nodes": bn.size(),
    "total_arcs": bn.sizeArcs(),
    "severity_parents": meta.get("severity_parents", []),
    "outcome_nodes": [INJ_NODE, DMG_NODE],
    "outcome_states": {INJ_NODE: INJ_STATES, DMG_NODE: DMG_STATES},
    "build_window": "1982-2006",
    "build_accidents": len(ds),
}
bn_json_path = EXPORT / "frozen_bn.json"
bn_json_path.write_text(json.dumps(bn_json, indent=2))
print(f"  wrote {bn_json_path.name}  ({bn_json_path.stat().st_size / 1e6:.1f} MB)")

# ──────────────────────────────────────────────────────────────────────────────
# 2. NODES.JSON — every node, states, description
# ──────────────────────────────────────────────────────────────────────────────
nodes_list = []
for nid in sorted(bn.nodes()):
    v = bn.variable(nid)
    name = v.name()
    states = [v.label(i) for i in range(v.domainSize())]
    if name == INJ_NODE:
        desc = "Worst personnel injury level in this accident (4-state)."
    elif name == DMG_NODE:
        desc = "Aircraft damage level in this accident (4-state)."
    elif name.startswith("person: "):
        desc = f"Human-factors node: {name[8:]} was identified in findings."
    elif states == ["Yes", "No"]:
        desc = f"Event/finding node: '{name}' occurred in the accident sequence."
    else:
        desc = ""
    nodes_list.append({"name": name, "states": states, "description": desc})
(EXPORT / "nodes.json").write_text(json.dumps(nodes_list, indent=2))
print(f"  wrote nodes.json  ({len(nodes_list)} nodes)")

# ──────────────────────────────────────────────────────────────────────────────
# 3. PICK 10 NARRATIVES — prefer longer, messier ones from 2007-2019 held-out
# ──────────────────────────────────────────────────────────────────────────────
full = json.loads((PROC / "refined_dataset.json").read_text())
window = set(json.loads((PROC / "refined_dataset_1982_2006.json").read_text()))
held = [(k, v) for k, v in full.items()
        if k not in window and len(str(v.get("narr_accf") or "").strip()) >= 200]
held.sort(key=lambda t: -len(str(t[1].get("narr_accf", ""))))
random.seed(42)
pool = held[:80]
random.shuffle(pool)
selected = pool[:10]

narr_dir = EXPORT / "narratives"
narr_dir.mkdir(exist_ok=True)
ev_ids = []
for ev_id, inc in selected:
    narr = str(inc.get("narr_accf", ""))
    safe_name = ev_id.replace("/", "_")
    (narr_dir / f"{safe_name}.txt").write_text(narr)
    ev_ids.append(ev_id)
print(f"  wrote {len(ev_ids)} narratives (longest: "
      f"{max(len(str(inc['narr_accf'])) for _, inc in selected)} chars)")

# ──────────────────────────────────────────────────────────────────────────────
# 4. CODED FIELDS CSV for those 10
# ──────────────────────────────────────────────────────────────────────────────
import query_to_bn as qb
names = set(n for n in bn.names() if n not in (INJ_NODE, DMG_NODE))
import main_app

rows = []
for ev_id, inc in selected:
    text = qb.redact_severity_phrases(str(inc["narr_accf"])[:4000])
    parsed = qb.parse_query_to_bn_evidence(text, names, dataset=ds, semantic=False)
    hard_nodes = sorted(parsed["confidence"].keys())
    soft = {l: c for l, c, _ in
            qb.retrieval_facts(text, names, main_app.refined_dataset)}
    soft_nodes = sorted(soft.keys())

    coded_occs = []
    for f in ("sequence_of_events", "findings", "occurrences"):
        for e in (inc.get(f) or []):
            if isinstance(e, str):
                coded_occs.append(e.strip().lower())
            elif isinstance(e, dict):
                for val in e.values():
                    if isinstance(val, str):
                        coded_occs.append(val.strip().lower())
    coded_occs = sorted(set(coded_occs))

    rows.append({
        "ev_id": ev_id,
        "injury_coded": INJ_STATES[injury_state(inc)],
        "damage_coded": DMG_STATES[damage_state(inc)] if damage_state(inc) is not None else "",
        "narr_length": len(str(inc.get("narr_accf", ""))),
        "ntsb_coded_labels": "; ".join(coded_occs[:20]),
        "pipeline_hard_evidence": "; ".join(hard_nodes),
        "pipeline_soft_evidence": "; ".join(f"{n} ({soft[n]:.2f})" for n in soft_nodes),
        "n_hard": len(hard_nodes),
        "n_soft": len(soft_nodes),
    })

csv_path = EXPORT / "coded_fields.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"  wrote coded_fields.csv  ({len(rows)} rows)")

# ──────────────────────────────────────────────────────────────────────────────
# 5. COPY SOURCE FILES (coding logic)
# ──────────────────────────────────────────────────────────────────────────────
src_dir = EXPORT / "source"
src_dir.mkdir(exist_ok=True)
copies = [
    (FROZEN / "code/query_to_bn.py", "query_to_bn.py"),
    (FROZEN / "code/bn_upgraded.py", "bn_upgraded.py"),
    (FROZEN / "tests/bn_build_ours.py", "bn_build_ours.py"),
    (REPO / "shared/code/prognosis.py", "prognosis.py"),
    (REPO / "shared/code/zhang_diagnosis.py", "zhang_diagnosis.py"),
]
for src, dst in copies:
    if src.exists():
        shutil.copy2(src, src_dir / dst)
        print(f"  copied {src.name} -> source/{dst}")
    else:
        print(f"  WARNING: {src} not found")

print("\n✅ export complete")

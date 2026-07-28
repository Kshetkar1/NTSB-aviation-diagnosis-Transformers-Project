#!/usr/bin/env python3
"""Analysis of the narrative-only rebuild (tests/llm_recode_dataset.py).

Builds two parallel datasets over the SAME accidents (those with a real
narrative and a cached LLM record):

    CODED : the investigators' record (occurrence chain, findings, damage,
            injury) -- what Zhang's analysis consumed
    LLM   : the record reconstructed by gpt-4o-mini from the narrative ALONE

and compares, at every level of the analysis:

  1. outcome extraction  : damage + highest-injury accuracy vs the coded fields
  2. Table 7             : P(cause | fire) fire-cause distribution, both ways,
                           with Zhang's published anchors
  3. root priors (Eq 6)  : occurrence count / 184,517,128 for key nodes
  4. the full Boolean BN : Zhang's Section 4 recipe on each dataset; Table 9
                           forward anchors + LOEP downstream column

Writes docs/LLM_NARRATIVE_ONLY_REBUILD.md.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/llm_recode_analysis.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import prognosis as pg  # noqa: E402
import bn_build_ours as builder  # noqa: E402

CACHE = ROOT / "outputs" / "llm_recode"
OUT_MD = ROOT / "docs" / "LLM_NARRATIVE_ONLY_REBUILD.md"

lines: list[str] = []


def emit(s: str = "") -> None:
    print(s)
    lines.append(s)


def synth_incident(rec: dict) -> dict:
    """LLM record -> incident dict shaped for pg.build_edges."""
    seq = [{"Occurrence_No": str(i + 1), "Occurrence_Description": o}
           for i, o in enumerate(rec["occurrences"])]
    finds = [{"finding_description": f["finding"],
              "Occurrence_No": str(f["occ"])} for f in rec["findings"]]
    return {"sequence_of_events": seq, "findings": finds,
            "damage": rec["damage"], "ev_highest_injury": rec["injury"]}


def fire_causes(ds: dict):
    """Zhang's Table 7 rule: causes of fire = findings attached to the fire
    occurrence + the occurrence immediately preceding fire in the chain.
    Returns (n_fire_accidents, {cause: n_accidents})."""
    count = defaultdict(set)
    fire_evs = set()
    for ev, inc in ds.items():
        descs, nos = pg._ordered_occurrences(inc)
        fire_idx = [i for i, d in enumerate(descs) if d == "fire"]
        if not fire_idx:
            continue
        fire_evs.add(ev)
        fire_nos = {nos[i] for i in fire_idx}
        for f in inc.get("findings", []):
            d = pg._s(f.get("finding_description")).lower()
            if d and pg._s(f.get("Occurrence_No")) in fire_nos:
                count[d].add(ev)
        for i in fire_idx:
            if i >= 1 and descs[i - 1] != "fire":
                count[descs[i - 1]].add(ev)
    return len(fire_evs), {c: len(evs) for c, evs in count.items()}


def root_prior_counts(ds: dict):
    """from-occurrence accident counts per node (Zhang's occurrenceTimesFromID)."""
    edge_events, _ = pg.build_graph(ds)
    from_counts = defaultdict(set)
    for (a, _b), evs in edge_events.items():
        from_counts[a].update(evs)
    return {n: len(evs) for n, evs in from_counts.items()}


def posterior(bn, target, evidence=None):
    import pyagrum as gum
    ie = gum.LazyPropagation(bn)
    if evidence:
        ie.setEvidence(evidence)
    ie.addTarget(target)
    ie.makeInference()
    v = bn.variable(target)
    yes = [i for i in range(v.domainSize()) if v.label(i) == "Yes"][0]
    return float(ie.posterior(target)[yes])


def g4(x):
    if x is None:
        return "--"
    return f"{x:.4g}"


def main():
    full = pg.load_dataset()
    recs = {}
    for f in CACHE.glob("*.json"):
        ev = f.stem
        if ev in full:
            recs[ev] = json.loads(f.read_text())

    coded = {ev: full[ev] for ev in recs}
    llmds = {ev: synth_incident(r) for ev, r in recs.items()}
    emit("# Narrative-only rebuild: LLM re-codes the dataset, "
         "Zhang's analysis re-runs")
    emit()
    emit(f"gpt-4o-mini read the factual narrative of **{len(recs)}** accidents "
         f"(of {len(full)} in the 1982-2006 window; the rest have no usable "
         "free text) and reconstructed the coded record -- occurrence chain, "
         "findings, damage, injury -- from the narrative ALONE. Everything "
         "below compares the investigators' coded data (CODED) against the "
         "LLM's narrative-only records (LLM) over the SAME accidents.")
    emit()

    # ================= 1. outcome extraction ====================================
    dmg_ok = dmg_tot = dmg_unk = 0
    inj_ok = inj_tot = 0
    for ev, r in recs.items():
        true_d = pg._s(full[ev].get("damage")).upper()
        if r["damage"] == "UNK":
            dmg_unk += 1
        elif true_d in ("DEST", "SUBS", "MINR", "NONE"):
            dmg_tot += 1
            dmg_ok += (r["damage"] == true_d)
        true_i = pg._s(full[ev].get("ev_highest_injury")).upper()
        if true_i in ("FATL", "SERS", "MINR", "NONE"):
            inj_tot += 1
            inj_ok += (r["injury"] == true_i)
    emit("## 1. Outcome extraction from the narrative")
    emit()
    emit("| Field | LLM vs coded record | Note |")
    emit("|---|---|---|")
    emit(f"| Aircraft damage | **{dmg_ok}/{dmg_tot} = {dmg_ok/max(dmg_tot,1):.0%}** "
         f"| LLM abstained (UNK) on {dmg_unk} narratives that don't state damage |")
    emit(f"| Highest injury | **{inj_ok}/{inj_tot} = {inj_ok/max(inj_tot,1):.0%}** "
         f"| default 'none' when the narrative reports nobody hurt |")
    emit()

    # ================= 2. Table 7 ================================================
    n_fire_c, cc = fire_causes(coded)
    n_fire_l, cl = fire_causes(llmds)
    emit("## 2. Table 7 -- P(cause | fire), the paper's centerpiece table")
    emit()
    emit(f"Fire accidents found: CODED **{n_fire_c}**, LLM narrative-only "
         f"**{n_fire_l}** (Zhang's full-window count: 102; this subset holds "
         "the accidents that have narratives).")
    emit()
    union = sorted(set(cc) | set(cl),
                   key=lambda c: -(cc.get(c, 0) + cl.get(c, 0)))
    emit("| Cause (top 20 by combined support) | CODED n | CODED P | LLM n | LLM P |")
    emit("|---|---|---|---|---|")
    for c in union[:20]:
        emit(f"| {c[:58]} | {cc.get(c, 0)} | "
             f"{cc.get(c, 0)/max(n_fire_c,1):.4f} | {cl.get(c, 0)} | "
             f"{cl.get(c, 0)/max(n_fire_l,1):.4f} |")
    vec_c = np.array([cc.get(c, 0) for c in union], dtype=float)
    vec_l = np.array([cl.get(c, 0) for c in union], dtype=float)
    rho = spearmanr(vec_c, vec_l).statistic
    top10_c = {c for c in sorted(cc, key=lambda x: -cc[x])[:10]}
    top10_l = {c for c in sorted(cl, key=lambda x: -cl[x])[:10]}
    emit()
    emit(f"- Spearman rank correlation over all {len(union)} causes: "
         f"**{rho:.3f}**")
    emit(f"- Top-10 overlap: **{len(top10_c & top10_l)}/10**")
    l1 = float(np.abs(vec_c / max(n_fire_c, 1)
                      - vec_l / max(n_fire_l, 1)).sum())
    emit(f"- L1 distance between the two P(cause | fire) vectors: **{l1:.3f}**")
    emit()

    # ================= 3. root priors ============================================
    emit("## 3. Root priors (Eq. 6: occurrence count / 184,517,128 flights)")
    emit()
    pc = root_prior_counts(coded)
    pl = root_prior_counts(llmds)
    KEY = ["loss of engine power", "fire", "in flight collision with object",
           "gear collapsed", "loss of control - in flight",
           "in flight encounter with weather"]
    emit("| Node | CODED n | LLM n | CODED prior | LLM prior |")
    emit("|---|---|---|---|---|")
    for k in KEY:
        a, b = pc.get(k, 0), pl.get(k, 0)
        emit(f"| {k} | {a} | {b} | {a/builder.TOTAL_FLIGHTS:.3g} | "
             f"{b/builder.TOTAL_FLIGHTS:.3g} |")
    common = set(pc) & set(pl)
    rho_p = spearmanr([pc[n] for n in sorted(common)],
                      [pl[n] for n in sorted(common)]).statistic
    emit()
    emit(f"- {len(common)} nodes appear in both graphs; Spearman rank "
         f"correlation of their occurrence counts: **{rho_p:.3f}**")
    emit()

    # ================= 4. full BN rebuild ========================================
    emit("## 4. Full Bayesian network, Zhang's Section 4 recipe, both datasets")
    emit()
    bn_c, meta_c = builder.build_network(coded)
    bn_l, meta_l = builder.build_network(llmds)
    emit(f"- CODED network: **{meta_c['nodes']} nodes, {meta_c['arcs']} arcs**")
    emit(f"- LLM narrative-only network: **{meta_l['nodes']} nodes, "
         f"{meta_l['arcs']} arcs**")
    emit()

    LOEP = "loss of engine power"

    def anchor(bn, evname, target=LOEP):
        if evname not in bn.names() or target not in bn.names():
            return None
        try:
            return posterior(bn, target, {evname: "Yes"})
        except Exception:
            return None

    emit("### Table 9 forward anchors: P(loss of engine power | cause)")
    emit()
    emit("| Evidence | Zhang | CODED net | LLM net |")
    emit("|---|---|---|---|")
    ANCH = [
        ("engine instrument", 0.95),
        ("combustion assembly, combustion liner", 0.50),
        ("fluid, oil grade", 0.95),
    ]
    for evname, z in ANCH:
        emit(f"| {evname} | {z} | {g4(anchor(bn_c, evname))} | "
             f"{g4(anchor(bn_l, evname))} |")
    emit()

    emit("### Table 9 downstream column: P(target | loss of engine power)")
    emit()
    emit("| Target | Zhang | CODED net | LLM net |")
    emit("|---|---|---|---|")
    DOWN = [
        ("forced landing", 0.1429),
        ("ditching", 4.61e-3),
        ("gear collapsed", 5.18e-3),
    ]
    for tname, z in DOWN:
        vc = anchor(bn_c, LOEP, tname) if tname in bn_c.names() else None
        vl = anchor(bn_l, LOEP, tname) if tname in bn_l.names() else None
        emit(f"| {tname} | {z} | {g4(vc)} | {g4(vl)} |")
    emit()

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

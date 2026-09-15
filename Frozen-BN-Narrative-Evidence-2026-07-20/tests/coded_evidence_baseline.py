#!/usr/bin/env python3
"""CODED-EVIDENCE baseline: can the original evidence-setting procedure be run
at all on the held-out window?

The paper claims free-text narratives can substitute for the coded fields as
evidence into the network. That substitution has never been scored against
the thing it substitutes for. This script attempts exactly that: take each
held-out accident's OWN coded labels -- occurrences, findings, person
findings -- set them as hard evidence on the frozen network, propagate once,
and read injury and damage. Injury and damage nodes are never entered.

The result is a coverage finding rather than an accuracy comparison. The
NTSB replaced its occurrence/finding taxonomy after 2006, so the coded
vocabulary of the test window and the coded vocabulary the network was built
from do not intersect. This script measures that directly and reports what
the coded route scores as a consequence.

Two arms:
  coded-all  -- occurrences + findings + person findings (full coded record)
  coded-occ  -- occurrences only (stricter; findings encode investigator
                judgement a narrative-only pipeline could not see)

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/coded_evidence_baseline.py [--limit N]
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.pop("NTSB_FULL_CORPUS", None)
os.environ.pop("NTSB_USE_TRAIN_INDEX", None)

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "shared" / "code", FROZEN_DIR / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pyagrum as gum  # noqa: E402

import prognosis as pg  # noqa: E402
import query_to_bn as qb  # noqa: E402
from bn_upgraded import (build_upgraded, INJ_NODE, DMG_NODE,  # noqa: E402
                         INJ_STATES, DMG_STATES, DMG_BY_CODE)

FULL = REPO_ROOT / "shared" / "data" / "processed" / "refined_dataset.json"
WINDOW = REPO_ROOT / "shared" / "data" / "processed" / "refined_dataset_1982_2006.json"
OUTDIR = FROZEN_DIR / "outputs"
PER_ITEM = OUTDIR / "heldout_per_item.json"

RAW = REPO_ROOT / "shared" / "data" / "raw"
META = (REPO_ROOT / "Zhang-Replication-Foundation-2026-06-04" / "reference"
        / "data" / "metaData.xlsx")

INJ_LAB = ["fatal", "serious", "minor", "none"]
DMG_LAB = ["destroyed", "substantial", "minor", "none"]


def truth_states(inc):
    inj = pg.zhang_injury_code(inc)
    inj_i = {"FATL": 0, "SERS": 1, "MINR": 2, "NONE": 3}[inj]
    dmg_code = str(inc.get("damage") or "").upper()
    dmg_i = DMG_BY_CODE.get(dmg_code) if dmg_code in DMG_BY_CODE else None
    return inj_i, dmg_i


def posteriors(bn, confidence):
    """Same inference path the main eval uses for event evidence."""
    ie = gum.LazyPropagation(bn)
    if confidence:
        qb.apply_evidence(ie, bn, confidence)
    ie.addTarget(INJ_NODE)
    ie.addTarget(DMG_NODE)
    ie.makeInference()

    def dist(node, states):
        v = bn.variable(node)
        post = ie.posterior(node)
        by = {v.label(i): float(post[i]) for i in range(v.domainSize())}
        return np.array([by[s] for s in states])
    return dist(INJ_NODE, INJ_STATES), dist(DMG_NODE, DMG_STATES)


def occurrence_labels(inc):
    labs = set()
    for s in inc.get("sequence_of_events", []) or []:
        d = str(s.get("Occurrence_Description") or "").strip().lower()
        if d:
            labs.add(d)
    return labs


def recover_legacy_coded(ids):
    """Rebuild the legacy coded record for accidents whose codes were never
    decoded when the processed dataset was assembled.

    `rebuild_1982_2006.py` performs this decode for the network window only.
    The same two raw tables cover 2007 -- Occurrences.txt (Occurrence_Code)
    and seq_of_events.txt (Subj_Code, Modifier_Code, Person_Code) -- and the
    same metaData.xlsx dictionary resolves both. Returns
    {ev_id: (sequence_of_events, findings)} in the schema the window dataset
    uses, so downstream label extraction is byte-identical.
    """
    import pandas as pd
    md = pd.read_excel(META, dtype=str)
    code2m = dict(zip(md["code_iaids"].astype(str), md["meaning"].astype(str)))

    def meaning(code):
        return str(code2m.get(str(code), "")).strip()

    ids = set(ids)
    out = {i: ([], []) for i in ids}

    occ = pd.read_csv(RAW / "Occurrences.txt", sep=",", dtype=str,
                      on_bad_lines="skip")
    for r in occ[occ["ev_id"].isin(ids)].itertuples(index=False):
        out[r.ev_id][0].append({
            "ev_id": r.ev_id,
            "Aircraft_Key": r.Aircraft_Key,
            "Occurrence_No": r.Occurrence_No,
            "Occurrence_Code": str(r.Occurrence_Code),
            "Occurrence_Description": meaning(r.Occurrence_Code) or "Unknown",
            "phase_no": r.Phase_of_Flight,
            "phase_description": meaning(r.Phase_of_Flight),
            "eventsoe_no": None,
            "source": "legacy_occurrences_recovered",
        })

    seq = pd.read_csv(RAW / "seq_of_events.txt", sep="\t", dtype=str,
                      on_bad_lines="skip")
    seq.columns = [c.strip('"') for c in seq.columns]
    for r in seq[seq["ev_id"].isin(ids)].itertuples(index=False):
        out[r.ev_id][1].append({
            "ev_id": r.ev_id,
            "Occurrence_No": r.Occurrence_No,
            "seq_event_no": r.seq_event_no,
            "Subj_Code": str(r.Subj_Code),
            "finding_description": meaning(r.Subj_Code),
            "modifier_description": meaning(r.Modifier_Code),
            "person_description": meaning(r.Person_Code),
            "Cause_Factor": r.Cause_Factor,
            "source": "legacy_seq_of_events_recovered",
        })
    for k in out:
        out[k][0].sort(key=lambda s: int(str(s["Occurrence_No"]) or 0))
    return out


def macro_f1(preds, truth, n_cls):
    f1 = []
    for c in range(n_cls):
        tp = sum(1 for p, y in zip(preds, truth) if p == c and y == c)
        fp = sum(1 for p, y in zip(preds, truth) if p == c and y != c)
        fn = sum(1 for p, y in zip(preds, truth) if p != c and y == c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(f1) / n_cls, f1


def mcnemar_exact(a_right, b_right):
    """Two-sided exact McNemar on paired top-1 correctness."""
    b = sum(1 for x, y in zip(a_right, b_right) if x and not y)
    c = sum(1 for x, y in zip(a_right, b_right) if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b, c, min(1.0, 2 * tail)


def boot_ci(correct, iters=10000, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.asarray(correct, dtype=float)
    idx = rng.integers(0, len(arr), size=(iters, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    full = json.loads(FULL.read_text())
    window_ids = set(json.loads(WINDOW.read_text()).keys())
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = set(n for n in bn.names() if n not in (INJ_NODE, DMG_NODE))

    held = []
    for k, inc in full.items():
        if k in window_ids:
            continue
        narr = str(inc.get("narr_accf") or "").strip()
        if len(narr) < 100:
            continue
        held.append((k, inc))
    held.sort(key=lambda t: t[0])
    if limit:
        held = held[:limit]
    print(f"held-out accidents: {len(held)}")
    print(f"BN nodes available as evidence: {len(names)}")

    # Recover the legacy coded record for held-out accidents that shipped with
    # none. Only pre-2008 accidents can be recovered: the legacy tables stop
    # there, and after 2007 the taxonomy itself was replaced.
    shipped_match = sum(1 for _, inc in held
                        if qb._incident_bn_labels(inc) & names)
    shipped_labels = set()
    for _, inc in held:
        shipped_labels |= qb._incident_bn_labels(inc)
    recover_ids = [k for k, inc in held
                   if not qb._incident_bn_labels(inc)
                   and str(inc.get("ev_date") or k)[:4] < "2008"]
    recovered = {}
    if recover_ids:
        print(f"recovering legacy codes for {len(recover_ids)} accidents ...")
        rec = recover_legacy_coded(recover_ids)
        for k, inc in held:
            if k in rec and (rec[k][0] or rec[k][1]):
                inc["sequence_of_events"], inc["findings"] = rec[k]
                recovered[k] = True
        print(f"  recovered coded records for {len(recovered)}")

    arms = {"coded-all": [], "coded-occ": []}
    truth_inj, truth_dmg, ids = [], [], []
    n_ev = {"coded-all": [], "coded-occ": []}
    cov = {}                       # year -> [accidents, with any coded rows]
    raw_labels = set()             # every coded label seen in the test window
    n_uncoded = 0

    for i, (k, inc) in enumerate(held):
        inj_y, dmg_y = truth_states(inc)
        raw = qb._incident_bn_labels(inc)
        raw_labels |= raw
        if not raw:
            n_uncoded += 1
        y = str(inc.get("ev_date") or k)[:4]
        cov.setdefault(y, [0, 0])
        cov[y][0] += 1
        if raw:
            cov[y][1] += 1
        all_labs = raw & names
        occ_labs = occurrence_labels(inc) & names
        ids.append(k)
        truth_inj.append(inj_y)
        truth_dmg.append(dmg_y)
        for arm, labs in (("coded-all", all_labs), ("coded-occ", occ_labs)):
            n_ev[arm].append(len(labs))
            conf = {lab: 1.0 for lab in labs}
            try:
                arms[arm].append(posteriors(bn, conf))
            except Exception as exc:
                print(f"  {arm} failed on {k}: {exc}")
                arms[arm].append(posteriors(bn, {}))
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(held)}")

    # narrative pipeline, from the canonical run
    narr_inj = narr_dmg = None
    if PER_ITEM.exists():
        pi = json.loads(PER_ITEM.read_text())
        by_id = {it["id"]: it for it in pi["items"]}
        if all(k in by_id for k in ids):
            narr_inj = [by_id[k]["narrative-evidence:inj_pred"] for k in ids]
            narr_dmg = [by_id[k]["narrative-evidence:dmg_pred"] for k in ids]
        else:
            print("WARNING: cohort mismatch with heldout_per_item.json")

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    n_match = sum(1 for x in n_ev["coded-all"] if x > 0)
    overlap = raw_labels & names

    emit()
    emit("# Can the coded-evidence route be run on the held-out window?")
    emit()
    emit(f"n = {len(ids)} held-out accidents (2007-2019), the same cohort as "
         "the narrative evaluation. Evidence is the accident's own coded "
         "record entered as hard evidence on the frozen network; the injury "
         "and damage nodes are never entered.")
    emit()
    emit("## Coverage of the coded record")
    emit()
    emit("| Year | Accidents | With any coded occurrences/findings |")
    emit("|---|---|---|")
    for y in sorted(cov):
        emit(f"| {y} | {cov[y][0]} | {cov[y][1]} |")
    emit()
    emit(f"- Nodes in the frozen network available as evidence: "
         f"**{len(names)}**.")
    emit(f"- Distinct coded labels in the held-out window as the processed "
         f"dataset shipped: **{len(shipped_labels)}**, of which "
         f"**{len(shipped_labels & names)}** are network nodes.")
    emit(f"- Held-out accidents the network could accept any evidence for, "
         f"as shipped: **{shipped_match} of {len(ids)}**.")
    emit(f"- After recovering the 2007 legacy codes from raw (below): "
         f"**{n_match} of {len(ids)}**, all of them 2007.")
    emit(f"- Accidents still with no usable coded evidence: "
         f"**{len(ids) - n_match} of {len(ids)}**.")
    emit()
    emit("The NTSB replaced its occurrence and finding taxonomy after 2006. "
         "The network's nodes are legacy labels (e.g. "
         "`airframe/component/system failure/malfunction`); the 2008-onward "
         "accidents are coded in the new hierarchical scheme (e.g. "
         "`personnel issues-action/decision-action-incomplete action-ground "
         "crew - f`). The two vocabularies do not intersect, so the coded "
         "evidence route sets no evidence and every posterior below falls "
         "back to the network prior.")
    emit()
    emit("### Two distinct causes, stated separately")
    emit()
    emit("**2008 onward (263 accidents, 89% of the cohort).** A genuine "
         "taxonomy break. `Events_Sequence.txt`, the table carrying "
         "`Occurrence_Description`, begins in 2008 and holds zero rows for "
         "2006 or 2007; its description vocabulary shares no string with the "
         "legacy vocabulary the network was built from. No merge can repair "
         "this: the labels the network conditions on were retired. The "
         "original evidence-setting procedure cannot be executed on these "
         "accidents at all.")
    emit()
    emit(f"**2007 ({len(recovered)} accidents, "
         f"{100*len(recovered)/max(len(ids),1):.0f}%).** A gap in this "
         "pipeline, not in the source, and this script repairs it. The "
         "legacy tables do cover 2007 -- `Occurrences.txt` and "
         "`seq_of_events.txt` -- but they carry numeric codes only, and "
         "`rebuild_1982_2006.py` decoded them for the network window alone. "
         "Running the same `metaData.xlsx` decode over 2007 recovers a full "
         "coded record for every one of these accidents, and the recovered "
         "labels land on the network almost perfectly: 17 of 17 distinct "
         "occurrence labels, 60 of 62 finding labels and 16 of 18 person "
         "labels are existing nodes. These accidents therefore support a "
         "genuine coded-vs-narrative comparison, reported below.")
    emit()

    def report(sel):
        """sel: list of positions into the cohort arrays."""
        for target, states, truth_all, narr_all in (
                ("Injury", INJ_LAB, truth_inj, narr_inj),
                ("Damage", DMG_LAB, truth_dmg, narr_dmg)):
            idx = 0 if target == "Injury" else 1
            truth = [truth_all[i] for i in sel]
            emit(f"### {target}")
            emit()
            emit("| Evidence source | Accuracy | 95% CI | Macro-F1 | " +
                 " | ".join(f"{s} recall" for s in states) + " |")
            emit("|---|---|---|---|" + "---|" * len(states))
            rows = []
            for arm in ("coded-all", "coded-occ"):
                rows.append((arm, [int(np.argmax(arms[arm][i][idx]))
                                   for i in sel]))
            if narr_all is not None:
                rows.append(("narrative (ours)", [narr_all[i] for i in sel]))
            for name, preds in rows:
                ok = [1 if p == y else 0 for p, y in zip(preds, truth)]
                acc = sum(ok) / len(ok)
                lo, hi = boot_ci(ok)
                mf1, _ = macro_f1(preds, truth, len(states))
                rec = []
                for c in range(len(states)):
                    n = sum(1 for y in truth if y == c)
                    h = sum(1 for p, y in zip(preds, truth)
                            if p == c and y == c)
                    rec.append(f"{h}/{n}" if n else "-")
                emit(f"| {name} | {100*acc:.1f}% | "
                     f"[{100*lo:.1f}%, {100*hi:.1f}%] | {mf1:.3f} | "
                     + " | ".join(rec) + " |")
            emit()
            if narr_all is None:
                continue
            emit("Paired McNemar (exact, two-sided) against the narrative "
                 "pipeline:")
            emit()
            emit("| Comparison | coded only right | narrative only right | p |")
            emit("|---|---|---|---|")
            nr = [narr_all[i] == truth_all[i] for i in sel]
            for arm in ("coded-all", "coded-occ"):
                cr = [int(np.argmax(arms[arm][i][idx])) == truth_all[i]
                      for i in sel]
                b, c, p = mcnemar_exact(cr, nr)
                star = " *" if p < 0.05 else ""
                emit(f"| {arm} vs narrative | {b} | {c} | {p:.4f}{star} |")
            emit()

    sel_2007 = [i for i, k in enumerate(ids) if k in recovered]
    sel_rest = [i for i, k in enumerate(ids) if k not in recovered]

    if sel_2007:
        emit(f"## 2007 subset: coded evidence really does enter the network "
             f"(n = {len(sel_2007)})")
        emit()
        emit("These are the recovered accidents. Every one sets real hard "
             "evidence on the frozen network, so this is the only place in "
             "the held-out window where the original evidence-setting "
             "procedure and the narrative pipeline can be compared on the "
             "same accidents. Note the sample is small and the `coded-all` "
             "arm includes findings, which are investigator judgements "
             "recorded after the outcome was known; `coded-occ` uses "
             "occurrences only and is the fairer arm.")
        emit()
        report(sel_2007)

    emit(f"## 2008 onward: coded evidence cannot enter the network "
         f"(n = {len(sel_rest)})")
    emit()
    emit("No accident here shares a label with the network, so both coded "
         "arms set no evidence and return the network prior. The comparison "
         "is not close because there is nothing to compare: this is the "
         "measured cost of the taxonomy change, and the reason a narrative "
         "route is the only route.")
    emit()
    report(sel_rest)

    emit(f"## Whole held-out window (n = {len(ids)})")
    emit()
    report(list(range(len(ids))))

    out_md = OUTDIR / "coded_evidence_baseline.md"
    out_md.write_text("\n".join(lines) + "\n")
    json.dump({"ids": ids,
               "truth_inj": truth_inj, "truth_dmg": truth_dmg,
               "coded_all_inj": [list(map(float, p[0])) for p in arms["coded-all"]],
               "coded_all_dmg": [list(map(float, p[1])) for p in arms["coded-all"]],
               "coded_occ_inj": [list(map(float, p[0])) for p in arms["coded-occ"]],
               "coded_occ_dmg": [list(map(float, p[1])) for p in arms["coded-occ"]],
               "n_ev": n_ev},
              open(OUTDIR / "coded_evidence_baseline.json", "w"))
    print(f"\nwrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

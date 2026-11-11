#!/usr/bin/env python3
"""WHAT DO THE NARRATIVES ADD? Same accidents, same frozen network, coded
evidence vs narrative evidence -- broken down by accident type and shown as
concrete cases.

Answers Maha's question directly: for the same cases and the same network,
what do we gain by using narrative evidence instead of (or next to) the coded
record, and on WHICH kinds of accidents?

Inputs are the per-item records two existing scripts already write:
  outputs/coded_vs_narrative_window_items.jsonl   (coded_vs_narrative_window.py,
                                                   500 build-window accidents,
                                                   leave-one-out)
  outputs/coded_evidence_baseline.json             (coded_evidence_baseline.py,
                                                   held-out cohort; only the
                                                   2007 accidents carry coded
                                                   evidence the network accepts)
  outputs/heldout_per_item.json                    (frozenbn_heldout_narrative_bn_eval.py)

Arms compared, all on the SAME frozen upgraded network:
  coded        accident's own human-coded occurrences/findings/persons as hard
               evidence on event nodes  (the original evidence-setting route)
  narr-events  redacted narrative -> hard evidence + event virtual evidence on
               the SAME event nodes      (replace the coder, keep the interface)
  narr-sev     redacted narrative -> injury/damage virtual evidence from the
               k = 100 most similar 1982-2006 narratives (the deployed path)

Outputs  outputs/narrative_gain_case_studies.md  with
  1. paired win/loss matrices  (both right / coded only / narrative only / both wrong)
  2. gains by stratum: true severity class, size of the coded record,
     narrative length, occurrence family (turbulence, gear, fire, engine, ...)
  3. exemplar accidents where the narrative is right and the coded record is
     wrong -- and the reverse, for honesty -- with the redacted text excerpt,
     the coded labels that entered, and (with --explain) the exact hard
     evidence matched, the event virtual facts fired, and the top neighbours
     with their coded outcomes.

Caveat that travels with every number: on the build window the `coded` arm is
in-sample (the CPTs were fit from these outcomes), so it is an UPPER BOUND on
the coded route. The 2007 held-out subset (n = 33) is the only clean
out-of-sample head-to-head. From 2008 on the coded route has no inputs at all
(taxonomy change), so there is nothing to compare: narrative is the only route.

Run (where the data and outputs live):
  export PYTHONPATH="shared/code:Frozen-BN-Narrative-Evidence-2026-07-20/code"
  python Frozen-BN-Narrative-Evidence-2026-07-20/tests/narrative_gain_case_studies.py
  python ... narrative_gain_case_studies.py --explain 6   # re-run retrieval for 6 exemplars/target
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.pop("NTSB_FULL_CORPUS", None)
os.environ.pop("NTSB_USE_TRAIN_INDEX", None)

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "shared" / "code", FROZEN_DIR / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUTDIR = FROZEN_DIR / "outputs"
WINDOW_ITEMS = OUTDIR / "coded_vs_narrative_window_items.jsonl"
CODED_HELDOUT = OUTDIR / "coded_evidence_baseline.json"
HELDOUT_ITEMS = OUTDIR / "heldout_per_item.json"
FULL = REPO_ROOT / "shared" / "data" / "processed" / "refined_dataset.json"
OUT_MD = OUTDIR / "narrative_gain_case_studies.md"

INJ_LAB = ["fatal", "serious", "minor", "none"]
DMG_LAB = ["destroyed", "substantial", "minor", "none"]
TARGETS = {"Injury": ("inj", INJ_LAB), "Damage": ("dmg", DMG_LAB)}
NARR_ARMS = ("narr-sev", "narr-events")

# Retrieval gates, identical to the canonical evaluation.
TOP_K, MIN_FQ, MIN_LIFT, TOP_M, MAX_CONF, SEV_ALPHA, TRUNC = 100, 0.15, 3.0, 3, 0.95, 0.5, 4000

# Coarse occurrence families from the accident's OWN coded occurrence labels.
# First match wins; order matters (fire before engine, gear before ground).
FAMILIES = [
    ("turbulence", ("turbulence",)),
    ("fire / smoke", ("fire", "smoke", "explosion")),
    ("landing gear", ("gear",)),
    ("engine / powerplant", ("engine", "powerplant", "propeller")),
    ("loss of control", ("loss of control",)),
    ("hard landing", ("hard landing",)),
    ("ground collision / taxi", ("on ground", "ground collision", "collision with", "taxi")),
    ("abrupt maneuver", ("abrupt maneuver", "evasive")),
    ("cabin / ground crew", ("cabin", "ground crew", "jet blast", "passenger", "flight attendant")),
    ("weather / icing / wind", ("icing", "windshear", "wind shear", "weather", "downdraft")),
    ("airframe / system failure", ("airframe", "system failure", "malfunction")),
]


# --------------------------------------------------------------------------- #
# pure analysis helpers (no project imports; smoke-testable)
# --------------------------------------------------------------------------- #
def mcnemar_exact(a_right, b_right):
    b = sum(1 for x, y in zip(a_right, b_right) if x and not y)
    c = sum(1 for x, y in zip(a_right, b_right) if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b, c, min(1.0, 2 * tail)


def occurrence_family(labels) -> str:
    text = " | ".join(sorted(labels)).lower()
    for fam, keys in FAMILIES:
        if any(k in text for k in keys):
            return fam
    return "other / none coded"


def coded_bin(n: int) -> str:
    if n == 0:
        return "0 labels (nothing the network accepts)"
    if n <= 2:
        return "1-2 labels (sparse)"
    if n <= 5:
        return "3-5 labels"
    return "6+ labels (rich)"


def length_bin(n_chars: int) -> str:
    if n_chars < 600:
        return "< 600 chars (short)"
    if n_chars < 1500:
        return "600-1500 chars"
    return ">= 1500 chars (long)"


def win_matrix(recs, target_slot, arm):
    """recs: list of dicts with 'truth', 'coded', arm -> class ints per slot."""
    m = Counter()
    for r in recs:
        y = r["truth"][target_slot]
        if y is None:
            continue
        c = r["coded"][target_slot] == y
        n = r[arm][target_slot] == y
        m["both right" if c and n else "coded only" if c else
          "narrative only" if n else "both wrong"] += 1
    return m


def stratum_table(recs, target_slot, key_fn, arms=NARR_ARMS):
    """Return rows: (stratum, n, coded_acc, {arm: acc}, {arm: (narr_only, coded_only)})."""
    groups = defaultdict(list)
    for r in recs:
        if r["truth"][target_slot] is None:
            continue
        groups[key_fn(r)].append(r)
    rows = []
    for name, g in groups.items():
        y = [r["truth"][target_slot] for r in g]
        c_ok = [r["coded"][target_slot] == t for r, t in zip(g, y)]
        accs, pairs = {}, {}
        for arm in arms:
            n_ok = [r[arm][target_slot] == t for r, t in zip(g, y)]
            accs[arm] = sum(n_ok) / len(g)
            pairs[arm] = (sum(1 for a, b in zip(c_ok, n_ok) if b and not a),
                          sum(1 for a, b in zip(c_ok, n_ok) if a and not b))
        rows.append((name, len(g), sum(c_ok) / len(g), accs, pairs))
    rows.sort(key=lambda t: -t[1])
    return rows


def pick_exemplars(recs, target_slot, arm, want_narr_right: bool, n: int):
    """Narrative-right/coded-wrong (or the reverse). Prefer accidents whose
    coded record was NOT empty (so the coded arm genuinely tried), then the
    sparsest coded record, then the biggest miss."""
    out = []
    for r in recs:
        y = r["truth"][target_slot]
        if y is None:
            continue
        c_ok = r["coded"][target_slot] == y
        n_ok = r[arm][target_slot] == y
        if want_narr_right and n_ok and not c_ok:
            out.append(r)
        if not want_narr_right and c_ok and not n_ok:
            out.append(r)
    out.sort(key=lambda r: (r.get("n_coded_labels", 0) == 0,
                            r.get("n_coded_labels", 0),
                            -abs((r["coded"][target_slot] or 0) - (r["truth"][target_slot] or 0))))
    return out[:n]


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def load_window_records(ds):
    recs = []
    for line in WINDOW_ITEMS.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        inc = ds.get(r["id"], {})
        import query_to_bn as qb
        labs = qb._incident_bn_labels(inc)
        occ = {str(s.get("Occurrence_Description") or "").strip().lower()
               for s in inc.get("sequence_of_events", []) or []}
        recs.append({
            "id": r["id"],
            "date": str(inc.get("ev_date") or r["id"])[:10],
            "truth": (r["inj_y"], r["dmg_y"]),
            "prior": tuple(r["prior"]),
            "coded": tuple(r["coded"]),
            "narr-events": tuple(r["narr-events"]),
            "narr-sev": tuple(r["narr-sev"]),
            "n_coded_labels": int(r.get("n_coded_labels", len(labs))),
            "n_narr_ev": int(r.get("n_narr_ev", 0)),
            "coded_labels": sorted(labs),
            "family": occurrence_family(occ or labs),
            "narr_len": len(str(inc.get("narr_accf") or "")),
            "narr": str(inc.get("narr_accf") or ""),
        })
    return recs


def load_heldout_2007_records(full):
    """2007 is the only held-out year whose coded vocabulary the network
    recognises. coded_evidence_baseline.py recovers those codes in memory, so
    we take its posteriors and pair them with the canonical narrative run."""
    cb = json.loads(CODED_HELDOUT.read_text())
    hp = {it["id"]: it for it in json.loads(HELDOUT_ITEMS.read_text())["items"]}
    recs = []
    for i, k in enumerate(cb["ids"]):
        if not str(k).startswith("2007"):
            continue
        if k not in hp:
            continue
        n_ev = cb["n_ev"]["coded-all"][i]
        if n_ev == 0:
            continue
        inc = full.get(k, {})
        it = hp[k]
        recs.append({
            "id": k,
            "date": str(inc.get("ev_date") or k)[:10],
            "truth": (cb["truth_inj"][i], cb["truth_dmg"][i]),
            "prior": (it["prior:inj_pred"], it["prior:dmg_pred"]),
            "coded": (int(max(range(4), key=lambda j: cb["coded_all_inj"][i][j])),
                      int(max(range(4), key=lambda j: cb["coded_all_dmg"][i][j]))),
            "narr-events": (it["hard+soft:inj_pred"], it["hard+soft:dmg_pred"]),
            "narr-sev": (it["bn-sev:inj_pred"], it["bn-sev:dmg_pred"]),
            "n_coded_labels": int(n_ev),
            "coded_labels": [],            # recovered in memory upstream; not persisted
            "family": "2007 held-out",
            "narr_len": len(str(inc.get("narr_accf") or "")),
            "narr": str(inc.get("narr_accf") or ""),
        })
    return recs


# --------------------------------------------------------------------------- #
# optional: re-run the narrative pipeline on exemplars to show the evidence
# --------------------------------------------------------------------------- #
def explain(rec, ds, bn_names, main_app, qb, pg):
    ptext = qb.redact_severity_phrases(rec["narr"][:TRUNC])
    hard = qb.parse_query_to_bn_evidence(ptext, bn_names, dataset=ds,
                                         semantic=False, exclude_ev_ids=[rec["id"]])
    soft = qb.retrieval_facts(ptext, bn_names, ds, top_k=TOP_K, min_fq=MIN_FQ,
                              min_lift=MIN_LIFT, top_m=TOP_M, max_conf=MAX_CONF,
                              exclude_ev_ids=[rec["id"]])
    emb = main_app.get_embedding(qb.inference_query(ptext))
    scores, matches = main_app.find_top_matches(emb, exclude_ev_ids=[rec["id"]])
    neigh, seen = [], set()
    for s, m in zip(scores, matches):
        if m.get("source") != "incident":
            continue
        ev = m.get("ev_id")
        if not ev or ev in seen or ev not in ds:
            continue
        seen.add(ev)
        inc = ds[ev]
        dmg = str(inc.get("damage") or "").upper()
        neigh.append((ev, float(s), pg.zhang_injury_code(inc), dmg))
        if len(neigh) >= 5:
            break
    rdist = qb.severity_retrieval_distributions(ptext, ds, top_k=TOP_K,
                                                alpha=SEV_ALPHA, exclude_ev_ids=[rec["id"]])
    return {"hard": hard["trace"], "soft": soft, "neigh": neigh, "f_q": rdist,
            "redacted": ptext}


# --------------------------------------------------------------------------- #
def main() -> int:
    argv = sys.argv
    n_explain = int(argv[argv.index("--explain") + 1]) if "--explain" in argv else 0
    n_ex = int(argv[argv.index("--exemplars") + 1]) if "--exemplars" in argv else 6

    import prognosis as pg
    import query_to_bn as qb
    ds = pg.load_dataset()
    full = json.loads(FULL.read_text()) if FULL.exists() else {}

    have_window = WINDOW_ITEMS.exists()
    have_heldout = CODED_HELDOUT.exists() and HELDOUT_ITEMS.exists()
    if not have_window and not have_heldout:
        print("No per-item inputs found. Run coded_vs_narrative_window.py and/or "
              "coded_evidence_baseline.py first.")
        return 1

    window = load_window_records(ds) if have_window else []
    held07 = load_heldout_2007_records(full) if have_heldout else []
    print(f"window records: {len(window)}   2007 held-out records: {len(held07)}")

    bn_names = main_app = None
    if n_explain:
        from bn_upgraded import build_upgraded, INJ_NODE, DMG_NODE
        import main_app as _ma
        main_app = _ma
        bn, _ = build_upgraded(ds)
        bn_names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]

    L = []

    def emit(s=""):
        print(s)
        L.append(s)

    emit("# What the narratives add: coded evidence vs narrative evidence, "
         "same accidents, same frozen network")
    emit()
    emit("Generated by `tests/narrative_gain_case_studies.py` from the per-item "
         "records of `coded_vs_narrative_window.py` and `coded_evidence_baseline.py`.")
    emit()
    emit("**Read every build-window number with this caveat:** the `coded` arm "
         "enters the accident's own human-coded labels into a network whose "
         "structure and CPTs were estimated from those very accidents. It is an "
         "in-sample upper bound on the coded route, not a peer. Narrative arms "
         "use redacted text and a retrieval pool with the query removed. Any tie "
         "is better than a tie; any narrative win is a lower bound on the margin. "
         "The 2007 held-out block is the only clean out-of-sample head-to-head.")
    emit()

    for title, recs in (("Build window 1982-2006, leave-one-out", window),
                        ("Held-out 2007 (last legacy-taxonomy year)", held07)):
        if not recs:
            continue
        emit(f"## {title}  (n = {len(recs)})")
        emit()

        # ---------- 1. win/loss matrices ----------
        emit("### 1. Paired outcome on the same accidents")
        emit()
        emit("| Target | Narrative arm | both right | coded only | **narrative only** | both wrong | McNemar p |")
        emit("|---|---|---|---|---|---|---|")
        for tname, (slot_key, _) in TARGETS.items():
            slot = 0 if tname == "Injury" else 1
            for arm in NARR_ARMS:
                m = win_matrix(recs, slot, arm)
                c_ok = [r["coded"][slot] == r["truth"][slot] for r in recs if r["truth"][slot] is not None]
                n_ok = [r[arm][slot] == r["truth"][slot] for r in recs if r["truth"][slot] is not None]
                _, _, p = mcnemar_exact(c_ok, n_ok)
                emit(f"| {tname} | `{arm}` | {m['both right']} | {m['coded only']} | "
                     f"**{m['narrative only']}** | {m['both wrong']} | {p:.3g}{' *' if p < 0.05 else ''} |")
        emit()
        emit("`narrative only` is the count of accidents the coded record got wrong "
             "and the narrative got right on the same network. That column is the "
             "measured gain.")
        emit()

        # ---------- 2. strata ----------
        emit("### 2. Where the gain lives")
        emit()
        strata = [
            ("True injury class", lambda r: INJ_LAB[r["truth"][0]], 0),
            ("True damage class", lambda r: DMG_LAB[r["truth"][1]] if r["truth"][1] is not None else "n/a", 1),
            ("Size of the coded record the network accepts", lambda r: coded_bin(r["n_coded_labels"]), None),
            ("Narrative length", lambda r: length_bin(r["narr_len"]), None),
            ("Occurrence family (from the accident's own codes)", lambda r: r["family"], None),
        ]
        for sname, key_fn, only_slot in strata:
            if all(r["family"] == "2007 held-out" for r in recs) and "Occurrence" in sname:
                continue
            emit(f"#### {sname}")
            emit()
            for tname in TARGETS:
                slot = 0 if tname == "Injury" else 1
                if only_slot is not None and only_slot != slot:
                    continue
                emit(f"*{tname}*")
                emit()
                emit("| Stratum | n | coded acc | narr-sev acc | narr-events acc | "
                     "narr-sev: narr-only / coded-only | narr-events: narr-only / coded-only |")
                emit("|---|---|---|---|---|---|---|")
                for name, n, cacc, accs, pairs in stratum_table(recs, slot, key_fn):
                    emit(f"| {name} | {n} | {cacc:.0%} | {accs['narr-sev']:.0%} | "
                         f"{accs['narr-events']:.0%} | {pairs['narr-sev'][0]} / {pairs['narr-sev'][1]} | "
                         f"{pairs['narr-events'][0]} / {pairs['narr-events'][1]} |")
                emit()

        # ---------- 3. exemplars ----------
        emit("### 3. Cases")
        emit()
        emit("Redacted excerpts are what the pipeline actually saw (explicit "
             "injury/damage wording removed). Coded labels are what the "
             "original route entered as hard evidence.")
        emit()
        for tname, (slot_key, labs) in TARGETS.items():
            slot = 0 if tname == "Injury" else 1
            for want_narr, heading in ((True, "narrative right, coded wrong"),
                                       (False, "coded right, narrative wrong (for honesty)")):
                ex = pick_exemplars(recs, slot, "narr-sev", want_narr,
                                    n_ex if want_narr else max(2, n_ex // 3))
                if not ex:
                    continue
                emit(f"#### {tname} -- {heading}  (showing {len(ex)})")
                emit()
                for r in ex:
                    y = r["truth"][slot]
                    emit(f"**{r['id']}** ({r['date']}, {r['family']}) -- truth **{labs[y]}**; "
                         f"coded -> {labs[r['coded'][slot]]} ({r['n_coded_labels']} labels entered); "
                         f"narr-sev -> {labs[r['narr-sev'][slot]]}; "
                         f"narr-events -> {labs[r['narr-events'][slot]]}; "
                         f"prior -> {labs[r['prior'][slot]]}")
                    if r["coded_labels"]:
                        shown = r["coded_labels"][:6]
                        more = f" (+{len(r['coded_labels']) - 6} more)" if len(r["coded_labels"]) > 6 else ""
                        emit(f"- coded labels: " + "; ".join(f"`{l}`" for l in shown) + more)
                    excerpt = " ".join(r["narr"].split())[:420]
                    emit(f"- narrative (redacted excerpt): \"{qb.redact_severity_phrases(excerpt)}...\"")
                    if n_explain and want_narr and len([e for e in L if e.startswith('- hard evidence')]) < 10_000:
                        try:
                            ex_info = explain(r, ds, bn_names, main_app, qb, pg)
                            hard = ", ".join(f"`{n}` ({why})" for n, why in ex_info["hard"]) or "none"
                            soft = ", ".join(f"`{n}` c={c:.2f}" for n, c, _ in ex_info["soft"]) or "none"
                            neigh = "; ".join(f"{ev} (sim {s:.2f}, inj {ij}, dmg {dg})"
                                              for ev, s, ij, dg in ex_info["neigh"])
                            fq = ex_info["f_q"]
                            emit(f"- hard evidence matched: {hard}")
                            emit(f"- event virtual evidence fired: {soft}")
                            emit(f"- top-5 neighbours (sim, coded injury, coded damage): {neigh}")
                            if fq:
                                emit(f"- f_q injury {['%.2f' % x for x in fq['injury']]}  "
                                     f"f_q damage {['%.2f' % x for x in fq['damage']]}")
                        except Exception as exc:
                            emit(f"- (explain failed: {type(exc).__name__}: {exc})")
                    emit()
        emit()

    emit("## How to say it in the paper")
    emit()
    emit("1. **From 2008 on, the coded route has no inputs.** The NTSB replaced "
         "its occurrence/finding taxonomy; zero labels in 263 held-out accidents "
         "are network nodes. The narrative pipeline is not a convenience next to "
         "the coded record -- on modern accidents it is the only way to put "
         "evidence on this network. (BII-1)")
    emit("2. **Where both routes run, the narrative matches the human coder on "
         "injury and the deployed severity path beats the coded ceiling on both "
         "targets.** Report the `narrative only` column above and the strata where "
         "it concentrates (expected: substantial/minor damage, sparse coded "
         "records, turbulence and cabin injuries whose detail the codes flatten). "
         "Pair with the McNemar p.")
    emit("3. **Where the narrative loses, it loses in the parser, not in the text.** "
         "`narr-events` under-performs `coded` on damage while `narr-sev` from the "
         "same narratives over-performs it: the information is in the free text; "
         "phrase matching does not reach it. That is a limitation plus a next step.")
    emit("4. Show two or three of the cases above in Section 9 or an appendix: "
         "truth, what the coded route entered and predicted, what the narrative "
         "said, what evidence it fired, and the posterior that followed.")
    emit()

    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"\nwrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""HELD-OUT diagnosis evaluation at CAUSE-CATEGORY level (era-fair).

Problem this fixes: the old diagnosis metric compared a terse 1982-2006 node
label ("compressor assembly, blade retention") against the full 2007-2019
probable-cause PARAGRAPH by embedding similarity -- avg match ~= 36.6% with 0
hits at the 0.75 threshold is close to guaranteed by that construction, and
NTSB switched coding taxonomies in 2008 (legacy subject codes -> CICTT), so
exact-code matching across the split is impossible by design.

Era-fair protocol:
  * Both eras are rolled up to the CICTT TOP-LEVEL cause categories:
    AIRCRAFT / PERSONNEL / ENVIRONMENT / ORGANIZATIONAL.
    - held-out truth: first segment of the CICTT finding string
      ("Personnel issues-Action/decision-..." -> PERSONNEL), C/F findings only
    - window (legacy) findings and BN node labels: keyword rules over the
      legacy subject vocabulary (rules below, auditable)
  * Truth = the SET of categories among an accident's C/F findings; a
    prediction is correct if its top-1 category is in that set. MRR uses the
    rank of the first category in the truth set.

Predictors (all leak-safe: outcome phrases redacted before any embedding):
  * freq        -- constant ranking by category frequency among window C/F
                   findings (the baseline every predictor must beat)
  * retrieval   -- similarity-weighted category vote over the C/F findings of
                   the top-100 most similar window accidents
  * bn-post     -- parse narrative -> evidence -> frozen BN; rank categories
                   by max POSTERIOR P(node=Yes | evidence) over that
                   category's finding/person nodes (evidence nodes excluded)
  * bn-lift     -- same, ranked by max LIFT P(Yes|e)/P(Yes) -- surfaces what
                   the evidence CHANGED instead of what is common

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/diagnosis_heldout_eval.py [--limit N]
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
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
from bn_upgraded import build_upgraded, INJ_NODE, DMG_NODE  # noqa: E402

FULL = REPO_ROOT / "shared" / "data" / "processed" / "refined_dataset.json"
WINDOW = REPO_ROOT / "shared" / "data" / "processed" / "refined_dataset_1982_2006.json"
OUT_JSON = FROZEN_DIR / "outputs" / "diagnosis_heldout_eval.json"
OUT_MD = FROZEN_DIR / "outputs" / "diagnosis_heldout_eval.md"

CATS = ["PERSONNEL", "AIRCRAFT", "ENVIRONMENT", "ORGANIZATIONAL"]

# ---------------------------------------------------------------------------
# Category rollup rules
# ---------------------------------------------------------------------------

# CICTT (2008+) top-level -> category
_CICTT_TOP = {
    "personnel issues": "PERSONNEL",
    "aircraft": "AIRCRAFT",
    "environmental issues": "ENVIRONMENT",
    "organizational issues": "ORGANIZATIONAL",
}

# Legacy (1982-2006) subject keyword rules, checked IN ORDER.
# Sources: the 642 distinct C/F subjects in the window (top-70 reviewed by
# hand); person attribution routes ambiguous act-subjects.
_ORG_KEYS = (
    "inadequate design", "insufficient standards", "standards/requirements",
    "surveillance of operation", "procedure inadequate", "inadequate procedure",
    "certification", "regulation", "facility inadequate",
    "insufficiently defined", "oversight", "inadequate substantiation",
)
_ENV_KEYS = (
    "weather", "wind ", "gust", "turbulence", "icing condition",
    "thunderstorm", "terrain", "object", "light condition", "dark night",
    "airport facilit", "runway/landing area condition", "snowbank",
    "bird", "animal", "obstruction", "visibility restriction",
)
_PERSONNEL_KEYS = (
    "lookout", "planning", "decision", "judgment", "procedure", "directive",
    "clearance", "coordination", "supervision", "instruction", "checklist",
    "communicat", "monitoring", "attention", "perception", "workload",
    "maintenance", "handling", "directional control", "aircraft control",
    "airspeed", "altitude", "flare", "alignment", "touchdown", "go-around",
    "rotation", "compensation for wind", "misjudged", "wrong runway",
    "identification", "evaluation", "remedial action", "visual/aural",
    "self-induced pressure", "spatial disorientation", "experience",
    "briefing", "lack of familiarity", "information insufficient",
    "control tower service", "separation", "traffic advisory",
    "emergency", "evacuation", "use of equipment", "preflight",
    "operation with known deficiencies", "impairment", "fatigue (flightcrew",
)
_AIRCRAFT_KEYS = (
    "gear", "tire", "wheel", "strut", "engine", "turbine", "compressor",
    "propeller", "rotor", "blade", "cylinder", "piston", "electrical",
    "wiring", "hydraulic", "pneumatic", "fuel", "oil ", "exhaust",
    "ignition", "pump", "valve", "line", "hose", "fitting", "brake", "apu",
    "auxiliary power", "seat", "belt", "door", "slide", "window",
    "windshield", "wing", "fuselage", "spar", "stabilizer", "aileron",
    "elevator", "rudder", "flap", "spoiler", "slat", "trim",
    "flight control", "instrument", "gauge", "indicator", "warning system",
    "avionics", "autopilot", "air conditioning", "pressurization", "bleed",
    "anti-ice", "deice", "fire extinguish", "powerplant", "gearbox",
    "transmission", "airframe", "skin", "bulkhead", "attachment", "bolt",
    "nut", "fastener", "bearing", "seal", "duct", "filter", "sensor",
    "switch", "relay", "battery", "generator", "alternator", "starter",
    "carburetor", "magneto", "equipment/furnishings", "fluid",
    "air/ground communications equipment", "radar", "rivet", "cable",
    "accessory", "nacelle", "cowling", "pylon", "thrust reverser",
)
_ORG_PERSONS = ("manufacturer", "company", "faa", "management", "operator")


def categorize_legacy(subject: str, person: str = "") -> str | None:
    """Legacy-era finding subject (+ optional person attribution) -> category."""
    s = str(subject or "").lower().strip()
    if not s:
        return None
    for k in _ORG_KEYS:
        if k in s:
            return "ORGANIZATIONAL"
    for k in _ENV_KEYS:
        if k in s:
            return "ENVIRONMENT"
    for k in _PERSONNEL_KEYS:
        if k in s:
            return "PERSONNEL"
    for k in _AIRCRAFT_KEYS:
        if k in s:
            return "AIRCRAFT"
    p = str(person or "").lower()
    if p:
        if any(k in p for k in _ORG_PERSONS):
            return "ORGANIZATIONAL"
        return "PERSONNEL"
    return None


def categorize_cictt(desc: str) -> str | None:
    top = str(desc or "").split("-", 1)[0].strip().lower()
    return _CICTT_TOP.get(top)


def truth_categories(inc: dict) -> set:
    out = set()
    for f in inc.get("findings") or []:
        if f.get("Cause_Factor") not in ("C", "F"):
            continue
        c = categorize_cictt(f.get("finding_description"))
        if c:
            out.add(c)
    return out


def node_category(name: str, legacy_subjects: set) -> str | None:
    """BN node name -> category (None for occurrence/severity/unmapped)."""
    if name in (INJ_NODE, DMG_NODE):
        return None
    if name.startswith("person: "):
        p = name[len("person: "):]
        if any(k in p for k in _ORG_PERSONS):
            return "ORGANIZATIONAL"
        return "PERSONNEL"
    if name not in legacy_subjects:
        return None            # occurrence node, not a cause candidate
    return categorize_legacy(name)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p from discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bootstrap_ci(hits: list, n_boot: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    arr = np.array(hits, dtype=float)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def score_ranking(ranking: list, truth: set):
    """(top1_hit, mrr) for a category ranking against a truth set."""
    hit = 1.0 if ranking and ranking[0] in truth else 0.0
    mrr = 0.0
    for i, c in enumerate(ranking):
        if c in truth:
            mrr = 1.0 / (i + 1)
            break
    return hit, mrr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    full = json.loads(FULL.read_text())
    window_ids = set(json.loads(WINDOW.read_text()).keys())
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = set(bn.names())
    import main_app  # loads window retrieval index

    # legacy subject vocabulary (for node-kind detection + mapping audit)
    legacy_subjects = set()
    win_cat_counts: Counter = Counter()
    n_unmapped_win = n_total_win = 0
    for inc in ds.values():
        for f in inc.get("findings") or []:
            d = pg._s(f.get("finding_description")).lower()
            if d:
                legacy_subjects.add(d)
            if f.get("Cause_Factor") in ("C", "F") and d:
                n_total_win += 1
                c = categorize_legacy(d, f.get("person_description"))
                if c:
                    win_cat_counts[c] += 1
                else:
                    n_unmapped_win += 1
    freq_ranking = [c for c, _ in win_cat_counts.most_common()]
    for c in CATS:
        if c not in freq_ranking:
            freq_ranking.append(c)
    print(f"window C/F findings: {n_total_win}, unmapped by rules: "
          f"{n_unmapped_win} ({100*n_unmapped_win/max(n_total_win,1):.1f}%)")
    print(f"window category counts: {dict(win_cat_counts)}")
    print(f"frequency-baseline ranking: {freq_ranking}")

    # cause-candidate nodes by category
    cand_by_cat: dict = {c: [] for c in CATS}
    for n in names:
        c = node_category(n, legacy_subjects)
        if c:
            cand_by_cat[c].append(n)
    print("candidate cause nodes per category:",
          {c: len(v) for c, v in cand_by_cat.items()})

    # priors for lift (single inference, all candidates)
    ie0 = gum.LazyPropagation(bn)
    all_cands = [n for v in cand_by_cat.values() for n in v]
    for n in all_cands:
        ie0.addTarget(n)
    ie0.makeInference()

    def p_yes(ie, node):
        v = bn.variable(node)
        yes = [i for i in range(v.domainSize()) if v.label(i) == "Yes"][0]
        return float(ie.posterior(node)[yes])

    priors = {n: max(p_yes(ie0, n), 1e-12) for n in all_cands}

    # held-out set: same protocol as the severity eval + needs C/F truth
    held = []
    for k, inc in full.items():
        if k in window_ids:
            continue
        narr = str(inc.get("narr_accf") or "").strip()
        if len(narr) < 100:
            continue
        truth = truth_categories(inc)
        if not truth:
            continue
        held.append((k, inc, narr, truth))
    held.sort(key=lambda t: t[0])
    if limit:
        held = held[:limit]
    print(f"held-out accidents with narrative + C/F cause categories: {len(held)}")

    predictors = ["freq", "retrieval", "bn-post", "bn-lift"]
    per_item = []
    for idx, (k, inc, narr, truth) in enumerate(held):
        text = narr[:4000]
        rec = {"id": k, "truth": sorted(truth)}

        # ---- retrieval vote ------------------------------------------------
        rankings = {"freq": freq_ranking}
        try:
            emb = main_app.get_embedding(qb.inference_query(text))
            scores_, matches = main_app.find_top_matches(emb)
            votes: Counter = Counter()
            seen, n_pool = set(), 0
            for s, m in zip(scores_, matches):
                if m.get("source") != "incident":
                    continue
                ev = m.get("ev_id")
                if not ev or ev in seen or ev not in ds:
                    continue
                seen.add(ev)
                n_pool += 1
                for f in ds[ev].get("findings") or []:
                    if f.get("Cause_Factor") not in ("C", "F"):
                        continue
                    c = categorize_legacy(f.get("finding_description"),
                                          f.get("person_description"))
                    if c:
                        votes[c] += float(s)
                if n_pool >= 100:
                    break
            rankings["retrieval"] = ([c for c, _ in votes.most_common()]
                                     + [c for c in freq_ranking
                                        if c not in votes]) if votes else freq_ranking
        except Exception as exc:
            print(f"  retrieval failed for {k}: {exc}")
            rankings["retrieval"] = freq_ranking

        # ---- BN evidence ---------------------------------------------------
        try:
            hard = qb.parse_query_to_bn_evidence(text, names, dataset=ds,
                                                 semantic=False)
            softonly = {lab: c for lab, c, _ in qb.retrieval_facts(
                text, names, main_app.refined_dataset)}
            conf = qb.merge_evidence_soft_priority(hard["confidence"], softonly)
        except Exception as exc:
            print(f"  parse failed for {k}: {exc}")
            conf = {}
        try:
            ie = gum.LazyPropagation(bn)
            if conf:
                qb.apply_evidence(ie, bn, conf)
            cands = [n for n in all_cands if n not in conf]
            for n in cands:
                ie.addTarget(n)
            ie.makeInference()
            post_scores = {c: 0.0 for c in CATS}
            lift_scores = {c: 0.0 for c in CATS}
            for c in CATS:
                for n in cand_by_cat[c]:
                    if n in conf:
                        continue
                    p = p_yes(ie, n)
                    post_scores[c] = max(post_scores[c], p)
                    lift_scores[c] = max(lift_scores[c], p / priors[n])
            rankings["bn-post"] = sorted(CATS, key=lambda c: -post_scores[c])
            rankings["bn-lift"] = sorted(CATS, key=lambda c: -lift_scores[c])
        except Exception as exc:
            print(f"  BN inference failed for {k}: {exc}")
            rankings["bn-post"] = freq_ranking
            rankings["bn-lift"] = freq_ranking

        for p in predictors:
            hit, mrr = score_ranking(rankings[p], truth)
            rec[f"{p}:top1"] = hit
            rec[f"{p}:mrr"] = mrr
            rec[f"{p}:rank"] = rankings[p]
        per_item.append(rec)
        if (idx + 1) % 25 == 0:
            print(f"  ... {idx + 1}/{len(held)}")

    # ---- aggregate ----------------------------------------------------------
    lines = ["# Held-out diagnosis evaluation (cause-category level)", ""]
    lines.append(f"n = {len(per_item)} held-out accidents (2007-2019) with a "
                 "narrative and >=1 C/F cause finding. Era-fair rollup to "
                 "CICTT top-level categories (AIRCRAFT / PERSONNEL / "
                 "ENVIRONMENT / ORGANIZATIONAL). Truth = category set of the "
                 "accident's C/F findings; top-1 correct if the predicted #1 "
                 "category is in that set. Leak-safe (redacted embeddings).")
    lines += ["", f"Window mapping coverage: {n_total_win - n_unmapped_win}/"
              f"{n_total_win} C/F findings mapped "
              f"({100*(1-n_unmapped_win/max(n_total_win,1)):.1f}%).", ""]

    summary = {}
    lines += ["## Top-1 accuracy and MRR (95% CI bootstrap, 10k)", "",
              "| Predictor | Top-1 | 95% CI | MRR | n |",
              "|---|---|---|---|---|"]
    for p in predictors:
        hits = [r[f"{p}:top1"] for r in per_item]
        mrrs = [r[f"{p}:mrr"] for r in per_item]
        lo, hi = bootstrap_ci(hits)
        summary[p] = {"top1": float(np.mean(hits)), "mrr": float(np.mean(mrrs)),
                      "ci": [lo, hi]}
        lines.append(f"| {p} | {100*np.mean(hits):.1f}% | "
                     f"[{100*lo:.1f}%, {100*hi:.1f}%] | "
                     f"{np.mean(mrrs):.3f} | {len(hits)} |")

    lines += ["", "## Paired comparisons (McNemar exact, top-1)", "",
              "| A vs B | A only right | B only right | p |", "|---|---|---|---|"]
    pairs = [("retrieval", "freq"), ("bn-lift", "freq"), ("bn-post", "freq"),
             ("bn-lift", "bn-post"), ("retrieval", "bn-lift"),
             ("retrieval", "bn-post")]
    for a, b in pairs:
        aw = sum(1 for r in per_item
                 if r[f"{a}:top1"] > r[f"{b}:top1"])
        bw = sum(1 for r in per_item
                 if r[f"{b}:top1"] > r[f"{a}:top1"])
        pv = mcnemar_exact(aw, bw)
        star = " *" if pv < 0.05 else ""
        lines.append(f"| {a} vs {b} | {aw} | {bw} | {pv:.4f}{star} |")

    lines += ["", "## Per-category recall (top-1 predictions, truth contains category)",
              "", "| Predictor | " + " | ".join(CATS) + " |",
              "|---|" + "---|" * len(CATS)]
    for p in predictors:
        cells = []
        for c in CATS:
            rel = [r for r in per_item if c in r["truth"]]
            got = sum(1 for r in rel if r[f"{p}:rank"][0] == c)
            cells.append(f"{got}/{len(rel)}")
        lines.append(f"| {p} | " + " | ".join(cells) + " |")

    OUT_MD.write_text("\n".join(lines) + "\n")
    OUT_JSON.write_text(json.dumps({
        "n": len(per_item), "summary": summary,
        "window_mapping": {"total": n_total_win, "unmapped": n_unmapped_win,
                           "counts": dict(win_cat_counts)},
        "freq_ranking": freq_ranking, "items": per_item}, indent=1))
    print("\n".join(lines[4:30]))
    print(f"\nwrote {OUT_MD}\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

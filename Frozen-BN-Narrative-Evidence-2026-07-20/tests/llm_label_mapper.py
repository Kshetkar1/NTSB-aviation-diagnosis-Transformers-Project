#!/usr/bin/env python3
"""LLM Semantic Label Mapper — can an LLM bridge the taxonomy gap?

The phrase parser gets 9.3% recall on the 12 severity-parent labels because
NTSB taxonomy names ("airframe/component/system failure/malfunction") never
appear in narratives. This experiment tests whether an LLM can read a
redacted narrative and correctly identify which of the 12 parent labels apply.

Evaluation:
  Phase 1 — Label-level: precision, recall, F1 against human-coded labels
  Phase 2 — End-to-end: route LLM labels through the BN → severity accuracy

Cohort: n=300 leave-one-out from the 1982-2006 build window (same as D3).
Test set (2007-2019) is NEVER touched.

Run:
    python3 tests/llm_label_mapper.py
    python3 tests/llm_label_mapper.py --n 50   # quick test with 50 accidents
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT.parent / "shared" / "code"))

os.environ.pop("NTSB_FULL_CORPUS", None)

from openai import OpenAI
from config import get_openai_api_key, LLM_MODEL

import numpy as np
import pyagrum as gum
import prognosis as pg
from bn_upgraded import (build_upgraded, INJ_NODE, DMG_NODE,
                         INJ_STATES, DMG_STATES, injury_state, damage_state,
                         last_occurrence)
import query_to_bn as qb

OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# The 12 severity parents (shared by injury and damage nodes)
SEVERITY_PARENTS = [
    "loss of engine power (total) - mechanical failure/malfunction",
    "hard landing",
    "in flight collision with terrain/water",
    "in flight encounter with weather",
    "in flight collision with object",
    "near collision between aircraft",
    "fire",
    "on ground/water collision with object",
    "on ground/water collision with terrain/water",
    "airframe/component/system failure/malfunction",
    "dragged wing, rotor, pod, float or tail/skid",
    "miscellaneous/other",
]

PROMPT_TEMPLATE = """You are an NTSB aviation accident analyst. Given an accident narrative, identify which of the following event categories occurred in this accident.

THE 12 EVENT CATEGORIES (use these EXACT labels):
1. loss of engine power (total) - mechanical failure/malfunction
2. hard landing
3. in flight collision with terrain/water
4. in flight encounter with weather
5. in flight collision with object
6. near collision between aircraft
7. fire
8. on ground/water collision with object
9. on ground/water collision with terrain/water
10. airframe/component/system failure/malfunction
11. dragged wing, rotor, pod, float or tail/skid
12. miscellaneous/other

RULES:
- Only select categories that ACTUALLY OCCURRED in the accident, based on the narrative.
- "airframe/component/system failure/malfunction" includes any structural failure, component breakage, system malfunction, or mechanical issue with the aircraft.
- "hard landing" means the landing itself was abnormally forceful, not just any landing.
- "fire" includes post-crash fire, in-flight fire, or any fire during the accident sequence.
- "miscellaneous/other" is a catch-all for events that don't fit other categories.
- Do NOT guess. If the narrative doesn't describe an event, don't include it.
- Return ONLY a JSON array of the exact label strings that apply. No explanation.

NARRATIVE:
{narrative}

RESPOND WITH ONLY A JSON ARRAY, e.g.: ["fire", "hard landing"]"""


def get_ground_truth(inc: dict, bn_names: set) -> set:
    """Which of the 12 severity parents are active for this accident?"""
    active = set()
    # Check occurrences
    for occ in inc.get("sequence_of_events", []):
        desc = str(occ.get("Occurrence_Description", "")).strip().lower()
        if desc in bn_names:
            active.add(desc)
    # Check findings
    for f in inc.get("findings", []):
        desc = str(f.get("finding_description", "")).strip().lower()
        if desc in bn_names:
            active.add(desc)
    # Also check last occurrence
    last = last_occurrence(inc)
    if last and last in bn_names:
        active.add(last)
    return active & set(SEVERITY_PARENTS)


def llm_predict(client: OpenAI, narrative: str, model: str) -> set:
    """Ask the LLM which of the 12 labels apply to this narrative."""
    redacted = qb.redact_severity_phrases(narrative[:4000])
    prompt = PROMPT_TEMPLATE.format(narrative=redacted)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            text = resp.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            labels = json.loads(text)
            if isinstance(labels, list):
                valid = set()
                for lab in labels:
                    lab_lower = lab.strip().lower()
                    for p in SEVERITY_PARENTS:
                        if lab_lower == p.lower():
                            valid.add(p)
                            break
                return valid
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt < 2:
                time.sleep(1)
                continue
            print(f"  WARNING: failed to parse LLM response: {text[:100]}")
        except Exception as e:
            if "rate_limit" in str(e).lower():
                time.sleep(5)
                continue
            raise
    return set()


def bn_severity_posterior(bn, evidence: dict, target: str, states: list) -> int:
    """Get the argmax severity state given evidence."""
    ie = gum.LazyPropagation(bn)
    valid = {k: v for k, v in evidence.items() if k in bn.names()}
    if valid:
        ie.setEvidence(valid)
    ie.addTarget(target)
    ie.makeInference()
    post = ie.posterior(target)
    return int(np.argmax([float(post[i]) for i in range(len(states))]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="Number of accidents to test")
    parser.add_argument("--model", default=None, help="LLM model (default: config LLM_MODEL)")
    parser.add_argument("--heldout", action="store_true",
                        help="Score the 2007-2019 held-out cohort instead of the "
                             "build window. NOTE: the CICTT recoding means 0 of 296 "
                             "held-out accidents carry coded labels in the network's "
                             "vocabulary, so label-level P/R and the coded arm are "
                             "not computable there and are skipped.")
    args = parser.parse_args()

    model = args.model or LLM_MODEL
    print(f"LLM Semantic Label Mapper Experiment")
    print(f"Model: {model}, n={args.n}")
    print(f"=" * 60)

    # Build network
    print("\nBuilding network …", flush=True)
    ds = pg.load_dataset()
    bn, meta = build_upgraded(ds)
    bn_names = set(bn.names())
    print(f"  {bn.size()} nodes, {bn.sizeArcs()} arcs")

    if args.heldout:
        # Same cohort definition as frozenbn_heldout_narrative_bn_eval.py:
        # in the full corpus, not in the window, narrative >= 100 chars.
        PROC = ROOT.parent / "shared" / "data" / "processed"
        full = json.loads((PROC / "refined_dataset.json").read_text())
        window_ids = set(json.loads(
            (PROC / "refined_dataset_1982_2006.json").read_text()).keys())
        cohort = [(k, v) for k, v in full.items()
                  if k not in window_ids
                  and len(str(v.get("narr_accf") or "").strip()) >= 100]
        cohort.sort(key=lambda t: t[0])
        if args.n:
            cohort = cohort[:args.n]
        print(f"  cohort: held-out 2007-2019, {len(cohort)} accidents")
        print(f"  NOTE: 0 of these carry coded labels in the network vocabulary")
        print(f"        (CICTT recoding, BII-1) -- label-level P/R and the coded")
        print(f"        arm are not computable and are skipped.\n")
    else:
        candidates = [(k, v) for k, v in ds.items()
                      if len(str(v.get("narr_accf", "")).strip()) >= 100]
        random.seed(42)
        random.shuffle(candidates)
        cohort = candidates[:args.n]
        print(f"  cohort: build window 1982-2006, {len(cohort)} accidents "
              f"with narratives >=100 chars\n")

    client = OpenAI(api_key=get_openai_api_key())

    # Phase 1: Label-level evaluation
    print("PHASE 1: Label-level evaluation")
    print("-" * 40)

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    total_tp, total_fp, total_fn = 0, 0, 0

    # Also collect phrase-parser results for comparison
    phrase_tp, phrase_fp, phrase_fn = 0, 0, 0

    # Phase 2 accumulators
    inj_correct = {"llm": 0, "phrase": 0, "coded": 0, "prior": 0}
    dmg_correct = {"llm": 0, "phrase": 0, "coded": 0, "prior": 0}
    n_inj, n_dmg = 0, 0

    results = []

    for i, (ev_id, inc) in enumerate(cohort):
        narr = str(inc.get("narr_accf", ""))
        ground = get_ground_truth(inc, bn_names)

        # LLM prediction
        llm_pred = llm_predict(client, narr, model)

        # Phrase parser prediction (existing pipeline)
        redacted = qb.redact_severity_phrases(narr[:4000])
        parsed = qb.parse_query_to_bn_evidence(redacted, bn_names, dataset=ds, semantic=False)
        phrase_pred = set(parsed.get("confidence", {}).keys()) & set(SEVERITY_PARENTS)

        # Label-level scoring
        for lab in SEVERITY_PARENTS:
            in_ground = lab in ground
            in_llm = lab in llm_pred
            in_phrase = lab in phrase_pred

            if in_ground and in_llm:
                tp[lab] += 1; total_tp += 1
            elif in_llm and not in_ground:
                fp[lab] += 1; total_fp += 1
            elif in_ground and not in_llm:
                fn[lab] += 1; total_fn += 1

            if in_ground and in_phrase:
                phrase_tp += 1
            elif in_phrase and not in_ground:
                phrase_fp += 1
            elif in_ground and not in_phrase:
                phrase_fn += 1

        # Phase 2: End-to-end severity through BN
        inj_true = injury_state(inc)
        dmg_true_val = damage_state(inc)

        # LLM evidence
        llm_ev = {lab: "Yes" for lab in llm_pred if lab in bn_names}
        # Phrase evidence
        phrase_ev = {lab: "Yes" for lab in phrase_pred if lab in bn_names}
        # Coded evidence (ground truth)
        coded_ev = {lab: "Yes" for lab in ground if lab in bn_names}

        # Injury predictions
        n_inj += 1
        inj_prior = bn_severity_posterior(bn, {}, INJ_NODE, INJ_STATES)
        inj_llm = bn_severity_posterior(bn, llm_ev, INJ_NODE, INJ_STATES)
        inj_phrase = bn_severity_posterior(bn, phrase_ev, INJ_NODE, INJ_STATES)
        inj_coded = bn_severity_posterior(bn, coded_ev, INJ_NODE, INJ_STATES)

        if inj_prior == inj_true: inj_correct["prior"] += 1
        if inj_llm == inj_true: inj_correct["llm"] += 1
        if inj_phrase == inj_true: inj_correct["phrase"] += 1
        if inj_coded == inj_true: inj_correct["coded"] += 1

        # Damage predictions
        if dmg_true_val is not None:
            n_dmg += 1
            dmg_prior = bn_severity_posterior(bn, {}, DMG_NODE, DMG_STATES)
            dmg_llm = bn_severity_posterior(bn, llm_ev, DMG_NODE, DMG_STATES)
            dmg_phrase = bn_severity_posterior(bn, phrase_ev, DMG_NODE, DMG_STATES)
            dmg_coded = bn_severity_posterior(bn, coded_ev, DMG_NODE, DMG_STATES)

            if dmg_prior == dmg_true_val: dmg_correct["prior"] += 1
            if dmg_llm == dmg_true_val: dmg_correct["llm"] += 1
            if dmg_phrase == dmg_true_val: dmg_correct["phrase"] += 1
            if dmg_coded == dmg_true_val: dmg_correct["coded"] += 1

        results.append({
            "ev_id": ev_id,
            "ground_truth": sorted(ground),
            "llm_pred": sorted(llm_pred),
            "phrase_pred": sorted(phrase_pred),
            "inj_true": INJ_STATES[inj_true],
            "inj_llm": INJ_STATES[inj_llm],
            "inj_phrase": INJ_STATES[inj_phrase],
            "inj_coded": INJ_STATES[inj_coded],
        })

        if (i + 1) % 10 == 0:
            llm_rec = total_tp / max(total_tp + total_fn, 1) * 100
            llm_prec = total_tp / max(total_tp + total_fp, 1) * 100
            print(f"  [{i+1}/{len(cohort)}] LLM recall {llm_rec:.1f}%  "
                  f"precision {llm_prec:.1f}%  "
                  f"injury {inj_correct['llm']}/{n_inj}  "
                  f"damage {dmg_correct['llm']}/{n_dmg}", flush=True)

    # ─── RESULTS ─────────────────────────────────────────────────────────
    def prf(tp_val, fp_val, fn_val):
        p = tp_val / max(tp_val + fp_val, 1)
        r = tp_val / max(tp_val + fn_val, 1)
        f = 2 * p * r / max(p + r, 1e-9)
        return p, r, f

    llm_p, llm_r, llm_f = prf(total_tp, total_fp, total_fn)
    ph_p, ph_r, ph_f = prf(phrase_tp, phrase_fp, phrase_fn)

    print("\n" + "=" * 60)
    print("PHASE 1 RESULTS: Label-level (12 severity parents)")
    print("=" * 60)

    if args.heldout:
        print("\n  SKIPPED. 0 of the 296 held-out accidents carry coded labels in")
        print("  the network's vocabulary (CICTT recoding, BII-1), so there is no")
        print("  ground truth to score against on this cohort.")
    else:
        print(f"\n  {'Method':30s} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10}")
        print(f"  {'Phrase parser':30s} {ph_p:10.1%} {ph_r:10.1%} {ph_f:10.1%}")
        print(f"  {'LLM label mapper':30s} {llm_p:10.1%} {llm_r:10.1%} {llm_f:10.1%}")

        print(f"\n  Per-label recall (LLM):")
        for lab in SEVERITY_PARENTS:
            t = tp[lab]; f = fn[lab]
            rec = t / max(t + f, 1)
            total = t + f
            if total > 0:
                print(f"    {lab:55s}  {rec:5.1%}  ({t}/{total})")

    print("\n" + "=" * 60)
    print("PHASE 2 RESULTS: End-to-end severity through BN")
    print("=" * 60)

    arms = ["prior", "phrase", "llm"] if args.heldout else \
           ["prior", "phrase", "llm", "coded"]

    print(f"\n  {'Arm':30s} {'Injury acc':>12} {'Damage acc':>12}")
    print(f"  {'─'*30} {'─'*12} {'─'*12}")
    for arm in arms:
        ia = inj_correct[arm] / max(n_inj, 1) * 100
        da = dmg_correct[arm] / max(n_dmg, 1) * 100
        marker = " <- LLM" if arm == "llm" else ""
        print(f"  {arm:30s} {ia:11.1f}% {da:11.1f}%{marker}")

    if args.heldout:
        print(f"\n  'coded' arm omitted: no held-out accident has coded evidence")
        print(f"  in the network vocabulary, so it is identical to 'prior'.")

    print(f"\n  n_injury={n_inj}, n_damage={n_dmg}")

    # Delta
    llm_inj = inj_correct["llm"] / max(n_inj, 1) * 100
    llm_dmg = dmg_correct["llm"] / max(n_dmg, 1) * 100
    ph_inj = inj_correct["phrase"] / max(n_inj, 1) * 100
    ph_dmg = dmg_correct["phrase"] / max(n_dmg, 1) * 100
    cod_inj = inj_correct["coded"] / max(n_inj, 1) * 100
    cod_dmg = dmg_correct["coded"] / max(n_dmg, 1) * 100

    print(f"\n  LLM vs phrase parser:  injury {llm_inj - ph_inj:+.1f} pp,  "
          f"damage {llm_dmg - ph_dmg:+.1f} pp")
    if not args.heldout:
        print(f"  LLM vs human coded:   injury {llm_inj - cod_inj:+.1f} pp,  "
              f"damage {llm_dmg - cod_dmg:+.1f} pp")

    # Save results
    suffix = "_heldout" if args.heldout else "_window"
    out_json = OUT_DIR / f"llm_label_mapper_results{suffix}.json"
    summary = {
        "model": model,
        "cohort": "heldout_2007_2019" if args.heldout else "window_1982_2006",
        "label_level_computable": not args.heldout,
        "n": len(cohort),
        "label_level": {
            "llm": {"precision": llm_p, "recall": llm_r, "f1": llm_f},
            "phrase": {"precision": ph_p, "recall": ph_r, "f1": ph_f},
        },
        "end_to_end": {
            "injury": {arm: inj_correct[arm] / max(n_inj, 1) for arm in inj_correct},
            "damage": {arm: dmg_correct[arm] / max(n_dmg, 1) for arm in dmg_correct},
            "n_injury": n_inj, "n_damage": n_dmg,
        },
        "per_accident": results[:20],
    }
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {out_json.relative_to(ROOT)}")

    # Markdown summary
    out_md = OUT_DIR / f"llm_label_mapper{suffix}.md"
    cohort_name = ("held-out 2007-2019" if args.heldout
                   else "build window 1982-2006")
    with open(out_md, "w") as f:
        f.write(f"# LLM Semantic Label Mapper Results\n\n")
        f.write(f"Model: {model} | cohort: {cohort_name} | n={len(cohort)}\n\n")
        f.write(f"## Label-level (12 severity parents)\n\n")
        if args.heldout:
            f.write("Not computable on this cohort. 0 of 296 held-out accidents "
                    "carry coded labels in the network's vocabulary (CICTT "
                    "recoding, BII-1), so there is no ground truth to score "
                    "against.\n\n")
        else:
            f.write(f"| Method | Precision | Recall | F1 |\n|---|---|---|---|\n")
            f.write(f"| Phrase parser | {ph_p:.1%} | {ph_r:.1%} | {ph_f:.1%} |\n")
            f.write(f"| **LLM mapper** | **{llm_p:.1%}** | **{llm_r:.1%}** | "
                    f"**{llm_f:.1%}** |\n\n")
        f.write(f"## End-to-end severity (through BN event nodes)\n\n")
        f.write(f"| Arm | Injury | Damage |\n|---|---|---|\n")
        for arm in arms:
            ia = inj_correct[arm] / max(n_inj, 1) * 100
            da = dmg_correct[arm] / max(n_dmg, 1) * 100
            f.write(f"| {arm} | {ia:.1f}% | {da:.1f}% |\n")
        if args.heldout:
            f.write("\n`coded` arm omitted: no held-out accident has coded "
                    "evidence in the network vocabulary, so it is identical "
                    "to `prior`.\n")
        f.write(f"\nLLM vs phrase: injury {llm_inj - ph_inj:+.1f} pp, "
                f"damage {llm_dmg - ph_dmg:+.1f} pp\n")
        if not args.heldout:
            f.write(f"LLM vs coded: injury {llm_inj - cod_inj:+.1f} pp, "
                    f"damage {llm_dmg - cod_dmg:+.1f} pp\n")
    print(f"  wrote {out_md.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())

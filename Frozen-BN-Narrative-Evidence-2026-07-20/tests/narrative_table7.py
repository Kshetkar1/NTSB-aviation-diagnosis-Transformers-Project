#!/usr/bin/env python3
"""Zhang's Table 7, rebuilt from narratives instead of coded fields.

Zhang's Table 7 answers the diagnosis question: GIVEN A FIRE, what caused it?
His value for a cause is a count over coded records,

    P(cause | fire) = #(fire accidents whose coded record carries that cause)
                      / 102

and `zhang_diagnosis.empirical_cause_distribution("fire", cause_factor_only=True)`
reproduces all 85 of his published rows to +/- 0.0005. That reproduction uses
CODED FIELDS ONLY -- no narrative touches it. It proves the pipeline is
faithful; it says nothing about whether narratives can do the same job.

This script adds the missing columns. For the same 102 fire accidents it
re-derives the same table using ONLY the factual narrative, under four
different ways of getting from prose to NTSB cause labels:

    phrase     deterministic phrase matching (the shipped parser)
    retrieval  similarity-weighted labels from the k-NN pool
    bn         narrative evidence entered on the frozen network, conditioned
               on fire = Yes, reading posteriors over the cause nodes
    llm        an LLM given the narrative and Zhang's own cause vocabulary

Each arm predicts a SET of causes per accident, then the identical counting
rule is applied, so every column is directly comparable to Zhang's.

Agreement with Zhang is reported as Spearman rank correlation, top-10
overlap, total variation distance and L1 -- not accuracy, because Table 7 is
a distribution, not a classification.

Leakage: retrieval excludes each query accident from its own neighbour pool.
The word "fire" is NOT masked -- we condition on fire being known, exactly as
Zhang does; the task is to recover the CAUSE, not to detect the fire.

Run:
    python3 tests/narrative_table7.py [--limit N] [--no-bn] [--no-llm]
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT.parent / "shared" / "code"))

os.environ.pop("NTSB_FULL_CORPUS", None)

import prognosis as pg
import query_to_bn as qb
import zhang_diagnosis as zd

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

TOP_SHOW = 20
MIN_NARR = 100


PROMPT = """You are an NTSB accident analyst. Below is the factual narrative of an
aviation accident that involved a FIRE.

Your task: identify which of the listed NTSB cause/factor labels were
contributory to this accident, based ONLY on the narrative.

CANDIDATE LABELS:
{labels}

RULES:
- Select only labels the narrative actually supports.
- The narrative uses plain English; the labels are NTSB taxonomy terms. Map
  the meaning, not the wording. ("the alternator wire shorted" ->
  "Electrical system, electric wiring")
- NTSB investigators assign a contributory cause sparingly: on average fewer
  than two per accident. Prefer precision over coverage.
- Do NOT guess. If the narrative gives no evidence for a label, omit it.
- Give each selection a confidence in [0,1]: how sure you are that an NTSB
  investigator would have recorded this label as contributory.

NARRATIVE:
{narrative}

Respond with ONLY a JSON object mapping the exact label strings to
confidences, ranked most to least confident, e.g.
{{"Airframe/component/system failure/malfunction": 0.9, "Fluid, fuel": 0.4}}
"""


def llm_causes(client, narrative, labels, model):
    """Ask the LLM which cause labels apply, WITH confidences.

    Confidences are what make the calibration step possible: Zhang's table
    averages ~1.7 contributory causes per accident, and an uncalibrated LLM
    returns roughly twice that, which inflates every cell. Returning scores
    instead of a bare set lets the operating point be chosen afterwards
    without re-querying the model.
    """
    numbered = "\n".join(f"{i+1}. {l}" for i, l in enumerate(labels))
    prompt = PROMPT.format(labels=numbered, narrative=narrative[:4000])
    valid = {l.lower(): l for l in labels}
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            text = resp.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            picked = json.loads(text.strip())
            if isinstance(picked, list):      # tolerate the older array form
                picked = {p: 1.0 for p in picked if isinstance(p, str)}
            return {valid[k.lower()]: float(v) for k, v in picked.items()
                    if isinstance(k, str) and k.lower() in valid}
        except Exception as e:
            if attempt == 2:
                print(f"      llm failed: {e.__class__.__name__}")
                return {}
            import time
            time.sleep(2 * (attempt + 1))
    return {}


def agreement(ref: dict, arm: dict, all_causes: list) -> dict:
    """Compare an arm's P(cause|fire) against Zhang's."""
    from scipy import stats
    r = np.array([ref.get(c, 0.0) for c in all_causes])
    a = np.array([arm.get(c, 0.0) for c in all_causes])
    rho, p = stats.spearmanr(r, a) if len(all_causes) > 2 else (np.nan, np.nan)
    # distributions are not normalised (a cause set per accident), so compare
    # shape after normalising to sum 1
    rn = r / r.sum() if r.sum() else r
    an = a / a.sum() if a.sum() else a
    tv = 0.5 * np.abs(rn - an).sum()
    top_ref = {c for c in sorted(all_causes, key=lambda c: -ref.get(c, 0))[:10]}
    top_arm = {c for c in sorted(all_causes, key=lambda c: -arm.get(c, 0))[:10]}
    return {
        "spearman_rho": float(rho), "spearman_p": float(p),
        "total_variation": float(tv),
        "l1": float(np.abs(r - a).sum()),
        "top10_overlap": len(top_ref & top_arm),
        "n_causes_predicted": int((a > 0).sum()),
    }


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    use_bn = "--no-bn" not in sys.argv
    use_llm = "--no-llm" not in sys.argv

    print("Loading 1982-2006 window ...", flush=True)
    ds = pg.load_dataset()

    # ── Zhang / coded reference column ────────────────────────────────────
    ref = zd.empirical_cause_distribution("fire", dataset=ds,
                                          cause_factor_only=True)
    zhang_col = {d["cause"]: d["probability"] for d in ref["causes"]}
    n_fire_total = ref["outcome_count"]
    print(f"  coded reference: {n_fire_total} fire accidents, "
          f"{len(zhang_col)} causes (Zhang's Table 7)")

    # the fire cohort itself
    targets = zd.OUTCOME_ALIASES.get("fire", {"fire"})
    fire_ids = [k for k, inc in ds.items() if zd._has_outcome(inc, targets)]
    fire_ids.sort()
    cohort = [k for k in fire_ids
              if len(str(ds[k].get("narr_accf") or "").strip()) >= MIN_NARR]
    print(f"  of those, {len(cohort)} have a factual narrative "
          f">= {MIN_NARR} chars")
    if limit:
        cohort = cohort[:limit]
        print(f"  limited to {len(cohort)}")

    cause_vocab = sorted(zhang_col)          # Zhang's own 85 labels
    vocab_lower = {c.lower(): c for c in cause_vocab}

    # ── network, only if the BN arm is on ─────────────────────────────────
    bn = names = None
    if use_bn:
        from bn_upgraded import build_upgraded
        print("  building frozen network ...", flush=True)
        bn, _ = build_upgraded(ds)
        names = list(bn.names())
        # Zhang's published labels are Title Case; network nodes are lower.
        node_lower = {n.lower(): n for n in names}
        bn_cause_nodes = {c: node_lower[c.lower()] for c in cause_vocab
                          if c.lower() in node_lower}
        print(f"  {len(bn_cause_nodes)} of {len(cause_vocab)} Zhang causes "
              f"exist as network nodes")
    else:
        from bn_upgraded import build_upgraded  # noqa: F401
        names = cause_vocab
        bn_cause_nodes = {}

    client = model = None
    if use_llm:
        from openai import OpenAI
        from config import get_openai_api_key, LLM_MODEL
        client = OpenAI(api_key=get_openai_api_key())
        model = LLM_MODEL
        print(f"  LLM arm: {model}")

    arms = ["phrase", "retrieval"]
    if use_bn:
        arms.append("bn")
    if use_llm:
        arms.append("llm")

    hits = {a: defaultdict(set) for a in arms}
    llm_scores: dict[str, dict] = {}
    print(f"\nScoring {len(cohort)} fire accidents across arms {arms} ...",
          flush=True)

    import pyagrum as gum
    for i, ev_id in enumerate(cohort):
        inc = ds[ev_id]
        narr = str(inc.get("narr_accf") or "")[:4000]

        # ---- phrase ------------------------------------------------------
        try:
            parsed = qb.parse_query_to_bn_evidence(narr, names, dataset=ds,
                                                   semantic=False)
            conf = parsed.get("confidence", {})
            for lab in conf:
                if lab.lower() in vocab_lower:
                    hits["phrase"][vocab_lower[lab.lower()]].add(ev_id)
        except Exception:
            pass

        # ---- retrieval ---------------------------------------------------
        try:
            import main_app
            facts = qb.retrieval_facts(narr, names, main_app.refined_dataset,
                                       exclude_ev_ids={ev_id})
            for lab, c, _ in facts:
                if lab.lower() in vocab_lower:
                    hits["retrieval"][vocab_lower[lab.lower()]].add(ev_id)
        except Exception:
            pass

        # ---- bn ----------------------------------------------------------
        if use_bn and bn_cause_nodes:
            try:
                ie = gum.LazyPropagation(bn)
                ev = {n: c for n, c in conf.items() if n in set(names)}
                ev.pop("fire", None)
                qb.apply_evidence(ie, bn, ev)
                if "fire" in set(names):
                    ie.addEvidence("fire", "Yes")
                for node in bn_cause_nodes.values():
                    ie.addTarget(node)
                ie.makeInference()
                for zlabel, node in bn_cause_nodes.items():
                    v = bn.variable(node)
                    yi = [k for k in range(v.domainSize())
                          if v.label(k) == "Yes"]
                    if yi and float(ie.posterior(node)[yi[0]]) >= 0.5:
                        hits["bn"][zlabel].add(ev_id)
                del ie
            except Exception:
                pass

        # ---- llm ---------------------------------------------------------
        if use_llm:
            scored = llm_causes(client, narr, cause_vocab, model)
            llm_scores[ev_id] = scored
            for c in scored:
                hits["llm"][c].add(ev_id)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(cohort)}]", flush=True)

    # ── build columns and compare ─────────────────────────────────────────
    n = len(cohort)
    cols = {a: {c: len(v) / n for c, v in hits[a].items()} for a in arms}

    # ── calibration: match Zhang's attribution rate ───────────────────────
    # Zhang's table carries ~1.735 contributory causes per fire accident. An
    # arm emitting a different total mass cannot match his cells no matter how
    # good its semantics, so the LLM's confidences are thresholded to put the
    # same total mass on the table. This is a single scalar fitted to one
    # published aggregate -- not per-cause tuning.
    zhang_mass = sum(zhang_col.values())
    if use_llm and llm_scores:
        pairs = sorted(((s, ev, c) for ev, d in llm_scores.items()
                        for c, s in d.items()), reverse=True)
        keep = int(round(zhang_mass * n))
        cal = defaultdict(set)
        for s, ev, c in pairs[:keep]:
            cal[c].add(ev)
        cols["llm-cal"] = {c: len(v) / n for c, v in cal.items()}
        arms = arms + ["llm-cal"]
        thr = pairs[keep - 1][0] if 0 < keep <= len(pairs) else 0.0
        print(f"\n  calibrated LLM: kept top {keep} (cause, accident) pairs "
              f"= {zhang_mass:.3f}/accident, confidence cut {thr:.2f}")

    all_causes = sorted(set(zhang_col) | {c for a in arms for c in cols[a]})

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say("\n" + "=" * 100)
    say("ZHANG'S TABLE 7, REBUILT FROM NARRATIVES")
    say("=" * 100)
    say(f"\n  Fire accidents: {n_fire_total} coded / {n} with narratives "
        f"(this table uses the {n} narrative accidents for every arm).")
    say(f"  Cause vocabulary: Zhang's own {len(cause_vocab)} labels.")
    if use_bn:
        say(f"  Of those, {len(bn_cause_nodes)} exist as nodes in the frozen "
            f"network -- the BN arm can only ever name those.")
    say("")

    hdr = f"  {'cause':<52} {'Zhang':>7}" + "".join(
        f"{a:>11}" for a in arms)
    say(hdr)
    say("  " + "-" * (52 + 7 + 11 * len(arms)))
    for c in sorted(all_causes, key=lambda c: -zhang_col.get(c, 0))[:TOP_SHOW]:
        row = f"  {c[:52]:<52} {zhang_col.get(c,0):7.4f}"
        for a in arms:
            row += f"{cols[a].get(c,0):11.4f}"
        say(row)

    say("\n" + "=" * 100)
    say("AGREEMENT WITH ZHANG'S PUBLISHED COLUMN")
    say("=" * 100)
    say(f"\n  {'arm':<12} {'Spearman rho':>13} {'p':>10} {'top-10 ovlp':>12} "
        f"{'tot.var.':>10} {'L1':>8} {'#causes named':>15}")
    say(f"  {'-'*12} {'-'*13} {'-'*10} {'-'*12} {'-'*10} {'-'*8} {'-'*15}")
    agg = {}
    for a in arms:
        m = agreement(zhang_col, cols[a], all_causes)
        agg[a] = m
        say(f"  {a:<12} {m['spearman_rho']:13.3f} {m['spearman_p']:10.4f} "
            f"{m['top10_overlap']:9d}/10 {m['total_variation']:10.3f} "
            f"{m['l1']:8.2f} {m['n_causes_predicted']:12d}/{len(cause_vocab)}")

    say("""
  Spearman rho  -- does the arm rank causes the way Zhang does?
  top-10 ovlp   -- how many of Zhang's ten most common causes it also ranks top-10
  tot.var.      -- distance between the two shapes after normalising (0 = identical)
  #causes named -- coverage; an arm that names 3 causes cannot reproduce a table of 85
""")

    # ── per-cell closeness, in the Appendix A format ──────────────────────
    best = max((a for a in arms),
               key=lambda a: agg[a]["spearman_rho"]
               if not np.isnan(agg[a]["spearman_rho"]) else -9)
    say("=" * 100)
    say(f"PER-CELL CLOSENESS -- Zhang against the best narrative arm ({best})")
    say("=" * 100)
    say(f"\n  {'#':>3}  {'Contributory factor':<50} {'Zhang':>8} "
        f"{'Narr':>8} {'|Diff|':>8}")
    say(f"  {'-'*3}  {'-'*50} {'-'*8} {'-'*8} {'-'*8}")
    ordered = sorted(zhang_col, key=lambda c: -zhang_col[c])
    diffs = []
    for i, c in enumerate(ordered[:25], 1):
        zv, av = zhang_col[c], cols[best].get(c, 0.0)
        diffs.append(abs(zv - av))
        say(f"  {i:>3}  {c[:50]:<50} {zv:8.4f} {av:8.4f} {abs(zv-av):8.4f}")
    alld = [abs(zhang_col[c] - cols[best].get(c, 0.0)) for c in zhang_col]
    say(f"\n  Across all {len(zhang_col)} of Zhang's causes: "
        f"mean |Diff| {np.mean(alld):.4f}, median {np.median(alld):.4f}, "
        f"max {np.max(alld):.4f}")
    say(f"  Cells within 0.02 of Zhang: "
        f"{sum(1 for d in alld if d <= 0.02)}/{len(alld)}   "
        f"within 0.05: {sum(1 for d in alld if d <= 0.05)}/{len(alld)}")
    say(f"  (the coded reproduction matches every cell to +/-0.0005)")
    say("")

    (OUT / "narrative_table7.md").write_text(
        "# Zhang Table 7 from narratives\n\n```\n" + "\n".join(lines) + "\n```\n")
    (OUT / "narrative_table7.json").write_text(json.dumps({
        "n_fire_coded": n_fire_total, "n_fire_narrative": n,
        "cause_vocab_size": len(cause_vocab),
        "zhang": zhang_col, "arms": cols, "agreement": agg,
        "zhang_mass_per_accident": zhang_mass,
        "llm_raw_scores": llm_scores,
    }, indent=2))
    print("  wrote outputs/narrative_table7.{md,json}")


if __name__ == "__main__":
    raise SystemExit(main())

"""FIX 2 - Auto-gating: route conditioning OFF where it would hurt (READ-ONLY eval).

The §14 validation showed conditioning helps on SPECIFIC-mechanism causes but
HURTS on incidents whose only true cause is a generic catch-all (top-1 -20.7 pp),
because there the unconditioned frequency prior is already optimal. A trustworthy
tool must not condition when it would hurt.

This script evaluates candidate **per-query gates** (all observable at inference,
none require the true cause) on the validated LOO records, and reports combined
top-1 / MRR / Brier for: (i) ungated conditioning, (ii) prior, (iii) each gate.
A gate "fires" => fall back to the unconditioned prior for that incident; else
keep the conditioned prediction. We report per-subgroup (generic-true vs
specific-true) so we can see whether the gate removes the harm without regressing
the specific-cause win.

The winning rule is then implemented in ``zhang_diagnosis.gated_diagnose``.

Outputs: docs/gating_results.json
Run:  python3.11 tests/gating_validation.py   (offline, cached embeddings)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "docs" / ".mplcache"))

import qc_common as qc  # noqa: E402

DOCS = ROOT / "docs"
RESULTS_PATH = DOCS / "gating_results.json"
G = qc.GENERIC
MARGIN_THR = 0.08  # selective_diagnose default (SELECTIVE_MARGIN_THRESHOLD)


def _gen(c):
    return c in G


def _spec_order(order):
    return [c for c in order if c not in G]


def cond_margin(rec):
    """Specific-cause confidence margin of the conditioned distribution."""
    spec = _spec_order(rec["cond_order"])
    if not spec:
        return 0.0
    p0 = rec["cond_prob"].get(spec[0], 0.0)
    p1 = rec["cond_prob"].get(spec[1], 0.0) if len(spec) >= 2 else 0.0
    return p0 - p1


# --- gates: True => fall back to the unconditioned prior ----------------------
GATES = {
    "ungated_cond": lambda r: False,           # always conditioned (baseline)
    "always_prior": lambda r: True,            # always prior (baseline)
    "cond_top1_generic": lambda r: bool(r["cond_order"]) and _gen(r["cond_order"][0]),
    "unc_top1_generic": lambda r: bool(r["unc_order"]) and _gen(r["unc_order"][0]),
    "unc_gen_and_cond_spec": (
        lambda r: bool(r["unc_order"]) and _gen(r["unc_order"][0])
        and bool(r["cond_order"]) and not _gen(r["cond_order"][0])),
    "low_margin": lambda r: cond_margin(r) < MARGIN_THR,
    "low_margin_or_cond_generic": (
        lambda r: cond_margin(r) < MARGIN_THR
        or (bool(r["cond_order"]) and _gen(r["cond_order"][0]))),
    # refined: only fall back to a generic-leaning prior when conditioning is ALSO
    # weak (low specific margin) -- i.e. don't let a *confident* specific
    # conditioned prediction be overridden.
    "unc_gen_and_low_margin": (
        lambda r: bool(r["unc_order"]) and _gen(r["unc_order"][0])
        and cond_margin(r) < MARGIN_THR),
    "unc_gen_and_cond_spec_lowmargin": (
        lambda r: bool(r["unc_order"]) and _gen(r["unc_order"][0])
        and bool(r["cond_order"]) and not _gen(r["cond_order"][0])
        and cond_margin(r) < MARGIN_THR),
}


# The gate implemented in zhang_diagnosis.gated_diagnose (see report §16): fall back
# to the prior ONLY when the population prior's top cause is a generic catch-all AND
# the conditioned prediction lacks a confident specific margin.
CHOSEN_GATE = "unc_gen_and_low_margin"

# Oracle (upper bound; NOT usable at inference -- needs the true cause): route
# generic-true incidents to the prior, specific-true to conditioning.
def _oracle(r):
    return r["generic_only"]


def selected(rec, gate):
    """Return (order, prob) of the distribution the gate selects."""
    if gate(rec):
        return rec["unc_order"], rec["unc_prob"]
    return rec["cond_order"], rec["cond_prob"]


def eval_gate(records, gate):
    top1 = mrr = brier = 0.0
    n = len(records)
    fired = 0
    for r in records:
        if gate(r):
            fired += 1
        order, prob = selected(r, gate)
        true = r["true_all"]
        top1 += 1.0 if qc.top1(order, true) else 0.0
        mrr += qc.mrr(order, true)
        conf = qc.top1_conf(order, prob)
        correct = 1.0 if (order and order[0] in true) else 0.0
        brier += (conf - correct) ** 2
    return {"n": n, "fired": fired, "fire_rate": fired / n if n else 0.0,
            "top1": top1 / n if n else 0.0, "mrr": mrr / n if n else 0.0,
            "brier": brier / n if n else 0.0}


def eval_all(records, label):
    gtrue = [r for r in records if r["generic_only"]]
    strue = [r for r in records if not r["generic_only"]]
    out = {"label": label, "n": len(records), "n_generic_true": len(gtrue),
           "n_specific_true": len(strue), "chosen_gate": CHOSEN_GATE, "gates": {}}
    for name, gate in {**GATES, "ORACLE_true_cause": _oracle}.items():
        out["gates"][name] = {
            "overall": eval_gate(records, gate),
            "generic_true": eval_gate(gtrue, gate) if gtrue else None,
            "specific_true": eval_gate(strue, gate) if strue else None,
        }
    return out


def main():
    records = qc.build_records(top_n_incidents=100)
    A = [r for r in records if qc.is_A(r)]
    Aclean = [r for r in records if qc.is_clean_A(r)]

    summary = {"margin_threshold": MARGIN_THR,
               "A_clean": eval_all(Aclean, "A-clean (leakage-free factual)"),
               "A_all": eval_all(A, "A-all (factual)")}
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"results -> {RESULTS_PATH}")

    for blk in (summary["A_clean"], summary["A_all"]):
        print("\n" + "=" * 92)
        print(f"{blk['label']}   n={blk['n']}  "
              f"(generic-true={blk['n_generic_true']}, specific-true={blk['n_specific_true']})")
        print("=" * 92)
        print(f"{'gate':30} {'fire%':>6} {'top1':>7} {'MRR':>7} {'Brier':>7} "
              f"{'gen-t1':>7} {'spec-t1':>8}")
        print("-" * 92)
        for name, g in blk["gates"].items():
            o = g["overall"]
            gt = g["generic_true"]["top1"] if g["generic_true"] else float("nan")
            st = g["specific_true"]["top1"] if g["specific_true"] else float("nan")
            print(f"{name:30} {o['fire_rate']*100:5.1f}% {o['top1']:7.3f} "
                  f"{o['mrr']:7.3f} {o['brier']:7.3f} {gt:7.3f} {st:8.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Calibration of evidence confidences: when a parser says a fact holds with
confidence c, how often is the fact actually in the accident's coded record?

Population: 1982-2006 accidents with a factual narrative AND coded labels
(so ground truth exists). For each accident the narrative is parsed two ways:

    LLM        llm_evidence.llm_parse_evidence -- asserted facts (c = 1.0)
               and implied facts (the LLM's own c in [0.05, 0.95])
    RETRIEVAL  query_to_bn.retrieval_facts     -- soft facts whose c = f_q,
               the fact's similarity-weighted frequency among the 100 most
               similar accidents (a measured number)

Every (confidence, fact-in-coded-record?) pair goes into a reliability
diagram; expected calibration error (ECE) is reported per source.

TWO truth signals are reported, because "fact in the coded record" is an
imperfect target -- the coded record contains investigation findings the
narrative cannot know, and occasionally omits narrative facts:

    STRICT   fact present in the accident's coded record (conservative --
             a genuinely-narrated fact the NTSB never coded counts as wrong)
    LENIENT  coded record OR spoken verbatim in the narrative text (every
             content word of the node's core appears in the narrative)

If a parser stays miscalibrated under BOTH signals, the miscalibration is
real, not an artifact of the conservative target. Both parsers face the same
signals either way, so the comparison is always fair.

Writes docs/figures/confidence_calibration.png and prints the binned table.

Run (needs OPENAI_API_KEY + network):
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/confidence_calibration.py [--n 120] [--workers 6]
"""
from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
_SHARED = REPO_ROOT / "shared" / "code"
_FROZEN_CODE = FROZEN_DIR / "code"
for _p in (_SHARED, _FROZEN_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

import prognosis as pg  # noqa: E402
import query_to_bn as qb  # noqa: E402
from llm_evidence import llm_parse_evidence  # noqa: E402
from bn_upgraded import build_upgraded, INJ_NODE, DMG_NODE  # noqa: E402

BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.999),
        (0.999, 1.001)]
BIN_LABELS = ["0-.2", ".2-.4", ".4-.6", ".6-.8", ".8-<1", "1.0 (hard)"]


def bin_of(c):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= c < hi:
            return i
    return len(BINS) - 1


def ece(pairs):
    """Expected calibration error over (confidence, correct) pairs."""
    if not pairs:
        return None
    tot = len(pairs)
    err = 0.0
    for i in range(len(BINS)):
        sub = [(c, ok) for c, ok in pairs if bin_of(c) == i]
        if not sub:
            continue
        conf = np.mean([c for c, _ in sub])
        acc = np.mean([ok for _, ok in sub])
        err += len(sub) / tot * abs(conf - acc)
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]

    pool = []
    for ev, inc in ds.items():
        narr = str(inc.get("narr_accf") or "").strip()
        truth = qb._incident_bn_labels(inc) & set(names)
        if len(narr) >= 300 and len(truth) >= 3:
            pool.append((ev, inc, narr, truth))
    random.Random(11).shuffle(pool)
    pool = pool[:args.n]
    print(f"calibration population: {len(pool)} in-window accidents "
          f"(narrative >= 300 chars, >= 3 coded labels)")

    llm_pairs, ret_pairs = [], []          # strict truth (coded record)
    llm_pairs_len, ret_pairs_len = [], []  # lenient truth (coded OR verbatim)

    def spoken(node, nq_tokens):
        core = set(qb._node_core(node).split()) - qb._STOPWORDS
        return bool(core) and core <= nq_tokens

    def work(item):
        ev, inc, narr, truth = item
        nq_tokens = set(qb._norm(narr).split())
        out_llm, out_ret = [], []
        try:
            llm = llm_parse_evidence(narr[:2400], names)
            for node, c in llm["confidence"].items():
                strict = node in truth
                out_llm.append((c, strict, strict or spoken(node, nq_tokens)))
        except Exception:
            pass
        try:
            for node, c, _why in qb.retrieval_facts(narr, names, ds):
                strict = node in truth
                out_ret.append((c, strict, strict or spoken(node, nq_tokens)))
        except Exception:
            pass
        return out_llm, out_ret

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for out_llm, out_ret in ex.map(work, pool):
            llm_pairs.extend((c, s) for c, s, _l in out_llm)
            llm_pairs_len.extend((c, l) for c, _s, l in out_llm)
            ret_pairs.extend((c, s) for c, s, _l in out_ret)
            ret_pairs_len.extend((c, l) for c, _s, l in out_ret)

    print(f"\ncollected: {len(llm_pairs)} LLM facts, "
          f"{len(ret_pairs)} retrieval soft facts\n")

    print(f"{'bin':>10} | {'LLM n':>6} {'LLM conf':>9} {'LLM acc':>8} | "
          f"{'RET n':>6} {'RET conf':>9} {'RET acc':>8}")
    rows = []
    for i, lab in enumerate(BIN_LABELS):
        lp = [(c, ok) for c, ok in llm_pairs if bin_of(c) == i]
        rp = [(c, ok) for c, ok in ret_pairs if bin_of(c) == i]
        lc = np.mean([c for c, _ in lp]) if lp else float("nan")
        la = np.mean([ok for _, ok in lp]) if lp else float("nan")
        rc = np.mean([c for c, _ in rp]) if rp else float("nan")
        ra = np.mean([ok for _, ok in rp]) if rp else float("nan")
        rows.append((lab, len(lp), lc, la, len(rp), rc, ra))
        print(f"{lab:>10} | {len(lp):>6} {lc:>9.3f} {la:>8.3f} | "
              f"{len(rp):>6} {rc:>9.3f} {ra:>8.3f}")

    # raw pairs saved so the figure can be re-rendered without new LLM calls
    import json
    (ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs" / "calibration_pairs.json").write_text(json.dumps({
        "n_narratives": len(pool),
        "llm_strict": llm_pairs, "llm_lenient": llm_pairs_len,
        "ret_strict": ret_pairs, "ret_lenient": ret_pairs_len,
    }))

    e_llm, e_ret = ece(llm_pairs), ece(ret_pairs)
    e_llm_len, e_ret_len = ece(llm_pairs_len), ece(ret_pairs_len)
    print(f"\nECE (lower = better calibrated):")
    print(f"  STRICT truth (coded record):        "
          f"LLM {e_llm:.3f}   retrieval {e_ret:.3f}")
    print(f"  LENIENT truth (coded OR verbatim):  "
          f"LLM {e_llm_len:.3f}   retrieval {e_ret_len:.3f}")
    print("(strict is conservative -- a genuinely-narrated fact the NTSB never "
          "coded counts as wrong; lenient also credits facts spoken verbatim "
          "in the narrative. Both parsers face the same signals.)")

    # ---- figure -----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for pairs, color, label, marker, solid in (
            (llm_pairs, "#d62728",
             f"LLM, strict truth (ECE {e_llm:.2f})", "o", True),
            (ret_pairs, "#1f77b4",
             f"retrieval f_q, strict truth (ECE {e_ret:.2f})", "s", True),
            (llm_pairs_len, "#d62728",
             f"LLM, lenient truth (ECE {e_llm_len:.2f})", "o", False),
            (ret_pairs_len, "#1f77b4",
             f"retrieval f_q, lenient truth (ECE {e_ret_len:.2f})", "s", False)):
        xs, ys, ns = [], [], []
        for i in range(len(BINS)):
            sub = [(c, ok) for c, ok in pairs if bin_of(c) == i]
            if not sub:
                continue
            xs.append(np.mean([c for c, _ in sub]))
            ys.append(np.mean([ok for _, ok in sub]))
            ns.append(len(sub))
        sizes = [30 + 170 * n / max(ns) for n in ns]
        alpha = 0.85 if solid else 0.35
        ax.scatter(xs, ys, s=sizes, color=color, label=label, marker=marker,
                   zorder=3, alpha=alpha)
        ax.plot(xs, ys, color=color, lw=1, alpha=alpha * 0.6,
                ls="-" if solid else "--")
    ax.set_xlabel("claimed confidence that the fact holds")
    ax.set_ylabel("fraction of facts actually in the coded record")
    ax.set_title("Evidence-confidence calibration\n"
                 f"({len(pool)} narratives, 1982-2006; strict = coded, "
                 "lenient = coded/verbatim)", fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    out_png = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "docs_FrozenBN" / "figures" / "confidence_calibration.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"\nwrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

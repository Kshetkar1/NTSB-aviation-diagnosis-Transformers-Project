#!/usr/bin/env python3
"""
MULTI-STEP ESCALATION CHAIN DEMO (PROGNOSIS)
============================================
The advisor asked for the *multi-step* prognosis story, not just single forward cells:

    "a small initial issue -> what % escalates into a bigger issue -> and beyond,"
    with a probability AT EACH HOP.

This script delivers that, fully data-backed from the corrected 1982-2006 window
(`data/processed/refined_dataset_1982_2006.json`):

  1. Builds GLOBAL step transition probabilities P(next | current) from consecutive
     Occurrence_No-ordered Occurrence_Description pairs across every incident (same
     counting semantics as main_app._transition_probabilities, computed over the whole
     dataset). Every transition carries its support N.
  2. DATA-DRIVEN ranking: enumerates all length-2 chains and ranks them by their weakest
     step support (min N), so the data -- not a hardcoded guess -- decides what is well
     supported.
  3. Presents 2-3 concrete end-to-end escalation chains. For each hop it prints the event,
     the raw transition probability P(step|prev), the support N, and the running cumulative
     path probability (product of steps).
  4. RAW/honest estimate (with N) for every hop, PLUS a semantic-neighbour smoothed
     estimate for sparse hops (small N), consistent with prognosis.py's honest framing.

Run (framework python + network for the semantic lane):
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 tests/prognosis_chain_demo.py
    ... --no-semantic     # skip the embedding lane (no network needed)
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import prognosis as pg  # noqa: E402

# A transition with fewer than this many observations is "sparse" -> we additionally
# surface a semantic-neighbour smoothed estimate for it.
SPARSE_N = 15

# Generic, low-information buckets we keep OUT of greedy walks (they are real events but
# read as "other", not as an escalation). Curated explicit chains may still pass through
# them only if the data strongly supports it.
GENERIC = {"miscellaneous/other"}


def _fmt_p(p: float) -> str:
    return f"{p:0.3f}"


def _short(label: str, width: int = 52) -> str:
    return label if len(label) <= width else label[: width - 1] + "\u2026"


def print_ranked_chains(tc: dict, hops: int = 2, top: int = 12) -> None:
    print("\n" + "=" * 90)
    print(f"DATA-DRIVEN RANKING: best-supported length-{hops} chains (ranked by weakest "
          f"step N)")
    print("=" * 90)
    ranked = pg.rank_multistep_chains(tc, hops=hops, top=top, min_step_n=1)
    for i, ch in enumerate(ranked, 1):
        arrow = ""
        for s in ch["steps"]:
            arrow += f" --(p={_fmt_p(s['p'])}, N={s['n']})--> [{_short(s['to'], 40)}]"
        head = f"[{_short(ch['events'][0], 40)}]"
        print(f"\n {i:2d}. minN={ch['min_n']:>3}  cumulative={_fmt_p(ch['cumulative'])}")
        print(f"      {head}{arrow}")


def maybe_semantic(steps, main_app, k: int = 50):
    """Attach a semantic smoothed estimate to any sparse hop. Network required; failures
    are reported inline, never fatal."""
    if main_app is None:
        return
    for s in steps:
        if s["n"] >= SPARSE_N:
            continue
        try:
            est = pg.semantic_step_estimate(s["from"], s["to"], k=k, main_app=main_app)
            s["semantic"] = est
        except Exception as exc:  # network / index / key problems -> report, keep going
            s["semantic_error"] = f"{type(exc).__name__}: {exc}"


def print_chain(title: str, ch: dict, main_app=None) -> None:
    print("\n" + "-" * 90)
    print(f"CHAIN: {title}")
    print("-" * 90)
    maybe_semantic(ch["steps"], main_app)

    # One-line readable chain (the advisor's requested format).
    line = f"[{_short(ch['events'][0])}]"
    cum = 1.0
    for s in ch["steps"]:
        cum *= s["p"]
        line += f"\n   --(p={_fmt_p(s['p'])}, N={s['n']})--> [{_short(s['to'])}]"
    print(line)
    print(f"   cumulative path probability = {_fmt_p(ch['cumulative'])}")

    # Per-hop detail table.
    print("\n   per-hop detail:")
    running = 1.0
    for j, s in enumerate(ch["steps"], 1):
        running *= s["p"]
        sparse = " [SPARSE]" if s["n"] < SPARSE_N else ""
        print(f"     hop {j}: P(next|prev) = {_fmt_p(s['p'])}  "
              f"(N={s['n']} of {s['denom']} transitions out, "
              f"{s['n_incidents']} distinct incidents){sparse}")
        print(f"            from: {s['from']}")
        print(f"            to  : {s['to']}")
        print(f"            cumulative so far = {_fmt_p(running)}")
        if "semantic" in s:
            est = s["semantic"]
            val = "n/a" if est.get("value") is None else _fmt_p(est["value"])
            print(f"            semantic-smoothed P(reach '{_short(s['to'], 30)}' | "
                  f"~'{_short(s['from'], 30)}') = {val} "
                  f"(k={est.get('k')}, hits={est.get('n_outcome')})")
        elif "semantic_error" in s:
            print(f"            semantic-smoothed: unavailable ({s['semantic_error']})")


def sanity_checks(chains) -> None:
    print("\n" + "=" * 90)
    print("SANITY CHECKS")
    print("=" * 90)
    ok = True
    for ch in chains:
        prod = 1.0
        for s in ch["steps"]:
            prod *= s["p"]
            if not (0.0 <= s["p"] <= 1.0):
                print(f"  FAIL: step prob out of [0,1]: {s['p']}")
                ok = False
            if s["n"] < 0 or s["n"] > s["denom"]:
                print(f"  FAIL: support N={s['n']} not in [0, denom={s['denom']}]")
                ok = False
        if abs(prod - ch["cumulative"]) > 1e-9:
            print(f"  FAIL: cumulative {ch['cumulative']} != product {prod}")
            ok = False
    print("  All probabilities in [0,1]; cumulative == product of steps; N within "
          f"[0, denom]: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(2)


def main() -> int:
    use_semantic = "--no-semantic" not in sys.argv

    print("Loading corrected window dataset ...", flush=True)
    ds = pg.load_dataset()
    tc = pg.build_transition_counts(ds)
    print(f"  incidents total           : {len(ds)}")
    print(f"  incidents with >=2 events : {tc['n_incidents_used']} "
          f"(the rest are single-event timelines -> no transition, legacy truncation)")
    print(f"  distinct transition edges : {len(tc['pair_counts'])}")

    # 1) Let the data show which roots have the richest multi-step support.
    print_ranked_chains(tc, hops=2, top=12)

    # 2) Optional semantic lane (needs network + embedding index).
    main_app = None
    if use_semantic:
        try:
            import main_app as _m
            if getattr(_m, "DATA_LOADED", False):
                main_app = _m
                print("\n[semantic lane ENABLED -- embedding index loaded]")
            else:
                print("\n[semantic lane SKIPPED -- knowledge base not loaded]")
        except Exception as exc:
            print(f"\n[semantic lane SKIPPED -- import failed: {exc}]")
    else:
        print("\n[semantic lane DISABLED via --no-semantic]")

    # 3) Concrete escalation chains -- selected FROM the ranking above (well-supported,
    #    and readable as "small issue -> bigger issue -> beyond"). These are explicit
    #    paths scored against the data, not hardcoded probabilities.
    curated = [
        ("System malfunction -> loss of control -> crash", [
            "airframe/component/system failure/malfunction",
            "loss of control - in flight",
            "in flight collision with terrain/water",
        ]),
        ("Engine power loss -> forced landing -> crash", [
            "loss of engine power (total) - nonmechanical",
            "forced landing",
            "in flight collision with terrain/water",
        ]),
        ("System malfunction -> loss of control on ground -> collision", [
            "airframe/component/system failure/malfunction",
            "loss of control - on ground/water",
            "on ground/water collision with terrain/water",
        ]),
    ]

    chains = []
    for title, events in curated:
        ch = pg.chain_from_events(events, tc)
        print_chain(title, ch, main_app=main_app)
        chains.append(ch)

    # 4) Also show one fully DATA-CHOSEN greedy chain from a clean engine seed (no
    #    hardcoded path -- greedy follows the best-supported hop, skipping generic buckets).
    print("\n" + "#" * 90)
    print("BONUS: greedy data-chosen chain (highest-N hop each step, generic buckets "
          "skipped)")
    print("#" * 90)
    greedy = pg.multistep_chain(
        "loss of engine power (total) - mechanical failure/malfunction",
        tc, hops=3, avoid=GENERIC,
    )
    print_chain("greedy from 'loss of engine power (total) - mechanical failure/malfunction'",
                greedy, main_app=main_app)
    chains.append(greedy)

    sanity_checks(chains)
    print("\nDONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

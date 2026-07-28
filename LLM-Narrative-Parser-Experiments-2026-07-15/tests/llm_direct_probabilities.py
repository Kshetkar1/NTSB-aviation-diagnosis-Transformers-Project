#!/usr/bin/env python3
"""Can the LLM REPLACE the Bayesian network? -- the direct test.

For every BN-dependent published result (Table 9 grid, Section 5.2 cumulative
evidence, Fig 12 stages), ask gpt-4o-mini to produce the PROBABILITIES ITSELF:
no network, no dataset, just the same conditioning statement Zhang's queries
use, on the same population (NTSB Part-121 accidents, 1982-2006).

Compare three ways for every cell:
    ZHANG     the paper's number
    OUR BN    upgraded network, exact inference (already validated)
    LLM       the model's own probability estimate

Scores each method against Zhang by order-of-magnitude error
|log10(x / zhang)| -- the right metric when values span 1e-3 .. 1.

Run (needs OPENAI_API_KEY + network):
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/llm_direct_probabilities.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import pyagrum as gum  # noqa: E402

import prognosis as pg  # noqa: E402
import query_to_bn as qb  # noqa: E402
from bn_upgraded import build_upgraded, INJ_NODE, DMG_NODE  # noqa: E402

MODEL = "gpt-4o-mini"
OUT_MD = ROOT / "docs" / "LLM_DIRECT_VS_BN.md"

_SYSTEM = """\
You are an expert on U.S. NTSB Part-121 (scheduled air carrier) accident
statistics for 1982-2006. You will be given conditioning evidence about an
accident and a list of target outcomes. Estimate the conditional probability
of each target given the evidence, over that accident population.

Return JSON only: {"probabilities": {"<target>": <float>, ...}}
Give your best numeric estimate for every target (never refuse, never null).
"""

LOEP = "loss of engine power"

# (scenario name, evidence description for the LLM, evidence nodes for the BN)
SCENARIOS = [
    ("Table 9 col: inoperative engine instruments",
     "The accident involves inoperative/malfunctioning engine instruments.",
     {"engine instrument": 1.0}),
    ("Table 9 col: combustion liner failure",
     "The accident involves a failure of the combustion liner in the "
     "combustion assembly.",
     {"combustion assembly, combustion liner": 1.0}),
    ("Table 9 col: improper oil usage",
     "The accident involves use of an improper oil grade.",
     {"fluid, oil grade": 1.0}),
    ("Table 9 col: engine instruments + improper oil",
     "The accident involves BOTH inoperative engine instruments AND use of "
     "an improper oil grade.",
     {"engine instrument": 1.0, "fluid, oil grade": 1.0}),
    ("Table 9 col: loss of engine power",
     "The accident involves a loss of engine power.",
     {LOEP: 1.0}),
]

# targets: (LLM label, BN (node,state), Zhang values per scenario column)
TARGETS = [
    ("loss of engine power", (LOEP, "Yes"),
     [0.95, 0.50, 0.95, 0.99, 1.0]),
    ("forced landing", ("forced landing", "Yes"),
     [0.1357, 0.0714, 0.1357, 0.1471, 0.1429]),
    ("ditching", ("ditching", "Yes"),
     [4.37e-3, 2.30e-3, 4.37e-3, 4.57e-3, 4.61e-3]),
    ("gear collapsed", ("gear collapsed", "Yes"),
     [9.60e-3, 2.30e-3, 4.37e-3, 9.82e-3, 5.18e-3]),
    ("aircraft destroyed", (DMG_NODE, "destroyed aircraft"),
     [1.33e-2, 2.30e-3, 4.37e-3, 1.35e-2, 5.59e-3]),
    ("substantial aircraft damage", (DMG_NODE, "substantial damage"),
     [4.60e-2, 3.63e-3, 6.09e-3, 4.63e-2, 1.66e-2]),
    ("serious injury (highest)", (INJ_NODE, "serious injury"),
     [6.23e-2, 7.68e-4, 1.46e-3, 6.23e-2, 8.22e-3]),
    ("no injury", (INJ_NODE, "no injury"),
     [0.9431, 0.9978, 0.9958, 0.9429, 0.9899]),
]

SEC52 = [
    ("failure of the main gear strut",
     {"landing gear, main gear strut": 1.0}, 0.25),
    ("failure of the main gear strut AND the emergency extension assembly",
     {"landing gear, main gear strut": 1.0,
      "landing gear, emergency extension assembly": 1.0}, 0.682),
    ("failure of the main gear strut, the emergency extension assembly AND "
     "the gear locking mechanism",
     {"landing gear, main gear strut": 1.0,
      "landing gear, emergency extension assembly": 1.0,
      "landing gear, gear locking mechanism": 1.0}, 0.777),
    ("failure of the main gear strut, the emergency extension assembly, the "
     "gear locking mechanism AND the main gear attachment",
     {"landing gear, main gear strut": 1.0,
      "landing gear, emergency extension assembly": 1.0,
      "landing gear, gear locking mechanism": 1.0,
      "landing gear, main gear attachment": 1.0}, 0.894),
]

lines: list[str] = []


def emit(s: str = "") -> None:
    print(s)
    lines.append(s)


def g4(x):
    return "--" if x is None else f"{x:.4g}"


def ask_llm(client, evidence_text: str, target_labels):
    user = (f"EVIDENCE: {evidence_text}\n\n"
            "TARGETS (estimate P(target | evidence) for each):\n"
            + "\n".join(f"- {t}" for t in target_labels)
            + "\n\nReturn the JSON.")
    resp = client.chat.completions.create(
        model=MODEL, temperature=0, seed=7,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": user}],
    )
    try:
        probs = json.loads(resp.choices[0].message.content)["probabilities"]
    except (json.JSONDecodeError, KeyError):
        probs = {}
    out = {}
    for t in target_labels:
        try:
            out[t] = float(probs.get(t))
        except (TypeError, ValueError):
            out[t] = None
    return out


def bn_posts(bn, evidence_conf, targets):
    ie = gum.LazyPropagation(bn)
    qb.apply_evidence(ie, bn, evidence_conf)
    for n, _ in targets:
        ie.addTarget(n)
    ie.makeInference()
    out = []
    for n, state in targets:
        v = bn.variable(n)
        i = [k for k in range(v.domainSize()) if v.label(k) == state][0]
        out.append(float(ie.posterior(n)[i]))
    return out


def oom_err(x, z):
    """order-of-magnitude error |log10(x/z)|; None if not computable."""
    if x is None or x <= 0 or z <= 0:
        return None
    return abs(math.log10(x / z))


def main():
    import main_app
    client = main_app.get_client()
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)

    emit("# Can the LLM replace the Bayesian network? Direct probability test")
    emit()
    emit(f"{MODEL} is asked for the SAME conditional probabilities Zhang's "
         "BN-dependent tables publish -- no network, no dataset, just its own "
         "knowledge. Error metric: order-of-magnitude error "
         "|log10(estimate / Zhang)| (0 = exact, 1 = off by 10x).")
    emit()

    errs_bn, errs_llm = [], []

    # ------------------ Table 9 grid --------------------------------------------
    tgt_nodes = [t[1] for t in TARGETS]
    tgt_labels = [t[0] for t in TARGETS]
    for ci, (name, ev_text, ev_bn) in enumerate(SCENARIOS):
        llm = ask_llm(client, ev_text, tgt_labels)
        bnv = bn_posts(bn, ev_bn, tgt_nodes)
        emit(f"## {name}")
        emit()
        emit("| Target | Zhang | Our BN | LLM direct | BN oom-err | LLM oom-err |")
        emit("|---|---|---|---|---|---|")
        for (label, _, zvals), b in zip(TARGETS, bnv):
            z = zvals[ci]
            l = llm.get(label)
            eb, el = oom_err(b, z), oom_err(l, z)
            if eb is not None:
                errs_bn.append(eb)
            if el is not None:
                errs_llm.append(el)
            emit(f"| {label} | {g4(z)} | {g4(b)} | {g4(l)} | "
                 f"{g4(eb)} | {g4(el)} |")
        emit()

    # ------------------ Section 5.2 ----------------------------------------------
    emit("## Section 5.2 -- cumulative gear evidence, P(main gear collapsed)")
    emit()
    emit("| Evidence | Zhang | Our BN | LLM direct | BN oom-err | LLM oom-err |")
    emit("|---|---|---|---|---|---|")
    llm_seq = []
    for ev_text, ev_bn, z in SEC52:
        llm = ask_llm(client, f"The accident involves a {ev_text}.",
                      ["main gear collapsed"])
        l = llm.get("main gear collapsed")
        llm_seq.append(l)
        b = bn_posts(bn, ev_bn, [("main gear collapsed", "Yes")])[0]
        eb, el = oom_err(b, z), oom_err(l, z)
        if eb is not None:
            errs_bn.append(eb)
        if el is not None:
            errs_llm.append(el)
        emit(f"| {ev_text[:52]} | {g4(z)} | {g4(b)} | {g4(l)} | "
             f"{g4(eb)} | {g4(el)} |")
    mono = all(a is not None and b is not None and b >= a
               for a, b in zip(llm_seq, llm_seq[1:]))
    emit()
    emit(f"- LLM sequence monotonically increasing (as evidence accumulates)? "
         f"**{'YES' if mono else 'NO'}** (Zhang and the BN both increase)")
    emit()

    # ------------------ summary ---------------------------------------------------
    import statistics as st
    emit("## Score")
    emit()
    emit("| Method | median oom-err | mean oom-err | worst | n cells |")
    emit("|---|---|---|---|---|")
    for tag, errs in (("Our BN", errs_bn), ("LLM direct", errs_llm)):
        emit(f"| {tag} | {st.median(errs):.3f} | {st.mean(errs):.3f} | "
             f"{max(errs):.3f} | {len(errs)} |")
    emit()
    emit("oom-err 0.30 = off by 2x, 1.00 = off by 10x, 2.00 = off by 100x.")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""One-stop diagnosis demo for the advisor meeting.

Shows, in order:
  1. Zhang reproduction   -- P(cause | fire) global == Table 7 (102 accidents)
  2. Generalization        -- any outcome, detected from data (no hard-coding)
  3. Query-specific        -- same outcome, different wording -> different causes
  4. Conditional diagnosis -- P(cause | outcome AND conditions)
  5. Self-consistency      -- retrieval converges to the global table

Sections 3 and 5 use the embedding API (retrieval). Run:
  python tests/diagnose_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import zhang_diagnosis as zd  # noqa: E402


def _print(r, k=6):
    if r.get("error"):
        print("   ERROR:", r["error"])
        return
    print(f"   outcome={r['outcome']!r}  accidents={r['outcome_count']}", end="")
    if r.get("matched_conditions"):
        print(f"  conditions={r['matched_conditions']}", end="")
    print()
    for c in r["causes"][:k]:
        print(f"      {c['probability']*100:5.1f}%  ({c['n']:>2})  {c['cause']}")


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main(skip_network: bool = False) -> None:
    section("1. ZHANG REPRODUCTION  --  P(cause | fire), global == Table 7")
    _print(zd.diagnose("what is the probability of fire?", mode="global"))

    section("2. GENERALIZATION  --  any outcome, detected from data")
    for q in ["why did the aircraft lose engine power?",
              "what caused the landing gear to collapse?",
              "causes of a forced landing"]:
        print(f"\n[{q}]")
        _print(zd.diagnose(q, mode="global"), k=4)

    section("4. CONDITIONAL DIAGNOSIS  --  P(cause | outcome AND conditions)")
    print("\n[baseline: P(cause | fire)]")
    _print(zd.diagnose("probability of fire", mode="global"), k=4)
    print("\n[P(cause | fire AND electrical wiring)]")
    _print(zd.diagnose_conditional("what caused the fire?",
                                   conditions=["electrical wiring"], mode="global"), k=4)
    print("\n[parsed: 'given an electrical wiring problem, what caused the fire?']")
    _print(zd.diagnose_conditional(
        "given an electrical wiring problem, what caused the fire?", mode="global"), k=4)

    if skip_network:
        print("\n(skipping retrieval sections 3 & 5 -- no network)")
        return

    section("3. QUERY-SPECIFIC  --  same outcome (fire), different wording")
    for q in ["fire caused by an electrical short or wiring problem",
              "fire that broke out following a brake or landing gear failure"]:
        print(f"\n[{q}]")
        _print(zd.diagnose(q, mode="retrieval", top_n_incidents=120), k=4)

    section("5. SELF-CONSISTENCY  --  retrieval -> global Table 7 as the net widens")
    g = {c["cause"]: c["probability"]
         for c in zd.diagnose("probability of fire", top_n=999, mode="global")["causes"]}
    for tn in (120, 400, 2000):
        r = zd.diagnose("what is the probability of fire?", top_n=999,
                        mode="retrieval", top_n_incidents=tn)
        rd = {c["cause"]: c["probability"] for c in r["causes"]}
        maxd = max((abs(g[c] - rd.get(c, 0)) for c in g), default=0)
        print(f"   top_n_incidents={tn:>4}  fire-subset={r['outcome_count']:>3}  "
              f"max|delta p| vs Table 7 = {maxd:.3f}")


if __name__ == "__main__":
    main(skip_network="--offline" in sys.argv)

#!/usr/bin/env python3
"""Step 9 — Recreate Zhang's Table 9 / Fig 12 / Table 5 with Lane 1 counting.

For each Zhang scenario, count the empirical P(outcome | evidence) directly from
the NTSB cases (with n + Wilson CI) and put it side by side with Zhang's value.

The honest headline: counting does NOT reproduce Zhang's absolute probabilities,
and the dominant reason is the DENOMINATOR — Zhang normalizes over ~184M flights,
this dataset is 2,243 accidents only. Prior-dominated outcomes (esp. "no injury")
differ by construction. Emits outputs/step09_recreate_zhang.{json,md}.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpt_config import OUTPUT_DIR, REFINED_DATA_PATH, ZHANG_RECREATION_TARGETS_PATH  # noqa: E402
from stats import empirical_p, wilson_ci  # noqa: E402

MIN_N = 30  # below this, magnitude is "sparse — directional only"


def _eventsoe(inc: dict) -> set[int]:
    out = set()
    for e in inc.get("sequence_of_events") or []:
        if isinstance(e, dict) and e.get("eventsoe_no") is not None:
            try:
                out.add(int(e["eventsoe_no"]))
            except (TypeError, ValueError):
                pass
    return out


def _outcome_fn(field: str):
    if field == "damage_severe":
        return lambda i: str(i.get("damage") or "").upper() in ("SUBS", "DEST")
    if field == "no_injury":
        return lambda i: str(i.get("ev_highest_injury") or "").upper() == "NONE"
    if field == "serious_fatal":
        return lambda i: str(i.get("ev_highest_injury") or "").upper() in ("SERS", "FATL")
    raise ValueError(f"unknown outcome field {field}")


def _cohort(incs: list[dict], sc: dict) -> list[dict]:
    if sc["evidence_kind"] == "occurrence":
        codes = set(sc["evidence_codes"])
        return [i for i in incs if _eventsoe(i) & codes]
    if sc["evidence_kind"] == "finding_regex":
        pat = re.compile(sc["evidence_regex"], re.I)
        return [
            i for i in incs
            if any(isinstance(f, dict) and pat.search(str(f.get("finding_description") or ""))
                   for f in (i.get("findings") or []))
        ]
    raise ValueError(sc["evidence_kind"])


def main() -> None:
    data = json.loads(REFINED_DATA_PATH.read_text(encoding="utf-8"))
    incs = list(data.values())
    spec = json.loads(ZHANG_RECREATION_TARGETS_PATH.read_text(encoding="utf-8"))

    results = []
    for sc in spec["scenarios"]:
        coh = _cohort(incs, sc)
        n = len(coh)
        rows = []
        for t in sc["targets"]:
            fn = _outcome_fn(t["field"])
            k = sum(1 for i in coh if fn(i))
            p = empirical_p(k, n)
            lo, hi = wilson_ci(k, n)
            zh = t["zhang"]
            in_band = (n >= MIN_N and p == p and abs(p - zh) <= 0.20)
            rows.append({
                "outcome": t["outcome"], "k": k, "n": n,
                "p": p, "ci": [lo, hi], "zhang": zh,
                "gap": abs(p - zh) if p == p else None,
                "verdict": "within 0.20" if in_band else ("sparse" if n < MIN_N else "gap"),
            })
        results.append({"scenario": sc["name"], "n_cohort": n, "rows": rows})

    out = {
        "denominator_note": spec["denominator_note"],
        "provenance": spec["provenance"],
        "min_n": MIN_N,
        "scenarios": results,
    }
    (OUTPUT_DIR / "step09_recreate_zhang.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# Step 9 — Recreating Zhang's tables with Lane 1 counting",
        "",
        "**Headline:** counting does **not** reproduce Zhang's absolute probabilities.",
        f"**Dominant reason:** {spec['denominator_note']}",
        "",
    ]
    for s in results:
        lines += [f"## {s['scenario']}  (cohort n={s['n_cohort']})", "",
                  "| Outcome | counted P | 95% CI | n | Zhang | gap | verdict |",
                  "|---------|-----------|--------|---|-------|-----|---------|"]
        for r in s["rows"]:
            ci = f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]" if r["p"] == r["p"] else "n/a"
            ps = f"{r['p']:.3f}" if r["p"] == r["p"] else "n/a"
            gp = f"{r['gap']:.3f}" if r["gap"] is not None else "n/a"
            lines.append(f"| {r['outcome']} | {ps} | {ci} | {r['n']} | {r['zhang']:.4f} | {gp} | {r['verdict']} |")
        lines.append("")
    lines += [
        "## What this means",
        "",
        "- **Not a bug:** the gap is mostly the 184M-flight vs 2,243-accident denominator, plus coding era + Beta smoothing.",
        "- **What IS comparable:** ordering, relative risk vs base rate, and causal-tight conditionals — not absolute priors.",
        "- **To match absolute probabilities** you would need the flight-level denominator Zhang used (not in this accident-only data).",
    ]
    (OUTPUT_DIR / "step09_recreate_zhang.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Step 9 → outputs/step09_recreate_zhang.md")
    for s in results:
        for r in s["rows"]:
            ps = f"{r['p']:.3f}" if r["p"] == r["p"] else "n/a"
            print(f"  {s['scenario'][:28]:28} {r['outcome']:20} mine={ps} zhang={r['zhang']:.3f} n={r['n']} [{r['verdict']}]")


if __name__ == "__main__":
    main()

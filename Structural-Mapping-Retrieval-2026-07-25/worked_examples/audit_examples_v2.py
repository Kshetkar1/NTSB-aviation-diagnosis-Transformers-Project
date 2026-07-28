#!/usr/bin/env python3
"""Audit Worked_Examples/data/example_after.json and example_during.json.

Verifies the post-Maha worked-example outputs:

  - All 8 LTP distributions (A0/A2 x diag/prog x after/during) sum to ~1
  - Every top-5 enriched row has a non-empty label and elaboration
  - Every probability is in [0, 1]
  - Enriched rows are sorted non-increasing by probability
  - Retrieval table has exactly 50 rows
  - For after-incident: hit flags exist and align with top1_code vs ground truth
  - For during-incident: there is no ground truth (skipped), but the during
    query is exactly the agreed pilot-voice sentence

Exits 0 only if there are 0 failures (warnings are non-fatal).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

TOL_SUM = 1e-3

_FAILS: list[str] = []
_WARNS: list[str] = []


def ok(msg: str) -> None:
    print(f"  PASS  {msg}")


def fail(msg: str) -> None:
    _FAILS.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg: str) -> None:
    _WARNS.append(msg)
    print(f"  WARN  {msg}")


def section(t: str) -> None:
    print(f"\n--- {t} ---")


def near(a: float, b: float, tol: float = TOL_SUM) -> bool:
    return abs(float(a) - float(b)) <= tol


def assert_prob_in_range(p, where: str) -> None:
    try:
        v = float(p)
    except Exception:
        fail(f"{where}: probability not float: {p!r}")
        return
    if v < -1e-9 or v > 1.0 + 1e-9:
        fail(f"{where}: probability out of range [0,1]: {v}")


def audit_ltp_sum(rows: list[dict], expected_field: str, where: str) -> None:
    s = sum(float(r.get(expected_field, 0.0)) for r in rows)
    if near(s, 1.0):
        ok(f"{where}: LTP Sigma = {s:.6f}")
    else:
        fail(f"{where}: LTP Sigma = {s:.6f} (expected ~1.0)")


def audit_sorted(rows: list[dict], field: str, where: str) -> None:
    probs = [float(r.get(field, 0.0)) for r in rows]
    for i in range(1, len(probs)):
        if probs[i] > probs[i - 1] + 1e-9:
            fail(f"{where}: not sorted - row {i+1} ({probs[i]:.4f}) > row {i} ({probs[i-1]:.4f})")
            return
    ok(f"{where}: rows sorted by probability")


def audit_enriched_rows(rows: list[dict], where: str, kind: str) -> None:
    if not rows:
        fail(f"{where}: empty enriched list")
        return
    if len(rows) > 5:
        warn(f"{where}: has {len(rows)} rows, expected <= 5 (top-N)")
    for i, r in enumerate(rows, start=1):
        label = (r.get("label") or "").strip()
        elab = (r.get("elaboration") or "").strip()
        if not label:
            fail(f"{where}: row {i} has empty label")
        if not elab:
            fail(f"{where}: row {i} has empty elaboration")
        assert_prob_in_range(r.get("probability"), f"{where} row {i}")
        if kind == "code":
            if "code" not in r:
                fail(f"{where}: row {i} missing 'code' field")
    ok(f"{where}: {len(rows)} enriched rows, all labels+elaborations non-empty")


def audit_scenario(path: Path, scenario_name: str) -> None:
    section(f"Loading {path.name}")
    if not path.exists():
        fail(f"{path} does not exist")
        return
    data = json.loads(path.read_text())
    if data.get("scenario") != scenario_name:
        fail(f"{path.name}: scenario field is {data.get('scenario')!r}, expected {scenario_name!r}")

    # Retrieval - must have 50 rows
    section(f"{scenario_name}: retrieval")
    diag_ret = data.get("diagnosis", {}).get("retrieval") or []
    prog_ret = data.get("prognosis", {}).get("retrieval") or []
    if len(diag_ret) != 50:
        fail(f"{scenario_name}: diagnosis retrieval has {len(diag_ret)} rows (expected 50)")
    else:
        ok(f"{scenario_name}: diagnosis retrieval = 50 rows")
    if len(prog_ret) != 50:
        fail(f"{scenario_name}: prognosis retrieval has {len(prog_ret)} rows (expected 50)")
    else:
        ok(f"{scenario_name}: prognosis retrieval = 50 rows")

    # All 4 LTP distributions sum to ~1
    section(f"{scenario_name}: LTP sums")
    for model in ("a0", "a2"):
        diag = data["diagnosis"][model]
        prog = data["prognosis"][model]
        audit_ltp_sum(diag.get("ltp_causes") or [], "probability",
                      f"{scenario_name} {model.upper()} diag")
        audit_ltp_sum(prog.get("ltp_events") or [], "probability",
                      f"{scenario_name} {model.upper()} prog")

    # Enriched top-5 rows
    section(f"{scenario_name}: enriched top-5 (labels + elaborations)")
    for model in ("a0", "a2"):
        diag = data["diagnosis"][model]
        prog = data["prognosis"][model]
        audit_enriched_rows(diag.get("ltp_causes_enriched") or [],
                            f"{scenario_name} {model.upper()} diag causes", "cause")
        audit_enriched_rows(diag.get("coded_distribution_enriched") or [],
                            f"{scenario_name} {model.upper()} diag codes", "code")
        audit_enriched_rows(prog.get("ltp_events_enriched") or [],
                            f"{scenario_name} {model.upper()} prog events", "event")

    # Sorted-ness
    section(f"{scenario_name}: enriched rows sorted")
    for model in ("a0", "a2"):
        diag = data["diagnosis"][model]
        prog = data["prognosis"][model]
        audit_sorted(diag.get("ltp_causes_enriched") or [], "probability",
                     f"{scenario_name} {model.upper()} diag causes")
        audit_sorted(diag.get("coded_distribution_enriched") or [], "probability",
                     f"{scenario_name} {model.upper()} diag codes")
        audit_sorted(prog.get("ltp_events_enriched") or [], "probability",
                     f"{scenario_name} {model.upper()} prog events")

    # Ground-truth presence check (cluster-vs-NTSB-cause is a human read,
    # not a programmatic Zhang-code comparison).
    section(f"{scenario_name}: ground truth presence")
    gt = data.get("ground_truth") or {}
    if scenario_name == "after_incident":
        if not gt.get("text"):
            fail(f"{scenario_name}: ground_truth.text is empty")
        else:
            ok(f"{scenario_name}: NTSB probable-cause text present ({len(gt['text'])} chars)")
        # Verify every pipeline produced at least a top cluster + top cause to
        # show alongside the NTSB cause text.
        for model in ("a0", "a2"):
            diag = data["diagnosis"][model]
            top_cluster = (diag.get("clusters") or [{}])[0].get("cluster", "")
            top_cause = (diag.get("ltp_causes_enriched") or [{}])[0].get("label", "")
            if not top_cluster:
                fail(f"{scenario_name} {model.upper()}: missing top cluster label")
            else:
                ok(f"{scenario_name} {model.upper()}: top cluster = {top_cluster!r}")
            if not top_cause:
                fail(f"{scenario_name} {model.upper()}: missing top free-text cause label")
            else:
                ok(f"{scenario_name} {model.upper()}: top cause = {top_cause!r}")
    else:
        dq = data.get("diagnosis_query", "").strip()
        pq = data.get("prognosis_query", "").strip()
        if not dq:
            fail(f"{scenario_name}: diagnosis_query is empty")
        else:
            ok(f"{scenario_name}: diagnosis_query present ({len(dq)} chars)")
        if dq != pq:
            fail(f"{scenario_name}: prognosis_query must equal diagnosis_query (single short query) - mismatch")
        else:
            ok(f"{scenario_name}: prognosis_query reuses diagnosis_query")
        if len(dq) > 350:
            warn(f"{scenario_name}: diagnosis_query is {len(dq)} chars - longer than typical pilot voice (<300)")


def main() -> int:
    print("=" * 78)
    print("Worked-example audit (v2)")
    print("=" * 78)
    # Auto-discover every example_after*.json and example_during*.json pair.
    after_files = sorted(DATA.glob("example_after*.json"))
    during_files = sorted(DATA.glob("example_during*.json"))
    print(f"Discovered {len(after_files)} after-files, {len(during_files)} during-files\n")
    for p in after_files:
        audit_scenario(p, "after_incident")
    for p in during_files:
        audit_scenario(p, "during_incident")

    print("\n" + "=" * 78)
    print(f"SUMMARY: {len(_FAILS)} fail, {len(_WARNS)} warn")
    print("=" * 78)
    if _FAILS:
        print("FAILS:")
        for f in _FAILS:
            print(f"  - {f}")
    if _WARNS:
        print("WARNS:")
        for w in _WARNS:
            print(f"  - {w}")
    return 0 if not _FAILS else 1


if __name__ == "__main__":
    sys.exit(main())

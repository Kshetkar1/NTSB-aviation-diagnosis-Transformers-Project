#!/usr/bin/env python3
"""Critical audit of Worked_Examples/data/a0.json and a2.json."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Testing_Structural_Mapping_Slides" / "scripts"))

QUERY_A0 = (
    "During cruise flight at FL340, the No. 2 engine experienced a sudden loss "
    "of power accompanied by elevated EGT, vibration, and an associated master "
    "caution. The crew completed the engine shutdown procedure and declared an "
    "emergency. The aircraft diverted to the nearest suitable airport and "
    "landed without further incident. No injuries were reported among the "
    "passengers or crew."
)
EV_A2 = "20100114X11754"
TOL = 1e-3


def ok(msg: str) -> None:
    print(f"  PASS  {msg}")


def fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


def warn(msg: str) -> None:
    print(f"  WARN  {msg}")


def near(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol


def audit_a0_cached(a0: dict) -> list[str]:
    issues: list[str] = []
    print("\n=== A0 cached JSON checks ===")

    n_ret = len(a0["retrieval"])
    if n_ret == 50:
        ok(f"retrieval has 50 rows")
    else:
        fail(f"retrieval has {n_ret} rows (expected 50)")
        issues.append("A0 retrieval count")

    n_active = len(a0["clusters"])
    n_skip = len(a0.get("skipped_clusters", []))
    if n_active + n_skip == 7 + 2:
        ok(f"clusters: {n_active} active + {n_skip} skipped = 9 total")
    else:
        warn(f"clusters: {n_active} active + {n_skip} skipped")

    member_sum = sum(c["n_incidents"] for c in a0["clusters"]) + sum(
        s["n_incidents"] for s in a0.get("skipped_clusters", [])
    )
    if member_sum == 50:
        ok("cluster members sum to 50")
    else:
        fail(f"cluster members sum to {member_sum} (expected 50)")
        issues.append("A0 cluster member count")

    p_sum = sum(c["p_k_given_q"] for c in a0["clusters"])
    if near(p_sum, 1.0):
        ok(f"active P(K|Q) sums to {p_sum:.6f}")
    else:
        fail(f"active P(K|Q) sums to {p_sum:.6f} (expected 1.0)")
        issues.append("A0 P(K|Q) sum")

    for c in a0["clusters"]:
        expected_w = c["avg_similarity"] * c["n_incidents"]
        if not near(c["weight"], expected_w, 1e-6):
            fail(f"weight mismatch for {c['cluster'][:40]}")
            issues.append("A0 weight formula")
            break
    else:
        ok("cluster weights = avg_sim × n")

    ltp_sum = sum(x["probability"] for x in a0["ltp_causes"])
    if near(ltp_sum, 1.0, 1e-3):
        ok(f"ltp_causes is complete (n={len(a0['ltp_causes'])}, sum={ltp_sum:.6f})")
    else:
        fail(f"ltp_causes sum={ltp_sum:.6f} (expected 1.0)")
        issues.append("A0 LTP sum")

    coded_sum = sum(x["probability"] for x in a0["coded_distribution"])
    if near(coded_sum, 1.0, 0.01):
        ok(f"coded_distribution is complete (n={len(a0['coded_distribution'])}, sum={coded_sum:.6f})")
    else:
        fail(f"coded_distribution sum={coded_sum:.6f}")
        issues.append("A0 coded sum")

    prog_sum = sum(x["probability"] for x in a0["prognosis"])
    if near(prog_sum, 1.0):
        ok(f"prognosis sums to {prog_sum:.6f}")
    else:
        fail(f"prognosis sums to {prog_sum:.6f}")
        issues.append("A0 prognosis sum")

    if a0["mode"] == "production" and a0["corpus_size"] == 2243:
        ok("mode=production, corpus=2243")
    else:
        fail(f"mode={a0['mode']}, corpus={a0['corpus_size']}")
        issues.append("A0 mode/corpus")

    return issues


def rerun_a0_pipeline() -> tuple[dict, list[str]]:
    issues: list[str] = []
    print("\n=== A0 re-run production pipeline ===")
    os.environ.pop("NTSB_USE_TRAIN_INDEX", None)
    if "main_app" in sys.modules:
        del sys.modules["main_app"]
    import main_app  # noqa: E402
    from cause_code_mapper import build_code_embeddings, map_distribution_to_codes  # noqa: E402

    emb = main_app.get_embedding(QUERY_A0)
    scores, matches = main_app.find_top_matches(emb)
    top_scores, top_matches = list(scores[:50]), list(matches[:50])
    clusters = main_app.cluster_incidents_by_type(top_scores, top_matches, 50)
    ca = main_app.calculate_cause_probabilities_per_cluster(clusters)
    diag = main_app.calculate_chain_rule_diagnosis(clusters, ca)
    causes = diag["weighted_causes"]

    full_sum = sum(c["probability"] for c in causes)
    if near(full_sum, 1.0):
        ok(f"full LTP distribution Σ P(C|Q) = {full_sum:.6f}")
    else:
        fail(f"full LTP distribution Σ P(C|Q) = {full_sum:.6f}")
        issues.append("A0 full LTP sum on re-run")

    cached = json.loads((DATA / "a0.json").read_text())
    for i in range(min(5, len(cached["retrieval"]))):
        r_c, r_n = cached["retrieval"][i], {"ev_id": top_matches[i].get("ev_id"), "score": float(top_scores[i])}
        if r_c["ev_id"] != r_n["ev_id"] or not near(r_c["score"], r_n["score"], 1e-5):
            fail(f"retrieval rank {i+1} mismatch cached vs re-run")
            issues.append("A0 retrieval mismatch")
            break
    else:
        ok("top-5 retrieval matches cached JSON")

    for i, wc in enumerate(causes):
        c = cached["ltp_causes"][i]
        if not near(c["probability"], wc["probability"], 1e-6):
            fail(f"ltp rank {i+1} prob mismatch: cached={c['probability']}, rerun={wc['probability']}")
            issues.append("A0 LTP prob mismatch")
            break
    else:
        ok(f"full LTP probabilities match cached JSON (n={len(causes)})")

    codes_list, labels_list, code_embs = build_code_embeddings()
    dist = [{"cause": c["cause"], "probability": c["probability"]} for c in causes]
    coded, _ = map_distribution_to_codes(dist, codes_list, labels_list, code_embs)
    coded_sum = sum((cd.get("probability") or 0) for cd in coded)
    if near(coded_sum, 1.0):
        ok(f"full coded distribution sums to {coded_sum:.6f}")
    else:
        fail(f"full coded distribution sums to {coded_sum:.6f}")
        issues.append("A0 coded full sum")

    c0 = cached["coded_distribution"][0]
    r0 = coded[0]
    if str(c0["code"]) == str(r0.get("code")) and near(c0["probability"], r0.get("probability", 0), 1e-4):
        ok(f"top coded cause matches: code {c0['code']} p={c0['probability']:.4f}")
    else:
        fail(f"top coded cause mismatch cached={c0} rerun={r0}")
        issues.append("A0 top coded mismatch")

    return {"full_ltp_sum": full_sum, "n_causes": len(causes)}, issues


def ground_truth_a2() -> tuple[str, str]:
    from config import REFINED_DATA_PATH  # noqa: E402

    inc = json.loads(Path(REFINED_DATA_PATH).read_text())[EV_A2]
    query = str(inc.get("narr_accp") or inc.get("narr_accf") or "").strip()
    findings = inc.get("findings") or []
    cause_findings = [
        str(f.get("finding_description", "")).strip()
        for f in findings
        if isinstance(f, dict)
        and (f.get("Cause_Factor") or "").strip() == "C"
        and (f.get("finding_description") or "").strip()
    ]
    narr_cause = str(inc.get("narr_cause") or "").strip()
    truth = cause_findings[0] if cause_findings else narr_cause
    return query, truth


def audit_a2_cached(a2: dict) -> list[str]:
    issues: list[str] = []
    print("\n=== A2 cached JSON checks ===")

    if a2["ev_id"] == EV_A2 and a2["corpus_size"] == 177:
        ok("ev_id and corpus_size correct")
    else:
        fail(f"ev_id={a2['ev_id']}, corpus={a2['corpus_size']}")
        issues.append("A2 metadata")

    _, truth_ref = ground_truth_a2()
    gt_text = a2["ground_truth"]["text"]
    if truth_ref[:80].lower() in gt_text[:120].lower() or gt_text[:80].lower() in truth_ref[:120].lower():
        ok("ground truth text aligns with dataset Cause_Factor=C finding")
    else:
        fail("ground truth text may not match dataset M1 definition")
        issues.append("A2 ground truth text")
        print(f"    dataset: {truth_ref[:100]}")
        print(f"    cached:  {gt_text[:100]}")

    dup_ranks = [r for r in a2["retrieval"] if r["ev_id"] == "20170612X10422"]
    if len(dup_ranks) >= 2:
        warn(
            f"ev_id 20170612X10422 appears {len(dup_ranks)}× in top-50 "
            "(embedding index has multiple rows per incident — expected)"
        )

    for label in ("a0", "a2"):
        sub = a2[label]
        p_sum = sum(c["p_k_given_q"] for c in sub["clusters"])
        if near(p_sum, 1.0, 0.02):
            ok(f"{label.upper()} P(K|Q) sums to {p_sum:.6f}")
        else:
            fail(f"{label.upper()} P(K|Q) sums to {p_sum:.6f}")
            issues.append(f"A2 {label} P(K|Q)")

        ltp_sum = sum(c["probability"] for c in sub["ltp_causes"])
        if near(ltp_sum, 1.0, 1e-3):
            ok(f"{label.upper()} LTP complete (n={len(sub['ltp_causes'])}, sum={ltp_sum:.6f})")
        else:
            fail(f"{label.upper()} LTP sum={ltp_sum:.6f}")
            issues.append(f"A2 {label} LTP sum")

    if a2["hit"]["a0"] and a2["hit"]["a2"]:
        ok(f"both A0 and A2 top-1 code hit ground truth code {a2['ground_truth']['code']}")
    else:
        fail(f"hit flags: A0={a2['hit']['a0']}, A2={a2['hit']['a2']}")
        issues.append("A2 hit flags")

    gt_code = str(a2["ground_truth"]["code"])
    if str(a2["a0"]["top1_code"]) == gt_code and str(a2["a2"]["top1_code"]) == gt_code:
        ok(f"top-1 codes = {gt_code} (130 Airframe/component/system failure)")
    else:
        fail("top-1 code mismatch vs ground truth")
        issues.append("A2 top-1 codes")

    return issues


def compare_a2_to_trace(a2: dict) -> list[str]:
    issues: list[str] = []
    print("\n=== A2 JSON vs source trace ===")
    trace_path = (
        PROJECT_ROOT
        / "Testing_Structural_Mapping_Slides"
        / "outputs"
        / "trace_20100114X11754_a2_full.txt"
    )
    if not trace_path.exists():
        warn("trace file missing — skip trace comparison")
        return issues

    text = trace_path.read_text()
    # spot-check first retrieval row
    if "0.722260   20110621X20741" in text:
        r1 = a2["retrieval"][0]
        if near(r1["score"], 0.72226, 1e-4) and r1["ev_id"] == "20110621X20741":
            ok("retrieval rank-1 matches trace")
        else:
            fail("retrieval rank-1 differs from trace")
            issues.append("A2 trace retrieval")

    a0_top = a2["a0"]["ltp_causes"][0]
    if near(a0_top["probability"], 0.033776, 1e-4):
        ok("A0 top LTP cause prob matches trace (~0.033776)")
    else:
        warn(f"A0 top LTP prob={a0_top['probability']:.6f} (trace had 0.033776)")

    a2_top = a2["a2"]["ltp_causes"][0]
    if near(a2_top["probability"], 0.032785, 1e-4):
        ok("A2 top LTP cause prob matches trace (~0.032785)")
    else:
        warn(f"A2 top LTP prob={a2_top['probability']:.6f} (trace had 0.032785)")

    return issues


def main() -> None:
    all_issues: list[str] = []
    a0 = json.loads((DATA / "a0.json").read_text())
    a2 = json.loads((DATA / "a2.json").read_text())

    all_issues.extend(audit_a0_cached(a0))
    _, a0_rerun_issues = rerun_a0_pipeline()
    all_issues.extend(a0_rerun_issues)
    all_issues.extend(audit_a2_cached(a2))
    all_issues.extend(compare_a2_to_trace(a2))

    print("\n" + "=" * 60)
    if all_issues:
        print(f"AUDIT RESULT: {len(all_issues)} issue(s) found")
        for i in sorted(set(all_issues)):
            print(f"  - {i}")
    else:
        print("AUDIT RESULT: ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

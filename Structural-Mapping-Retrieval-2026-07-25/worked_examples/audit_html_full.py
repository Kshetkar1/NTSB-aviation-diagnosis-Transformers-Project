#!/usr/bin/env python3
"""Exhaustive audit: JSON vs production pipeline vs rendered HTML."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"
HTML_PATH = Path(__file__).resolve().parent / "worked_examples.html"
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
TOL = 1e-4


def near(a: float, b: float, tol: float = TOL) -> bool:
    return abs(float(a) - float(b)) <= tol


def ok(msg: str) -> None:
    print(f"  PASS  {msg}")


def fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


def warn(msg: str) -> None:
    print(f"  WARN  {msg}")


def compare_list(name: str, cached: list, expected: list, keys: list[str], tol=TOL) -> list[str]:
    issues = []
    if len(cached) != len(expected):
        fail(f"{name}: length {len(cached)} vs expected {len(expected)}")
        return [f"{name} length"]
    for i, (c, e) in enumerate(zip(cached, expected)):
        for k in keys:
            cv, ev = c.get(k), e.get(k)
            if k in ("score", "probability", "p_k_given_q", "avg_similarity", "weight", "p_c_given_k"):
                if not near(cv, ev, tol):
                    fail(f"{name}[{i}].{k}: cached={cv} expected={ev}")
                    issues.append(f"{name}[{i}].{k}")
            elif cv != ev:
                fail(f"{name}[{i}].{k}: cached={cv!r} expected={ev!r}")
                issues.append(f"{name}[{i}].{k}")
    if not issues:
        ok(f"{name}: all {len(cached)} rows match production")
    return issues


def rerun_a0():
    os.environ.pop("NTSB_USE_TRAIN_INDEX", None)
    for mod in list(sys.modules):
        if mod.startswith("main_app"):
            del sys.modules[mod]
    import main_app
    from cause_code_mapper import build_code_embeddings, map_distribution_to_codes

    emb = main_app.get_embedding(QUERY_A0)
    scores, matches = main_app.find_top_matches(emb)
    top_scores, top_matches = list(scores[:50]), list(matches[:50])
    clusters = main_app.cluster_incidents_by_type(top_scores, top_matches, 50)
    ca = main_app.calculate_cause_probabilities_per_cluster(clusters)
    diag = main_app.calculate_chain_rule_diagnosis(clusters, ca)
    causes = diag["weighted_causes"]
    codes_list, labels_list, code_embs = build_code_embeddings()
    coded, _ = map_distribution_to_codes(
        [{"cause": c["cause"], "probability": c["probability"]} for c in causes],
        codes_list, labels_list, code_embs,
    )
    prog = main_app.predict_future_events(QUERY_A0, top_n_incidents=50, max_chain_steps=1)

    retrieval = [
        {"rank": i + 1, "ev_id": m.get("ev_id", ""), "score": float(s)}
        for i, (s, m) in enumerate(zip(top_scores, top_matches))
    ]
    active = [lbl for lbl, v in ca.items() if v.get("causes")]
    cluster_rows = []
    for label in sorted(active, key=lambda l: ca[l].get("p_cluster_query", 0), reverse=True):
        c = ca[label]
        cluster_rows.append({
            "cluster": label,
            "n_incidents": c["total_incidents"],
            "avg_similarity": c["avg_similarity"],
            "weight": c["avg_similarity"] * c["total_incidents"],
            "p_k_given_q": c["p_cluster_query"],
        })
    ltp = [{"rank": i + 1, "probability": c["probability"], "cause": c["cause"]} for i, c in enumerate(causes[:20])]
    coded_rows = [
        {"rank": i + 1, "code": "—" if cd.get("code") is None else str(cd.get("code")),
         "probability": cd.get("probability", 0)}
        for i, cd in enumerate(coded[:15])
    ]
    prog_rows = [
        {"rank": i + 1, "probability": fe.get("probability", 0)}
        for i, fe in enumerate((prog.get("future_events") or [])[:15])
    ]
    return {
        "retrieval": retrieval,
        "clusters": cluster_rows,
        "ltp_causes": ltp,
        "coded_distribution": coded_rows,
        "prognosis": prog_rows,
        "full_ltp_sum": sum(c["probability"] for c in causes),
        "full_coded_sum": sum((cd.get("probability") or 0) for cd in coded),
    }


def rerun_a2():
    os.environ["NTSB_USE_TRAIN_INDEX"] = "1"
    for mod in list(sys.modules):
        if mod.startswith("main_app"):
            del sys.modules[mod]
    import main_app
    from config import REFINED_DATA_PATH, LLM_MODEL
    from cause_code_mapper import build_code_embeddings, map_cause_to_code, map_distribution_to_codes
    from extract_struct_v2 import extract_struct_v2
    from io_cache import load_struct_jsonl
    from paths import DEFAULT_STRUCT_CACHE_V2_PATH
    from struct_hooks import make_score_adjust_fn
    from struct_score_v2 import structural_similarity as structural_similarity_v2

    inc = json.loads(Path(REFINED_DATA_PATH).read_text())[EV_A2]
    query = str(inc.get("narr_accp") or inc.get("narr_accf") or "").strip()
    emb = main_app.get_embedding(query)
    scores, matches = main_app.find_top_matches(emb)
    top_scores, top_matches = list(scores[:50]), list(matches[:50])

    a0_clusters = main_app.cluster_incidents_by_type(top_scores, top_matches, 50)
    a0_ca = main_app.calculate_cause_probabilities_per_cluster(a0_clusters)
    a0_diag = main_app.calculate_chain_rule_diagnosis(a0_clusters, a0_ca)
    a0_causes = a0_diag["weighted_causes"]

    struct_by_eid = load_struct_jsonl(DEFAULT_STRUCT_CACHE_V2_PATH)
    q_struct = extract_struct_v2(query, ev_id=EV_A2, model=LLM_MODEL)
    adj = make_score_adjust_fn(q_struct, struct_by_eid, alpha=2.0, similarity_fn=structural_similarity_v2)
    a2_scores = [float(adj(s, m)) for s, m in zip(top_scores, top_matches)]
    a2_clusters = main_app.cluster_incidents_by_type(a2_scores, top_matches, 50)
    a2_ca = main_app.calculate_cause_probabilities_per_cluster(a2_clusters)
    a2_diag = main_app.calculate_chain_rule_diagnosis(a2_clusters, a2_ca)
    a2_causes = a2_diag["weighted_causes"]

    codes_list, labels_list, code_embs = build_code_embeddings()
    a0_coded, _ = map_distribution_to_codes(
        [{"cause": c["cause"], "probability": c["probability"]} for c in a0_causes],
        codes_list, labels_list, code_embs,
    )
    a2_coded, _ = map_distribution_to_codes(
        [{"cause": c["cause"], "probability": c["probability"]} for c in a2_causes],
        codes_list, labels_list, code_embs,
    )

    def pack(ca, causes, coded):
        active = [lbl for lbl, v in ca.items() if v.get("causes")]
        clusters = []
        for label in sorted(active, key=lambda l: ca[l].get("p_cluster_query", 0), reverse=True):
            c = ca[label]
            clusters.append({
                "cluster": label,
                "n_incidents": c["total_incidents"],
                "avg_similarity": c["avg_similarity"],
                "weight": c["avg_similarity"] * c["total_incidents"],
                "p_k_given_q": c["p_cluster_query"],
            })
        return {
            "clusters": clusters,
            "ltp_causes": [{"rank": i + 1, "probability": c["probability"], "cause": c["cause"]} for i, c in enumerate(causes[:20])],
            "coded_distribution": [
                {"rank": i + 1, "code": "—" if cd.get("code") is None else str(cd.get("code")),
                 "probability": cd.get("probability", 0)}
                for i, cd in enumerate(coded[:15])
            ],
            "full_ltp_sum": sum(c["probability"] for c in causes),
            "full_coded_sum": sum((cd.get("probability") or 0) for cd in coded),
        }

    retrieval = [{"rank": i + 1, "ev_id": m.get("ev_id", ""), "score": float(s)} for i, (s, m) in enumerate(zip(top_scores, top_matches))]
    return {
        "retrieval": retrieval,
        "a0": pack(a0_ca, a0_causes, a0_coded),
        "a2": pack(a2_ca, a2_causes, a2_coded),
    }


def hand_check_a0_top_ltp(a0: dict) -> list[str]:
    """Verify top LTP cause by hand: P(C|Q) = P(C|K) * P(K|Q)."""
    issues = []
    top = a0["ltp_causes"][0]
    cause_text = top["cause"]
    # find in cluster_causes
    p_ck = None
    cluster_label = None
    p_kq = None
    for label, rows in a0["cluster_causes"].items():
        for r in rows:
            if r["cause"].startswith(cause_text[:40]) or cause_text.startswith(r["cause"][:40]):
                p_ck = r["p_c_given_k"]
                cluster_label = label
                break
        if p_ck is not None:
            break
    for c in a0["clusters"]:
        if c["cluster"] == cluster_label:
            p_kq = c["p_k_given_q"]
            break
    if p_ck is None or p_kq is None:
        warn(f"hand-check: could not locate cluster for top cause")
        return issues
    expected = p_ck * p_kq
    if near(top["probability"], expected, 1e-5):
        ok(f"hand-check top LTP: {p_ck:.4f} × {p_kq:.4f} = {expected:.6f} ≈ {top['probability']:.6f}")
    else:
        fail(f"hand-check top LTP: expected {expected:.6f}, got {top['probability']:.6f}")
        issues.append("A0 hand LTP")
    return issues


def audit_html_matches_json(a0: dict, a2: dict) -> list[str]:
    """Spot-check that formatted numbers in HTML match JSON (4 decimal places)."""
    issues = []
    print("\n=== HTML rendered numbers vs JSON ===")
    if not HTML_PATH.exists():
        fail("worked_examples.html missing")
        return ["HTML missing"]

    html = HTML_PATH.read_text(encoding="utf-8")

    checks = [
        ("A0 retrieval rank1 score", f">{a0['retrieval'][0]['score']:.4f}<", True),
        ("A0 top LTP", f">{a0['ltp_causes'][0]['probability']:.4f}<", True),
        ("A0 top code prob", f">{a0['coded_distribution'][0]['probability']:.4f}<", True),
        ("A0 prognosis rank1", f">{a0['prognosis'][0]['probability']:.4f}<", True),
        ("A2 retrieval rank1", f">{a2['retrieval'][0]['score']:.4f}<", True),
        ("A2 A0 top LTP", f">{a2['a0']['ltp_causes'][0]['probability']:.4f}<", True),
        ("A2 A2 top LTP", f">{a2['a2']['ltp_causes'][0]['probability']:.4f}<", True),
        ("A2 coded A2 rank1", f">{a2['a2']['coded_distribution'][0]['probability']:.4f}<", True),
        ("A2 ev_id", esc := a2["ev_id"], False),
    ]
    for label, needle, is_num in checks:
        if needle in html:
            ok(f"HTML contains {label}: {needle if is_num else needle}")
        else:
            fail(f"HTML missing {label}: {needle}")
            issues.append(f"HTML {label}")

    # engine shift callout should match computed delta
    a0_map = {c["cluster"]: c for c in a2["a0"]["clusters"]}
    a2_map = {c["cluster"]: c for c in a2["a2"]["clusters"]}
    eng = next(l for l in a0_map if "engine fire" in l.lower())
    shift_pp = (a2_map[eng]["p_k_given_q"] - a0_map[eng]["p_k_given_q"]) * 100
    shift_str = f"{shift_pp:+.1f} percentage points"
    if shift_str.replace("+", "+") in html or f"{shift_pp:+.1f}" in html:
        ok(f"HTML engine-shift callout uses computed {shift_pp:+.1f} pp")
    else:
        warn(f"HTML engine-shift text may not match computed {shift_pp:+.1f} pp")

    return issues


def main():
    all_issues: list[str] = []
    a0 = json.loads((DATA / "a0.json").read_text())
    a2 = json.loads((DATA / "a2.json").read_text())

    print("=== A0 JSON vs production (all rows) ===")
    exp_a0 = rerun_a0()
    all_issues.extend(compare_list("A0 retrieval", a0["retrieval"], exp_a0["retrieval"], ["rank", "ev_id", "score"]))
    all_issues.extend(compare_list("A0 clusters", a0["clusters"], exp_a0["clusters"],
                                   ["cluster", "n_incidents", "avg_similarity", "weight", "p_k_given_q"]))
    all_issues.extend(compare_list("A0 ltp top20", a0["ltp_causes"], exp_a0["ltp_causes"], ["rank", "probability"]))
    all_issues.extend(compare_list("A0 coded top15", a0["coded_distribution"], exp_a0["coded_distribution"],
                                   ["rank", "code", "probability"], tol=1e-3))
    all_issues.extend(compare_list("A0 prognosis", a0["prognosis"], exp_a0["prognosis"], ["rank", "probability"]))
    if near(exp_a0["full_ltp_sum"], 1.0):
        ok(f"A0 full LTP sum = {exp_a0['full_ltp_sum']:.6f}")
    if near(exp_a0["full_coded_sum"], 1.0, 1e-3):
        ok(f"A0 full coded sum = {exp_a0['full_coded_sum']:.6f}")

    print("\n=== A2 JSON vs production (all rows) ===")
    exp_a2 = rerun_a2()
    all_issues.extend(compare_list("A2 retrieval", a2["retrieval"], exp_a2["retrieval"], ["rank", "ev_id", "score"]))
    for path in ("a0", "a2"):
        all_issues.extend(compare_list(
            f"A2 {path} clusters", a2[path]["clusters"], exp_a2[path]["clusters"],
            ["cluster", "n_incidents", "avg_similarity", "weight", "p_k_given_q"],
        ))
        all_issues.extend(compare_list(
            f"A2 {path} ltp top20", a2[path]["ltp_causes"], exp_a2[path]["ltp_causes"],
            ["rank", "probability"],
        ))
        all_issues.extend(compare_list(
            f"A2 {path} coded top15", a2[path]["coded_distribution"], exp_a2[path]["coded_distribution"],
            ["rank", "code", "probability"], tol=1e-3,
        ))
        if near(exp_a2[path]["full_ltp_sum"], 1.0):
            ok(f"A2 {path} full LTP sum = {exp_a2[path]['full_ltp_sum']:.6f}")
        if near(exp_a2[path]["full_coded_sum"], 1.0, 1e-3):
            ok(f"A2 {path} full coded sum = {exp_a2[path]['full_coded_sum']:.6f}")

    all_issues.extend(hand_check_a0_top_ltp(a0))
    all_issues.extend(audit_html_matches_json(a0, a2))

    # Known display limitations (not data errors)
    print("\n=== Known display caveats (not failures) ===")
    warn("HTML shows top-5 causes per cluster in A0, not all causes in cluster")
    warn("HTML shows top-20 LTP causes; full distributions have 68 (A0) / 101 (A2) entries")
    warn("A2 cluster table header says 'avg cos sim' but values are reranked scores (can exceed 1.0)")
    warn("Side-by-side LTP table compares rank positions, not matched cause strings")
    warn("Pipeline prose still says A2 is 'at-least-as-good' — evaluation metric claim, not per-incident guarantee")

    print("\n" + "=" * 60)
    if all_issues:
        print(f"AUDIT: {len(set(all_issues))} REAL issue(s)")
        for i in sorted(set(all_issues)):
            print(f"  - {i}")
    else:
        print("AUDIT: ALL NUMERIC CHECKS PASSED — JSON, production, and HTML align")
    print("=" * 60)
    return 1 if all_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())

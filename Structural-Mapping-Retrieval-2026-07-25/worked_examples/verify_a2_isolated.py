#!/usr/bin/env python3
"""Verify a2.json vs production in an isolated process (train index only)."""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "Testing_Structural_Mapping_Slides" / "scripts")]
os.environ["NTSB_USE_TRAIN_INDEX"] = "1"

import main_app  # noqa: E402
from config import REFINED_DATA_PATH, LLM_MODEL  # noqa: E402
from cause_code_mapper import build_code_embeddings, map_distribution_to_codes  # noqa: E402
from extract_struct_v2 import extract_struct_v2  # noqa: E402
from io_cache import load_query_struct_jsonl, load_struct_jsonl, sha256_text  # noqa: E402
from paths import DEFAULT_QUERY_CACHE_V2_PATH, DEFAULT_STRUCT_CACHE_V2_PATH  # noqa: E402
from struct_hooks import make_score_adjust_fn  # noqa: E402
from struct_score_v2 import structural_similarity as structural_similarity_v2  # noqa: E402

EV = "20100114X11754"
cached = json.loads((Path(__file__).parent / "data" / "a2.json").read_text())
inc = json.loads(Path(REFINED_DATA_PATH).read_text())[EV]
query = str(inc.get("narr_accp") or inc.get("narr_accf") or "").strip()
emb = main_app.get_embedding(query)
scores, matches = main_app.find_top_matches(emb)
top_scores, top_matches = list(scores[:50]), list(matches[:50])

def near(a, b, t=1e-4):
    return abs(float(a) - float(b)) <= t

failures = []
for i in range(50):
    c = cached["retrieval"][i]
    if c["ev_id"] != top_matches[i]["ev_id"] or not near(c["score"], top_scores[i]):
        failures.append(f"retrieval[{i+1}]")
        break
else:
    print("PASS retrieval x50")

for label, use_a2 in [("a0", False), ("a2", True)]:
    ts, tm = top_scores, top_matches
    if use_a2:
        struct = load_struct_jsonl(DEFAULT_STRUCT_CACHE_V2_PATH)
        query_cache = load_query_struct_jsonl(DEFAULT_QUERY_CACHE_V2_PATH)
        q_hash = sha256_text(query)
        qs = query_cache[q_hash] if q_hash in query_cache else extract_struct_v2(query, ev_id=EV, model=LLM_MODEL)
        adj = make_score_adjust_fn(qs, struct, 2.0, similarity_fn=structural_similarity_v2)
        ts = [float(adj(s, m)) for s, m in zip(top_scores, top_matches)]
    cl = main_app.cluster_incidents_by_type(ts, tm, 50)
    ca = main_app.calculate_cause_probabilities_per_cluster(cl)
    diag = main_app.calculate_chain_rule_diagnosis(cl, ca)
    causes = diag["weighted_causes"]
    codes_list, labels_list, code_embs = build_code_embeddings()
    coded, _ = map_distribution_to_codes(
        [{"cause": c["cause"], "probability": c["probability"]} for c in causes],
        codes_list, labels_list, code_embs,
    )
    sub = cached[label]
    active = sorted(
        (lbl for lbl, v in ca.items() if v.get("causes")),
        key=lambda l: ca[l]["p_cluster_query"],
        reverse=True,
    )
    if len(sub["clusters"]) != len(active):
        failures.append(f"{label} cluster count")
    else:
        for j, lbl in enumerate(active):
            cc, exp = sub["clusters"][j], ca[lbl]
            if cc["cluster"] != lbl or not near(cc["p_k_given_q"], exp["p_cluster_query"]):
                failures.append(f"{label} cluster[{j+1}]")
                break
        else:
            print(f"PASS {label} clusters x{len(active)}")
    for j in range(20):
        if not near(sub["ltp_causes"][j]["probability"], causes[j]["probability"]):
            failures.append(f"{label} ltp[{j+1}]")
            break
    else:
        print(f"PASS {label} full LTP (n={len(causes)}, sum={s:.6f})")
    if not near(sum(c["probability"] for c in causes), 1.0):
        failures.append(f"{label} ltp sum")
    else:
        print(f"PASS {label} ltp sum=1.0")
    if str(sub["top1_code"]) != str(coded[0].get("code")):
        failures.append(f"{label} top1 code")
    else:
        print(f"PASS {label} top1 code={sub['top1_code']}")

print("FAILURES:", failures or "none")
raise SystemExit(1 if failures else 0)

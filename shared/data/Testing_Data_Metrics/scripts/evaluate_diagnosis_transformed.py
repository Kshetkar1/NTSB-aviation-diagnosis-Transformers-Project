# Train-only retrieval index MUST be set before any import of config/main_app.
import os

os.environ["NTSB_USE_TRAIN_INDEX"] = "1"

import argparse
import json
import math
import sys
from pathlib import Path

import eval_common

# Softer thresholds: plain-text vs plain-text (calibrate separately from structural 0.75).
MATCH_THRESHOLD_M1_PLAIN = float(os.getenv("EVAL_TRANSFORM_THRESHOLD_M1", "0.65"))
MATCH_THRESHOLD_M2_PLAIN = float(os.getenv("EVAL_TRANSFORM_THRESHOLD_M2", "0.55"))

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))

TESTING_ROOT = PROJECT_ROOT / "data" / "Testing_Data_Metrics"
TEST_IDS_PATH = TESTING_ROOT / "splits" / "test_ev_ids.txt"
OUTPUT_DIR = TESTING_ROOT / "outputs"
CSV_PATH = OUTPUT_DIR / "eval_results_transformed.csv"
JSONLINES_PATH = OUTPUT_DIR / "eval_results_transformed.json"
SUMMARY_PATH = OUTPUT_DIR / "eval_results_transformed_summary.json"

from config import LLM_MODEL, REFINED_DATA_PATH  # noqa: E402

import main_app  # noqa: E402

narr_cause_truth_chunks = eval_common.narr_cause_truth_chunks
pipe_join = eval_common.pipe_join
load_test_ids = eval_common.load_test_ids
ev_ids_in_output_csv = eval_common.ev_ids_in_output_csv


def _llm_chat(system: str, user: str, max_tokens: int = 600) -> str:
    try:
        client = main_app.get_client()
        r = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[warn] LLM call failed: {e}", flush=True)
        return ""


def transform_query_for_search(raw_query: str) -> str:
    """Query transformation: richer terminology for embedding / retrieval (no new facts)."""
    if not raw_query or not str(raw_query).strip():
        return ""
    system = (
        "You rewrite aviation accident narratives for semantic search and retrieval. "
        "Preserve all facts from the user text; use clear NTSB/aviation terminology where it fits. "
        "Do not invent events, causes, or outcomes. Output only the rewritten narrative, no preamble."
    )
    out = _llm_chat(system, raw_query.strip(), max_tokens=4000)
    return out if out else raw_query.strip()


def structural_cause_to_plain(structural_line: str) -> str:
    """Turn NTSB hierarchical finding text into 1–2 plain-English sentences (for metric comparison only)."""
    if not structural_line or not str(structural_line).strip():
        return ""
    system = (
        "You convert NTSB-style hierarchical cause labels into one or two clear English sentences "
        "describing the causal factor. Do not add facts not implied by the label. "
        "Output only the plain-language description, no preamble."
    )
    out = _llm_chat(system, structural_line.strip(), max_tokens=400)
    return out if out else structural_line.strip()


def max_cosine_sim(pred: str, truths: list[str]) -> float:
    return eval_common.max_cosine_sim(pred, truths, main_app.get_embedding)


def recall_at_k_m1(preds: list[str], truths: list[str], k: int) -> bool:
    return eval_common.recall_at_k(
        preds, truths, k, MATCH_THRESHOLD_M1_PLAIN, main_app.get_embedding
    )


def recall_at_k_m2(preds: list[str], truths: list[str], k: int) -> bool:
    return eval_common.recall_at_k(
        preds, truths, k, MATCH_THRESHOLD_M2_PLAIN, main_app.get_embedding
    )


def mrr_score_m1(preds: list[str], truths: list[str], k: int) -> float:
    return eval_common.mrr_score(
        preds, truths, k, MATCH_THRESHOLD_M1_PLAIN, main_app.get_embedding
    )


def mrr_score_m2(preds: list[str], truths: list[str], k: int) -> float:
    return eval_common.mrr_score(
        preds, truths, k, MATCH_THRESHOLD_M2_PLAIN, main_app.get_embedding
    )


def build_query(inc: dict):
    accp = inc.get("narr_accp") or ""
    accf = inc.get("narr_accf") or ""
    if accp and str(accp).strip():
        return str(accp).strip(), "narr_accp"
    if accf and str(accf).strip():
        return str(accf).strip(), "narr_accf"
    return None, None


def append_csv_row(row: dict, fieldnames: list[str]) -> None:
    eval_common.append_csv_row(row, fieldnames, CSV_PATH, OUTPUT_DIR)


def append_json_record(record: dict) -> None:
    eval_common.append_json_record(record, JSONLINES_PATH, OUTPUT_DIR)


def main() -> None:
    global CSV_PATH, JSONLINES_PATH, SUMMARY_PATH

    ap = argparse.ArgumentParser(
        description="Diagnosis eval: transformed query + plain-text causes (M1/M2, train-only index)."
    )
    ap.add_argument("--n", default="10", help='Number of test incidents or "all".')
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument(
        "--output-stem",
        default="eval_results_transformed",
        help="Writes <stem>.csv, <stem>.json, <stem>_summary.json under outputs/.",
    )
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    stem = (args.output_stem or "eval_results_transformed").strip() or "eval_results_transformed"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_PATH = OUTPUT_DIR / f"{stem}.csv"
    JSONLINES_PATH = OUTPUT_DIR / f"{stem}.json"
    SUMMARY_PATH = OUTPUT_DIR / f"{stem}_summary.json"

    if not main_app.DATA_LOADED:
        raise RuntimeError(
            "main_app could not load train embeddings / merged_dataset_train.json. "
            "Run create_splits.py, 2_generate_embeddings.py --train, 2b --train."
        )
    if not REFINED_DATA_PATH.is_file():
        raise FileNotFoundError(f"Full dataset not found: {REFINED_DATA_PATH}")

    test_ids = load_test_ids(TEST_IDS_PATH)
    with open(REFINED_DATA_PATH, "r", encoding="utf-8") as f:
        full_dataset: dict = json.load(f)

    if args.n == "all":
        batch = test_ids[args.offset :]
    else:
        n = int(args.n)
        batch = test_ids[args.offset : args.offset + n]

    if args.resume:
        done = ev_ids_in_output_csv(CSV_PATH)
        batch = [eid for eid in batch if eid not in done]
        print(f"[resume] {len(done)} on disk; {len(batch)} to run", flush=True)

    fieldnames = [
        "ev_id",
        "query_source",
        "query_full_text",
        "query_transformed",
        "top_prediction_cause_structural",
        "top_prediction_detailed",
        "top_prediction_prob",
        "top5_detailed_full",
        "m1_truth_structural",
        "m1_truth_detailed",
        "m1_match_pct",
        "m1_hit",
        "m1_recall5",
        "mrr_m1",
        "m2_truth_full",
        "m2_match_pct",
        "m2_hit",
        "m2_recall5",
        "mrr_m2",
    ]

    for ev_id in batch:
        inc = full_dataset.get(ev_id)
        if not inc:
            print(f"[skip] {ev_id}: not in full dataset", flush=True)
            continue

        query_raw, qsrc = build_query(inc)
        if not query_raw:
            print(f"[skip] {ev_id}: no narr_accp / narr_accf", flush=True)
            continue

        print(f"[…] {ev_id} query transform", flush=True)
        query_t = transform_query_for_search(query_raw)
        if not query_t:
            query_t = query_raw

        diag = main_app.diagnose_with_conditional_probabilities(
            query_t, top_n=10, top_n_incidents=50
        )
        causes = diag.get("weighted_causes") or []
        if diag.get("error"):
            print(f"[warn] {ev_id}: {diag.get('error')}", flush=True)

        top5_s = [str(c.get("cause", "")).strip() for c in causes[:5] if c.get("cause")]
        top1_s = top5_s[0] if top5_s else ""
        top1_prob = causes[0].get("probability", 0.0) if causes else 0.0

        print(f"[…] {ev_id} plain-language expansion ({len(top5_s)} preds + M1)", flush=True)
        top1_d = structural_cause_to_plain(top1_s) if top1_s else ""
        top5_d = [structural_cause_to_plain(p) for p in top5_s]
        top5_d = [p for p in top5_d if p]

        findings = inc.get("findings") or []
        m1_list_s = [
            str(f.get("finding_description", "")).strip()
            for f in findings
            if isinstance(f, dict) and (f.get("Cause_Factor") or "").strip() == "C"
            and (f.get("finding_description") or "").strip()
        ]
        m1_list_d = [structural_cause_to_plain(x) for x in m1_list_s]
        m1_list_d = [x for x in m1_list_d if x]

        narr_cause = inc.get("narr_cause") or ""
        narr_s = str(narr_cause).strip() if narr_cause else ""
        m2_chunks = narr_cause_truth_chunks(narr_s)

        def score_m1_plain(truths: list[str]):
            if not truths:
                return "N/A", "", "", ""
            pct = max_cosine_sim(top1_d, truths) * 100.0 if top1_d else float("nan")
            pct_s = "" if math.isnan(pct) else f"{pct:.4f}"
            hit = (
                ""
                if not top1_d or math.isnan(pct)
                else ("1" if pct / 100.0 >= MATCH_THRESHOLD_M1_PLAIN else "0")
            )
            r5 = "1" if recall_at_k_m1(top5_d, truths, 5) else "0"
            mrr = mrr_score_m1(top5_d, truths, 5)
            return pct_s, hit, r5, f"{mrr:.6f}"

        def score_m2_plain(truths: list[str]):
            if not truths:
                return "N/A", "", "", ""
            pct = max_cosine_sim(top1_d, truths) * 100.0 if top1_d else float("nan")
            pct_s = "" if math.isnan(pct) else f"{pct:.4f}"
            hit = (
                ""
                if not top1_d or math.isnan(pct)
                else ("1" if pct / 100.0 >= MATCH_THRESHOLD_M2_PLAIN else "0")
            )
            r5 = "1" if recall_at_k_m2(top5_d, truths, 5) else "0"
            mrr = mrr_score_m2(top5_d, truths, 5)
            return pct_s, hit, r5, f"{mrr:.6f}"

        m1_struct_s = pipe_join(m1_list_s) if m1_list_s else "N/A"
        m1_det_s = pipe_join(m1_list_d) if m1_list_d else "N/A"
        m2_truth_s = narr_s if m2_chunks else "N/A"

        if not m1_list_d:
            m1_pct, m1_hit, m1_r5, m1_mrr = ("N/A", "", "", "")
        else:
            m1_pct, m1_hit, m1_r5, m1_mrr = score_m1_plain(m1_list_d)

        if not m2_chunks:
            m2_pct, m2_hit, m2_r5, m2_mrr = ("N/A", "", "", "")
        else:
            m2_pct, m2_hit, m2_r5, m2_mrr = score_m2_plain(m2_chunks)

        row = {
            "ev_id": ev_id,
            "query_source": qsrc or "",
            "query_full_text": query_raw,
            "query_transformed": query_t,
            "top_prediction_cause_structural": top1_s,
            "top_prediction_detailed": top1_d,
            "top_prediction_prob": f"{top1_prob:.6f}",
            "top5_detailed_full": pipe_join(top5_d) if top5_d else "N/A",
            "m1_truth_structural": m1_struct_s,
            "m1_truth_detailed": m1_det_s,
            "m1_match_pct": m1_pct,
            "m1_hit": m1_hit,
            "m1_recall5": m1_r5,
            "mrr_m1": m1_mrr,
            "m2_truth_full": m2_truth_s,
            "m2_match_pct": m2_pct,
            "m2_hit": m2_hit,
            "m2_recall5": m2_r5,
            "mrr_m2": m2_mrr,
        }

        append_csv_row(row, fieldnames)
        append_json_record(row)
        print(f"[ok] {ev_id}", flush=True)

    summary = eval_common.recompute_summary_two_metrics(
        CSV_PATH, MATCH_THRESHOLD_M1_PLAIN, MATCH_THRESHOLD_M2_PLAIN
    )
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"Plain-text eval summary ({CSV_PATH.name})")
    print(
        f"Thresholds: M1={MATCH_THRESHOLD_M1_PLAIN}  M2={MATCH_THRESHOLD_M2_PLAIN}",
        flush=True,
    )
    print("=" * 72)
    hdr = f"{'Metric':<36} {'Eligible':>10} {'Avg Match%':>12} {'Top-1 Acc':>12} {'Recall@5':>10} {'MRR':>10}"
    print(hdr)
    print("-" * len(hdr))
    for name, key in (
        ("M1 — detailed (plain vs plain)", "M1_detailed_plain_text"),
        ("M2 — detailed pred vs narr_cause", "M2_narrative_plain_vs_plain"),
    ):
        m = summary["metrics"][key]
        elig = m.get("eligible", 0)
        am = m.get("avg_match_pct")
        t1 = m.get("top1_accuracy")
        r5 = m.get("recall_at_5")
        mr = m.get("mrr")
        am_s = f"{am:.2f}" if am is not None else "—"
        t1_s = f"{t1*100:.1f}%" if t1 is not None else "—"
        r5_s = f"{r5*100:.1f}%" if r5 is not None else "—"
        mr_s = f"{mr:.4f}" if mr is not None else "—"
        print(f"{name:<36} {elig:>10} {am_s:>12} {t1_s:>12} {r5_s:>10} {mr_s:>10}")


if __name__ == "__main__":
    main()

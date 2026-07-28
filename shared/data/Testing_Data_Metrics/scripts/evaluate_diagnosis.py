# Train-only retrieval index MUST be set before any import of config/main_app.
import os

os.environ["NTSB_USE_TRAIN_INDEX"] = "1"

import argparse
import json
import math
import sys
from pathlib import Path

import eval_common

MATCH_THRESHOLD = 0.75

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))

TESTING_ROOT = PROJECT_ROOT / "data" / "Testing_Data_Metrics"
TEST_IDS_PATH = TESTING_ROOT / "splits" / "test_ev_ids.txt"
OUTPUT_DIR = TESTING_ROOT / "outputs"
CSV_PATH = OUTPUT_DIR / "eval_results.csv"
JSONLINES_PATH = OUTPUT_DIR / "eval_results.json"
SUMMARY_PATH = OUTPUT_DIR / "eval_summary.json"

from config import REFINED_DATA_PATH  # noqa: E402

import main_app  # noqa: E402

narr_cause_truth_chunks = eval_common.narr_cause_truth_chunks
pipe_join = eval_common.pipe_join
load_test_ids = eval_common.load_test_ids
ev_ids_in_output_csv = eval_common.ev_ids_in_output_csv


def max_cosine_sim(pred: str, truths: list[str]) -> float:
    return eval_common.max_cosine_sim(pred, truths, main_app.get_embedding)


def recall_at_k(preds: list[str], truths: list[str], k: int) -> bool:
    return eval_common.recall_at_k(preds, truths, k, MATCH_THRESHOLD, main_app.get_embedding)


def mrr_score(preds: list[str], truths: list[str], k: int) -> float:
    return eval_common.mrr_score(preds, truths, k, MATCH_THRESHOLD, main_app.get_embedding)


def build_query(inc: dict):
    accp = inc.get("narr_accp") or ""
    accf = inc.get("narr_accf") or ""
    if accp and str(accp).strip():
        return str(accp).strip(), "narr_accp"
    if accf and str(accf).strip():
        return str(accf).strip(), "narr_accf"
    return None, None


def recompute_summary_from_csv() -> dict:
    return eval_common.recompute_summary_structural_only(CSV_PATH, MATCH_THRESHOLD)


def append_csv_row(row: dict, fieldnames: list[str]) -> None:
    eval_common.append_csv_row(row, fieldnames, CSV_PATH, OUTPUT_DIR)


def append_json_record(record: dict) -> None:
    eval_common.append_json_record(record, JSONLINES_PATH, OUTPUT_DIR)


def main() -> None:
    global CSV_PATH, JSONLINES_PATH, SUMMARY_PATH

    ap = argparse.ArgumentParser(description="Diagnosis eval vs M1/M2 structural (train-only index).")
    ap.add_argument(
        "--n",
        default="10",
        help='Number of test incidents (integer) or "all".',
    )
    ap.add_argument("--offset", type=int, default=0, help="Skip this many test ids from the list.")
    ap.add_argument(
        "--output-stem",
        default="eval_results",
        help="Under outputs/: writes <stem>.csv, <stem>.json, <stem>_summary.json (default: eval_results).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip ev_ids already present in the output CSV (same --output-stem).",
    )
    args = ap.parse_args()

    stem = (args.output_stem or "eval_results").strip() or "eval_results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_PATH = OUTPUT_DIR / f"{stem}.csv"
    JSONLINES_PATH = OUTPUT_DIR / f"{stem}.json"
    SUMMARY_PATH = OUTPUT_DIR / f"{stem}_summary.json"

    if not main_app.DATA_LOADED:
        raise RuntimeError(
            "main_app could not load train embeddings / merged_dataset_train.json. "
            "Run: python data/Testing_Data_Metrics/scripts/create_splits.py then "
            "python data/preprocessing/2_generate_embeddings.py --train and "
            "python data/preprocessing/2b_precompute_diagnostic_data.py --train"
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
        before = len(batch)
        batch = [eid for eid in batch if eid not in done]
        print(
            f"[resume] {CSV_PATH.name}: {len(done)} ev_ids on disk; "
            f"{before} -> {len(batch)} to run after skipping existing",
            flush=True,
        )

    fieldnames = [
        "ev_id",
        "query_source",
        "query_full_text",
        "top_prediction_cause",
        "top_prediction_prob",
        "m1_truth_full",
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

        query, qsrc = build_query(inc)
        if not query:
            print(f"[skip] {ev_id}: no narr_accp / narr_accf", flush=True)
            continue

        diag = main_app.diagnose_with_conditional_probabilities(query, top_n=10, top_n_incidents=50)
        causes = diag.get("weighted_causes") or []
        if diag.get("error"):
            print(f"[warn] {ev_id}: {diag.get('error')}", flush=True)

        top5 = [str(c.get("cause", "")).strip() for c in causes[:5] if c.get("cause")]
        top1 = top5[0] if top5 else ""
        top1_prob = causes[0].get("probability", 0.0) if causes else 0.0

        findings = inc.get("findings") or []
        m1_list = [
            str(f.get("finding_description", "")).strip()
            for f in findings
            if isinstance(f, dict) and (f.get("Cause_Factor") or "").strip() == "C"
            and (f.get("finding_description") or "").strip()
        ]
        narr_cause = inc.get("narr_cause") or ""
        narr_s = str(narr_cause).strip() if narr_cause else ""
        m2_chunks = narr_cause_truth_chunks(narr_s)

        def score_metric(truths: list[str]):
            if not truths:
                return "N/A", "", "", ""
            pct = max_cosine_sim(top1, truths) * 100.0 if top1 else float("nan")
            pct_s = "" if math.isnan(pct) else f"{pct:.4f}"
            hit = (
                ""
                if not top1 or math.isnan(pct)
                else ("1" if pct / 100.0 >= MATCH_THRESHOLD else "0")
            )
            r5 = "1" if recall_at_k(top5, truths, 5) else "0"
            mrr = mrr_score(top5, truths, 5)
            mrr_s = f"{mrr:.6f}"
            return pct_s, hit, r5, mrr_s

        m1_truth_s = pipe_join(m1_list) if m1_list else "N/A"
        m2_truth_s = narr_s if m2_chunks else "N/A"

        if not m1_list:
            m1_pct, m1_hit, m1_r5, m1_mrr = ("N/A", "", "", "")
        else:
            m1_pct, m1_hit, m1_r5, m1_mrr = score_metric(m1_list)

        if not m2_chunks:
            m2_pct, m2_hit, m2_r5, m2_mrr = ("N/A", "", "", "")
        else:
            m2_pct, m2_hit, m2_r5, m2_mrr = score_metric(m2_chunks)

        row = {
            "ev_id": ev_id,
            "query_source": qsrc or "",
            "query_full_text": query,
            "top_prediction_cause": top1,
            "top_prediction_prob": f"{top1_prob:.6f}",
            "m1_truth_full": m1_truth_s,
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

        print(f"[ok] {ev_id}  top1={top1[:80]!r}…" if len(top1) > 80 else f"[ok] {ev_id}  top1={top1!r}", flush=True)

    summary = recompute_summary_from_csv()
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"Cumulative summary (all rows in {CSV_PATH.name})")
    print("=" * 72)
    hdr = f"{'Metric':<28} {'Eligible':>10} {'Avg Match%':>12} {'Top-1 Acc':>12} {'Recall@5':>10} {'MRR':>10}"
    print(hdr)
    print("-" * len(hdr))
    for name, key in (
        ("M1 — findings (C only)", "M1_findings_C_only"),
        ("M2 — full probable cause (narr_cause)", "M2_probable_cause_narrative_full"),
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
        print(f"{name:<28} {elig:>10} {am_s:>12} {t1_s:>12} {r5_s:>10} {mr_s:>10}")


if __name__ == "__main__":
    main()

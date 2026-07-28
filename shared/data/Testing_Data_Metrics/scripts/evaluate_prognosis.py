# Train-only retrieval index MUST be set before any import of config/main_app.
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ["NTSB_USE_TRAIN_INDEX"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))

TESTING_ROOT = PROJECT_ROOT / "data" / "Testing_Data_Metrics"
TEST_IDS_PATH = TESTING_ROOT / "splits" / "test_ev_ids.txt"
OUTPUT_DIR = TESTING_ROOT / "outputs"

from config import REFINED_DATA_PATH  # noqa: E402

import main_app  # noqa: E402


def load_test_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run create_splits.py first.")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [x for x in lines if x]


def sequence_descriptions(inc: dict) -> list[str]:
    seq = inc.get("sequence_of_events") or []
    return [str(e.get("Occurrence_Description") or "").strip() for e in seq]


def norm(s: str) -> str:
    return main_app._normalize_occurrence_text(s)


def eval_transition(query: str, truth_next: str, top_n_incidents: int):
    """Returns exact_hit, recall5, top1_text, top1_prob, n_aligned, n_downstream."""
    r = main_app.predict_future_events(query, top_n_incidents=top_n_incidents, max_chain_steps=1)
    dist = r.get("future_events") or []
    tnorm = norm(truth_next)
    if not dist or not tnorm:
        return False, False, "", float("nan"), len(r.get("aligned_incidents") or []), r.get(
            "downstream_incident_count", 0
        )

    top1 = dist[0]["event"]
    p1 = float(dist[0]["probability"])
    top5_events = [d["event"] for d in dist[:5]]
    exact = norm(top1) == tnorm
    r5 = any(norm(x) == tnorm for x in top5_events)
    return exact, r5, top1, p1, len(r.get("aligned_incidents") or []), r.get("downstream_incident_count", 0)


def soft_hit(truth_next: str, pred: str, threshold: float) -> bool:
    if not truth_next or not pred:
        return False
    a = main_app.get_embedding(truth_next)
    b = main_app.get_embedding(pred)
    import numpy as np

    return float(np.dot(np.asarray(a), np.asarray(b))) >= threshold


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prognosis eval: next sequence event prediction (train-only index)."
    )
    ap.add_argument("--n", default="20", help='Test incidents to use (int) or "all".')
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument(
        "--max-positions",
        type=int,
        default=8,
        help="Max sequence indices to evaluate per test incident (query = event i, truth = i+1).",
    )
    ap.add_argument("--top-n-incidents", type=int, default=50)
    ap.add_argument(
        "--soft-threshold",
        type=float,
        default=None,
        help="If set (e.g. 0.85), also compute soft_hit via embedding cosine pred vs truth.",
    )
    ap.add_argument("--output-stem", default="eval_prognosis", help="Writes outputs/<stem>.csv and _summary.json")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = (args.output_stem or "eval_prognosis").strip() or "eval_prognosis"
    csv_path = OUTPUT_DIR / f"{stem}.csv"
    summary_path = OUTPUT_DIR / f"{stem}_summary.json"

    if not main_app.DATA_LOADED:
        raise RuntimeError(
            "Train-only knowledge base not loaded. Run create_splits.py, then "
            "2_generate_embeddings.py --train and ensure merged_dataset_train.json exists."
        )
    if not REFINED_DATA_PATH.is_file():
        raise FileNotFoundError(f"Full dataset not found: {REFINED_DATA_PATH}")

    test_ids = load_test_ids(TEST_IDS_PATH)
    with open(REFINED_DATA_PATH, "r", encoding="utf-8") as f:
        full_dataset: dict = json.load(f)

    if args.n == "all":
        batch = test_ids[args.offset :]
    else:
        batch = test_ids[args.offset : args.offset + int(args.n)]

    rows_out = []
    n_exact = 0
    n_r5 = 0
    n_soft = 0
    n_soft_evaluated = 0
    n_total = 0
    n_skipped_empty = 0

    fieldnames = [
        "ev_id",
        "step_index",
        "query_event",
        "truth_next",
        "top1_pred",
        "top1_prob",
        "exact_hit",
        "recall5_hit",
        "n_aligned",
        "n_downstream",
        "soft_hit",
    ]

    for ev_id in batch:
        inc = full_dataset.get(ev_id)
        if not inc:
            continue
        descs = sequence_descriptions(inc)
        if len(descs) < 2:
            n_skipped_empty += 1
            continue
        max_i = min(len(descs) - 2, args.max_positions - 1)
        for i in range(max_i + 1):
            q = descs[i]
            truth = descs[i + 1]
            if not q or not truth:
                continue
            exact, r5, top1, p1, n_al, n_dn = eval_transition(
                q, truth, args.top_n_incidents
            )
            sh = ""
            if args.soft_threshold is not None and top1:
                n_soft_evaluated += 1
                s = soft_hit(truth, top1, args.soft_threshold)
                sh = "1" if s else "0"
                if s:
                    n_soft += 1
            n_total += 1
            n_exact += 1 if exact else 0
            n_r5 += 1 if r5 else 0
            rows_out.append(
                {
                    "ev_id": ev_id,
                    "step_index": i,
                    "query_event": q,
                    "truth_next": truth,
                    "top1_pred": top1,
                    "top1_prob": "" if math.isnan(p1) else f"{p1:.6f}",
                    "exact_hit": "1" if exact else "0",
                    "recall5_hit": "1" if r5 else "0",
                    "n_aligned": n_al,
                    "n_downstream": n_dn,
                    "soft_hit": sh,
                }
            )
            print(
                f"{ev_id} step{i} exact={exact} r5={r5} top1_prob={p1:.3f} downstream={n_dn}",
                flush=True,
            )

    exact_acc = n_exact / n_total if n_total else 0.0
    r5_acc = n_r5 / n_total if n_total else 0.0
    soft_acc = (
        (n_soft / n_soft_evaluated) if n_soft_evaluated else None
    )

    summary = {
        "n_test_incidents_requested": len(batch),
        "n_transition_rows": n_total,
        "n_skipped_fewer_than_2_events": n_skipped_empty,
        "exact_accuracy": exact_acc,
        "recall5_accuracy": r5_acc,
        "soft_accuracy": soft_acc,
        "n_soft_evaluated": n_soft_evaluated,
        "soft_threshold": args.soft_threshold,
        "train_only_index": True,
        "max_positions_per_incident": args.max_positions,
    }

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

"""Shared helpers for diagnosis evaluation scripts (structural + transformed)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

NARR_CAUSE_CHUNK_CHARS = 10000


def narr_cause_truth_chunks(narr_cause: str, min_chars: int = 10) -> list[str]:
    """Split full probable-cause narrative into embeddable chunks (max cosine over chunks)."""
    s = str(narr_cause).strip() if narr_cause else ""
    if len(s) < min_chars:
        return []
    if len(s) <= NARR_CAUSE_CHUNK_CHARS:
        return [s]
    chunks: list[str] = []
    step = NARR_CAUSE_CHUNK_CHARS
    i = 0
    while i < len(s):
        chunks.append(s[i : i + step])
        i += step
    return chunks


def _vec(text: str, get_embedding: Callable[[str], list]) -> np.ndarray:
    emb = get_embedding(text)
    return np.asarray(emb, dtype=np.float64)


def max_cosine_sim(pred: str, truths: list[str], get_embedding: Callable[[str], list]) -> float:
    if not truths or not pred or not str(pred).strip():
        return float("nan")
    pv = _vec(pred, get_embedding)
    best = -1.0
    for t in truths:
        if not t or not str(t).strip():
            continue
        tv = _vec(str(t), get_embedding)
        best = max(best, float(np.dot(pv, tv)))
    return best


def recall_at_k(
    preds: list[str],
    truths: list[str],
    k: int,
    threshold: float,
    get_embedding: Callable[[str], list],
) -> bool:
    if not truths:
        return False
    for p in preds[:k]:
        s = max_cosine_sim(p, truths, get_embedding)
        if not math.isnan(s) and s >= threshold:
            return True
    return False


def mrr_score(
    preds: list[str],
    truths: list[str],
    k: int,
    threshold: float,
    get_embedding: Callable[[str], list],
) -> float:
    if not truths:
        return 0.0
    for i, p in enumerate(preds[:k], start=1):
        s = max_cosine_sim(p, truths, get_embedding)
        if not math.isnan(s) and s >= threshold:
            return 1.0 / i
    return 0.0


def pipe_join(items: list[str]) -> str:
    cleaned = [str(x).replace("|", " ").strip() for x in items if x and str(x).strip()]
    return " | ".join(cleaned) if cleaned else "N/A"


def load_test_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run create_splits.py first.")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [x for x in lines if x]


def ev_ids_in_output_csv(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    ids: set[str] = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eid = (row.get("ev_id") or "").strip()
            if eid:
                ids.add(eid)
    return ids


def append_csv_row(
    row: dict,
    fieldnames: list[str],
    csv_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.is_file()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        if new_file:
            w.writeheader()
        w.writerow(row)


def append_json_record(record: dict, json_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list = []
    if json_path.is_file():
        try:
            records = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                records = []
        except json.JSONDecodeError:
            records = []
    records.append(record)
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def recompute_summary_two_metrics(
    csv_path: Path,
    match_threshold_m1: float,
    match_threshold_m2: float,
) -> dict:
    """Aggregate M1 and M2 only (columns m1_*, m2_*)."""
    if not csv_path.is_file():
        return {"metrics": {}, "row_count": 0}

    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    def agg(metric: str) -> dict:
        pct_key = f"{metric}_match_pct"
        hit_key = f"{metric}_hit"
        r5_key = f"{metric}_recall5"
        mrr_key = f"mrr_{metric}"

        if metric == "m1":
            elig = [r for r in rows if r.get("m1_truth_detailed", "").strip() not in ("", "N/A")]
        else:
            elig = [r for r in rows if r.get("m2_truth_full", "").strip() not in ("", "N/A")]

        if not elig:
            return {
                "eligible": 0,
                "avg_match_pct": None,
                "top1_accuracy": None,
                "recall_at_5": None,
                "mrr": None,
            }

        pcts: list[float] = []
        hits: list[float] = []
        r5s: list[float] = []
        mrrs: list[float] = []

        for r in elig:
            p_raw = r.get(pct_key, "").strip()
            if p_raw and p_raw != "N/A":
                try:
                    pcts.append(float(p_raw))
                except ValueError:
                    pass
            h = r.get(hit_key, "").strip()
            if h in ("1", "0", "True", "False"):
                hits.append(1.0 if h in ("1", "True") else 0.0)
            r5 = r.get(r5_key, "").strip()
            if r5 in ("1", "0", "True", "False"):
                r5s.append(1.0 if r5 in ("1", "True") else 0.0)
            m = r.get(mrr_key, "").strip()
            if m and m != "N/A":
                try:
                    mrrs.append(float(m))
                except ValueError:
                    pass

        return {
            "eligible": len(elig),
            "avg_match_pct": sum(pcts) / len(pcts) if pcts else None,
            "top1_accuracy": sum(hits) / len(hits) if hits else None,
            "recall_at_5": sum(r5s) / len(r5s) if r5s else None,
            "mrr": sum(mrrs) / len(mrrs) if mrrs else None,
        }

    summary = {
        "row_count": len(rows),
        "metrics": {
            "M1_detailed_plain_text": agg("m1"),
            "M2_narrative_plain_vs_plain": agg("m2"),
        },
        "match_threshold_m1": match_threshold_m1,
        "match_threshold_m2": match_threshold_m2,
    }
    return summary


def recompute_summary_structural_only(csv_path: Path, match_threshold: float) -> dict:
    """Aggregate M1/M2 for structural eval (m1_truth_full, m2_truth_full)."""
    if not csv_path.is_file():
        return {"metrics": {}, "row_count": 0}

    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    def agg(metric: str) -> dict:
        pct_key = f"{metric}_match_pct"
        hit_key = f"{metric}_hit"
        r5_key = f"{metric}_recall5"
        mrr_key = f"mrr_{metric}"

        if metric == "m1":
            elig = [r for r in rows if r.get("m1_truth_full", "").strip() not in ("", "N/A")]
        else:
            elig = [r for r in rows if r.get("m2_truth_full", "").strip() not in ("", "N/A")]

        if not elig:
            return {
                "eligible": 0,
                "avg_match_pct": None,
                "top1_accuracy": None,
                "recall_at_5": None,
                "mrr": None,
            }

        pcts: list[float] = []
        hits: list[float] = []
        r5s: list[float] = []
        mrrs: list[float] = []

        for r in elig:
            p_raw = r.get(pct_key, "").strip()
            if p_raw and p_raw != "N/A":
                try:
                    pcts.append(float(p_raw))
                except ValueError:
                    pass
            h = r.get(hit_key, "").strip()
            if h in ("1", "0", "True", "False"):
                hits.append(1.0 if h in ("1", "True") else 0.0)
            r5 = r.get(r5_key, "").strip()
            if r5 in ("1", "0", "True", "False"):
                r5s.append(1.0 if r5 in ("1", "True") else 0.0)
            m = r.get(mrr_key, "").strip()
            if m and m != "N/A":
                try:
                    mrrs.append(float(m))
                except ValueError:
                    pass

        return {
            "eligible": len(elig),
            "avg_match_pct": sum(pcts) / len(pcts) if pcts else None,
            "top1_accuracy": sum(hits) / len(hits) if hits else None,
            "recall_at_5": sum(r5s) / len(r5s) if r5s else None,
            "mrr": sum(mrrs) / len(mrrs) if mrrs else None,
        }

    return {
        "row_count": len(rows),
        "metrics": {
            "M1_findings_C_only": agg("m1"),
            "M2_probable_cause_narrative_full": agg("m2"),
        },
        "match_threshold": match_threshold,
    }

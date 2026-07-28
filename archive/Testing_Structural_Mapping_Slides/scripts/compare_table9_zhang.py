#!/usr/bin/env python3
"""
Compare our model against Zhang Table 9: Loss of Engine Power.

DIAGNOSIS row: P(loss of engine power | evidence) — from diagnose_with_conditional_probabilities()
PROGNOSIS rows: P(forced landing), P(gear collapsed), etc. — from predict_future_events()

Uses EMBEDDING-BASED matching (not substring) to align prognosis events to Zhang's labels.
Runs BOTH A0 (embedding-only) and A2 (structural mapping) for each evidence set.
Outputs formatted .xlsx.

Usage:
    python compare_table9_zhang.py
"""
from __future__ import annotations

import os
os.environ["NTSB_USE_TRAIN_INDEX"] = "1"

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPTS))

import main_app  # noqa: E402
from config import LLM_MODEL, get_openai_api_key  # noqa: E402
from openai import OpenAI  # noqa: E402

import numpy as np  # noqa: E402
from cause_code_mapper import (  # noqa: E402
    build_code_embeddings, map_distribution_to_codes,
    match_events_to_targets, match_causes_to_targets,
)
from paths import EVAL_OUTPUT_DIR  # noqa: E402

from extract_struct_v2 import extract_struct_v2  # noqa: E402
from io_cache import load_struct_jsonl  # noqa: E402
from paths import DEFAULT_STRUCT_CACHE_V2_PATH  # noqa: E402
from struct_hooks import make_score_adjust_fn  # noqa: E402
from struct_score_v2 import structural_similarity as structural_similarity_v2  # noqa: E402

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    print("WARNING: openpyxl not installed. Will output CSV only. pip install openpyxl")


# ============================================================
# Zhang Table 9 reference values (from the paper)
# ============================================================
TABLE_9_EVIDENCE_SETS = [
    "Inoperative engine instruments",
    "Combustion liner failure",
    "Improper oil usage",
    "Inop. instruments + Improper oil",
    "All three causes",
]

ZHANG_TABLE_9 = {
    "Loss of engine power":         [0.95,      0.50,       0.95,      0.99,      1.0],
    "Forced landing":               [13.57e-2,  7.14e-2,    13.57e-2,  14.71e-2,  14.29e-2],
    "Ditching":                     [4.37e-3,   2.30e-3,    4.37e-3,   4.57e-3,   4.61e-3],
    "Gear collapsed":               [9.60e-3,   2.30e-3,    4.37e-3,   9.82e-3,   5.18e-3],
    "Other gear collapsed":         [4.80e-3,   2.30e-3,    4.37e-3,   5.00e-3,   4.66e-3],
    "Destroyed aircraft":           [1.33e-2,   2.30e-3,    4.37e-3,   1.35e-2,   5.59e-3],
    "Substantial aircraft damage":  [4.60e-2,   3.63e-3,    6.09e-3,   4.63e-2,   1.66e-2],
    "Minor aircraft damage":        [9.34e-3,   1.54e-3,    2.92e-3,   9.47e-3,   3.78e-3],
    "Serious injury":               [6.23e-2,   7.68e-4,    1.46e-3,   6.23e-2,   8.22e-3],
    "No injury":                    [94.31e-2,  99.78e-2,   99.58e-2,  94.29e-2,  98.99e-2],
}

QUERIES = [
    (
        "The aircraft experienced inoperative engine instruments during flight. "
        "The engine instruments failed and were not providing accurate readings."
    ),
    (
        "The aircraft experienced a combustion liner failure during flight. "
        "The combustion assembly liner in the engine cracked and failed."
    ),
    (
        "The aircraft had improper oil usage. The engine oil was of improper "
        "grade or contaminated, affecting engine performance."
    ),
    (
        "The aircraft experienced both inoperative engine instruments and "
        "improper oil usage. The engine instruments failed while the engine "
        "oil was also contaminated or of improper grade."
    ),
    (
        "The aircraft experienced loss of engine power due to multiple causes: "
        "inoperative engine instruments, combustion liner failure, and "
        "improper oil usage all contributed to the engine failure."
    ),
]


def run_diagnosis(query, score_adjust_fn=None):
    r = main_app.diagnose_with_conditional_probabilities(
        query, top_n=10, top_n_incidents=50, score_adjust_fn=score_adjust_fn,
    )
    return r.get("weighted_causes", [])


def run_prognosis(query, score_adjust_fn=None):
    r = main_app.predict_future_events(
        query, top_n_incidents=50, max_chain_steps=1, score_adjust_fn=score_adjust_fn,
    )
    return r.get("future_events", [])


def find_coded_probability(coded_dist, target):
    """Find probability for a target label in a coded distribution using embedding similarity."""
    target_emb = np.array(main_app.get_embedding(target))
    best_prob = 0.0
    best_sim = -1.0
    for cd in coded_dist:
        label = cd.get("label", "")
        if not label:
            continue
        label_emb = np.array(main_app.get_embedding(label))
        norm = np.linalg.norm(target_emb) * np.linalg.norm(label_emb)
        sim = float(np.dot(target_emb, label_emb) / norm) if norm > 0 else 0.0
        if sim > best_sim:
            best_sim = sim
            best_prob = cd.get("probability", 0.0)
    return best_prob if best_sim >= 0.5 else 0.0


def build_a2_adjust_fn(query, struct_by_eid):
    q_struct = extract_struct_v2(query, "zhang_table9_query", model=LLM_MODEL)
    return make_score_adjust_fn(
        q_struct, struct_by_eid, alpha=0.5,
        similarity_fn=structural_similarity_v2,
    )


def write_xlsx(output_dir, event_names, all_results):
    """Write formatted Excel file."""
    if not HAS_OPENPYXL:
        return None
    wb = Workbook()
    ws = wb.active
    ws.title = "Table 9 Comparison"

    # Styles
    header_fill = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")
    zhang_fill = PatternFill(start_color="D6E4F0", end_color="D6E4F0", fill_type="solid")
    a0_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    a2_fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF", size=10)
    bold_font = Font(bold=True, size=10)
    thin_border = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"), bottom=Side(style="thin"),
    )

    # Header row 1: evidence set groups
    ws.cell(row=1, column=1, value="Event").font = header_font
    ws.cell(row=1, column=1).fill = header_fill
    ws.cell(row=1, column=2, value="Type").font = header_font
    ws.cell(row=1, column=2).fill = header_fill
    col = 3
    for ev_label in TABLE_9_EVIDENCE_SETS:
        for sub in ["Zhang BN", "A0", "A2"]:
            cell = ws.cell(row=1, column=col, value=f"{sub} — {ev_label}")
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", wrap_text=True)
            cell.border = thin_border
            col += 1

    # Data rows
    for row_idx, event_name in enumerate(event_names, start=2):
        ws.cell(row=row_idx, column=1, value=event_name).font = bold_font
        ws.cell(row=row_idx, column=1).border = thin_border
        etype = "DIAGNOSIS" if event_name == "Loss of engine power" else "prognosis"
        ws.cell(row=row_idx, column=2, value=etype).border = thin_border

        col = 3
        for j, ev_label in enumerate(TABLE_9_EVIDENCE_SETS):
            r = all_results[j]
            zhang_p = r[f"{event_name}_zhang"]
            a0_p = r[f"{event_name}_a0"]
            a2_p = r[f"{event_name}_a2"]

            for val, fill in [(zhang_p, zhang_fill), (a0_p, a0_fill), (a2_p, a2_fill)]:
                cell = ws.cell(row=row_idx, column=col, value=round(val, 6))
                cell.number_format = "0.0000"
                cell.fill = fill
                cell.border = thin_border
                cell.alignment = Alignment(horizontal="center")
                col += 1

    # Column widths
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 12
    for c in range(3, col):
        ws.column_dimensions[chr(64 + c) if c <= 26 else "A" + chr(64 + c - 26)].width = 16

    xlsx_path = output_dir / "comparison_table9.xlsx"
    wb.save(xlsx_path)
    return xlsx_path


def main():
    if not main_app.DATA_LOADED:
        raise RuntimeError("Train index not loaded.")

    output_dir = EVAL_OUTPUT_DIR / "zhang_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading structural cache for A2...", flush=True)
    struct_by_eid = load_struct_jsonl(DEFAULT_STRUCT_CACHE_V2_PATH)
    print(f"  Loaded {len(struct_by_eid)} cached structures.", flush=True)

    codes, labels, code_embs = build_code_embeddings()

    print("\n" + "=" * 70)
    print("TABLE 9: Loss of Engine Power")
    print("  DIAGNOSIS (top row) + PROGNOSIS (downstream events)")
    print("  Zhang BN vs A0 vs A2 — embedding-based event matching")
    print("=" * 70)

    event_names = list(ZHANG_TABLE_9.keys())
    # Prognosis target labels — matched DIRECTLY, not through 54 occurrence codes.
    # These include damage severity and injury nodes that are NOT occurrence codes.
    prognosis_targets = [e for e in event_names if e != "Loss of engine power"]
    all_results = []

    for i, (ev_label, query) in enumerate(zip(TABLE_9_EVIDENCE_SETS, QUERIES)):
        print(f"\n{'='*50}")
        print(f"Evidence {i+1}/{len(QUERIES)}: {ev_label}")
        print(f"{'='*50}")

        # --- A0 ---
        print("  A0 diagnosis...", flush=True)
        a0_diag = run_diagnosis(query)
        # Map diagnosis to occurrence codes (for "Loss of engine power" row)
        a0_diag_dist = [{"cause": c["cause"], "probability": c["probability"]} for c in a0_diag]
        a0_diag_coded, _ = map_distribution_to_codes(a0_diag_dist, codes, labels, code_embs)

        print("  A0 prognosis...", flush=True)
        a0_prog = run_prognosis(query)
        # Direct match against Zhang's target labels (NOT through occurrence codes)
        a0_prog_matched = match_events_to_targets(a0_prog, prognosis_targets)

        # --- A2 ---
        print("  A2 diagnosis...", flush=True)
        a2_adj = build_a2_adjust_fn(query, struct_by_eid)
        a2_diag = run_diagnosis(query, score_adjust_fn=a2_adj)
        a2_diag_dist = [{"cause": c["cause"], "probability": c["probability"]} for c in a2_diag]
        a2_diag_coded, _ = map_distribution_to_codes(a2_diag_dist, codes, labels, code_embs)

        print("  A2 prognosis...", flush=True)
        a2_prog = run_prognosis(query, score_adjust_fn=a2_adj)
        a2_prog_matched = match_events_to_targets(a2_prog, prognosis_targets)

        result = {"evidence": ev_label, "query": query}

        for event_name in event_names:
            zhang_p = ZHANG_TABLE_9[event_name][i]

            if event_name == "Loss of engine power":
                # DIAGNOSIS — use occurrence-code mapping
                a0_p = find_coded_probability(a0_diag_coded, event_name)
                a2_p = find_coded_probability(a2_diag_coded, event_name)
                source = "diagnosis"
            else:
                # PROGNOSIS — use DIRECT target matching
                a0_p = a0_prog_matched.get(event_name, 0.0)
                a2_p = a2_prog_matched.get(event_name, 0.0)
                source = "prognosis"

            result[f"{event_name}_zhang"] = zhang_p
            result[f"{event_name}_a0"] = a0_p
            result[f"{event_name}_a2"] = a2_p

            print(f"  {event_name:35s}  Zhang={zhang_p:.4f}  A0={a0_p:.4f}  A2={a2_p:.4f}  [{source}]")

        result["a0_diag_coded_top10"] = a0_diag_coded[:10]
        result["a2_diag_coded_top10"] = a2_diag_coded[:10]
        result["a0_prog_raw_top10"] = [{"event": e["event"], "probability": round(e["probability"], 6)} for e in a0_prog[:10]]
        result["a2_prog_raw_top10"] = [{"event": e["event"], "probability": round(e["probability"], 6)} for e in a2_prog[:10]]

        all_results.append(result)

    # Write xlsx
    xlsx_path = write_xlsx(output_dir, event_names, all_results)

    # Write JSON
    json_path = output_dir / "comparison_table9.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*70}")
    print(f"Table 9 comparison complete.")
    if xlsx_path:
        print(f"  XLSX: {xlsx_path}")
    print(f"  JSON: {json_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

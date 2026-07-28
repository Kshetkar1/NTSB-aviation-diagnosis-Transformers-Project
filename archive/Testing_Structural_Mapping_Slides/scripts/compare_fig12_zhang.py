#!/usr/bin/env python3
"""
Compare our model against Zhang Fig 12: Pilot Error Influence Propagation.

Zhang shows: if pilot error is observed, what are the downstream probabilities
for hard landing, unstable approach, substantial damage, injuries, etc.?

Uses embedding-based event matching for both diagnosis and prognosis.
Runs both A0 and A2. Outputs formatted .xlsx.

Usage:
    python compare_fig12_zhang.py
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

from cause_code_mapper import match_events_to_targets, match_causes_to_targets  # noqa: E402
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
    print("WARNING: openpyxl not installed. Will output JSON only. pip install openpyxl")


# ============================================================
# Zhang Fig 12 reference values
# Three states: prior -> after pilot_error=1 -> after unstable_approach=1
# ============================================================
FIG_12_NODES = {
    "Pilot error":                      (2.24e-5,    1.0,              1.0),
    "Unstable approach":                (2.71e-8,    4.84e-3,          1.0),
    "Hard landing":                     (2.02e-7,    5.99e-2,          5.99e-2),
    "Improper flare":                   (1.52e-7,    4.12e-2,          4.12e-2),
    "Dragged wing/tail on runway":      (1.14e-7,    2.30e-2,          41.72e-2),
    "Substantial aircraft damage":      (3.22e-7,    4.58e-2,          24.64e-2),
    "No injury":                        (99.99e-2,   97.0e-2,          61.30e-2),
    "Minor injury":                     (1.57e-7,    1.34e-2,          1.34e-2),
}

QUERIES = {
    "pilot_error": (
        "The pilot in command made an error during the approach phase of flight. "
        "The pilot's actions were erroneous and contributed to the incident."
    ),
    "pilot_error_unstable_approach": (
        "The pilot in command made an error during the approach phase, resulting "
        "in an unstabilized approach. The approach was not maintained stable in "
        "terms of speed, descent rate, and flight path, leading to a dangerous situation."
    ),
}


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


def build_a2_adjust_fn(query, struct_by_eid):
    q_struct = extract_struct_v2(query, "zhang_fig12_query", model=LLM_MODEL)
    return make_score_adjust_fn(
        q_struct, struct_by_eid, alpha=0.5,
        similarity_fn=structural_similarity_v2,
    )


def write_xlsx(output_dir, conditions, all_results):
    if not HAS_OPENPYXL:
        return None
    wb = Workbook()
    ws = wb.active
    ws.title = "Fig 12 Comparison"

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

    # Header
    headers = ["Node", "Zhang Prior"]
    for cond_label, _ in conditions:
        headers.extend([f"Zhang — {cond_label}", f"A0 — {cond_label}", f"A2 — {cond_label}"])

    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        cell.border = thin_border

    # Data rows
    for row_idx, (node_name, (prior, _, _)) in enumerate(FIG_12_NODES.items(), start=2):
        ws.cell(row=row_idx, column=1, value=node_name).font = bold_font
        ws.cell(row=row_idx, column=1).border = thin_border

        cell = ws.cell(row=row_idx, column=2, value=prior)
        cell.number_format = "0.00E+00"
        cell.border = thin_border

        col = 3
        for j, (cond_label, _) in enumerate(conditions):
            r = all_results[j]
            zhang_p = r[f"{node_name}_zhang"]
            a0_p = r[f"{node_name}_a0"]
            a2_p = r[f"{node_name}_a2"]

            for val, fill in [(zhang_p, zhang_fill), (a0_p, a0_fill), (a2_p, a2_fill)]:
                cell = ws.cell(row=row_idx, column=col, value=round(val, 6))
                cell.number_format = "0.0000"
                cell.fill = fill
                cell.border = thin_border
                cell.alignment = Alignment(horizontal="center")
                col += 1

    ws.column_dimensions["A"].width = 35
    xlsx_path = output_dir / "comparison_fig12.xlsx"
    wb.save(xlsx_path)
    return xlsx_path


def main():
    if not main_app.DATA_LOADED:
        raise RuntimeError("Train index not loaded.")

    output_dir = EVAL_OUTPUT_DIR / "zhang_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading structural cache for A2...", flush=True)
    struct_by_eid = load_struct_jsonl(DEFAULT_STRUCT_CACHE_V2_PATH)
    node_targets = list(FIG_12_NODES.keys())

    print("\n" + "=" * 70)
    print("FIG 12: Pilot Error Influence Propagation")
    print("  Zhang BN vs A0 vs A2 (diagnosis + prognosis) — direct target matching")
    print("=" * 70)

    conditions = [
        ("After pilot error", "pilot_error"),
        ("After pilot error + unstable approach", "pilot_error_unstable_approach"),
    ]

    all_results = []

    for cond_label, cond_key in conditions:
        query = QUERIES[cond_key]
        print(f"\n{'='*50}")
        print(f"Condition: {cond_label}")
        print(f"{'='*50}")

        # A0
        print("  A0 diagnosis...", flush=True)
        a0_diag = run_diagnosis(query)
        a0_diag_matched = match_causes_to_targets(a0_diag, node_targets)
        print("  A0 prognosis...", flush=True)
        a0_prog = run_prognosis(query)
        a0_prog_matched = match_events_to_targets(a0_prog, node_targets)

        # A2
        print("  A2 diagnosis...", flush=True)
        a2_adj = build_a2_adjust_fn(query, struct_by_eid)
        a2_diag = run_diagnosis(query, score_adjust_fn=a2_adj)
        a2_diag_matched = match_causes_to_targets(a2_diag, node_targets)
        print("  A2 prognosis...", flush=True)
        a2_prog = run_prognosis(query, score_adjust_fn=a2_adj)
        a2_prog_matched = match_events_to_targets(a2_prog, node_targets)

        result = {"condition": cond_label, "query": query}

        for node_name, (prior, post_pe, post_ua) in FIG_12_NODES.items():
            zhang_p = post_pe if cond_key == "pilot_error" else post_ua

            # Best of diagnosis and prognosis (direct target matching)
            a0_p_diag = a0_diag_matched.get(node_name, 0.0)
            a0_p_prog = a0_prog_matched.get(node_name, 0.0)
            a0_p = max(a0_p_diag, a0_p_prog)

            a2_p_diag = a2_diag_matched.get(node_name, 0.0)
            a2_p_prog = a2_prog_matched.get(node_name, 0.0)
            a2_p = max(a2_p_diag, a2_p_prog)

            source = "diagnosis" if a0_p_diag >= a0_p_prog else "prognosis"

            result[f"{node_name}_zhang"] = zhang_p
            result[f"{node_name}_a0"] = a0_p
            result[f"{node_name}_a2"] = a2_p

            print(f"  {node_name:40s}  Zhang={zhang_p:.4e}  A0={a0_p:.4f}  A2={a2_p:.4f}  [{source}]")

        all_results.append(result)

    xlsx_path = write_xlsx(output_dir, conditions, all_results)

    json_path = output_dir / "comparison_fig12.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*70}")
    print(f"Fig 12 comparison complete.")
    if xlsx_path:
        print(f"  XLSX: {xlsx_path}")
    print(f"  JSON: {json_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

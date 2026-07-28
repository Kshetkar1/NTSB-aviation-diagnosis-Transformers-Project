#!/usr/bin/env python3
"""
Build the ground truth + A0 + A2 comparison table with FULL coded distributions.

Maha: "we could have just held out or leave one out thing and then say,
this is what actually happened. This is what these 3 models."

Jesse: "if you keep it the raw probability, then you can make a strong
comparison between the two."

For each of the 77 test incidents, outputs:
  Sheet 1 — Diagnosis: full probability over all 54 Zhang codes for A0 and A2
  Sheet 2 — Prognosis: top downstream events mapped to codes for A0 and A2
  Sheet 3 — Summary: accuracy stats, entropy, where A2 changed the answer

Requires Phase 1 JSON outputs (eval_A0_baseline.json, eval_A2_structural.json).
Run Phase 1 first:
    python eval_diagnosis_structural.py --n all --output-stem eval_A0_baseline
    python eval_diagnosis_structural.py --n all --structural --struct-version v2 --output-stem eval_A2_structural
"""
from __future__ import annotations

import os
os.environ["NTSB_USE_TRAIN_INDEX"] = "1"

import json
import math
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPTS))

from config import REFINED_DATA_PATH, LLM_MODEL, get_openai_api_key  # noqa: E402
from openai import OpenAI  # noqa: E402
from paths import EVAL_OUTPUT_DIR, TEST_IDS_PATH  # noqa: E402

import main_app  # noqa: E402
from cause_code_mapper import (  # noqa: E402
    ZHANG_OCCURRENCE_CODES,
    build_code_embeddings,
    map_cause_to_code,
    map_distribution_to_codes,
    map_events_to_codes,
)

from extract_struct_v2 import extract_struct_v2  # noqa: E402
from io_cache import load_struct_jsonl, load_query_struct_jsonl, sha256_text, append_jsonl  # noqa: E402
from paths import DEFAULT_STRUCT_CACHE_V2_PATH, DEFAULT_QUERY_CACHE_V2_PATH  # noqa: E402
from struct_hooks import make_score_adjust_fn  # noqa: E402
from struct_score_v2 import structural_similarity as structural_similarity_v2  # noqa: E402

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    print("WARNING: openpyxl not installed. pip install openpyxl")


def load_json_results(path: Path) -> dict:
    with open(path) as f:
        records = json.load(f)
    return {rec["ev_id"]: rec for rec in records}


def entropy(dist: list[dict]) -> float:
    H = 0.0
    for c in dist:
        p = c.get("probability", 0)
        if p > 0:
            H -= p * math.log2(p)
    return H


def build_full_code_vector(coded_dist: list[dict], all_codes: list[int]) -> dict[int | str, float]:
    """Convert coded distribution to {code: probability} dict covering all 54 codes + UNMAPPED."""
    code_map = {}
    unmapped_total = 0.0
    for cd in coded_dist:
        c = cd.get("code")
        if c is not None:
            code_map[c] = cd.get("probability", 0.0)
        else:
            unmapped_total += cd.get("probability", 0.0)
    # Ensure all codes present
    result = {}
    for code in all_codes:
        result[code] = code_map.get(code, 0.0)
    result["UNMAPPED"] = round(unmapped_total, 6)
    return result


BATCH_SIZE = 5  # Save Excel + checkpoint every N incidents


def _write_excel(output_dir: Path, rows_diag, rows_prog, rows_summary, all_codes, partial=False):
    """Write the 3-sheet Excel file. Called for both partial batch saves and the final save."""
    wb = Workbook()

    header_fill = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")
    gt_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    a0_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    a2_fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
    match_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF", size=9)
    bold_font = Font(bold=True, size=9)
    small_font = Font(size=8)
    thin_border = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"), bottom=Side(style="thin"),
    )

    # --- Sheet 1: Full Diagnosis Distributions ---
    ws1 = wb.active
    ws1.title = "Diagnosis — Full Distributions"

    diag_headers = ["ev_id", "ground_truth", "gt_code", "gt_label"]
    for c in all_codes:
        lbl = ZHANG_OCCURRENCE_CODES[c]
        diag_headers.append(f"A0_{c}_{lbl[:20]}")
        diag_headers.append(f"A2_{c}_{lbl[:20]}")
    diag_headers.append("A0_UNMAPPED")
    diag_headers.append("A2_UNMAPPED")

    for col_idx, h in enumerate(diag_headers, 1):
        cell = ws1.cell(row=1, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        cell.border = thin_border

    for row_idx, dr in enumerate(rows_diag, start=2):
        ws1.cell(row=row_idx, column=1, value=dr["ev_id"]).font = bold_font
        ws1.cell(row=row_idx, column=1).border = thin_border
        ws1.cell(row=row_idx, column=2, value=dr["ground_truth"]).border = thin_border
        ws1.cell(row=row_idx, column=3, value=dr["gt_code"]).border = thin_border
        ws1.cell(row=row_idx, column=3).fill = gt_fill
        ws1.cell(row=row_idx, column=4, value=dr["gt_label"]).border = thin_border
        ws1.cell(row=row_idx, column=4).fill = gt_fill

        col = 5
        for c in all_codes:
            a0_val = dr.get(f"A0_{c}", 0.0)
            a2_val = dr.get(f"A2_{c}", 0.0)

            cell_a0 = ws1.cell(row=row_idx, column=col, value=a0_val)
            cell_a0.number_format = "0.0000"
            cell_a0.font = small_font
            cell_a0.border = thin_border
            if a0_val > 0.01:
                cell_a0.fill = a0_fill
            if c == dr.get("gt_code") and a0_val > 0:
                cell_a0.fill = match_fill

            cell_a2 = ws1.cell(row=row_idx, column=col + 1, value=a2_val)
            cell_a2.number_format = "0.0000"
            cell_a2.font = small_font
            cell_a2.border = thin_border
            if a2_val > 0.01:
                cell_a2.fill = a2_fill
            if c == dr.get("gt_code") and a2_val > 0:
                cell_a2.fill = match_fill

            col += 2

        unmapped_fill = PatternFill(start_color="FAEEDA", end_color="FAEEDA", fill_type="solid")
        for label_key in ["A0_UNMAPPED", "A2_UNMAPPED"]:
            val = dr.get(label_key, 0.0)
            cell = ws1.cell(row=row_idx, column=col, value=val)
            cell.number_format = "0.0000"
            cell.font = small_font
            cell.border = thin_border
            if val > 0.01:
                cell.fill = unmapped_fill
            col += 1

    # --- Sheet 2: Prognosis ---
    ws2 = wb.create_sheet("Prognosis")
    prog_headers = ["ev_id", "query"]
    for rank in range(1, 6):
        prog_headers.append(f"A0_prog_{rank}_label")
        prog_headers.append(f"A0_prog_{rank}_prob")
    for rank in range(1, 6):
        prog_headers.append(f"A2_prog_{rank}_label")
        prog_headers.append(f"A2_prog_{rank}_prob")

    for col_idx, h in enumerate(prog_headers, 1):
        cell = ws2.cell(row=1, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = thin_border

    for row_idx, pr in enumerate(rows_prog, start=2):
        ws2.cell(row=row_idx, column=1, value=pr.get("ev_id", "")).border = thin_border
        ws2.cell(row=row_idx, column=2, value=pr.get("query", "")).border = thin_border
        col = 3
        for rank in range(1, 6):
            ws2.cell(row=row_idx, column=col, value=pr.get(f"A0_prog_{rank}_label", "")).border = thin_border
            cell = ws2.cell(row=row_idx, column=col + 1, value=pr.get(f"A0_prog_{rank}_prob", ""))
            cell.number_format = "0.0000"
            cell.border = thin_border
            col += 2
        for rank in range(1, 6):
            cell_lbl = ws2.cell(row=row_idx, column=col, value=pr.get(f"A2_prog_{rank}_label", ""))
            cell_lbl.border = thin_border
            cell_lbl.fill = a2_fill
            cell_val = ws2.cell(row=row_idx, column=col + 1, value=pr.get(f"A2_prog_{rank}_prob", ""))
            cell_val.number_format = "0.0000"
            cell_val.border = thin_border
            cell_val.fill = a2_fill
            col += 2

    # --- Sheet 3: Summary ---
    ws3 = wb.create_sheet("Summary")
    sum_headers = list(rows_summary[0].keys()) if rows_summary else []
    for col_idx, h in enumerate(sum_headers, 1):
        cell = ws3.cell(row=1, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = thin_border

    for row_idx, sr in enumerate(rows_summary, start=2):
        for col_idx, h in enumerate(sum_headers, 1):
            cell = ws3.cell(row=row_idx, column=col_idx, value=sr.get(h, ""))
            cell.border = thin_border
            if h == "A0_matches_gt" and sr.get(h) == 1:
                cell.fill = match_fill
            if h == "A2_matches_gt" and sr.get(h) == 1:
                cell.fill = match_fill

    # Summary stats at bottom
    total = len(rows_summary)
    if total > 0:
        a0_correct = sum(1 for r in rows_summary if r.get("A0_matches_gt") == 1)
        a2_correct = sum(1 for r in rows_summary if r.get("A2_matches_gt") == 1)
        a2_changed = sum(1 for r in rows_summary if r.get("A2_changed") == 1)

        gap = len(rows_summary) + 3
        status = "PARTIAL — run still in progress" if partial else "FINAL"
        ws3.cell(row=gap, column=1, value=f"SUMMARY STATISTICS ({status})").font = bold_font
        ws3.cell(row=gap + 1, column=1, value="Total test incidents processed")
        ws3.cell(row=gap + 1, column=2, value=total)
        ws3.cell(row=gap + 2, column=1, value="A0 correct (by code)")
        ws3.cell(row=gap + 2, column=2, value=f"{a0_correct}/{total} ({100*a0_correct/total:.1f}%)")
        ws3.cell(row=gap + 3, column=1, value="A2 correct (by code)")
        ws3.cell(row=gap + 3, column=2, value=f"{a2_correct}/{total} ({100*a2_correct/total:.1f}%)")
        ws3.cell(row=gap + 4, column=1, value="A2 changed top-1 from A0")
        ws3.cell(row=gap + 4, column=2, value=f"{a2_changed}/{total}")

    xlsx_path = output_dir / "three_model_comparison.xlsx"
    wb.save(xlsx_path)
    return xlsx_path


def load_checkpoint(path: Path) -> dict[str, dict]:
    """Load checkpoint JSONL → {ev_id: {diag: ..., prog: ..., summary: ...}}."""
    if not path.is_file():
        return {}
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = row.get("ev_id", "").strip()
            if eid:
                out[eid] = row
    return out


def append_checkpoint(path: Path, ev_id: str, diag_row: dict, prog_row: dict, summary_row: dict) -> None:
    """Append one incident's results to the checkpoint JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"ev_id": ev_id, "diag": diag_row, "prog": prog_row, "summary": summary_row}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def main():
    a0_path = EVAL_OUTPUT_DIR / "eval_A0_baseline.json"
    a2_path = EVAL_OUTPUT_DIR / "eval_A2_structural.json"

    if not a0_path.exists():
        print(f"Missing: {a0_path}")
        print("Run Phase 1 first: python eval_diagnosis_structural.py --n all --output-stem eval_A0_baseline")
        return
    if not a2_path.exists():
        print(f"Missing: {a2_path}")
        print("Run Phase 1 first: python eval_diagnosis_structural.py --n all --structural --struct-version v2 --output-stem eval_A2_structural")
        return

    print("Loading Phase 1 results...", flush=True)
    a0_data = load_json_results(a0_path)
    a2_data = load_json_results(a2_path)

    with open(REFINED_DATA_PATH, "r", encoding="utf-8") as f:
        full_dataset = json.load(f)

    with open(TEST_IDS_PATH, "r") as f:
        test_ids = [line.strip() for line in f if line.strip()]

    print(f"  A0 records: {len(a0_data)}")
    print(f"  A2 records: {len(a2_data)}")
    print(f"  Test IDs: {len(test_ids)}")

    codes, labels, code_embs = build_code_embeddings()
    all_codes = sorted(ZHANG_OCCURRENCE_CODES.keys())

    # Load structural cache for A2 prognosis
    print("  Loading structural cache for A2 prognosis...", flush=True)
    struct_by_eid = load_struct_jsonl(DEFAULT_STRUCT_CACHE_V2_PATH)
    print(f"  Loaded {len(struct_by_eid)} cached structures.", flush=True)

    # Load query struct cache — reuse LLM extractions from Step 2
    query_cache = load_query_struct_jsonl(DEFAULT_QUERY_CACHE_V2_PATH)
    print(f"  Loaded {len(query_cache)} cached query structs (from Step 2).", flush=True)
    client = OpenAI(api_key=get_openai_api_key())

    output_dir = EVAL_OUTPUT_DIR / "three_model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Checkpoint: load previous progress
    # ============================================================
    checkpoint_path = output_dir / "checkpoint.jsonl"
    checkpoint_data = load_checkpoint(checkpoint_path)
    if checkpoint_data:
        print(f"\n  RESUMING: found {len(checkpoint_data)} completed incidents in checkpoint.")
        print(f"  Remaining: {len(test_ids) - len(checkpoint_data)} incidents to process.\n")
    else:
        print(f"\n  Starting fresh — {len(test_ids)} incidents to process.\n")

    # ============================================================
    # Process all test incidents
    # ============================================================
    rows_diag = []  # Sheet 1: full coded diagnosis distributions
    rows_prog = []  # Sheet 2: prognosis
    rows_summary = []  # Sheet 3: summary

    # Pre-load checkpointed rows so the final Excel includes everything
    for ev_id_ck, ck in checkpoint_data.items():
        rows_diag.append(ck["diag"])
        rows_prog.append(ck["prog"])
        rows_summary.append(ck["summary"])

    new_since_save = 0  # track how many new incidents since last Excel save
    start_time = time.time()
    processed_new = 0

    for idx, ev_id in enumerate(test_ids):
        # Skip already-checkpointed incidents
        if ev_id in checkpoint_data:
            continue

        inc = full_dataset.get(ev_id)
        if not inc:
            continue

        processed_new += 1
        remaining = len(test_ids) - len(checkpoint_data) - processed_new
        elapsed = time.time() - start_time
        rate = processed_new / elapsed if elapsed > 0 else 0
        eta_min = (remaining / rate / 60) if rate > 0 else 0
        print(f"  [{processed_new + len(checkpoint_data)}/{len(test_ids)}] "
              f"Processing {ev_id}  "
              f"(ETA: {eta_min:.1f} min remaining)", flush=True)

        # Ground truth
        findings = inc.get("findings") or []
        cause_findings = [
            str(f.get("finding_description", "")).strip()
            for f in findings
            if isinstance(f, dict)
            and (f.get("Cause_Factor") or "").strip() == "C"
            and (f.get("finding_description") or "").strip()
        ]
        narr_cause = str(inc.get("narr_cause") or "").strip()
        ground_truth = cause_findings[0] if cause_findings else narr_cause

        if ground_truth:
            gt_code, gt_label, _ = map_cause_to_code(ground_truth, codes, labels, code_embs)
        else:
            gt_code, gt_label = None, "N/A"

        # A0 diagnosis
        a0_rec = a0_data.get(ev_id)
        a0_dist = a0_rec["probability_distribution"] if a0_rec else []
        a0_coded, _ = map_distribution_to_codes(a0_dist, codes, labels, code_embs)
        a0_code_vec = build_full_code_vector(a0_coded, all_codes)

        # A2 diagnosis
        a2_rec = a2_data.get(ev_id)
        a2_dist = a2_rec["probability_distribution"] if a2_rec else []
        a2_coded, _ = map_distribution_to_codes(a2_dist, codes, labels, code_embs)
        a2_code_vec = build_full_code_vector(a2_coded, all_codes)

        # Top-1 from coded distributions
        a0_top1_code = a0_coded[0]["code"] if a0_coded else None
        a0_top1_label = a0_coded[0]["label"] if a0_coded else "N/A"
        a0_top1_prob = a0_coded[0]["probability"] if a0_coded else 0.0

        a2_top1_code = a2_coded[0]["code"] if a2_coded else None
        a2_top1_label = a2_coded[0]["label"] if a2_coded else "N/A"
        a2_top1_prob = a2_coded[0]["probability"] if a2_coded else 0.0

        # Matches
        a0_match = 1 if (a0_top1_code is not None and a0_top1_code == gt_code) else 0
        a2_match = 1 if (a2_top1_code is not None and a2_top1_code == gt_code) else 0
        changed = 1 if a0_top1_code != a2_top1_code else 0

        # Build diagnosis row (full 54-code vector + UNMAPPED)
        diag_row = {
            "ev_id": ev_id,
            "ground_truth": ground_truth[:200] if ground_truth else "",
            "gt_code": gt_code,
            "gt_label": gt_label,
        }
        for c in all_codes:
            diag_row[f"A0_{c}"] = round(a0_code_vec[c], 6)
            diag_row[f"A2_{c}"] = round(a2_code_vec[c], 6)
        diag_row["A0_UNMAPPED"] = a0_code_vec.get("UNMAPPED", 0.0)
        diag_row["A2_UNMAPPED"] = a2_code_vec.get("UNMAPPED", 0.0)
        rows_diag.append(diag_row)

        # Build prognosis row
        # Get query text
        query = str(inc.get("narr_accp") or inc.get("narr_accf") or "").strip()
        prog_row = {
            "ev_id": ev_id,
            "query": query[:200],
        }
        if query and main_app.DATA_LOADED:
            # A0 prognosis
            try:
                a0_prog_result = main_app.predict_future_events(
                    query, top_n_incidents=50, max_chain_steps=1,
                )
                a0_prog = a0_prog_result.get("future_events", [])
                a0_prog_coded, _ = map_events_to_codes(a0_prog, codes, labels, code_embs)

                for rank, cd in enumerate(a0_prog_coded[:5], 1):
                    prog_row[f"A0_prog_{rank}_label"] = cd.get("label", "")
                    prog_row[f"A0_prog_{rank}_prob"] = round(cd.get("probability", 0), 6)
            except Exception as e:
                prog_row["A0_prog_error"] = str(e)

            # A2 prognosis (with structural reweighting — reuses cached query structs)
            try:
                q_hash = sha256_text(query)
                if q_hash in query_cache:
                    q_struct = query_cache[q_hash]
                else:
                    q_struct = extract_struct_v2(query, client=client, model=LLM_MODEL)
                    append_jsonl(DEFAULT_QUERY_CACHE_V2_PATH, {
                        "query_hash": q_hash, "struct": q_struct, "preview": query[:240],
                    })
                    query_cache[q_hash] = q_struct

                a2_adj = make_score_adjust_fn(
                    q_struct, struct_by_eid, alpha=0.5,
                    similarity_fn=structural_similarity_v2,
                )
                a2_prog_result = main_app.predict_future_events(
                    query, top_n_incidents=50, max_chain_steps=1,
                    score_adjust_fn=a2_adj,
                )
                a2_prog = a2_prog_result.get("future_events", [])
                a2_prog_coded, _ = map_events_to_codes(a2_prog, codes, labels, code_embs)

                for rank, cd in enumerate(a2_prog_coded[:5], 1):
                    prog_row[f"A2_prog_{rank}_label"] = cd.get("label", "")
                    prog_row[f"A2_prog_{rank}_prob"] = round(cd.get("probability", 0), 6)
            except Exception as e:
                prog_row["A2_prog_error"] = str(e)

        rows_prog.append(prog_row)

        # Summary row
        summary_row = {
            "ev_id": ev_id,
            "ground_truth": ground_truth[:150] if ground_truth else "",
            "gt_code": gt_code,
            "gt_label": gt_label,
            "A0_top1_label": a0_top1_label,
            "A0_top1_prob": round(a0_top1_prob, 6),
            "A0_top1_code": a0_top1_code,
            "A0_entropy": round(entropy(a0_coded), 4) if a0_coded else "",
            "A2_top1_label": a2_top1_label,
            "A2_top1_prob": round(a2_top1_prob, 6),
            "A2_top1_code": a2_top1_code,
            "A2_entropy": round(entropy(a2_coded), 4) if a2_coded else "",
            "A0_matches_gt": a0_match,
            "A2_matches_gt": a2_match,
            "A2_changed": changed,
        }
        rows_summary.append(summary_row)

        # ---- Checkpoint: save this incident immediately ----
        append_checkpoint(checkpoint_path, ev_id, diag_row, prog_row, summary_row)
        new_since_save += 1

        # ---- Batch save: write Excel every BATCH_SIZE incidents ----
        if new_since_save >= BATCH_SIZE:
            if HAS_OPENPYXL:
                _write_excel(output_dir, rows_diag, rows_prog, rows_summary, all_codes, partial=True)
                print(f"    >> Saved partial Excel ({len(rows_diag)} incidents so far)", flush=True)
            new_since_save = 0

    # ============================================================
    # Write final Excel with 3 sheets
    # ============================================================
    if HAS_OPENPYXL:
        xlsx_path = _write_excel(output_dir, rows_diag, rows_prog, rows_summary, all_codes, partial=False)
        print(f"\n  XLSX: {xlsx_path}")
    else:
        xlsx_path = None

    # Also write JSON for programmatic access
    json_path = output_dir / "three_model_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "diagnosis": rows_diag,
            "prognosis": rows_prog,
            "summary": rows_summary,
        }, f, indent=2, ensure_ascii=False, default=str)

    total = len(rows_summary)
    a0_correct = sum(1 for r in rows_summary if r.get("A0_matches_gt") == 1)
    a2_correct = sum(1 for r in rows_summary if r.get("A2_matches_gt") == 1)
    a2_changed = sum(1 for r in rows_summary if r.get("A2_changed") == 1)

    print(f"\n{'='*70}")
    print(f"Three-Model Comparison: {total} test incidents")
    print(f"  A0 top-1 matches truth (by code): {a0_correct}/{total} ({100*a0_correct/total:.1f}%)")
    print(f"  A2 top-1 matches truth (by code): {a2_correct}/{total} ({100*a2_correct/total:.1f}%)")
    print(f"  A2 changed top-1 from A0:         {a2_changed}/{total}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

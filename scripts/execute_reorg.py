#!/usr/bin/env python3
"""One-shot repo reorganization (Phases A + B moves/renames). Run from repo root."""
from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

FROZEN = REPO / "Frozen-BN-Narrative-Evidence-2026-07-20"
LLM = REPO / "LLM-Narrative-Parser-Experiments-2026-07-15"
STRUCT = REPO / "Structural-Mapping-Retrieval-2026-07-25"
EMBED = REPO / "Embedding-Similarity-Counting-Path-2026-03-25"
ZHANG = REPO / "Zhang-Replication-Foundation-2026-06-04"
BNCPT = REPO / "BN-CPT-Upgrade-Experiments-2026-06-19"
SHARED = REPO / "shared"
ARCHIVE = REPO / "archive"

APPROACH_SUBS = [
    "code", "apps", "tests", "docs_FrozenBN", "docs_LLM", "docs_StructMap",
    "docs_Embedding", "docs_Zhang", "docs_BNUpgrade", "presentations_FrozenBN",
    "presentations_StructMap", "presentations_Embedding", "paper_drafts",
    "outputs", "worked_examples", "scripts", "schema", "metrics",
    "methodology_audit", "reference",
]

FROZEN_TESTS = """
reproduce_all_examples.py zhang_conditional_method.py reproduce_fig3_from_tables.py
reproduce_table7.py reproduce_table7_full.py reproduce_fire_prior.py
heldout_narrative_bn_eval.py heldout_significance.py lr_baseline_heldout.py
test_narrative_to_bn.py combined_parser_validation.py
test_prognosis_invariants.py prognosis_chain_demo.py prognosis_table9.py
smoke_prognosis_transitions.py real_narrative_walkthrough.py real_narratives_demo.py
maha_narrative_demo.py trace_narrative_vs_direct.py sparse_robustness_validation.py
make_architecture_figure.py build_presentation_tables.py build_table4_analogue.py
build_diagnosis_figure.py build_sparse_decision_figure.py validate_conditional_and_beta_cdf.py
compare_zhang_probabilities_quick.py compare_table9_mine.py table9_semantic.py
compare_three_methods_fire.py test_train_preprocessing_artifacts.py
test_merged_dataset_split_properties.py rebuild_1982_2006.py add_findings_1982_2006.py
update_engine_dataset.py show_total_flights.py conftest.py
tree_demo.py recreate_easiest_examples.py verify_diagnosis_separation.py
exact_filter_validation.py confidence_calibration.py all_tables_tiered.py
parser_flaws_and_paraphrases.py codes_side_by_side.py narrative_negative_control.py
bn_build_ours.py bn_upgraded_full.py bn_upgraded_envelope.py bn_variance_envelope.py
bn_posterior_parity.py bn_full_comparison.py bn_three_way_tables.py
bn_zhang_injury_derivation.py all_tables_exact.py
""".split()

EMBED_TESTS = """
phase3_diagnose.py test_chain_rule_diagnosis.py qc_common.py
compare_lanes_fire.py convergence_to_zhang.py query_conditioning_validation.py
query_robustness.py run_fire_query.py close_loep_gap.py calibration_analysis.py
gating_validation.py head_to_head_diagnosis.py headtohead_scores.py
improve_narrative_diagnosis.py diagnose_demo.py ltp_zhang_recovery.py
loo_accuracy.py loo_lift.py loo_selective.py loo_specific.py derive_margin_threshold.py
00_embedding_validation.py 01_validate_clusters.py 02_incident_similarity_network.py
02b_query_similarity_validation.py 03_clustering_comparison.py 04_optimal_k_sweep.py
05_cause_normalization_proof.py 06_transition_probability_investigation.py
build_window_index_files.py embed_missing_window.py realign_window_vocab.py
test_diagnosis.py analyze_data_distribution.py explore_probabilities.py
compare_retrieval_zhang_denom.py verify_full_population.py
""".split()
FROZEN_TESTS = [t for t in FROZEN_TESTS if t not in EMBED_TESTS]

LLM_TESTS = """
check_anthropic.py llm_all_tables.py llm_direct_probabilities.py
llm_flaws_and_paraphrases.py llm_heldout_eval.py llm_model_ladder.py llm_narrative_eval.py
llm_only_identity.py llm_paraphrase_robustness.py llm_paraphrase_upgrade.py
llm_recode_analysis.py llm_recode_dataset.py paraphrase_bottleneck.py probe_claude_error.py
catchall_decomposition.py
""".split()

STRUCT_TESTS = ["test_struct_score_v2.py"]
ARCHIVE_TESTS = "quick_test.py quick_check.py quick_check_logic.py debug_map.py lift_check.py zhang_comparison_output.txt".split()

RENAMES = {
    REPO / "streamlit_tree_app.py": FROZEN / "apps" / "frozenbn_streamlit_diagnosis_prognosis_demo.py",
    REPO / "streamlit_app.py": EMBED / "apps" / "embed_streamlit_similarity_diagnosis_app.py",
    REPO / "tests/maha_narrative_demo.py": FROZEN / "tests" / "frozenbn_maha_narrative_evidence_ladder_demo.py",
    REPO / "tests/combined_parser_validation.py": FROZEN / "tests" / "frozenbn_tiered_parser_validation_11_scenarios.py",
    REPO / "tests/headtohead_scores.py": EMBED / "tests" / "embed_headtohead_retrieval_vs_zhang_counting.py",
    REPO / "tests/heldout_narrative_bn_eval.py": FROZEN / "tests" / "frozenbn_heldout_narrative_bn_eval.py",
}


def ensure_dirs():
    for base in (FROZEN, LLM, STRUCT, EMBED, ZHANG, BNCPT, SHARED, ARCHIVE):
        for sub in APPROACH_SUBS:
            (base / sub).mkdir(parents=True, exist_ok=True)
    (ARCHIVE / "scripts_legacy" / "tests").mkdir(parents=True, exist_ok=True)
    (SHARED / "code").mkdir(parents=True, exist_ok=True)
    (SHARED / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (SHARED / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (SHARED / "data" / "preprocessing").mkdir(parents=True, exist_ok=True)
    (SHARED / "reference").mkdir(parents=True, exist_ok=True)
    (SHARED / "evaluation").mkdir(parents=True, exist_ok=True)


def safe_move(src: Path, dst: Path):
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.move(str(src), str(dst))


def move_file(src: Path, dst: Path):
    if src.is_file():
        safe_move(src, dst)


def move_tree(src: Path, dst: Path):
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.move(str(src), str(dst))


def phase_a():
    ensure_dirs()

    # shared/code
    for fn in ("config.py", "main_app.py", "zhang_diagnosis.py"):
        move_file(REPO / fn, SHARED / "code" / fn)

    # shared/data
    move_tree(REPO / "data" / "raw", SHARED / "data" / "raw")
    move_tree(REPO / "data" / "processed", SHARED / "data" / "processed")
    move_tree(REPO / "data" / "preprocessing", SHARED / "data" / "preprocessing")
    move_tree(REPO / "evaluation", SHARED / "evaluation")

    # Frozen-BN code
    for fn in ("query_to_bn.py", "llm_evidence.py", "prognosis.py", "sparse_cpt.py", "ltp_zhang.py", "trees.py", "kill_pid.py"):
        move_file(REPO / fn, FROZEN / "code" / fn)
    move_file(REPO / "tests" / "bn_upgraded.py", FROZEN / "code" / "bn_upgraded.py")

    # Frozen-BN tests
    for fn in FROZEN_TESTS:
        move_file(REPO / "tests" / fn, FROZEN / "tests" / fn)

    for fn in LLM_TESTS:
        move_file(REPO / "tests" / fn, LLM / "tests" / fn)

    for fn in EMBED_TESTS:
        move_file(REPO / "tests" / fn, EMBED / "tests" / fn)

    for fn in STRUCT_TESTS:
        move_file(REPO / "tests" / fn, STRUCT / "tests" / fn)

    for fn in ARCHIVE_TESTS:
        ext = ".txt" if fn.endswith(".txt") else ".py"
        move_file(REPO / "tests" / fn, ARCHIVE / "scripts_legacy" / fn)

    # embedding html reports
    for fn in REPO.glob("tests/*.html"):
        safe_move(fn, EMBED / "tests" / fn.name)

    # apps
    move_file(REPO / "streamlit_tree_app.py", FROZEN / "apps" / "streamlit_tree_app.py")
    move_file(REPO / "streamlit_app.py", EMBED / "apps" / "streamlit_app.py")
    for fn in ("streamlit_app_1_31_26.py", "streamlit_app_withModernBert.py", "main_app_withModernBert.py", "travis.ipynb"):
        move_file(REPO / fn, EMBED / "code" / fn)

    # whole folders
    move_tree(REPO / "Worked_Examples", STRUCT / "worked_examples")
    move_tree(REPO / "Testing_Structural_Mapping", STRUCT / "_imported_Testing_Structural_Mapping")
    move_tree(REPO / "NTSB_improvements_june_19_2026", BNCPT / "_imported_improvements_june_19")
    move_tree(REPO / "Zhang_Replication_Runner", ZHANG / "scripts")
    move_tree(REPO / "Zhang's Approach 2026", ZHANG / "reference")
    move_tree(REPO / "methodology_audit_june_22_2026", EMBED / "methodology_audit")
    move_tree(REPO / "experiments" / "llm_agent", LLM / "code" / "llm_agent")
    move_tree(REPO / "experiments" / "transformers", EMBED / "code" / "transformers")

    move_tree(REPO / "NTSB_new_project_bundle_2026-03-25", ARCHIVE / "NTSB_new_project_bundle_2026-03-25")
    move_tree(REPO / "Testing_Structural_Mapping_Slides", ARCHIVE / "Testing_Structural_Mapping_Slides")
    move_tree(REPO / "DOCS_NEW_FILE", ARCHIVE / "DOCS_NEW_FILE")

    move_file(REPO / "A2_Results_Presentation.pptx", STRUCT / "presentations_StructMap" / "A2_Results_Presentation.pptx")
    move_file(REPO / "NTSB_shift_to_new_project_approach_2026-03-25.md", EMBED / "docs_Embedding" / "NTSB_shift_to_new_project_approach_2026-03-25.md")
    for fn in ("cursor_plan_get_my_probabilities.md", "cursor_plan_zhang_comparison.md"):
        move_file(REPO / fn, EMBED / "docs_Embedding" / fn)

    # metrics
    move_tree(REPO / "data" / "Testing_Data_Metrics", STRUCT / "metrics")

    # key outputs -> Frozen-BN
    if (REPO / "outputs").is_dir():
        for item in (REPO / "outputs").iterdir():
            if item.name in (".mplcache", "structmap_final_verdict", "structmap_improve", "structmap_refined", "llm_recode", "demo_screenshots"):
                if item.name.startswith("structmap"):
                    safe_move(item, STRUCT / "outputs" / item.name)
                elif item.name == "llm_recode":
                    safe_move(item, LLM / "outputs" / item.name)
                elif item.name == "demo_screenshots":
                    safe_move(item, FROZEN / "outputs" / item.name)
                continue
            if item.is_file():
                name = item.name
                if name.startswith("llm_") or "llm" in name and "heldout_eval_llm" in name:
                    safe_move(item, LLM / "outputs" / name)
                elif name in ("node_name_embeddings.json", "narrative_negative_control.md"):
                    safe_move(item, EMBED / "outputs" / name)
                else:
                    safe_move(item, FROZEN / "outputs" / name)

    # remaining tests -> archive legacy
    tests_dir = REPO / "tests"
    if tests_dir.is_dir():
        for item in list(tests_dir.iterdir()):
            if item.name == "__pycache__":
                shutil.rmtree(item, ignore_errors=True)
                continue
            dest = ARCHIVE / "scripts_legacy" / "tests" / item.name
            safe_move(item, dest)

    # empty data dir
    if (REPO / "data").is_dir() and not any((REPO / "data").iterdir()):
        (REPO / "data").rmdir()

    if (REPO / "experiments").is_dir() and not any((REPO / "experiments").iterdir()):
        (REPO / "experiments").rmdir()

    if (REPO / "outputs").is_dir() and not any((REPO / "outputs").iterdir()):
        (REPO / "outputs").rmdir()

    _move_docs()


def _move_docs():
    docs = REPO / "docs"
    if not docs.is_dir():
        return
    frozen_docs = {
        "PAPER_OUTLINE.md", "TABLE7_FULL_REPRODUCTION.md", "TABLE4_ANALYSIS.md", "TABLE4_OWN_VERSION.md",
        "ZHANG_REPRODUCTION_REPORT.md", "ZHANG_FULL_COVERAGE.md", "Professor_Implementation_Verification.md",
        "Professor_Step_By_Step_Guide.md", "Train_Test_Split_Plan.md", "PRESENTATION_TABLE4_TABLE7.md",
        "PRESENTATION_TABLE7_3SLIDES.md", "ALL_TABLES_EXACT_COMPARISON.md", "DIAGNOSIS_VALIDATION_REPORT.md",
        "Why_Low_Probabilities.md", "Chain_Rule_Implementation_Plan.md", "Diagnosis_Mode_Presentation_Guide.md",
        "Slides_18_19_Chain_Rule_Example.md", "TREES_DESIGN.md", "TREES_VALIDATION_REPORT.md",
        "EASIEST_TREE_RECREATION.md", "paper_writing_guide.md", "paper_writing_guide.docx",
        "paper_revision_checklist.docx", "presentation_table4.csv", "presentation_table7_full.csv",
        "presentation_table7_top12.csv", "table7_full_reproduction.csv", "table7_full_reproduction.xlsx",
        "conditional_and_beta_cdf_validation.json", "query_conditioning_results.json",
        "query_conditioning_validation.png", "query_robustness_results.json", "calibration_results.json",
        "sparse_robustness_results.json", "TERMINOLOGY_UPDATE.md", "exact_filter_validation.json",
        "_july26_extract.txt", "build_paper_revision_checklist.py", "extract_zhang_figures.py",
        "_md_to_docx.py", "build_restructured_draft.py",
    }
    frozen_drafts = {"Draft NTSB Paper RESTRUCTURED 7_22_26.docx", "NTSB Paper 6:4:26.docx"}
    struct_drafts = {
        "structural_mapping_paper_section.docx", "full_paper_v4.docx", "full_paper_v4_clean.docx",
        "full_paper_v4.pdf", "NTSB_Paper_April_20_26.docx", "full_paper_v5_draft.docx",
        "full_paper_v2.tex", "full_paper_v2.pdf",
    }
    frozen_pres = {
        "jesse_jul28_update.pptx", "maha_meeting_update.pptx", "friday_jul17_update.pptx",
        "tuesday_jul14_update.pptx", "tuesday_jul21_update.pptx", "tuesday_update.pptx",
        "friday_update.pptx", "diagnosis_slides.pptx", "diagnosis_slides_core.pptx",
        "diagnosis_slides.md", "friday_slide_bullets.md", "build_jesse_jul28_pptx.py",
        "build_maha_meeting_slides.py", "build_friday_jul17_pptx.py", "build_friday_pptx.py",
        "build_tuesday_jul14_pptx.py", "build_tuesday_jul21_pptx.py", "build_tuesday_pptx.py",
        "build_diagnosis_pptx.py", "build_diagnosis_core_pptx.py",
    }
    for fn in docs.iterdir():
        if fn.name.startswith("~$") or fn.name == ".DS_Store":
            continue
        if fn.name in frozen_drafts:
            safe_move(fn, FROZEN / "paper_drafts" / fn.name.replace(" ", "_"))
        elif fn.name in struct_drafts:
            safe_move(fn, STRUCT / "paper_drafts" / fn.name.replace(" ", "_"))
        elif fn.name in frozen_docs:
            safe_move(fn, FROZEN / "docs_FrozenBN" / fn.name)
        elif fn.name in frozen_pres:
            safe_move(fn, FROZEN / "presentations_FrozenBN" / fn.name)
        elif fn.suffix == ".pdf" and "Analysis" in fn.name:
            safe_move(fn, FROZEN / "presentations_FrozenBN" / fn.name)
        elif fn.name == "BN-NTSB RESS 2021.pdf":
            safe_move(fn, SHARED / "reference" / "BN-NTSB RESS 2021.pdf")
        elif fn.name.startswith("LLM") or fn.name == "ALL_TABLES_TIERED_PARSER.md":
            safe_move(fn, LLM / "docs_LLM" / fn.name)
        elif "structural" in fn.name.lower() or fn.name.startswith("sparse"):
            safe_move(fn, STRUCT / "docs_StructMap" / fn.name)
        elif fn.name in ("HEAD_TO_HEAD_SCORES.md", "head_to_head_scores_results.json", "gating_results.json"):
            safe_move(fn, EMBED / "docs_Embedding" / fn.name)
        elif fn.name == "meeting_walkthrough.html":
            safe_move(fn, STRUCT / "docs_StructMap" / fn.name)
        elif fn.name == "Project_plan.md":
            safe_move(fn, ARCHIVE / "DOCS_NEW_FILE" / fn.name if (ARCHIVE / "DOCS_NEW_FILE").exists() else EMBED / "docs_Embedding" / fn.name)
    if (docs / "figures").is_dir():
        safe_move(docs / "figures", FROZEN / "docs_FrozenBN" / "figures")
    if (docs / "presentations").is_dir():
        for p in (docs / "presentations").rglob("*"):
            if p.is_file():
                if "2026-07-28" in p.name:
                    safe_move(p, FROZEN / "docs_FrozenBN" / p.name)
                elif "2026-04-30" in p.name:
                    safe_move(p, EMBED / "docs_Embedding" / p.name)
                else:
                    safe_move(p, FROZEN / "presentations_FrozenBN" / p.name)
    if (docs / "slide_previews").is_dir():
        safe_move(docs / "slide_previews", FROZEN / "presentations_FrozenBN" / "slide_previews")
    if docs.is_dir() and not any(docs.iterdir()):
        docs.rmdir()


def phase_b_renames():
    for src, dst in RENAMES.items():
        # src may already have been moved
        if not src.exists():
            # try without rename if already at parent dest folder
            continue
        safe_move(src, dst)


def main():
    print("Phase A: structure...")
    phase_a()
    print("Phase B: renames...")
    phase_b_renames()
    print("Done. Run Phase C setup (config, pytest.ini, docs) separately.")


if __name__ == "__main__":
    main()

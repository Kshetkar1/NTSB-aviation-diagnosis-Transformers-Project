"""Paths for Testing_Structural_Mapping (repo root = parents[2] from scripts/)."""

from __future__ import annotations

from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
MAPPING_ROOT = _SCRIPTS.parent
PROJECT_ROOT = MAPPING_ROOT.parent

SCHEMA_DIR = MAPPING_ROOT / "schema"
DEFAULT_PROMPT_PATH = SCHEMA_DIR / "extraction_prompt_v1.txt"
EXTRACTION_PROMPT_V2_PATH = SCHEMA_DIR / "extraction_prompt_v2.txt"
DEFAULT_STRUCT_CACHE_PATH = MAPPING_ROOT / "cache" / "struct_train_v1.jsonl"
DEFAULT_QUERY_CACHE_PATH = MAPPING_ROOT / "cache" / "query_struct_v1.jsonl"
# A2: relational causal_chain + struct_score_v2 (separate caches from v1)
DEFAULT_STRUCT_CACHE_V2_PATH = MAPPING_ROOT / "cache" / "struct_train_v2.jsonl"
DEFAULT_QUERY_CACHE_V2_PATH = MAPPING_ROOT / "cache" / "query_struct_v2.jsonl"

TESTING_METRICS_ROOT = PROJECT_ROOT / "data" / "Testing_Data_Metrics"
TRAIN_IDS_PATH = TESTING_METRICS_ROOT / "splits" / "train_ev_ids.txt"
TEST_IDS_PATH = TESTING_METRICS_ROOT / "splits" / "test_ev_ids.txt"
EVAL_OUTPUT_DIR = MAPPING_ROOT / "outputs"

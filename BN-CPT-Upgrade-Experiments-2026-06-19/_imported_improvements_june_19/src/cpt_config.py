"""Paths and Zhang reference values for the empirical CPT work.

Self-contained: all inputs are vendored under this folder's ``data/`` directory.
No imports or paths reach outside ``NTSB_improvements_june_19_2026/``.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Vendored inputs (see data/ and README).
REFINED_DATA_PATH = DATA_DIR / "refined_dataset.json"
STRUCT_CACHE_PATH = DATA_DIR / "struct_cache_v2.jsonl"
ZHANG_TABLE9_GT_PATH = DATA_DIR / "zhang_table9_ground_truth.json"
ENGINE_VS_ZHANG_PATH = DATA_DIR / "engine_vs_zhang_table9.json"
ZHANG_RECREATION_TARGETS_PATH = DATA_DIR / "zhang_recreation_targets.json"

# Zhang Table 4 — P(x3=1 | brake wear, electrical overheat). PEDAGOGICAL example
# from the paper's Fig 2 (NOT part of the built NTSB.xdsl; Fire's only real parent
# there is the anti-ice node at 0.95). Kept as the honest "why it can't match" exhibit.
ZHANG_TABLE4 = {
    ("yes", "yes"): 0.99,
    ("yes", "no"): 0.93,
    ("no", "yes"): 0.95,
    ("no", "no"): 2e-9,
}

# Zhang Table 5 — P(x4=1 | fire). x4 proxy = substantial/destroyed damage.
ZHANG_TABLE5 = {
    "yes": 0.92,
    "no": 0.0,
}

TIER_B_MIN_N = 10
TIER_B_MAX_ABS_GAP = 0.20

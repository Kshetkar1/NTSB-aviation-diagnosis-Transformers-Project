"""
Configuration file for the Aviation Incident Diagnosis Engine.

Paths are relative to the repository root (parent of shared/).
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    _REPO_FOR_ENV = Path(__file__).resolve().parents[2]
    load_dotenv(_REPO_FOR_ENV / ".env")
except ImportError:
    pass

# --- Repository layout (post-reorg) ---
REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = REPO_ROOT / "shared"
DATA_DIR = SHARED_DIR / "data" / "processed"
FROZEN_BN_DIR = REPO_ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20"

# Zhang replication reference (vendored paper assets)
ZHANG_REF_DIR = REPO_ROOT / "Zhang-Replication-Foundation-2026-06-04" / "reference"
ZHANG_BTS_XLSX = ZHANG_REF_DIR / "data" / "table_01_37_061019.xlsx"
ZHANG_METADATA_XLSX = ZHANG_REF_DIR / "data" / "metaData.xlsx"
ZHANG_XDSL = ZHANG_REF_DIR / "NTSB.xdsl"
DOCS_FROZEN_DIR = FROZEN_BN_DIR / "docs_FrozenBN"

# Backward-compatible alias used across scripts
PROJECT_ROOT = REPO_ROOT

# --- OpenAI API Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_openai_api_key():
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it using: export OPENAI_API_KEY='your-key-here' "
            "or create a .env file at the repo root."
        )
    return OPENAI_API_KEY


LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- Data File Paths ---
MERGED_DATA_PATH = DATA_DIR / "merged_dataset.json"
_refined_json = DATA_DIR / "refined_dataset.json"
REFINED_DATA_PATH = _refined_json if _refined_json.is_file() else MERGED_DATA_PATH
TRAIN_MERGED_DATA_PATH = DATA_DIR / "merged_dataset_train.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
EMBEDDINGS_MAP_PATH = DATA_DIR / "embeddings_map.json"
WINDOW_DATA_PATH = DATA_DIR / "refined_dataset_1982_2006.json"
WINDOW_EMBEDDINGS_PATH = DATA_DIR / "embeddings_1982_2006.npy"
WINDOW_EMBEDDINGS_MAP_PATH = DATA_DIR / "embeddings_map_1982_2006.json"
CAUSE_STATS_PATH = DATA_DIR / "cause_statistics.json"
EMBEDDINGS_TRAIN_PATH = DATA_DIR / "embeddings_train.npy"
EMBEDDINGS_MAP_TRAIN_PATH = DATA_DIR / "embeddings_map_train.json"
CAUSE_STATS_TRAIN_PATH = DATA_DIR / "cause_statistics_train.json"
EMBEDDINGS_CHECKPOINT_TRAIN_PATH = DATA_DIR / "embeddings_checkpoint_train.npz"


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


USE_TRAIN_INDEX = _env_truthy("NTSB_USE_TRAIN_INDEX")
USE_FULL_CORPUS = _env_truthy("NTSB_FULL_CORPUS")
_window_ready = (
    WINDOW_DATA_PATH.is_file()
    and WINDOW_EMBEDDINGS_PATH.is_file()
    and WINDOW_EMBEDDINGS_MAP_PATH.is_file()
)
USE_ZHANG_WINDOW = _window_ready and not USE_FULL_CORPUS and not USE_TRAIN_INDEX

APPLY_CALIBRATION = not _env_truthy("NTSB_NO_CALIBRATION")
GATE_DIAGNOSIS_BY_DEFAULT = not _env_truthy("NTSB_NO_GATING")

# When True (default), never use stated injury/damage phrases or embed them for
# retrieval — prevents outcome-in-text leakage on severity prediction.
LEAK_SAFE_SEVERITY = not _env_truthy("NTSB_ALLOW_STATED_SEVERITY")

CALIBRATION_RESULTS_PATH = (
    FROZEN_BN_DIR / "docs_FrozenBN" / "calibration_results.json"
)
CALIBRATION_TEMPERATURE_FALLBACK = 0.473


def _load_calibration_temperature() -> float:
    try:
        import json
        data = json.loads(CALIBRATION_RESULTS_PATH.read_text(encoding="utf-8"))
        return float(data["recalibration"]["cond"]["temperature"]["T"])
    except Exception:
        return CALIBRATION_TEMPERATURE_FALLBACK


CALIBRATION_TEMPERATURE = _load_calibration_temperature()

if USE_TRAIN_INDEX:
    ACTIVE_INCIDENT_DATA_PATH = TRAIN_MERGED_DATA_PATH
    ACTIVE_EMBEDDINGS_PATH = EMBEDDINGS_TRAIN_PATH
    ACTIVE_EMBEDDINGS_MAP_PATH = EMBEDDINGS_MAP_TRAIN_PATH
    ACTIVE_INDEX_LABEL = "TRAIN-ONLY index (NTSB_USE_TRAIN_INDEX=1)"
elif USE_ZHANG_WINDOW:
    ACTIVE_INCIDENT_DATA_PATH = WINDOW_DATA_PATH
    ACTIVE_EMBEDDINGS_PATH = WINDOW_EMBEDDINGS_PATH
    ACTIVE_EMBEDDINGS_MAP_PATH = WINDOW_EMBEDDINGS_MAP_PATH
    ACTIVE_INDEX_LABEL = "Zhang window 1982-2006 (set NTSB_FULL_CORPUS=1 for full corpus)"
else:
    ACTIVE_INCIDENT_DATA_PATH = REFINED_DATA_PATH
    ACTIVE_EMBEDDINGS_PATH = EMBEDDINGS_PATH
    ACTIVE_EMBEDDINGS_MAP_PATH = EMBEDDINGS_MAP_PATH
    ACTIVE_INDEX_LABEL = "full corpus 1982-2019"

OUTPUT_DIR = FROZEN_BN_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def validate_paths():
    required_files = [REFINED_DATA_PATH, EMBEDDINGS_PATH, EMBEDDINGS_MAP_PATH]
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Required data files not found: {[str(f) for f in missing_files]}. "
            f"Please run the data processing scripts first."
        )

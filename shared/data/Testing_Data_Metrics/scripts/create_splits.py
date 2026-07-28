"""
Create train/test ev_id splits + train-only merged JSON for strict diagnosis eval.

Reads REFINED_DATA_PATH from config (refined_dataset.json if present, else merged_dataset.json).
Also writes a copy to data/processed/merged_dataset_train.json so
2_generate_embeddings.py --train and 2b_precompute_diagnostic_data.py --train work unchanged.

Run from repo root:
    python data/Testing_Data_Metrics/scripts/create_splits.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, REFINED_DATA_PATH, TRAIN_MERGED_DATA_PATH  # noqa: E402

TESTING_ROOT = PROJECT_ROOT / "data" / "Testing_Data_Metrics"
SPLITS_DIR = TESTING_ROOT / "splits"
TRAIN_DATA_DIR = TESTING_ROOT / "train_data"

TRAIN_TXT = SPLITS_DIR / "train_ev_ids.txt"
TEST_TXT = SPLITS_DIR / "test_ev_ids.txt"
TRAIN_JSON_TESTING = TRAIN_DATA_DIR / "merged_dataset_train.json"


def _non_empty(s: str | None) -> bool:
    return bool(s and str(s).strip())


def eligible_ev_ids(dataset: dict) -> list[str]:
    out: list[str] = []
    for ev_id, inc in dataset.items():
        findings = inc.get("findings") or []
        has_c = any(
            (f.get("Cause_Factor") or "").strip() == "C"
            for f in findings
            if isinstance(f, dict)
        )
        accp = inc.get("narr_accp") or ""
        accf = inc.get("narr_accf") or ""
        if not has_c:
            continue
        if not (_non_empty(accp) or _non_empty(accf)):
            continue
        out.append(ev_id)
    return out


def main() -> None:
    if not REFINED_DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {REFINED_DATA_PATH}")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(REFINED_DATA_PATH, "r", encoding="utf-8") as f:
        full = json.load(f)

    eligible = eligible_ev_ids(full)
    eligible.sort()
    random.seed(42)
    eligible_shuffled = eligible.copy()
    random.shuffle(eligible_shuffled)

    n = len(eligible_shuffled)
    n_train = int(0.7 * n)
    train_ids = sorted(eligible_shuffled[:n_train])
    test_ids = sorted(eligible_shuffled[n_train:])

    train_json = {eid: full[eid] for eid in train_ids}

    TRAIN_TXT.write_text("\n".join(train_ids) + ("\n" if train_ids else ""), encoding="utf-8")
    TEST_TXT.write_text("\n".join(test_ids) + ("\n" if test_ids else ""), encoding="utf-8")

    with open(TRAIN_JSON_TESTING, "w", encoding="utf-8") as f:
        json.dump(train_json, f, ensure_ascii=False, indent=2)

    # Mirror to data/processed for 2_generate_embeddings.py --train / 2b --train
    TRAIN_MERGED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_MERGED_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(train_json, f, ensure_ascii=False, indent=2)

    print(f"Source dataset: {REFINED_DATA_PATH}")
    print(f"Total incidents in file: {len(full)}")
    print(f"Eligible (≥1 Cause_Factor=C and narr_accp or narr_accf): {len(eligible)}")
    print(f"Train: {len(train_ids)}  |  Test: {len(test_ids)}")
    print(f"Wrote: {TRAIN_TXT}")
    print(f"Wrote: {TEST_TXT}")
    print(f"Wrote: {TRAIN_JSON_TESTING}")
    print(f"Wrote (mirror for preprocessing): {TRAIN_MERGED_DATA_PATH}")


if __name__ == "__main__":
    main()

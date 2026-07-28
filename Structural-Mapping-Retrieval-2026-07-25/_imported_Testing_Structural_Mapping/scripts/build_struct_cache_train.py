#!/usr/bin/env python3
"""
Build JSONL cache of incident_struct_v1 for train ev_ids (OpenAI).

Reads config.TRAIN_MERGED_DATA_PATH (no need for NTSB_USE_TRAIN_INDEX).
Default id list: data/Testing_Data_Metrics/splits/train_ev_ids.txt

Usage:
  cd <repo_root>
  export OPENAI_API_KEY=...
  python Testing_Structural_Mapping/scripts/build_struct_cache_train.py --limit 50
  python Testing_Structural_Mapping/scripts/build_struct_cache_train.py --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS.parent.parent))

from config import TRAIN_MERGED_DATA_PATH, get_openai_api_key, LLM_MODEL  # noqa: E402
from openai import OpenAI  # noqa: E402

from paths import (  # noqa: E402
    DEFAULT_PROMPT_PATH,
    DEFAULT_STRUCT_CACHE_PATH,
    TRAIN_IDS_PATH,
)
from extract_struct import (  # noqa: E402
    extract_struct_from_incident,
    load_prompt_template,
)
from io_cache import append_jsonl, iter_existing_ev_ids  # noqa: E402


def load_train_ids(path: Path, dataset_keys: set[str]) -> list[str]:
    if path.is_file():
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
        return [x for x in lines if x in dataset_keys]
    return sorted(dataset_keys)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache incident_struct_v1 for train ev_ids.")
    ap.add_argument("--output", type=Path, default=DEFAULT_STRUCT_CACHE_PATH)
    ap.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    ap.add_argument("--id-list", type=Path, default=TRAIN_IDS_PATH, help="Text file of ev_ids")
    ap.add_argument("--limit", type=int, default=None, help="Max new extractions this run")
    ap.add_argument("--resume", action="store_true", help="Skip ev_ids already in output JSONL")
    ap.add_argument("--model", default=LLM_MODEL)
    args = ap.parse_args()

    if not TRAIN_MERGED_DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing train merged JSON: {TRAIN_MERGED_DATA_PATH}")

    data = json.loads(TRAIN_MERGED_DATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Train merged dataset must be a JSON object keyed by ev_id")

    ids = load_train_ids(args.id_list, set(data.keys()))
    if args.resume:
        done = iter_existing_ev_ids(args.output)
        before = len(ids)
        ids = [i for i in ids if i not in done]
        print(f"[resume] skipped {before - len(ids)} already cached; {len(ids)} remaining")

    if args.limit is not None:
        ids = ids[: args.limit]

    tmpl = load_prompt_template(args.prompt)
    client = OpenAI(api_key=get_openai_api_key())
    ok = err = 0
    for i, eid in enumerate(ids, 1):
        inc = data.get(eid)
        if not inc:
            err += 1
            append_jsonl(
                args.output,
                {"ev_id": eid, "struct": None, "error": "missing in train JSON"},
            )
            continue
        try:
            st = extract_struct_from_incident(client, args.model, inc, tmpl)
            append_jsonl(args.output, {"ev_id": eid, "struct": st, "error": None})
            ok += 1
            print(f"[{i}/{len(ids)}] ok {eid}", flush=True)
        except Exception as ex:  # noqa: BLE001
            err += 1
            append_jsonl(
                args.output,
                {"ev_id": eid, "struct": None, "error": str(ex)},
            )
            print(f"[{i}/{len(ids)}] ERR {eid}: {ex}", flush=True)

    print(f"Done. ok={ok} err={err} -> {args.output}")


if __name__ == "__main__":
    main()

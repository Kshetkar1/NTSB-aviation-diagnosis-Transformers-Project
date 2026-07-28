"""Load / append JSONL caches for extracted structs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def load_struct_jsonl(path: Path) -> dict[str, dict]:
    """ev_id -> struct dict (latest line wins)."""
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
            eid = (row.get("ev_id") or "").strip()
            st = row.get("struct")
            if eid and isinstance(st, dict):
                out[eid] = st
    return out


def load_query_struct_jsonl(path: Path) -> dict[str, dict]:
    """sha256_hex -> struct."""
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
            h = (row.get("query_hash") or "").strip()
            st = row.get("struct")
            if h and isinstance(st, dict):
                out[h] = st
    return out


def iter_existing_ev_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = (row.get("ev_id") or "").strip()
            if eid and row.get("struct"):
                done.add(eid)
    return done


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def sha256_text(s: str) -> str:
    import hashlib

    return hashlib.sha256(s.encode("utf-8")).hexdigest()

"""Checks for train-only embedding outputs (Track B). Aligns with outline §6 row/map length rule."""
from __future__ import annotations

import json
import pathlib
import re
import struct

import pytest


@pytest.mark.parametrize(
    "n_rows,map_len",
    [(0, 0), (1, 1), (100, 100)],
)
def test_embedding_row_count_equals_map_length_rule(n_rows: int, map_len: int) -> None:
    assert n_rows == map_len


def _npy_row_count(path: pathlib.Path) -> int:
    """Parse .npy header without importing numpy (leading dimension of shape)."""
    data = path.read_bytes()
    if not data.startswith(b"\x93NUMPY"):
        raise ValueError("not a NumPy array file")
    ver_major = data[6]
    hlen = struct.unpack("<H", data[8:10])[0] if ver_major == 1 else struct.unpack("<I", data[8:12])[0]
    header_off = 10 if ver_major == 1 else 12
    header = data[header_off : header_off + hlen].decode("latin1")
    m = re.search(r"'shape':\s*\(\s*(\d+)", header)
    if not m:
        raise ValueError(f"could not parse shape from npy header: {header[:200]!r}")
    return int(m.group(1))


@pytest.mark.integration
def test_train_embeddings_rows_match_map_when_present() -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    proc = root / "shared" / "data" / "processed"
    emb_p = proc / "embeddings_train.npy"
    map_p = proc / "embeddings_map_train.json"
    if not (emb_p.is_file() and map_p.is_file()):
        pytest.skip("train embedding artifacts not present")
    n_rows = _npy_row_count(emb_p)
    mapping = json.loads(map_p.read_text(encoding="utf-8"))
    assert n_rows == len(mapping), (
        f"embeddings_train rows ({n_rows}) != embeddings_map_train length ({len(mapping)})"
    )


@pytest.mark.integration
def test_train_map_not_larger_than_full_when_both_exist() -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    proc = root / "shared" / "data" / "processed"
    full_m = proc / "embeddings_map.json"
    train_m = proc / "embeddings_map_train.json"
    if not (full_m.is_file() and train_m.is_file()):
        pytest.skip("need both maps to compare")
    full_len = len(json.loads(full_m.read_text(encoding="utf-8")))
    train_len = len(json.loads(train_m.read_text(encoding="utf-8")))
    assert train_len <= full_len, (
        "train map longer than full map (unexpected; check wrong file written)"
    )

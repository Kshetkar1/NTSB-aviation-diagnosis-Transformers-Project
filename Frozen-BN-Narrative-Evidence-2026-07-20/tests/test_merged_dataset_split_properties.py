"""
Property tests for train/test splits and train-only merged incident JSON.

Terminology: merged dataset = consolidated per-ev_id record (at data/processed/merged_dataset.json).
Fast tests run without the NTSB corpus; integration tests use on-disk split files when present.
"""
from __future__ import annotations

import json
import pathlib

import pytest


def assert_disjoint_train_test(train_ids: list[str], test_ids: list[str]) -> None:
    overlap = set(train_ids) & set(test_ids)
    assert not overlap, f"train and test overlap: {overlap}"


def assert_train_only_subset_of_full(
    full_merged: dict,
    train_only_merged: dict,
    test_ids: set[str],
) -> None:
    train_keys = set(train_only_merged.keys())
    assert train_keys.isdisjoint(test_ids), "train JSON contains a test ev_id key"
    assert train_keys <= set(full_merged.keys()), "train JSON has unknown ev_id vs full merged"
    for ev_id, payload in train_only_merged.items():
        assert full_merged[ev_id] == payload, f"payload mismatch for {ev_id}"


def assert_no_duplicate_split_entries(lines: list[str], *, context: str) -> None:
    seen: set[str] = set()
    dupes: list[str] = []
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        if s in seen:
            dupes.append(s)
        seen.add(s)
    assert not dupes, f"{context}: duplicate ev_ids: {dupes}"


def parse_ev_id_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


@pytest.mark.parametrize(
    "train,test",
    [
        (["a", "b"], ["c"]),
        (["ev1"], ["ev2", "ev3"]),
        ([str(i) for i in range(50)], [str(i) for i in range(50, 80)]),
        ([], ["only_test"]),
    ],
)
def test_train_test_disjoint_parametrized(train: list[str], test: list[str]) -> None:
    assert_disjoint_train_test(train, test)


@pytest.mark.parametrize(
    "full_keys,train_keys,test_keys",
    [
        ({"1", "2", "3"}, {"1", "2"}, {"3"}),
        ({"a", "b"}, {"a"}, {"b"}),
        ({str(i) for i in range(100)}, {str(i) for i in range(60)}, {str(i) for i in range(60, 100)}),
    ],
)
def test_train_only_merged_subset_parametrized(
    full_keys: set[str],
    train_keys: set[str],
    test_keys: set[str],
) -> None:
    assert train_keys.isdisjoint(test_keys)
    assert train_keys <= full_keys
    assert test_keys <= full_keys
    full_merged = {k: {"ev_id": k, "narr_accp": f"n-{k}"} for k in full_keys}
    train_only = {k: full_merged[k] for k in train_keys}
    assert_train_only_subset_of_full(full_merged, train_only, test_keys)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("ev1\nev2\n", ["ev1", "ev2"]),
        ("  ev1  \n\n ev2 \n", ["ev1", "ev2"]),
        ("", []),
        ("single", ["single"]),
    ],
)
def test_parse_ev_id_lines_parametrized(text: str, expected: list[str]) -> None:
    assert parse_ev_id_lines(text) == expected


def test_duplicate_detection_flags_problem() -> None:
    with pytest.raises(AssertionError):
        assert_no_duplicate_split_entries(["a", "b", "a"], context="train file")


@pytest.mark.parametrize(
    "lines",
    [
        ["x", "y", "z"],
        ["same", "other"],
    ],
)
def test_no_duplicates_ok(lines: list[str]) -> None:
    assert_no_duplicate_split_entries(lines, context="ok")


@pytest.mark.integration
def test_on_disk_splits_disjoint_when_present() -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    splits = root / "shared" / "data" / "Testing_Data_Metrics" / "splits"
    train_p = splits / "train_ev_ids.txt"
    test_p = splits / "test_ev_ids.txt"
    if not (train_p.is_file() and test_p.is_file()):
        pytest.skip("split files not present")
    train_ids = parse_ev_id_lines(train_p.read_text(encoding="utf-8"))
    test_ids = parse_ev_id_lines(test_p.read_text(encoding="utf-8"))
    assert_no_duplicate_split_entries(train_ids, context="train_ev_ids.txt")
    assert_no_duplicate_split_entries(test_ids, context="test_ev_ids.txt")
    assert_disjoint_train_test(train_ids, test_ids)


@pytest.mark.integration
def test_train_only_json_matches_full_when_present() -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    splits = root / "shared" / "data" / "Testing_Data_Metrics" / "splits"
    full_p = root / "shared" / "data" / "processed" / "merged_dataset.json"
    train_p = root / "shared" / "data" / "processed" / "merged_dataset_train.json"
    test_ids_p = splits / "test_ev_ids.txt"
    if not (full_p.is_file() and train_p.is_file() and test_ids_p.is_file()):
        pytest.skip("full merged, train-only merged, or test split not present")
    full_merged = json.loads(full_p.read_text(encoding="utf-8"))
    train_only = json.loads(train_p.read_text(encoding="utf-8"))
    test_ids = set(parse_ev_id_lines(test_ids_p.read_text(encoding="utf-8")))
    assert_train_only_subset_of_full(full_merged, train_only, test_ids)
    train_list_p = splits / "train_ev_ids.txt"
    if train_list_p.is_file():
        train_ids = parse_ev_id_lines(train_list_p.read_text(encoding="utf-8"))
        assert set(train_only.keys()) == set(train_ids), (
            "train JSON keys must match train_ev_ids.txt exactly"
        )

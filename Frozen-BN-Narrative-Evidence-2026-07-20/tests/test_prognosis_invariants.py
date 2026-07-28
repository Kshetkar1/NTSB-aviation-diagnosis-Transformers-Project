"""
Sanity checks for query-weighted prognosis (no network if DATA_LOADED false).

Run: pytest tests/test_prognosis_invariants.py -q
Full check (needs OPENAI_API_KEY + data): pytest tests/test_prognosis_invariants.py -q -m integration
"""

from __future__ import annotations

import math
import os
import pytest

# Optional integration tests
integration = pytest.mark.skipif(
    os.getenv("RUN_PROGNOSIS_INTEGRATION") != "1",
    reason="set RUN_PROGNOSIS_INTEGRATION=1 to run API-backed checks",
)


def test_normalize_occurrence_text_stable():
    import main_app

    assert main_app._normalize_occurrence_text("  A  B  ") == "a b"
    assert main_app._normalize_occurrence_text("") == ""


def test_sequence_prefix_matches():
    import main_app

    assert main_app._sequence_prefix_matches(["a", "b"], ["a"])
    assert not main_app._sequence_prefix_matches(["a"], ["a", "b"])


@integration
def test_predict_future_events_step1_probabilities_sum_to_one():
    import main_app

    if not main_app.DATA_LOADED:
        pytest.skip("knowledge base not loaded")

    os.environ.pop("NTSB_USE_TRAIN_INDEX", None)
    r = main_app.predict_future_events(
        "engine fire during takeoff", top_n_incidents=25, max_chain_steps=1
    )
    if r.get("error") or not r.get("future_events"):
        pytest.skip("no prognosis data for query")

    s = sum(float(x["probability"]) for x in r["future_events"])
    assert abs(s - 1.0) < 1e-5, f"step-1 probs sum to {s}, expected 1.0"

    for x in r["future_events"]:
        assert 0.0 <= x["probability"] <= 1.0


@integration
def test_evidence_ids_unique_per_outcome():
    import main_app

    if not main_app.DATA_LOADED:
        pytest.skip("knowledge base not loaded")

    os.environ.pop("NTSB_USE_TRAIN_INDEX", None)
    r = main_app.predict_future_events(
        "loss of engine power", top_n_incidents=30, max_chain_steps=1
    )
    for row in r.get("future_events") or []:
        ev = row.get("evidence") or []
        assert len(ev) == len(set(ev)), f"duplicate ev_id in evidence: {ev}"

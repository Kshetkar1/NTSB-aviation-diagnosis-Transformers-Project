"""Unit tests for causal-chain alignment (struct_score_v2)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "Testing_Structural_Mapping" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from struct_score_v2 import structural_similarity  # noqa: E402


def _step(role: str, system: str = "engine_mechanical", mechanism: str = "fatigue") -> dict:
    return {
        "step": 1,
        "element": "x",
        "role": role,
        "system": system,
        "mechanism": mechanism,
    }


def test_empty_chains_identity():
    a = {"causal_chain": [], "contributing_factors": [], "failure_pattern": "unknown"}
    assert structural_similarity(a, a) == 1.0


def test_identical_nonempty_chains_high_score():
    chain = [
        _step("initiating_event"),
        _step("propagation", "lubrication", "contamination"),
        _step("system_failure"),
    ]
    for i, s in enumerate(chain):
        s["step"] = i + 1
    st = {
        "causal_chain": chain,
        "contributing_factors": [{"element": "oil filter", "role": "latent", "system": "lubrication"}],
        "failure_pattern": "material_degradation_cascade",
    }
    sim = structural_similarity(st, st)
    assert sim >= 0.85


def test_disjoint_chains_lower_than_identity():
    a = {
        "causal_chain": [_step("initiating_event", "fuel", "deprivation")],
        "contributing_factors": [],
        "failure_pattern": "unknown",
    }
    b = {
        "causal_chain": [_step("outcome", "structural", "terrain_collision")],
        "contributing_factors": [],
        "failure_pattern": "unknown",
    }
    assert structural_similarity(a, b) < structural_similarity(a, a)

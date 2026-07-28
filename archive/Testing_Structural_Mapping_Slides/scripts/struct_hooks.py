"""Structural reweighting hook for main_app score_adjust_fn."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from reweight import reweighted_incident_score
from struct_score import struct_similarity as struct_similarity_v1


def _to_scalar(sim_out: float | tuple[float, Any]) -> float:
    if isinstance(sim_out, tuple):
        return float(sim_out[0])
    return float(sim_out)


def make_score_adjust_fn(
    query_struct: dict | None,
    struct_by_eid: dict[str, dict],
    alpha: float,
    *,
    similarity_fn: Callable[[dict, dict], float | tuple[float, Any]] | None = None,
) -> Callable[[float, dict], float]:
    """
    Compatible with main_app.diagnose_with_conditional_probabilities /
    predict_future_events score_adjust_fn.

    similarity_fn: defaults to v1 enum+jaccard (struct_similarity). Pass
    struct_score_v2.structural_similarity for A2 (causal chain alignment).
    """

    def fn(cosine_score: float, match: dict) -> float:
        if match.get("source") != "incident":
            return float(cosine_score)
        eid = match.get("ev_id")
        if not eid or not query_struct:
            return float(cosine_score)
        cand = struct_by_eid.get(eid)
        if not cand:
            return float(cosine_score)
        if similarity_fn is not None:
            sim = _to_scalar(similarity_fn(query_struct, cand))
        else:
            sim = _to_scalar(struct_similarity_v1(query_struct, cand))
        return reweighted_incident_score(cosine_score, sim, alpha)

    return fn

"""
Slide-formula variant of structural similarity for the A2 reranker.

This file lives only in Testing_Structural_Mapping_Slides/. It implements
the four formulas exactly as drawn on the slide deck (Formula 1-Formula 3),
with no factor-overlap term and no failure-pattern bonus. Formula 4
(score fusion with embeddings) is applied downstream in reweight.py.

Slide formulas:

  F1: sim(A, B) = 0.50 * role + 0.30 * mechanism + 0.20 * system
  F2: NW recurrence with constant gap_penalty = -0.05
  F3: chain_score = (strong + 0.5 * partial) / total
                    - 0.10 * (unmapped / total)
       where:
         strong   = aligned pairs with sim >= 0.70
         partial  = aligned pairs with 0.35 <= sim < 0.70
         unmapped = positions in either chain with no counterpart
         total    = len(chain_a) + len(chain_b)

The interface (function names, signatures) is kept identical to the
original struct_score_v2.py so that struct_hooks.make_score_adjust_fn
and eval_diagnosis_structural.py can use it as a drop-in replacement.
"""

from __future__ import annotations

import math
from typing import Any

# Adjacency tables are reused from the original implementation. The slide
# formula references "role", "mechanism", and "system" component scores
# without prescribing how unequal labels score, so we keep the same
# adjacency / compatibility lookups for fairness in the comparison.
ADJACENT_ROLES = {
    ("initiating_event", "propagation"): 1.0,
    ("propagation", "system_compromise"): 0.95,
    ("system_compromise", "system_failure"): 0.95,
    ("system_failure", "terminal_failure"): 0.9,
    ("terminal_failure", "operational_consequence"): 0.85,
    ("operational_consequence", "outcome"): 0.8,
    ("initiating_event", "system_compromise"): 0.7,
    ("propagation", "system_failure"): 0.75,
    ("system_failure", "operational_consequence"): 0.7,
}

SYSTEM_COMPATIBILITY = {
    ("engine_mechanical", "lubrication"): 0.85,
    ("lubrication", "engine_mechanical"): 0.85,
    ("fuel", "engine_mechanical"): 0.75,
    ("engine_mechanical", "fuel"): 0.75,
    ("hydraulic", "flight_controls"): 0.8,
    ("flight_controls", "hydraulic"): 0.8,
    ("electrical", "engine_mechanical"): 0.65,
    ("structural", "flight_controls"): 0.55,
}

MECHANISM_COMPATIBILITY = {
    ("maintenance_error", "material_degradation"): 0.7,
    ("material_degradation", "fatigue"): 0.85,
    ("fatigue", "corrosion"): 0.75,
    ("contamination", "material_degradation"): 0.8,
    ("thermal_damage", "overheating"): 0.9,
    ("overheating", "fire"): 0.85,
    ("fire", "structural"): 0.6,
    ("deprivation", "power_loss"): 0.9,
    ("power_loss", "loss_of_control"): 0.75,
    ("loss_of_control", "terrain_collision"): 0.85,
}

# Slide constants
GAP_PENALTY = -0.05
STRONG_THRESHOLD = 0.70
PARTIAL_THRESHOLD = 0.35
UNMAPPED_PENALTY_COEF = 0.10


def _role_adjacency_score(r1: str, r2: str) -> float:
    if r1 == r2:
        return 1.0
    key = (r1, r2)
    if key in ADJACENT_ROLES:
        return ADJACENT_ROLES[key]
    key2 = (r2, r1)
    if key2 in ADJACENT_ROLES:
        return ADJACENT_ROLES[key2] * 0.9
    return 0.3


def _system_compatibility(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    key = (s1, s2)
    if key in SYSTEM_COMPATIBILITY:
        return SYSTEM_COMPATIBILITY[key]
    return 0.4


def _mechanism_compatibility(m1: str, m2: str) -> float:
    if m1 == m2:
        return 1.0
    key = (m1, m2)
    if key in MECHANISM_COMPATIBILITY:
        return MECHANISM_COMPATIBILITY[key]
    key2 = (m2, m1)
    if key2 in MECHANISM_COMPATIBILITY:
        return MECHANISM_COMPATIBILITY[key2]
    return 0.35


def step_similarity(step_a: dict[str, Any], step_b: dict[str, Any]) -> float:
    """Slide F1: 0.50 * role + 0.30 * mechanism + 0.20 * system."""
    w_role, w_mech, w_sys = 0.50, 0.30, 0.20
    ra = str(step_a.get("role") or "unknown")
    rb = str(step_b.get("role") or "unknown")
    sa = str(step_a.get("system") or "aircraft")
    sb = str(step_b.get("system") or "aircraft")
    ma = str(step_a.get("mechanism") or "unknown")
    mb = str(step_b.get("mechanism") or "unknown")
    return (
        w_role * _role_adjacency_score(ra, rb)
        + w_mech * _mechanism_compatibility(ma, mb)
        + w_sys * _system_compatibility(sa, sb)
    )


def _nw_align(
    chain_a: list[dict[str, Any]],
    chain_b: list[dict[str, Any]],
) -> tuple[list[tuple[int | None, int | None]], float]:
    """Slide F2: NW global alignment with constant gap_penalty = -0.05."""
    n, m = len(chain_a), len(chain_b)
    if n == 0 and m == 0:
        return [], 0.0
    neg_inf = -1e18
    dp: list[list[float]] = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    bp: list[list[str]] = [[""] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + GAP_PENALTY
        bp[i][0] = "up"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + GAP_PENALTY
        bp[0][j] = "left"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i - 1][j - 1] + step_similarity(chain_a[i - 1], chain_b[j - 1])
            delete_ = dp[i - 1][j] + GAP_PENALTY
            insert_ = dp[i][j - 1] + GAP_PENALTY
            best = max(match, delete_, insert_)
            dp[i][j] = best
            if best == match:
                bp[i][j] = "diag"
            elif best == delete_:
                bp[i][j] = "up"
            else:
                bp[i][j] = "left"
    alignment: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bp[i][j] == "diag":
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or bp[i][j] == "up"):
            alignment.append((i - 1, None))
            i -= 1
        else:
            alignment.append((None, j - 1))
            j -= 1
    alignment.reverse()
    return alignment, dp[n][m]


def score_alignment(
    alignment: list[tuple[int | None, int | None]],
    chain_a: list[dict[str, Any]],
    chain_b: list[dict[str, Any]],
    factors_a: list[dict[str, Any]] | None = None,  # accepted but ignored
    factors_b: list[dict[str, Any]] | None = None,  # accepted but ignored
) -> float:
    """Slide F3: (strong + 0.5*partial)/total - 0.10*(unmapped/total).

    total = len(chain_a) + len(chain_b) (total positions across both chains).
    factor_overlap and pattern_bonus terms are intentionally NOT applied
    in the slide variant.
    """
    strong = 0
    partial = 0
    for ia, ib in alignment:
        if ia is not None and ib is not None:
            sim = step_similarity(chain_a[ia], chain_b[ib])
            if sim >= STRONG_THRESHOLD:
                strong += 1
            elif sim >= PARTIAL_THRESHOLD:
                partial += 1
    n_unmapped = sum(1 for ia, ib in alignment if ia is None or ib is None)
    total = len(chain_a) + len(chain_b)
    if total <= 0:
        return 1.0
    raw = (strong + 0.5 * partial) / total - UNMAPPED_PENALTY_COEF * (n_unmapped / total)
    return max(0.0, min(1.0, raw))


def structural_similarity(struct_a: dict[str, Any], struct_b: dict[str, Any]) -> float:
    """Top-level structural similarity using slide formulas only."""
    chain_a = struct_a.get("causal_chain") or []
    chain_b = struct_b.get("causal_chain") or []
    if not isinstance(chain_a, list):
        chain_a = []
    if not isinstance(chain_b, list):
        chain_b = []
    if not chain_a and not chain_b:
        return 1.0
    if not chain_a or not chain_b:
        return 0.0
    alignment, _ = _nw_align(chain_a, chain_b)
    return score_alignment(alignment, chain_a, chain_b)


def _is_finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def clip01(x: float) -> float:
    if not _is_finite(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))

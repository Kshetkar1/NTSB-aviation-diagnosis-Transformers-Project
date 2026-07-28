"""
Structural similarity v2: Needleman–Wunsch alignment on causal chains (role/system/mechanism)
plus factor overlap, pattern bonus, and unmapped penalty. Output in [0, 1].

Used by A2 (relational causal chain) with embedding fusion in struct_hooks.reweight.
"""

from __future__ import annotations

import math
from typing import Any

# Role adjacency (SME-inspired)
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

GAP_PENALTY = -0.3
MISMATCH_PENALTY = -0.5
CHAIN_WEIGHT = 0.85
FACTOR_WEIGHT = 0.15
UNMAPPED_PENALTY = 0.15
ALIGNMENT_WEIGHT = 0.80
PATTERN_BONUS_WEIGHT = 0.20

PATTERN_SYNONYMS = {
    "maintenance_latent_defect_cascade": {"latent_defect", "maintenance_cascade"},
    "material_degradation_cascade": {"fatigue_cascade", "corrosion_cascade"},
    "thermal_cascade": {"fire_cascade", "overheat_cascade"},
}


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
    w_role, w_sys, w_mech = 0.45, 0.35, 0.20
    ra = str(step_a.get("role") or "unknown")
    rb = str(step_b.get("role") or "unknown")
    sa = str(step_a.get("system") or "aircraft")
    sb = str(step_b.get("system") or "aircraft")
    ma = str(step_a.get("mechanism") or "unknown")
    mb = str(step_b.get("mechanism") or "unknown")
    return (
        w_role * _role_adjacency_score(ra, rb)
        + w_sys * _system_compatibility(sa, sb)
        + w_mech * _mechanism_compatibility(ma, mb)
    )


def _nw_align(
    chain_a: list[dict[str, Any]],
    chain_b: list[dict[str, Any]],
) -> tuple[list[tuple[int | None, int | None]], float]:
    """Needleman–Wunsch global alignment with affine-like gap via constant gap."""
    n, m = len(chain_a), len(chain_b)
    if n == 0 and m == 0:
        return [], 0.0
    # DP[i][j] = best score aligning first i of A with first j of B
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
    # Traceback
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


def _factor_overlap(factors_a: list[dict[str, Any]], factors_b: list[dict[str, Any]]) -> float:
    if not factors_a and not factors_b:
        return 1.0
    if not factors_a or not factors_b:
        return 0.0
    scores = []
    for fa in factors_a:
        best = 0.0
        ea = str(fa.get("element") or "").lower()
        ra = str(fa.get("role") or "unknown")
        sa = str(fa.get("system") or "aircraft")
        for fb in factors_b:
            eb = str(fb.get("element") or "").lower()
            rb = str(fb.get("role") or "unknown")
            sb = str(fb.get("system") or "aircraft")
            elem_sim = 1.0 if ea and eb and (ea in eb or eb in ea) else 0.3
            role_sim = 1.0 if ra == rb else 0.5
            sys_sim = _system_compatibility(sa, sb)
            combined = 0.4 * elem_sim + 0.3 * role_sim + 0.3 * sys_sim
            best = max(best, combined)
        scores.append(best)
    return sum(scores) / max(len(factors_a), 1)


def score_alignment(
    alignment: list[tuple[int | None, int | None]],
    chain_a: list[dict[str, Any]],
    chain_b: list[dict[str, Any]],
    factors_a: list[dict[str, Any]],
    factors_b: list[dict[str, Any]],
) -> float:
    strong = 0
    partial = 0
    for ia, ib in alignment:
        if ia is not None and ib is not None:
            sim = step_similarity(chain_a[ia], chain_b[ib])
            if sim >= 0.75:
                strong += 1
            elif sim >= 0.45:
                partial += 1
    denom = (len(chain_a) + len(chain_b)) / 2.0
    if denom <= 0:
        chain_score = 1.0
    else:
        chain_score = (strong + 0.5 * partial) / denom
        chain_score = max(0.0, min(1.0, chain_score))
    factor_score = _factor_overlap(factors_a, factors_b)
    n_unmapped = sum(1 for ia, ib in alignment if ia is None or ib is None)
    um_denom = len(chain_a) + len(chain_b)
    unmapped_frac = (n_unmapped / um_denom) if um_denom > 0 else 0.0
    um_penalty = UNMAPPED_PENALTY * unmapped_frac
    combined = CHAIN_WEIGHT * chain_score + FACTOR_WEIGHT * factor_score - um_penalty
    return max(0.0, min(1.0, combined))


def pattern_match_bonus(pattern_a: str, pattern_b: str) -> float:
    if pattern_a == pattern_b:
        return 1.0
    syns_a = PATTERN_SYNONYMS.get(pattern_a, set())
    syns_b = PATTERN_SYNONYMS.get(pattern_b, set())
    if pattern_a in syns_b or pattern_b in syns_a:
        return 0.85
    if syns_a & syns_b:
        return 0.75
    return 0.0


def structural_similarity(struct_a: dict[str, Any], struct_b: dict[str, Any]) -> float:
    chain_a = struct_a.get("causal_chain") or []
    chain_b = struct_b.get("causal_chain") or []
    if not isinstance(chain_a, list):
        chain_a = []
    if not isinstance(chain_b, list):
        chain_b = []
    factors_a = struct_a.get("contributing_factors") or []
    factors_b = struct_b.get("contributing_factors") or []
    if not chain_a and not chain_b:
        return 1.0
    if not chain_a or not chain_b:
        fa = _factor_overlap(factors_a, factors_b) if (factors_a or factors_b) else 0.0
        return max(0.0, min(1.0, FACTOR_WEIGHT * fa))

    alignment, _ = _nw_align(chain_a, chain_b)
    align_score = score_alignment(alignment, chain_a, chain_b, factors_a, factors_b)
    pa = str(struct_a.get("failure_pattern") or "unknown")
    pb = str(struct_b.get("failure_pattern") or "unknown")
    bonus = pattern_match_bonus(pa, pb)
    final = ALIGNMENT_WEIGHT * align_score + PATTERN_BONUS_WEIGHT * bonus
    return max(0.0, min(1.0, final))


def _is_finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def clip01(x: float) -> float:
    if not _is_finite(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))

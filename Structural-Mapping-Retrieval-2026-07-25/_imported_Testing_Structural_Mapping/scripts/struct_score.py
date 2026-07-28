"""Rule-based similarity between two incident struct dicts (0..1)."""

from __future__ import annotations

from typing import Any

ENUM_FIELDS = (
    "phase_of_flight",
    "aircraft_class",
    "primary_system",
    "failure_mode",
    "severity_outcome",
)

# Weights for enum agreement when both sides are non-unknown.
FIELD_WEIGHTS: dict[str, float] = {
    "phase_of_flight": 0.14,
    "aircraft_class": 0.10,
    "primary_system": 0.22,
    "failure_mode": 0.22,
    "severity_outcome": 0.12,
}
FACTOR_WEIGHT = 0.20


def _norm_enum(v: Any) -> str:
    s = str(v or "").strip().lower()
    return s if s else "unknown"


def _factor_tokens(struct: dict) -> set[str]:
    raw = struct.get("contributing_factors")
    if not isinstance(raw, list):
        return set()
    out: set[str] = set()
    for x in raw:
        t = str(x).strip().lower()
        if len(t) >= 2:
            out.add(t)
    return out


def struct_similarity(a: dict, b: dict) -> tuple[float, dict]:
    """
    Return (score in [0,1], breakdown dict for logging).
    Unknown on either side skips that enum mass (does not count against score).
    """
    breakdown: dict = {}
    earned = 0.0
    possible = 0.0

    for field in ENUM_FIELDS:
        w = FIELD_WEIGHTS[field]
        av, bv = _norm_enum(a.get(field)), _norm_enum(b.get(field))
        if av == "unknown" or bv == "unknown":
            breakdown[field] = {"a": av, "b": bv, "partial": True}
            continue
        possible += w
        if av == bv:
            earned += w
            breakdown[field] = {"match": True}
        else:
            breakdown[field] = {"match": False}

    fa, fb = _factor_tokens(a), _factor_tokens(b)
    if fa or fb:
        possible += FACTOR_WEIGHT
        inter = fa & fb
        union = fa | fb
        j = len(inter) / len(union) if union else 0.0
        earned += FACTOR_WEIGHT * j
        breakdown["contributing_factors_jaccard"] = round(j, 4)
    else:
        breakdown["contributing_factors_jaccard"] = None

    score = earned / possible if possible > 0 else 0.0
    return float(min(1.0, max(0.0, score))), breakdown

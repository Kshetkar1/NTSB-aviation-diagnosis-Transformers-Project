"""Combine embedding cosine score with structural similarity."""

from __future__ import annotations

import math


def reweighted_incident_score(cosine_score: float, struct_sim: float, alpha: float) -> float:
    """
    New weight = max(0, cosine_score) * exp(alpha * struct_sim).

    struct_sim in [0,1]. alpha=0 recovers baseline (up to max(0,cosine)).
    """
    base = max(0.0, float(cosine_score))
    s = min(1.0, max(0.0, float(struct_sim)))
    a = float(alpha)
    return base * math.exp(a * s)

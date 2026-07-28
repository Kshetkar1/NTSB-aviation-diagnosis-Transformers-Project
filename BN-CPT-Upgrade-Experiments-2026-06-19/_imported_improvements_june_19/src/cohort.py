"""Table 4 cohort filters — restrict to incidents in Zhang's subgraph."""

from __future__ import annotations

import re

from corpus import incident_corpus

# Incident is in the Table 4 subgraph if it discusses any of these themes.
TABLE4_RELEVANCE = re.compile(
    r"landing gear|normal brake|\bbrake\b|braking|"
    r"electrical|wiring|\bwire\b|overheat|"
    r"\bfire\b|explosion|fire/smoke|smoke \(non-impact\)",
    re.I,
)

MENTIONS_BRAKE_GEAR = re.compile(
    r"landing gear|normal brake|\bbrake\b|braking|gear.{0,30}brake",
    re.I,
)

MENTIONS_ELECTRICAL = re.compile(
    r"electrical|wiring|\bwire\b|generator|alternator|"
    r"battery|dc power|ac power|overheat",
    re.I,
)


def table4_cohort_flags(inc: dict) -> dict:
    corpus = incident_corpus(inc)
    acft_fire = str(inc.get("acft_fire") or "").strip().upper()
    fire_field = acft_fire in ("IFLT", "GRD", "BOTH")
    relevant = bool(TABLE4_RELEVANCE.search(corpus)) or fire_field
    snippet = ""
    m = TABLE4_RELEVANCE.search(corpus)
    if m:
        snippet = corpus[max(0, m.start() - 20) : m.end() + 40].strip()
    elif fire_field:
        snippet = f"acft_fire={acft_fire}"
    return {
        "table4_relevant": "1" if relevant else "0",
        "mentions_brake_gear": "1" if MENTIONS_BRAKE_GEAR.search(corpus) else "0",
        "mentions_electrical": "1" if MENTIONS_ELECTRICAL.search(corpus) else "0",
        "table4_relevance_snippet": snippet,
    }


def row_in_restricted_pool(row: dict) -> bool:
    return row.get("table4_relevant") == "1"


def row_eligible_for_cell(row: dict, brake: str, heat: str) -> bool:
    """Strict Table 4 cell membership (restricted cohort)."""
    if row.get("brake_wear") != brake or row.get("electrical_overheat") != heat:
        return False
    if brake == "no" and row.get("mentions_brake_gear") != "1":
        return False
    if heat == "no" and row.get("mentions_electrical") != "1":
        return False
    if brake == "yes" and row.get("mentions_brake_gear") != "1":
        # wear failure should appear with brake context
        return row.get("brake_wear") == "yes"
    if heat == "yes" and row.get("mentions_electrical") != "1":
        return row.get("electrical_overheat") == "yes"
    return True

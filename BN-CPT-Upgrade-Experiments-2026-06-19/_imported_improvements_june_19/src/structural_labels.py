"""Structural-mapping based labels (Layer 2) — uses causal-chain cache.

Answers Maha's question: instead of only matching related words in the
narrative, we read the extracted causal chain (element / system / mechanism)
for each incident and label the Table 4 variables from that structure.

Cache source: data/struct_cache_v2.jsonl (vendored; ~177 train incidents have
structure; others fall back to keyword).
"""

from __future__ import annotations

import json
import re

from cpt_config import STRUCT_CACHE_PATH

STRUCT_CACHE = STRUCT_CACHE_PATH

# mechanism / system enums that map to each Table 4 variable
BRAKE_SYSTEMS = {"landing_gear"}
BRAKE_MECHANISMS = {"mechanical_loosening", "wear", "fatigue", "structural_failure"}
BRAKE_TEXT = re.compile(r"brake|braking|wheel|tire|gear", re.I)
BRAKE_WEAR_TEXT = re.compile(r"worn|wear|degrad|fail|fatigue|crack", re.I)

ELEC_SYSTEMS = {"electrical_power", "electrical"}
# Thermal mechanisms that, *combined with an electrical context*, indicate
# electrical overheating. thermal_damage alone is NOT enough (it fires on
# engine / brake / structural systems too — the polysemy that caused over-firing).
OVERHEAT_MECHANISMS = {"overheating", "thermal_damage", "thermal_cascade", "overheat_cascade"}
# Electrical context: system enum OR element text naming an electrical component.
ELEC_TEXT = re.compile(r"electric|wiring|\bwire\b|circuit|\barc\b|short[- ]?circuit|bus\b|wire harness", re.I)
# Overheat/thermal signal in free text.
OVERHEAT_TEXT = re.compile(r"overheat|thermal|burn(?:ed|ing|t)?|melt|smoke|smolder", re.I)

FIRE_MECHANISMS = {"fire", "fire_cascade", "overheat_cascade"}
FIRE_TEXT = re.compile(r"\bfire\b|explosion|smoke|burn", re.I)


def _load_cache() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not STRUCT_CACHE.is_file():
        return out
    with open(STRUCT_CACHE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev_id = rec.get("ev_id")
            struct = rec.get("struct") or {}
            if ev_id and struct:
                out[ev_id] = struct
    return out


_CACHE: dict[str, dict] | None = None


def cache() -> dict[str, dict]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_cache()
    return _CACHE


def _chain_items(struct: dict) -> list[dict]:
    items = list(struct.get("causal_chain") or [])
    items += list(struct.get("contributing_factors") or [])
    return items


def struct_brake_wear(struct: dict) -> str:
    for it in _chain_items(struct):
        el = str(it.get("element") or "")
        sysv = str(it.get("system") or "")
        mech = str(it.get("mechanism") or "")
        if BRAKE_TEXT.search(el) and BRAKE_WEAR_TEXT.search(el):
            return "yes"
        if sysv in BRAKE_SYSTEMS and mech in BRAKE_MECHANISMS:
            return "yes"
    # brake/gear discussed but no wear signal → no
    for it in _chain_items(struct):
        if BRAKE_TEXT.search(str(it.get("element") or "")) or str(it.get("system") or "") in BRAKE_SYSTEMS:
            return "no"
    return "unknown"


def struct_overheat(struct: dict) -> str:
    """Electrical overheating (Zhang x2).

    A `yes` requires an **electrical context** AND a thermal/overheat signal.
    A standalone thermal_damage mechanism on a non-electrical system (engine,
    brake, structure) does NOT count — that was the over-firing bug. A real
    `no` branch fires when electrical is discussed without any overheat signal.
    """
    elec_seen = False
    for it in _chain_items(struct):
        el = str(it.get("element") or "")
        sysv = str(it.get("system") or "")
        mech = str(it.get("mechanism") or "")

        is_elec = sysv in ELEC_SYSTEMS or bool(ELEC_TEXT.search(el))
        is_overheat = mech in OVERHEAT_MECHANISMS or bool(OVERHEAT_TEXT.search(el))
        if is_elec:
            elec_seen = True

        # Explicit electrical overheating: electrical context + thermal signal.
        if is_elec and is_overheat:
            return "yes"
        # Mechanism literally "overheating" still needs an electrical tie-in.
        if mech == "overheating" and (is_elec or ELEC_TEXT.search(el)):
            return "yes"

    # Electrical discussed but no overheat anywhere → confident no.
    if elec_seen:
        return "no"
    return "unknown"


def struct_fire(struct: dict) -> str:
    fp = str(struct.get("failure_pattern") or "")
    if re.search(r"fire|thermal", fp, re.I):
        return "yes"
    for it in _chain_items(struct):
        mech = str(it.get("mechanism") or "")
        el = str(it.get("element") or "")
        if mech in FIRE_MECHANISMS:
            return "yes"
        if FIRE_TEXT.search(el):
            return "yes"
    return "unknown"


def structural_labels(ev_id: str) -> dict | None:
    struct = cache().get(ev_id)
    if not struct:
        return None
    return {
        "brake_wear_struct": struct_brake_wear(struct),
        "electrical_overheat_struct": struct_overheat(struct),
        "fire_struct": struct_fire(struct),
        "failure_pattern": str(struct.get("failure_pattern") or ""),
    }

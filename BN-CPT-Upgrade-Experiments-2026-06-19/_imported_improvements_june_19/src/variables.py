"""Layer 1 deterministic variable labels (yes / no / unknown)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from corpus import first_match_snippet, incident_corpus
from cohort import table4_cohort_flags
from structural_labels import structural_labels

Tri = str  # "yes" | "no" | "unknown"


@dataclass
class LabelResult:
    value: Tri
    method: str
    snippet: str


# --- Brake wear (x1) ---
BRAKE_CONTEXT = re.compile(
    r"brake|braking|normal brake|landing gear.*brake|gear.*brake",
    re.I,
)
BRAKE_WEAR_YES = [
    r"brake.{0,50}(worn|wear|degrad|fail|malfunction|damaged)",
    r"(worn|wear|degrad|fail|malfunction).{0,50}brake",
    r"normal brake system.{0,30}(worn|wear|fail)",
    r"landing gear.{0,40}brake.{0,40}(worn|wear|fail|malfunction)",
    r"parking brake.{0,40}(fail|malfunction|did not|inadvertent)",
]
BRAKE_WEAR_NO = [
    r"brake.{0,30}(normal|serviceable|satisfactory|no.{0,10}defect)",
    r"no.{0,15}(wear|worn|degrad).{0,20}brake",
]

# --- Electrical overheat (x2) ---
ELECTRICAL_CONTEXT = re.compile(
    r"electrical|wiring|wire|generator|alternator|battery|dc power|ac power",
    re.I,
)
ELECTRICAL_OVERHEAT_YES = [
    r"overheat",
    r"electrical.{0,40}(fire|smoke|arc|burn|hot)",
    r"wiring.{0,40}(overheat|hot|burn|arc|smoke|fire)",
    r"wire.{0,40}(overheat|hot|burn|arc)",
]
ELECTRICAL_OVERHEAT_NO = [
    r"electrical.{0,30}(normal|serviceable|no.{0,10}fault)",
    r"no.{0,15}overheat",
    r"wiring.{0,30}(normal|serviceable)",
]

FIRE_OCC_CODES = {"170", "171", "172"}
ACFT_FIRE_YES = {"IFLT", "GRD", "BOTH"}
FIRE_TEXT_YES = [
    r"\bfire\b",
    r"\bexplosion\b",
    r"fire/smoke",
    r"smoke \(non-impact\)",
]
FIRE_TEXT_NO = []  # handled by absence + acft_fire NONE

DAMAGE_YES = {"SUBS", "DEST", "substantial", "destroyed"}
DAMAGE_NO = {"NONE", "MIN", "none", "minor"}


def _tri_from_patterns(
    corpus: str,
    context: re.Pattern | None,
    yes_patterns: list[str],
    no_patterns: list[str],
    require_context: bool = False,
) -> LabelResult:
    if require_context and context and not context.search(corpus):
        return LabelResult("unknown", "rule", "")

    for pat in yes_patterns:
        if re.search(pat, corpus, re.I):
            return LabelResult("yes", "rule", first_match_snippet(corpus, [pat]))

    for pat in no_patterns:
        if re.search(pat, corpus, re.I):
            return LabelResult("no", "rule", first_match_snippet(corpus, [pat]))

    if require_context and context and context.search(corpus):
        return LabelResult("no", "rule", first_match_snippet(corpus, [context.pattern]))

    return LabelResult("unknown", "rule", "")


def _brake_wear_label(corpus: str) -> LabelResult:
    if not BRAKE_CONTEXT.search(corpus):
        return LabelResult("no", "rule", "no brake / landing gear brake mention")

    for pat in BRAKE_WEAR_YES:
        if re.search(pat, corpus, re.I):
            return LabelResult("yes", "rule", first_match_snippet(corpus, [pat]))

    # Brake discussed but no wear/failure signal → treat as "no wear failure" for CPT parent=0
    if re.search(r"brake", corpus, re.I):
        return LabelResult(
            "no",
            "rule",
            first_match_snippet(corpus, [r"brake"]),
        )
    return LabelResult("unknown", "rule", "")


def _electrical_overheat_label(corpus: str) -> LabelResult:
    has_elec = bool(ELECTRICAL_CONTEXT.search(corpus))
    for pat in ELECTRICAL_OVERHEAT_YES:
        if re.search(pat, corpus, re.I):
            return LabelResult("yes", "rule", first_match_snippet(corpus, [pat]))

    if has_elec:
        return LabelResult(
            "no",
            "rule",
            first_match_snippet(corpus, [r"electrical", r"wiring", r"wire"]),
        )
    return LabelResult("no", "rule", "no electrical / wiring mention")


def label_brake_wear(inc: dict) -> LabelResult:
    return _brake_wear_label(incident_corpus(inc))


def label_electrical_overheat(inc: dict) -> LabelResult:
    return _electrical_overheat_label(incident_corpus(inc))


def label_fire(inc: dict) -> LabelResult:
    c = incident_corpus(inc)
    af = str(inc.get("acft_fire") or "").strip().upper()
    if af in ACFT_FIRE_YES:
        return LabelResult("yes", "rule", f"acft_fire={af}")

    for step in inc.get("sequence_of_events") or []:
        if not isinstance(step, dict):
            continue
        code = str(step.get("Occurrence_Code") or "").strip()
        if code in FIRE_OCC_CODES:
            desc = str(step.get("Occurrence_Description") or "")
            return LabelResult("yes", "rule", f"code={code} {desc[:60]}")

    for pat in FIRE_TEXT_YES:
        if re.search(pat, c, re.I):
            return LabelResult("yes", "rule", first_match_snippet(c, [pat]))

    if af in ("NONE", "", "UNK", "UNKT") or af not in ACFT_FIRE_YES:
        if not any(re.search(p, c, re.I) for p in FIRE_TEXT_YES):
            return LabelResult("no", "rule", f"acft_fire={af or 'empty'}")

    return LabelResult("unknown", "rule", "")


def label_x4_damage(inc: dict) -> LabelResult:
    """Table 5 outcome proxy: substantial or destroyed aircraft damage."""
    c = incident_corpus(inc)
    dmg = str(inc.get("damage") or "").strip().upper()

    if dmg in ("SUBS", "DEST"):
        return LabelResult("yes", "rule", f"damage={dmg}")
    if re.search(r"substantial aircraft damage|destroyed aircraft", c, re.I):
        return LabelResult("yes", "rule", first_match_snippet(c, [r"substantial", r"destroyed"]))

    if dmg in ("NONE", "MIN", ""):
        if not re.search(r"substantial aircraft damage|destroyed aircraft", c, re.I):
            return LabelResult("no", "rule", f"damage={dmg or 'empty'}")

    return LabelResult("unknown", "rule", "")


def label_incident(ev_id: str, inc: dict) -> dict:
    bw = label_brake_wear(inc)
    eo = label_electrical_overheat(inc)
    fire = label_fire(inc)
    x4 = label_x4_damage(inc)
    cohort = table4_cohort_flags(inc)

    struct = structural_labels(ev_id) or {}
    has_struct = "1" if struct else "0"

    return {
        "ev_id": ev_id,
        "brake_wear": bw.value,
        "brake_wear_method": bw.method,
        "brake_wear_snippet": bw.snippet,
        "electrical_overheat": eo.value,
        "electrical_overheat_method": eo.method,
        "electrical_overheat_snippet": eo.snippet,
        "fire": fire.value,
        "fire_method": fire.method,
        "fire_snippet": fire.snippet,
        "x4_damage": x4.value,
        "x4_damage_method": x4.method,
        "x4_damage_snippet": x4.snippet,
        "damage_field": str(inc.get("damage") or ""),
        "acft_fire_field": str(inc.get("acft_fire") or ""),
        "has_struct": has_struct,
        "brake_wear_struct": struct.get("brake_wear_struct", ""),
        "electrical_overheat_struct": struct.get("electrical_overheat_struct", ""),
        "fire_struct": struct.get("fire_struct", ""),
        "failure_pattern": struct.get("failure_pattern", ""),
        **cohort,
    }

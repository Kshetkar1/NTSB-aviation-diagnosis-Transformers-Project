#!/usr/bin/env python3
"""
Map free-text cause descriptions to Zhang's NTSB occurrence codes.

Jesse: "what I'd be looking for, so we can compare directly, would be to
have them all in the same coding system."

Uses embedding cosine similarity (not substring matching) to find the
best-matching NTSB occurrence code for each cause string.
"""
from __future__ import annotations

import os
os.environ["NTSB_USE_TRAIN_INDEX"] = "1"

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import main_app  # noqa: E402

# Zhang's 54 occurrence codes (from metaData.xlsx, Table: Occurrences)
ZHANG_OCCURRENCE_CODES = {
    100: "Abrupt maneuver",
    110: "Altitude deviation, uncontrolled",
    120: "Cargo shift",
    130: "Airframe/component/system failure/malfunction",
    131: "Propeller failure/malfunction",
    132: "Rotor failure/malfunction",
    140: "Decompression",
    150: "Ditching",
    160: "Dragged wing, rotor, pod, float or tail/skid",
    170: "Fire/explosion",
    171: "Fire",
    172: "Explosion",
    180: "Forced landing",
    190: "Gear collapsed",
    191: "Main gear collapsed",
    192: "Nose gear collapsed",
    193: "Tail gear collapsed",
    194: "Complete gear collapsed",
    195: "Other gear collapsed",
    196: "Gear not extended",
    197: "Gear not retracted",
    198: "Gear retraction on ground",
    200: "Hard landing",
    210: "Hazardous materials leak/spill",
    220: "In flight collision with object",
    230: "In flight collision with terrain/water",
    231: "Wheels down landing in water",
    232: "Wheels up landing",
    240: "In flight encounter with weather",
    250: "Loss of control - in flight",
    260: "Loss of control - on ground/water",
    270: "Midair collision",
    271: "Collision between aircraft (other than midair)",
    280: "Near collision between aircraft",
    290: "Nose down",
    300: "Nose over",
    310: "On ground/water collision with object",
    320: "On ground/water collision with terrain/water",
    330: "On ground/water encounter with weather",
    340: "Overrun",
    350: "Loss of engine power",
    351: "Loss of engine power (total) - mechanical failure/malfunction",
    352: "Loss of engine power (partial) - mechanical failure/malfunction",
    353: "Loss of engine power (total) - nonmechanical",
    354: "Loss of engine power (partial) - nonmechanical",
    355: "Engine tearaway",
    360: "Propeller blast or jet exhaust/suction",
    370: "Propeller/rotor contact to person",
    380: "Roll over",
    390: "Undershoot",
    400: "Undetermined",
    410: "Vortex turbulence encountered",
    420: "Missing aircraft",
    430: "Miscellaneous/other",
}

_code_emb_cache: dict | None = None


def build_code_embeddings() -> tuple[list[int], list[str], np.ndarray]:
    """Embed all Zhang occurrence-code labels. Cached after first call."""
    global _code_emb_cache
    if _code_emb_cache is not None:
        return _code_emb_cache

    codes = list(ZHANG_OCCURRENCE_CODES.keys())
    labels = list(ZHANG_OCCURRENCE_CODES.values())

    print(f"  Embedding {len(labels)} Zhang occurrence-code labels...", flush=True)
    embeddings = [main_app.get_embedding(lbl) for lbl in labels]

    result = (codes, labels, np.array(embeddings))
    _code_emb_cache = result
    return result


def map_cause_to_code(
    cause_text: str,
    code_codes: list[int],
    code_labels: list[str],
    code_embeddings: np.ndarray,
    threshold: float = 0.40,
) -> tuple[int | None, str, float]:
    """
    Map a single free-text cause to the best Zhang occurrence code.

    Returns (code_number, code_label, cosine_similarity).
    If best similarity < threshold → (None, "UNMAPPED", similarity).
    """
    cause_emb = np.array(main_app.get_embedding(cause_text))
    norms = np.linalg.norm(code_embeddings, axis=1) * np.linalg.norm(cause_emb)
    norms = np.where(norms == 0, 1e-10, norms)
    sims = np.dot(code_embeddings, cause_emb) / norms

    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    if best_sim >= threshold:
        return code_codes[best_idx], code_labels[best_idx], best_sim
    return None, "UNMAPPED", best_sim


def map_event_to_code(
    event_text: str,
    code_codes: list[int],
    code_labels: list[str],
    code_embeddings: np.ndarray,
    threshold: float = 0.35,
) -> tuple[int | None, str, float]:
    """
    Map a prognosis event description to the best Zhang occurrence code.
    Lower threshold than causes since prognosis event text is more varied.
    """
    return map_cause_to_code(event_text, code_codes, code_labels, code_embeddings, threshold)


def map_distribution_to_codes(
    probability_distribution: list[dict],
    code_codes: list[int],
    code_labels: list[str],
    code_embeddings: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    """
    Map an entire probability distribution to Zhang's coding system.
    Aggregates probabilities when multiple causes map to the same code.

    Returns (coded_distribution, detailed_mappings).
    """
    code_probs: dict[str, float] = defaultdict(float)
    code_nums: dict[str, int] = {}
    mappings: list[dict] = []

    for entry in probability_distribution:
        cause = entry["cause"]
        prob = entry["probability"]

        code_num, code_label, sim = map_cause_to_code(
            cause, code_codes, code_labels, code_embeddings
        )
        code_probs[code_label] += prob
        if code_num is not None:
            code_nums[code_label] = code_num

        mappings.append({
            "original_cause": cause,
            "probability": prob,
            "mapped_code": code_num,
            "mapped_label": code_label,
            "similarity": round(sim, 4),
        })

    coded_distribution = sorted(
        [
            {"code": code_nums.get(lbl), "label": lbl, "probability": round(p, 6)}
            for lbl, p in code_probs.items()
        ],
        key=lambda x: -x["probability"],
    )
    return coded_distribution, mappings


def map_events_to_codes(
    future_events: list[dict],
    code_codes: list[int],
    code_labels: list[str],
    code_embeddings: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    """
    Map prognosis future_events to Zhang's coding system.
    Like map_distribution_to_codes but uses 'event' key instead of 'cause'.

    Returns (coded_distribution, detailed_mappings).
    """
    code_probs: dict[str, float] = defaultdict(float)
    code_nums: dict[str, int] = {}
    mappings: list[dict] = []

    for entry in future_events:
        event = entry.get("event", "")
        prob = entry.get("probability", 0.0)

        code_num, code_label, sim = map_event_to_code(
            event, code_codes, code_labels, code_embeddings
        )
        code_probs[code_label] += prob
        if code_num is not None:
            code_nums[code_label] = code_num

        mappings.append({
            "original_event": event,
            "probability": prob,
            "mapped_code": code_num,
            "mapped_label": code_label,
            "similarity": round(sim, 4),
        })

    coded_distribution = sorted(
        [
            {"code": code_nums.get(lbl), "label": lbl, "probability": round(p, 6)}
            for lbl, p in code_probs.items()
        ],
        key=lambda x: -x["probability"],
    )
    return coded_distribution, mappings


# ============================================================
# Direct matching: for Zhang BN nodes that are NOT occurrence codes
# (damage severity, injury severity, contributing factors, etc.)
# ============================================================
_target_emb_cache: dict[str, np.ndarray] = {}


def _get_target_embedding(label: str) -> np.ndarray:
    """Cache target label embeddings to avoid redundant API calls."""
    if label not in _target_emb_cache:
        _target_emb_cache[label] = np.array(main_app.get_embedding(label))
    return _target_emb_cache[label]


def match_events_to_targets(
    future_events: list[dict],
    target_labels: list[str],
    threshold: float = 0.30,
) -> dict[str, float]:
    """
    Match prognosis events directly against arbitrary target labels.

    This is for Zhang BN nodes that are NOT occurrence codes — damage
    severity (destroyed/substantial/minor), injury severity (fatal/serious/
    minor/none), contributing factors (pilot error, unstable approach), etc.

    For each target label, finds all prognosis events that best-match it
    (by embedding cosine similarity) and sums their probabilities.

    Unlike map_events_to_codes, this does NOT go through the 54 occurrence
    codes. It matches directly: prognosis event ↔ Zhang node label.

    Returns {target_label: aggregated_probability}.
    """
    if not future_events:
        return {lbl: 0.0 for lbl in target_labels}

    # Embed all target labels
    target_embs = np.array([_get_target_embedding(lbl) for lbl in target_labels])

    # Embed all prognosis events
    event_texts = [e.get("event", "") for e in future_events]
    event_probs = [e.get("probability", 0.0) for e in future_events]
    event_embs = np.array([np.array(main_app.get_embedding(t)) for t in event_texts])

    # For each event, find the best-matching target
    # similarity matrix: (n_events × n_targets)
    # Normalize
    t_norms = np.linalg.norm(target_embs, axis=1, keepdims=True)
    t_norms = np.where(t_norms == 0, 1e-10, t_norms)
    target_normed = target_embs / t_norms

    e_norms = np.linalg.norm(event_embs, axis=1, keepdims=True)
    e_norms = np.where(e_norms == 0, 1e-10, e_norms)
    event_normed = event_embs / e_norms

    sim_matrix = event_normed @ target_normed.T  # (n_events × n_targets)

    result = {lbl: 0.0 for lbl in target_labels}

    for i, (event_text, prob) in enumerate(zip(event_texts, event_probs)):
        best_target_idx = int(np.argmax(sim_matrix[i]))
        best_sim = float(sim_matrix[i, best_target_idx])

        if best_sim >= threshold:
            result[target_labels[best_target_idx]] += prob

    return result


def match_causes_to_targets(
    weighted_causes: list[dict],
    target_labels: list[str],
    threshold: float = 0.30,
) -> dict[str, float]:
    """
    Match diagnosis causes directly against arbitrary target labels.
    Same as match_events_to_targets but uses 'cause' key.
    """
    # Convert to event-like format for reuse
    events = [{"event": c.get("cause", ""), "probability": c.get("probability", 0.0)}
              for c in weighted_causes]
    return match_events_to_targets(events, target_labels, threshold)

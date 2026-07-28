"""
Extract structured causal chain (v2) from incident text via OpenAI Chat Completions.

Uses the same API style as extract_struct.py (chat.completions.create + json_object).
Output schema: causal_chain (role/system/mechanism per step), contributing_factors,
bystanders, failure_pattern. Used by A2 structural similarity (struct_score_v2).
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Repo root + this scripts dir (for extract_struct)
_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parents[1]
for _p in (_REPO_ROOT, _SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from openai import OpenAI

from extract_struct import build_incident_bundle
from paths import EXTRACTION_PROMPT_V2_PATH

VALID_ROLES = frozenset(
    {
        "initiating_event",
        "propagation",
        "system_compromise",
        "system_failure",
        "terminal_failure",
        "operational_consequence",
        "outcome",
    }
)
VALID_SYSTEMS = frozenset(
    {
        "engine_mechanical",
        "lubrication",
        "fuel",
        "hydraulic",
        "electrical",
        "structural",
        "flight_controls",
        "landing_gear",
        "propulsion",
        "fire_protection",
        "environment",
        "human_performance",
        "maintenance",
        "aircraft",
    }
)
VALID_MECHANISMS = frozenset(
    {
        "maintenance_error",
        "material_degradation",
        "fatigue",
        "corrosion",
        "contamination",
        "thermal_damage",
        "mechanical_loosening",
        "deprivation",
        "overheating",
        "fire",
        "separation",
        "power_loss",
        "loss_of_control",
        "terrain_collision",
        "procedural_error",
        "design_deficiency",
        "unknown",
    }
)
VALID_PATTERNS = frozenset(
    {
        "maintenance_latent_defect_cascade",
        "material_degradation_cascade",
        "thermal_cascade",
        "defense_in_depth_failure",
        "human_performance_chain",
        "environmental_encounter",
        "cross_system_propagation",
        "fuel_management_failure",
        "unknown",
    }
)


def load_prompt_template() -> str:
    return EXTRACTION_PROMPT_V2_PATH.read_text(encoding="utf-8")


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def normalize_schema_v2(raw: dict[str, Any], ev_id: str) -> dict[str, Any]:
    """Coerce LLM output into a stable v2 schema."""
    # Always use dataset id (LLM sometimes emits a different internal id string).
    out: dict[str, Any] = {
        "ev_id": str(ev_id),
        "causal_chain": [],
        "contributing_factors": [],
        "bystanders": [],
        "failure_pattern": "unknown",
    }
    chain = raw.get("causal_chain") or []
    if not isinstance(chain, list):
        chain = []
    for i, step in enumerate(chain):
        if not isinstance(step, dict):
            continue
        role = str(step.get("role") or "unknown").strip().lower().replace(" ", "_")
        if role not in VALID_ROLES:
            role = "unknown"
        system = str(step.get("system") or "aircraft").strip().lower().replace(" ", "_")
        if system not in VALID_SYSTEMS:
            system = "aircraft"
        mech = str(step.get("mechanism") or "unknown").strip().lower().replace(" ", "_")
        if mech not in VALID_MECHANISMS:
            mech = "unknown"
        out["causal_chain"].append(
            {
                "step": int(step.get("step") or i + 1),
                "element": str(step.get("element") or "").strip() or f"step_{i+1}",
                "role": role,
                "system": system,
                "mechanism": mech,
            }
        )
    for cf in raw.get("contributing_factors") or []:
        if not isinstance(cf, dict):
            continue
        # Roles here are free-form labels (e.g. outcome_amplifier); not restricted to causal_chain VALID_ROLES.
        r = str(cf.get("role") or "unknown").strip().lower().replace(" ", "_") or "unknown"
        sys_ = str(cf.get("system") or "aircraft").strip().lower().replace(" ", "_")
        if sys_ not in VALID_SYSTEMS:
            sys_ = "aircraft"
        out["contributing_factors"].append(
            {
                "element": str(cf.get("element") or "").strip() or "factor",
                "role": r,
                "system": sys_,
            }
        )
    for by in raw.get("bystanders") or []:
        if not isinstance(by, dict):
            continue
        sys_ = str(by.get("system") or "aircraft").strip().lower().replace(" ", "_")
        if sys_ not in VALID_SYSTEMS:
            sys_ = "aircraft"
        out["bystanders"].append(
            {
                "element": str(by.get("element") or "").strip() or "bystander",
                "system": sys_,
            }
        )
    fp = str(raw.get("failure_pattern") or "unknown").strip().lower().replace(" ", "_")
    if fp not in VALID_PATTERNS:
        fp = "unknown"
    out["failure_pattern"] = fp
    return out


def _default_struct_model() -> str:
    env = os.getenv("OPENAI_STRUCT_MODEL")
    if env:
        return env
    try:
        from config import LLM_MODEL

        return str(LLM_MODEL)
    except ImportError:
        return "gpt-4o-mini"


def call_llm(prompt: str, model: str | None = None) -> str:
    """Chat Completions (same stack as v1); avoids Responses API availability issues."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key)
    m = model or _default_struct_model()
    resp = client.chat.completions.create(
        model=m,
        messages=[
            {
                "role": "system",
                "content": (
                    "You output only valid JSON for aviation causal-chain structuring. "
                    "Follow the user's schema exactly."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("Empty response from OpenAI")
    return text


def extract_struct_v2(
    incident_text: str,
    ev_id: str,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    template = load_prompt_template()
    prompt = template.replace("{{INCIDENT_TEXT}}", incident_text[:120_000])
    raw_text = call_llm(prompt, model=model)
    raw_text = _strip_json_fence(raw_text)
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from LLM: {e}\n---\n{raw_text[:2000]}") from e
    if not isinstance(raw, dict):
        raise ValueError("LLM output is not a JSON object")
    return normalize_schema_v2(raw, ev_id)


def extract_struct_from_incident_v2(
    inc: dict[str, Any],
    ev_id: str,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """Bundle same fields as v1 extraction, then run v2 causal-chain prompt."""
    text = build_incident_bundle(inc)
    return extract_struct_v2(text, ev_id, model=model)

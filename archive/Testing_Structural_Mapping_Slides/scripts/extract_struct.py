"""OpenAI JSON extraction for incident_struct_v1."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

REQUIRED_KEYS = (
    "phase_of_flight",
    "aircraft_class",
    "primary_system",
    "failure_mode",
    "severity_outcome",
    "contributing_factors",
    "summary_10_words",
)


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_incident_bundle(inc: dict, max_chars: int = 7000) -> str:
    parts: list[str] = []
    for key in ("narr_accp", "narr_accf", "narr_cause"):
        t = str(inc.get(key) or "").strip()
        if t:
            parts.append(f"### {key}\n{t[:2800]}")
    seq = inc.get("sequence_of_events") or []
    lines: list[str] = []
    for e in seq[:10]:
        if isinstance(e, dict):
            d = str(e.get("Occurrence_Description") or "").strip()
            if d:
                lines.append(f"- {d[:600]}")
    if lines:
        parts.append("### sequence_of_events (first lines)\n" + "\n".join(lines))
    findings = inc.get("findings") or []
    fl: list[str] = []
    for f in findings[:8]:
        if isinstance(f, dict):
            fd = str(f.get("finding_description") or "").strip()
            if fd:
                fl.append(f"- {fd[:450]}")
    if fl:
        parts.append("### findings\n" + "\n".join(fl))
    text = "\n\n".join(parts)
    return text[:max_chars] if len(text) > max_chars else text


def _strip_code_fence(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _parse_json_object(raw: str) -> dict:
    s = _strip_code_fence(raw)
    return json.loads(s)


def normalize_struct(obj: Any) -> dict:
    if not isinstance(obj, dict):
        raise ValueError("struct must be a JSON object")
    out: dict = {}
    for k in REQUIRED_KEYS:
        out[k] = obj.get(k)
    if not isinstance(out["contributing_factors"], list):
        out["contributing_factors"] = []
    out["contributing_factors"] = [
        str(x).strip() for x in out["contributing_factors"] if str(x).strip()
    ][:10]
    out["summary_10_words"] = str(out.get("summary_10_words") or "").strip()[:120]
    for e in (
        "phase_of_flight",
        "aircraft_class",
        "primary_system",
        "failure_mode",
        "severity_outcome",
    ):
        out[e] = str(out.get(e) or "unknown").strip().lower() or "unknown"
    return out


def extract_struct_from_text(
    client: OpenAI,
    model: str,
    incident_text: str,
    prompt_template: str,
) -> dict:
    if not incident_text.strip():
        raise ValueError("empty incident_text")
    user = prompt_template.replace("{{INCIDENT_TEXT}}", incident_text)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You output only valid JSON for aviation incident structuring.",
            },
            {"role": "user", "content": user},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or ""
    parsed = _parse_json_object(raw)
    return normalize_struct(parsed)


def extract_struct_from_incident(
    client: OpenAI,
    model: str,
    inc: dict,
    prompt_template: str,
    max_chars: int = 7000,
) -> dict:
    bundle = build_incident_bundle(inc, max_chars=max_chars)
    return extract_struct_from_text(client, model, bundle, prompt_template)

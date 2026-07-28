"""
Enrich the worked-example JSONs with plain-English narratives.

For every top-N entry in:
  - diagnosis.{a0,a2}.ltp_causes
  - diagnosis.{a0,a2}.coded_distribution
  - prognosis.{a0,a2}.ltp_events

we ask GPT-4o-mini (temperature=0) to produce:
  - label        : short human label (<= 10 words)
  - elaboration  : one-sentence plain-English explanation

Results are CACHED deterministically in data/narrative_cache.json (keyed
by sha256(kind + raw_text)) so re-runs do not re-call the LLM.

Also deduplicates near-identical entries after rewriting: if two LLM
labels collide (case-insensitive exact match) the rows are merged and
the probability is summed; rank is recomputed.

Outputs are written back into the existing JSONs under new keys:
  diagnosis.{model}.ltp_causes_enriched
  diagnosis.{model}.coded_distribution_enriched
  prognosis.{model}.ltp_events_enriched

The raw arrays are preserved so the audit script still sees the
ground-truth pipeline numbers.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
SCRIPTS_DIR = PROJECT_ROOT / "Testing_Structural_Mapping_Slides" / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from config import LLM_MODEL  # noqa: E402
from main_app import get_client  # noqa: E402

CACHE_PATH = _HERE / "data" / "narrative_cache.json"
TOP_N = 5  # only enrich the top-N rows shown to Maha


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def _cache_key(kind: str, raw: str) -> str:
    h = hashlib.sha256(f"{kind}::{raw}".encode("utf-8")).hexdigest()
    return h


PROMPT_CAUSE = (
    "You are converting raw NTSB accident-cause taxonomy text into a clear, "
    "plain-English label that a pilot or investigator can understand at a "
    "glance. Stay faithful to the underlying technical meaning.\n\n"
    "Produce JSON with two fields:\n"
    "  - label        : <= 10 words, no jargon, no taxonomy hyphens\n"
    "  - elaboration  : one sentence (<= 30 words) explaining what this "
    "cause means in operational terms\n\n"
    "Examples:\n"
    "INPUT:  aircraft-aircraft power plant-engine (turbine/turboprop)-turbine section-failure - c\n"
    "OUTPUT: {\"label\": \"Turbine section structural failure\", \"elaboration\": "
    "\"A structural failure of the turbine section (rotating blades, disks, or stators) "
    "causes loss of thrust or, in severe cases, uncontained debris.\"}\n\n"
    "INPUT:  organizational issues-development-manufacture/production-equipment manufacture-manufacturer - c\n"
    "OUTPUT: {\"label\": \"Manufacturer / production defect\", \"elaboration\": "
    "\"A defect introduced during component manufacture or production led to the "
    "failure observed in service.\"}\n\n"
    "INPUT:  {raw}\n"
    "OUTPUT:"
)


PROMPT_EVENT = (
    "You are converting raw NTSB sequence-of-events text into a clear, "
    "plain-English label for a pilot. Stay faithful to the meaning.\n\n"
    "The input is usually formatted as 'Flight phase - Event description'.\n\n"
    "Produce JSON with two fields:\n"
    "  - label        : <= 10 words, plain English, includes the flight phase if specific\n"
    "  - elaboration  : one sentence (<= 30 words) explaining what this "
    "event means operationally and what the crew typically does next\n\n"
    "Examples:\n"
    "INPUT:  Initial climb - Engine shutdown\n"
    "OUTPUT: {\"label\": \"Engine shutdown during initial climb\", \"elaboration\": "
    "\"The affected engine is manually shut down by the crew shortly after takeoff, "
    "leaving the aircraft to operate on the remaining engine(s).\"}\n\n"
    "INPUT:  Landing - Fire/smoke (non-impact)\n"
    "OUTPUT: {\"label\": \"Fire or smoke during landing\", \"elaboration\": "
    "\"Fire or visible smoke is reported during the landing phase without impact, "
    "requiring evacuation and emergency-response procedures.\"}\n\n"
    "INPUT:  {raw}\n"
    "OUTPUT:"
)


PROMPT_CODE = (
    "You are converting an NTSB occurrence-code label into a plain-English "
    "explanation for a pilot. Stay faithful to the meaning.\n\n"
    "Produce JSON with two fields:\n"
    "  - label        : <= 10 words, plain English\n"
    "  - elaboration  : one sentence (<= 30 words) explaining what occurrences "
    "this code family typically covers\n\n"
    "Examples:\n"
    "INPUT:  130 - Airframe/component/system failure/malfunction\n"
    "OUTPUT: {\"label\": \"Airframe or system failure\", \"elaboration\": "
    "\"A physical airframe, engine, or onboard system failed or malfunctioned in "
    "flight, including engine fires, hydraulic loss, and structural failures.\"}\n\n"
    "INPUT:  352 - Loss of engine power (partial) - mechanical failure/malfunction\n"
    "OUTPUT: {\"label\": \"Partial engine power loss (mechanical)\", \"elaboration\": "
    "\"One engine partially loses thrust due to internal mechanical failure such as "
    "compressor stall, fuel-control fault, or turbine damage.\"}\n\n"
    "INPUT:  {raw}\n"
    "OUTPUT:"
)


PROMPTS = {
    "cause": PROMPT_CAUSE,
    "event": PROMPT_EVENT,
    "code": PROMPT_CODE,
}


def _call_llm(client, kind: str, raw: str) -> dict:
    """Single deterministic LLM call. Returns {"label": ..., "elaboration": ...}."""
    prompt = PROMPTS[kind].replace("{raw}", raw)
    resp = client.with_options(timeout=20.0).chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        timeout=20.0,
    )
    text = (resp.choices[0].message.content or "").strip()
    try:
        obj = json.loads(text)
    except Exception:
        # Fallback: try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(text[start : end + 1])
        else:
            raise RuntimeError(f"LLM did not return JSON: {text[:200]}")
    label = str(obj.get("label", "")).strip()
    elab = str(obj.get("elaboration", "")).strip()
    return {"label": label, "elaboration": elab}


_LLM_CALL_COUNT = 0


def _enrich_one(client, cache: dict, kind: str, raw: str) -> dict:
    global _LLM_CALL_COUNT
    key = _cache_key(kind, raw)
    if key in cache:
        return cache[key]
    _LLM_CALL_COUNT += 1
    print(f"      LLM call #{_LLM_CALL_COUNT} ({kind}): {raw[:60]!r}...", flush=True)
    result = _call_llm(client, kind, raw)
    cache[key] = result
    return result


def _enrich_causes(client, cache, items: list[dict]) -> list[dict]:
    """Enrich top-N causes. Dedup by enriched label (case-insensitive)."""
    enriched: list[dict] = []
    for r in items[:TOP_N]:
        raw = r.get("cause", "")
        out = _enrich_one(client, cache, "cause", raw)
        enriched.append({
            "rank": r.get("rank"),
            "probability": float(r.get("probability", 0.0)),
            "raw_cause": raw,
            "label": out["label"],
            "elaboration": out["elaboration"],
        })

    enriched = _dedup_and_resort(enriched, label_key="label", item_key="raw_cause")
    return enriched


def _enrich_events(client, cache, items: list[dict]) -> list[dict]:
    enriched: list[dict] = []
    for r in items[:TOP_N]:
        raw = r.get("event", "")
        out = _enrich_one(client, cache, "event", raw)
        enriched.append({
            "rank": r.get("rank"),
            "probability": float(r.get("probability", 0.0)),
            "raw_event": raw,
            "label": out["label"],
            "elaboration": out["elaboration"],
        })

    enriched = _dedup_and_resort(enriched, label_key="label", item_key="raw_event")
    return enriched


def _enrich_codes(client, cache, items: list[dict]) -> list[dict]:
    enriched: list[dict] = []
    for r in items[:TOP_N]:
        code = str(r.get("code", "-"))
        label = str(r.get("label", ""))
        raw = f"{code} - {label}" if code != "-" else (label or "UNMAPPED")
        out = _enrich_one(client, cache, "code", raw)
        enriched.append({
            "rank": r.get("rank"),
            "probability": float(r.get("probability", 0.0)),
            "code": code,
            "raw_label": label,
            "label": out["label"],
            "elaboration": out["elaboration"],
        })
    return enriched  # Codes do not need dedup (they are already unique)


def _dedup_and_resort(rows: list[dict], *, label_key: str, item_key: str) -> list[dict]:
    """If two rows have the same enriched label (case-insensitive), merge them
    and sum probabilities. Re-rank by combined probability."""
    by_label: dict[str, dict] = {}
    for r in rows:
        lab = r[label_key].strip().lower()
        if lab in by_label:
            existing = by_label[lab]
            existing["probability"] += r["probability"]
            existing.setdefault("merged_from", []).append(r.get(item_key, ""))
        else:
            by_label[lab] = {**r}

    merged = list(by_label.values())
    merged.sort(key=lambda x: x["probability"], reverse=True)
    for i, r in enumerate(merged, start=1):
        r["rank"] = i
    return merged


def _enrich_file(path: Path, client, cache) -> None:
    print(f"\nEnriching {path.name} ...", flush=True)
    data = json.loads(path.read_text())

    for model in ("a0", "a2"):
        diag = data["diagnosis"][model]
        prog = data["prognosis"][model]

        print(f"  [{model.upper()}] enrich diagnosis ltp_causes ({len(diag['ltp_causes'])} rows, top {TOP_N})", flush=True)
        diag["ltp_causes_enriched"] = _enrich_causes(client, cache, diag["ltp_causes"])
        _save_cache(cache)

        print(f"  [{model.upper()}] enrich coded_distribution ({len(diag['coded_distribution'])} rows, top {TOP_N})", flush=True)
        diag["coded_distribution_enriched"] = _enrich_codes(client, cache, diag["coded_distribution"])
        _save_cache(cache)

        print(f"  [{model.upper()}] enrich prognosis ltp_events ({len(prog['ltp_events'])} rows, top {TOP_N})", flush=True)
        prog["ltp_events_enriched"] = _enrich_events(client, cache, prog["ltp_events"])
        _save_cache(cache)

    path.write_text(json.dumps(data, indent=2))
    print(f"  Saved {path}", flush=True)


def main() -> None:
    client = get_client()
    cache = _load_cache()
    print(f"Loaded narrative cache: {len(cache)} entries")

    # Auto-discover every example_*.json so newly added incidents are enriched
    # automatically.
    candidates = sorted((_HERE / "data").glob("example_*.json"))
    if not candidates:
        print("No example_*.json files found - run precompute_examples.py first")
        return
    print(f"Found {len(candidates)} example file(s) to consider: "
          f"{[p.name for p in candidates]}")
    for path in candidates:
        _enrich_file(path, client, cache)
        _save_cache(cache)  # checkpoint after each file

    print(f"\nFinal cache size: {len(cache)} entries")
    print("DONE")


if __name__ == "__main__":
    main()

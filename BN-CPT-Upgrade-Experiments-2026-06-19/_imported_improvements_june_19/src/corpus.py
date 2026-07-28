"""Build searchable text corpus per incident."""

from __future__ import annotations


def incident_corpus(inc: dict) -> str:
    parts: list[str] = []
    for step in inc.get("sequence_of_events") or []:
        if not isinstance(step, dict):
            continue
        parts.append(str(step.get("Occurrence_Description") or ""))
        parts.append(str(step.get("Occurrence_Code") or ""))
    for f in inc.get("findings") or []:
        if isinstance(f, dict):
            parts.append(str(f.get("finding_description") or ""))
    parts.append(str(inc.get("narr_cause") or "")[:2000])
    parts.append(str(inc.get("damage") or ""))
    parts.append(str(inc.get("acft_fire") or ""))
    parts.append(str(inc.get("ev_highest_injury") or ""))
    return " ".join(parts).lower()


def first_match_snippet(corpus: str, patterns: list[str], window: int = 80) -> str:
    """Return text around first regex match for audit."""
    import re

    for pat in patterns:
        m = re.search(pat, corpus, re.I)
        if m:
            start = max(0, m.start() - 30)
            end = min(len(corpus), m.end() + window)
            return corpus[start:end].replace("\n", " ").strip()
    return ""

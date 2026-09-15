"""Narrative → structured evidence and outcome/seed fallbacks (no hard stop)."""

from __future__ import annotations

from collections import defaultdict

import zhang_diagnosis as zd


def list_outcome_cores(dataset) -> list[str]:
    """Common outcome families for manual picker when auto-detect fails."""
    cores: dict[str, int] = defaultdict(int)
    for inc in dataset.values():
        for s in inc.get("sequence_of_events", []):
            d = (s.get("Occurrence_Description") or "").strip()
            if d:
                cores[zd._core(d)] += 1
    # frequent cores first
    return [c for c, _ in sorted(cores.items(), key=lambda x: -x[1]) if c]


def outcome_from_core(core: str, dataset) -> tuple[str, set[str]] | None:
    """Build (name, targets) from a manually chosen outcome core."""
    if not core:
        return None
    cores: dict[str, set] = defaultdict(set)
    for lab in zd._occurrence_vocab(dataset):
        cores[zd._core(lab)].add(lab.lower())
    chosen_tok = set(core.split())
    targets: set = set()
    for c, labs in cores.items():
        if chosen_tok <= set(c.split()):
            targets |= labs
    if not targets:
        targets = cores.get(core, {core})
    return core, targets


def resolve_outcome(query: str, manual_core: str | None, dataset):
    """Auto-detect outcome, else use manual picker value."""
    det = zd.detect_outcome(query, dataset=dataset)
    if det is not None:
        return det, "auto-detected"
    if manual_core:
        built = outcome_from_core(manual_core, dataset)
        if built:
            return built, "manual picker"
    return None, "none"


def parse_narrative_evidence(query: str, bn_node_names, dataset):
    """Parse query to BN evidence nodes (hard + soft)."""
    import query_to_bn as qb

    try:
        parsed = qb.parse_query_to_bn_evidence(
            query, bn_node_names, dataset=dataset, semantic=True,
        )
        return parsed
    except Exception:
        return {"evidence": [], "trace": [], "confidence": {}}


def hard_evidence_set(parsed: dict) -> set[str]:
    conf = parsed.get("confidence") or {}
    out = set()
    for n in parsed.get("evidence") or []:
        if conf.get(n, 1.0) >= 0.999:
            out.add(n.lower())
    return out


def seed_events_from_parsed(parsed: dict) -> list[str]:
    """BN nodes parsed from narrative — used to seed prognosis when no outcome."""
    return list(parsed.get("evidence") or [])


def list_transition_seeds(dataset, top_k: int = 40) -> list[str]:
    """Frequent event labels that have forward transitions (prognosis picker)."""
    import prognosis as pg

    tc = pg.build_transition_counts(dataset)
    ranked = sorted(
        tc["from_counts"].items(), key=lambda x: -x[1],
    )
    return [lab for lab, n in ranked[:top_k] if n >= 2]

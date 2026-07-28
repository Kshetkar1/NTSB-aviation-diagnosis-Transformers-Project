"""Zhang-aligned empirical diagnosis (matches Table 7).

Computes P(cause | outcome) = count(cause & outcome) / count(outcome) over the
corrected 1982-2006 dataset, using Zhang's edge logic (buildOneGraphRep):
a cause -> outcome edge exists when a finding (subject) is attached to the
outcome occurrence, or is the occurrence immediately preceding it.

This is the same conditional DIRECTION as the retrieval engine's diagnosis
(P(cause | query)), but with Zhang's exact denominator (count of outcome
accidents) instead of retrieval similarity mass -- so it reproduces Table 7.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parent
DATASET = config.WINDOW_DATA_PATH

# Optional manual overrides: outcome name -> exact occurrence labels. Empty by
# default -- every outcome (including fire/explosion) is detected from the data.
# Kept only as the fallback for empirical_cause_distribution / sparse_cpt callers,
# whose .get(outcome, {outcome}) then resolves to the literal occurrence label
# (e.g. "fire" -> {"fire"} = 102 accidents, reproducing Table 7 exactly).
OUTCOME_ALIASES: dict[str, set] = {}

# Light query normalization so free-text wording maps to occurrence vocabulary.
_SYNONYMS = {
    "caught fire": "fire", "catches fire": "fire", "on fire": "fire",
    "burning": "fire", "burned": "fire", "ablaze": "fire", "smoke": "fire",
    "blew up": "explosion", "exploded": "explosion",
    "lose": "loss", "losing": "loss", "lost": "loss",
    "collapse": "collapsed", "collapsing": "collapsed",
    "ditched": "ditching", "ditch": "ditching",
}


def _resolve_dataset_path() -> Path:
    """Active global-mode dataset path, honoring config's window/full-corpus toggle.

    Reads config.ACTIVE_INCIDENT_DATA_PATH (which defaults to the 1982-2006 window
    when its index exists, and switches to the full corpus when NTSB_FULL_CORPUS=1).
    Falls back to the hardcoded Zhang-window DATASET if config lacks the attribute."""
    return Path(getattr(config, "ACTIVE_INCIDENT_DATA_PATH", DATASET))


def _load():
    return json.loads(_resolve_dataset_path().read_text(encoding="utf-8"))


def _occurrence_vocab(ds: dict) -> set:
    labels = set()
    for inc in ds.values():
        for s in inc.get("sequence_of_events", []):
            d = (s.get("Occurrence_Description") or "").strip()
            if d:
                labels.add(d)
    return labels


def _core(label: str) -> str:
    """Outcome 'family' key: drop parentheticals and ' - ...' qualifiers so all
    'Loss of engine power (total/partial) - ...' variants share one core."""
    s = re.sub(r"\(.*?\)", " ", label.lower())
    s = s.split(" - ")[0]
    return " ".join(s.split())


def _normalize_query(query: str) -> str:
    s = query.lower()
    for a, b in _SYNONYMS.items():
        s = s.replace(a, b)
    s = re.sub(r"[^a-z0-9/ ]", " ", s)  # drop punctuation; keep '/' for labels
    return " ".join(s.split())


def detect_outcome(query: str, dataset=None):
    """Detect the outcome in a free-text query, generalized to ANY NTSB outcome.

    Returns (name, targets) where targets is a set of lowercased occurrence
    labels (the outcome family), or None if nothing matches. Every outcome is
    detected against the dataset's occurrence vocabulary and expanded to its
    family (e.g. all 'loss of engine power' / 'gear collapsed' variants). Distinct
    labels stay distinct: 'fire' -> {fire} (102), separate from 'fire/explosion' (12)
    and 'explosion' (2). OUTCOME_ALIASES (empty by default) can force overrides.
    """
    nq = _normalize_query(query)
    for name, targets in OUTCOME_ALIASES.items():
        if re.search(rf"\b{re.escape(name)}\b", nq):
            return name, set(targets)

    ds = dataset if dataset is not None else _load()
    cores: dict[str, set] = defaultdict(set)
    for lab in _occurrence_vocab(ds):
        cores[_core(lab)].add(lab.lower())

    present = [c for c in cores if c and c in nq]
    chosen = max(present, key=len) if present else None
    if chosen is None:
        qtok = set(nq.split())
        best, best_score = None, 0.0
        for c in cores:
            ctok = set(c.split())
            if not ctok:
                continue
            score = len(ctok & qtok) / len(ctok)
            if score > best_score:
                best, best_score = c, score
        if best is None or best_score < 0.5:
            return None
        chosen = best

    # Expand to the family: any occurrence whose core contains all chosen tokens.
    chosen_tok = set(chosen.split())
    targets: set = set()
    for c, labs in cores.items():
        if chosen_tok <= set(c.split()):
            targets |= labs
    return chosen, targets


def parse_outcome(query: str):
    """Backward-compatible: return just the outcome name (or None)."""
    det = detect_outcome(query)
    return det[0] if det else None


def _cause_vocab(ds: dict) -> set:
    """All cause/evidence labels (findings + occurrences) used for conditions."""
    labels = set()
    for inc in ds.values():
        for f in inc.get("findings", []):
            d = (f.get("finding_description") or "").strip()
            if d:
                labels.add(d)
        for s in inc.get("sequence_of_events", []):
            d = (s.get("Occurrence_Description") or "").strip()
            if d:
                labels.add(d)
    return labels


def _incident_cause_labels(inc: dict) -> set:
    s = set()
    for f in inc.get("findings", []):
        s.add((f.get("finding_description") or "").strip().lower())
    for ev in inc.get("sequence_of_events", []):
        s.add((ev.get("Occurrence_Description") or "").strip().lower())
    return s


_STOP = {"the", "a", "an", "of", "was", "is", "were", "are", "in", "on", "to",
         "and", "or", "but", "with", "problem", "issue", "during", "flight",
         "aircraft", "there", "had", "has", "that", "this", "it", "its"}


def match_cause(text: str, ds: dict, min_score: float = 0.34):
    """Best-matching cause/evidence label for a free-text condition phrase.

    Ranks by how much of the CONDITION phrase the label covers (recall), then by
    how specific the label is (precision); deterministic via sorted iteration."""
    qtok = {t for t in _normalize_query(text).split() if t not in _STOP}
    if not qtok:
        return None
    best, best_key = None, (0.0, 0.0)
    for lab in sorted(_cause_vocab(ds)):
        ltok = set(_normalize_query(lab).split())
        inter = ltok & qtok
        if not inter:
            continue
        recall = len(inter) / len(qtok)
        precision = len(inter) / len(ltok)
        key = (recall, precision)
        if key > best_key:
            best, best_key = lab.lower(), key
    return best if best_key[0] >= min_score else None


_CONNECTORS = re.compile(r"\b(?:given|with|where|when|and|but|while|,)\b")


def _parse_conditions(query: str, outcome_name: str) -> list:
    """Light free-text parse: split on connectors, drop the outcome/question
    fragment, return candidate condition phrases."""
    q = query.lower()
    # remove the trailing question (e.g. 'what caused the fire?')
    q = re.split(r"\bwhat\b|\bwhy\b|\bcause", q)[0] if re.search(r"\bwhat\b|\bwhy\b|\bcause", q) else q
    frags = [f.strip() for f in _CONNECTORS.split(q) if f and f.strip()]
    out = []
    for f in frags:
        if outcome_name and outcome_name in f:
            continue
        if len(f.split()) < 2:  # skip stray words
            continue
        out.append(f)
    return out


def diagnose_conditional(query: str, conditions=None, top_n: int = 15,
                         mode: str = "global", top_n_incidents: int = 400,
                         dataset=None) -> dict:
    """Conditional diagnosis: P(cause | outcome AND conditions).

    Restricts the population to outcome accidents that ALSO contain every
    condition (a finding/occurrence present anywhere in the accident), then ranks
    causes with Zhang's edge logic + denominator over that subset.

    conditions: optional explicit list of condition phrases. If None, they are
    parsed from the query ('given X and Y, what caused Z').
    mode='global' counts over all eligible accidents; 'retrieval' first narrows to
    the query-relevant incidents (the engine lane).
    """
    if mode == "retrieval":
        import main_app
        ds = dataset if dataset is not None else main_app.refined_dataset
    else:
        ds = dataset if dataset is not None else _load()

    det = detect_outcome(query, dataset=ds)
    if det is None:
        return {"error": "no recognized outcome in query", "query": query}
    name, targets = det

    if conditions is None:
        conditions = _parse_conditions(query, name)
    matched = [(c, match_cause(c, ds)) for c in conditions]
    matched_labels = [m for _, m in matched if m]
    unmatched = [c for c, m in matched if not m]

    # candidate population
    if mode == "retrieval":
        import main_app
        q_emb = main_app.get_embedding(query)
        _, hits = main_app.find_top_matches(q_emb)
        pool, seen = [], set()
        for h in hits[:top_n_incidents]:
            if h.get("source") != "incident":
                continue
            ev = h.get("ev_id")
            if ev and ev not in seen:
                seen.add(ev)
                pool.append(ev)
    else:
        pool = list(ds)

    eligible = [ev for ev in pool
                if all(lbl in _incident_cause_labels(ds[ev]) for lbl in matched_labels)]

    res = empirical_cause_distribution(name, targets=targets, dataset=ds,
                                       restrict_ev_ids=eligible)
    res["query"] = query
    res["outcome"] = name
    res["outcome_labels"] = sorted(targets)
    res["mode"] = mode
    res["conditions"] = conditions
    res["matched_conditions"] = matched_labels
    res["unmatched_conditions"] = unmatched
    res["eligible_with_outcome"] = res["outcome_count"]
    res["causes"] = res["causes"][:top_n]
    return res


# --- Zhang-faithful contributory-factor labeling --------------------------------
# Zhang's Table 7 lists *contributory factors* to fire. In the NTSB legacy schema
# every finding carries a Cause_Factor flag: "C" (cause), "F" (factor), or blank
# (a non-causal descriptive finding, e.g. "Emergency procedure - Performed" or
# "Evacuation - Performed"). Zhang counts only C/F findings as contributory
# factors; the blank findings are responses/observations, not causes. Restricting
# to C/F is what reconciles the over-attribution (e.g. Emergency procedure 11->1,
# Evacuation 6->1) against Zhang's published counts. See
# docs/TABLE7_FULL_REPRODUCTION.md.
CONTRIBUTORY_CAUSE_FACTORS = {"C", "F"}

# Zhang's deriveNamebyCode() (his main.py) returns the literal string
# "Unknown quantity" for any Subj_Code missing from his code->meaning lookup. The
# refined dataset stores those same unresolved findings with a "nan"
# finding_description. Normalizing nan/empty -> "Unknown quantity" reproduces
# Zhang's exact unresolved-code convention (this is the single absent Table 7
# label, Subj_Code 92000, present on 2 fire findings as a Factor).
UNRESOLVED_FINDING_LABEL = "Unknown quantity"
_UNRESOLVED_RAW = {"", "nan", "none"}


def _is_contributory(fd: dict) -> bool:
    """True if a finding is flagged as a Cause ('C') or Factor ('F')."""
    return (str(fd.get("Cause_Factor") or "").strip().upper()
            in CONTRIBUTORY_CAUSE_FACTORS)


def _faithful_finding_label(fd: dict) -> str:
    """Finding label using Zhang's unresolved-code placeholder for nan/empty text."""
    raw = (str(fd.get("finding_description") or "")).strip()
    return UNRESOLVED_FINDING_LABEL if raw.lower() in _UNRESOLVED_RAW else raw


def _has_outcome(inc: dict, targets: set) -> bool:
    """True if any occurrence in the incident matches the outcome family."""
    return any((s.get("Occurrence_Description") or "").strip().lower() in targets
               for s in inc.get("sequence_of_events", []))


def _causes_into_outcome(inc: dict, targets: set,
                         cause_factor_only: bool = False) -> set:
    """Zhang edge logic: cause labels with an edge into the outcome occurrence.
    A cause is a finding attached to the outcome occurrence, or the occurrence
    immediately preceding it. Returns set of cause labels (dict meanings).

    cause_factor_only (default False, preserving all existing callers): when True,
    only findings flagged Cause/Factor count (Zhang's contributory factors) and
    nan/empty finding labels are normalized to "Unknown quantity" -- the faithful
    Zhang-Table-7 reproduction mode. The preceding-occurrence edge is unchanged
    (occurrences carry no Cause_Factor flag)."""
    seq = sorted(
        inc.get("sequence_of_events", []),
        key=lambda s: int(s.get("Occurrence_No") or 0),
    )
    if not seq:
        return set()
    descs = [s.get("Occurrence_Description", "") for s in seq]
    occ_nos = [str(s.get("Occurrence_No")) for s in seq]
    target_pos = [i for i, d in enumerate(descs) if (d or "").strip().lower() in targets]
    if not target_pos:
        return set()
    findings = inc.get("findings", [])
    causes = set()
    for i in target_pos:
        target_occ_no = occ_nos[i]
        for fd in findings:
            if str(fd.get("Occurrence_No")) == target_occ_no:
                if cause_factor_only:
                    if not _is_contributory(fd):
                        continue
                    label = _faithful_finding_label(fd)
                else:
                    label = (fd.get("finding_description") or "").strip()
                if label:
                    causes.add(label)
        if i >= 1:
            prev = (descs[i - 1] or "").strip()
            if prev:
                causes.add(prev)
    return causes


def empirical_cause_distribution(
    outcome: str, dataset=None, restrict_ev_ids=None, targets=None,
    cause_factor_only: bool = False,
) -> dict:
    """Zhang's Table-7-style P(cause | outcome) = count(cause & outcome) / count(outcome).

    restrict_ev_ids: optional iterable of ev_ids to restrict the population to
    (e.g., the fire accidents surfaced by retrieval). When None, uses ALL outcome
    accidents in the dataset (reproduces Zhang's Table 7).

    cause_factor_only (default False): faithful Zhang reproduction mode. When True,
    (1) only Cause/Factor findings count as contributory factors, (2) nan/empty
    finding labels normalize to "Unknown quantity", and (3) the denominator is the
    count of ALL outcome accidents (incidents containing the outcome occurrence),
    independent of whether any contributory factor survives the C/F filter -- this
    is Zhang's count(outcome) and reproduces the 102-fire denominator exactly. The
    default (False) preserves the historical behaviour for every existing caller."""
    ds = dataset if dataset is not None else _load()
    if targets is None:
        targets = OUTCOME_ALIASES.get(outcome, {outcome})
    allow = set(restrict_ev_ids) if restrict_ev_ids is not None else None

    outcome_accidents = []
    cause_count: dict[str, set] = defaultdict(set)

    for ev_id, inc in ds.items():
        if allow is not None and ev_id not in allow:
            continue
        causes = _causes_into_outcome(inc, targets,
                                      cause_factor_only=cause_factor_only)
        if cause_factor_only:
            # Denominator = count(outcome): every accident with the outcome
            # occurrence, even if all its findings were filtered out.
            if not _has_outcome(inc, targets):
                continue
        elif not causes:
            continue
        outcome_accidents.append(ev_id)
        for c in causes:
            cause_count[c].add(ev_id)

    total = len(outcome_accidents)
    dist = {c: len(evs) / total for c, evs in cause_count.items()} if total else {}
    return {
        "outcome": outcome,
        "outcome_count": total,
        "causes": sorted(
            ({"cause": c, "probability": p, "n": len(cause_count[c])}
             for c, p in dist.items()),
            key=lambda d: -d["probability"],
        ),
    }


# --- Confidence-aware (selective) diagnosis ------------------------------------
# The two dominant generic catch-all cause labels. The confidence signal (margin /
# top_prob) is computed over the SPECIFIC-mechanism ranking with these stripped,
# matching the validated selective experiment (tests/loo_selective.py), where the
# "top-1 / top-2 cause" used for the margin are specific causes, not catch-alls.
GENERIC_CAUSES = {
    "airframe/component/system failure/malfunction",
    "miscellaneous/other",
}

# Operational responses / post-incident actions belong in the PROGNOSIS tree
# (downstream of the outcome), not in a branching DIAGNOSIS tree (upstream causes).
# Zhang Table 7 can still list a few at very low rates (e.g. emergency procedure 1/102);
# we drop them from the diagnosis tree so cause vs consequence stay visually separated.
DIAGNOSIS_RESPONSE_LABELS = {
    "evacuation",
    "emergency procedure",
    "aborted takeoff",
    "panic",
    "fire extinguishing equipment",
}

# Default margin gate for selective_diagnose. Chosen as the pooled fire+gear
# margin at the top-25%-coverage operating point validated in tests/loo_selective.py
# (the coverage level where margin-gated retrieval beats the base-rate baseline for
# fire and gear collapse). Per-outcome 75th-percentile margins were fire=0.054 and
# gear=0.125; the pooled value (0.08) is the documented default. Overridable via the
# margin_threshold arg of selective_diagnose. See docs/ZHANG_REPRODUCTION_REPORT.md 10.6.
SELECTIVE_MARGIN_THRESHOLD = 0.08


# --- Calibration (temperature scaling) of the conditioned output ---------------
# Fitted temperature for the conditioned cause distribution (Guo et al. 2017),
# validated in tests/calibration_analysis.py (report §15): it ~halves held-out ECE
# (0.059 -> ~0.038) with no Brier cost. The value is sourced from
# docs/calibration_results.json via config (NOT a magic literal here); config also
# exposes the APPLY_CALIBRATION default toggle. Temperature scaling is strictly
# MONOTONIC, so it changes the probability MAGNITUDES (calibration) but never the
# cause RANKING (top-1 / MRR are preserved).
CALIBRATION_TEMPERATURE = float(getattr(config, "CALIBRATION_TEMPERATURE", 0.473))


def _temperature_scale(causes: list, temperature: float = None) -> list:
    """Temperature-scale a cause distribution (Guo et al. 2017; report §15).

    Mirrors tests/calibration_analysis.apply_temperature EXACTLY: treat each cause
    probability as a logit via log(p), divide by the temperature, then softmax over
    the FULL support. Returns a NEW list of cause dicts with recalibrated
    'probability' values ('n' counts untouched). With T>0 the map p -> softmax(log
    p / T) is strictly increasing, so the input ordering is preserved; the list is
    assumed already sorted by descending probability and is returned in the same
    order (no re-sort, so tie order is preserved too)."""
    if temperature is None:
        temperature = CALIBRATION_TEMPERATURE
    if not causes or not temperature or temperature <= 0:
        return causes
    logits = [math.log(max(float(c.get("probability") or 0.0), 1e-12)) / temperature
              for c in causes]
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    s = sum(exps) or 1.0
    out = []
    for c, e in zip(causes, exps):
        nc = dict(c)
        nc["probability"] = e / s
        out.append(nc)
    return out


def _specific_causes(causes: list) -> list:
    """Causes with the two dominant generic catch-all labels removed."""
    return [c for c in causes
            if str(c.get("cause") or "").lower() not in GENERIC_CAUSES]


def _confidence_signal(causes: list) -> tuple:
    """(top_prob, margin, top_cause) over the specific-mechanism ranking.

    margin = prob[top1] - prob[top2] (0.0 if fewer than 2 specific causes);
    top_prob = prob[top1] (0.0 if none); top_cause = label of top1 (or None)."""
    spec = _specific_causes(causes)
    if not spec:
        return 0.0, 0.0, None
    top_prob = float(spec[0].get("probability") or 0.0)
    if len(spec) >= 2:
        margin = top_prob - float(spec[1].get("probability") or 0.0)
    else:
        margin = 0.0
    return top_prob, margin, spec[0].get("cause")


def selective_diagnose(query: str = None, dataset=None,
                       margin_threshold: float = SELECTIVE_MARGIN_THRESHOLD,
                       top_n: int = 15, top_n_incidents: int = 120,
                       **kwargs) -> dict:
    """Confidence-aware per-incident diagnosis.

    Runs the retrieval diagnosis, then gates on the specific-cause margin:
      margin >= margin_threshold -> commit (confidence='high'), expose top_cause;
      margin <  margin_threshold -> abstain (confidence='low'), list the top
                                     candidate specific causes instead of one guess.

    This does NOT change population-level diagnosis (Table 7); it only adds a
    confidence wrapper around the retrieval lane. Returns a clean dict with keys:
    confidence, margin, top_prob, threshold, committed, top_cause, causes
    (plus query / outcome passthrough). The default threshold is the validated
    top-25%-coverage operating point for fire / gear collapse (see module constant).
    """
    res = diagnose_retrieval(query, top_n_incidents=top_n_incidents, top_n=top_n)
    if res.get("error"):
        return res

    top_prob = res.get("top_prob", 0.0)
    margin = res.get("margin", 0.0)
    spec = _specific_causes(res.get("causes", []))
    _, _, top_cause = _confidence_signal(res.get("causes", []))

    committed = margin >= margin_threshold
    return {
        "query": query,
        "outcome": res.get("outcome"),
        "outcome_count": res.get("outcome_count"),
        "confidence": "high" if committed else "low",
        "margin": margin,
        "top_prob": top_prob,
        "threshold": margin_threshold,
        "committed": committed,
        "top_cause": top_cause if committed else None,
        # When low-confidence, surface the top specific candidates so the engine can
        # say "multiple plausible causes" instead of a single misleading commit.
        "candidates": spec[:top_n] if not committed else [],
        "causes": res.get("causes", []),
        "mode": "retrieval",
    }


def gated_diagnose(query: str = None, dataset=None,
                   margin_threshold: float = SELECTIVE_MARGIN_THRESHOLD,
                   top_n: int = 15, top_n_incidents: int = 120,
                   **kwargs) -> dict:
    """Auto-gated diagnosis: condition on the narrative ONLY where it helps.

    The leakage-controlled LOO validation (report §14) showed query-conditioning
    ranks the true cause higher than the unconditioned population prior on
    SPECIFIC-mechanism causes, but HURTS on incidents whose only true cause is a
    generic catch-all (GENERIC_CAUSES), where the frequency prior is already
    optimal. This wrapper routes per query, using only signals observable at
    PREDICTION time (it never inspects the true cause):

      Gate (fall back to the unconditioned prior) iff BOTH hold:
        (1) the prior's top-1 cause is a generic catch-all  -- the "generic
            regime" where the population prior is hard to beat; AND
        (2) the conditioned specific-cause margin < margin_threshold -- i.e. the
            narrative gives no CONFIDENT specific mechanism to justify overriding.
      Otherwise commit to the conditioned (narrative-retrieval) distribution.

    Rationale (validated in tests/gating_validation.py, report §16): this is the
    most surgical gate found -- it recovers the harmed generic-true subgroup to
    the prior's accuracy while costing the specific-cause majority essentially
    nothing. The harm regime is rare (~2% of incidents), so gating makes the tool
    SAFE (never worse than the prior in the generic regime) rather than improving
    overall accuracy; see §16 for the honest trade-off.

    Returns the chosen distribution plus full provenance: gated (bool),
    gate_reason, the conditioned and prior top causes, and which path was taken.
    Existing diagnose/diagnose_retrieval/selective_diagnose behaviour is unchanged.
    """
    import main_app  # lazy: same retrieval index the conditioned lane uses

    cond = diagnose_retrieval(query, top_n_incidents=top_n_incidents, top_n=top_n)
    if cond.get("error"):
        return cond

    name = cond.get("outcome")
    targets = set(cond.get("outcome_labels") or [])
    ds = dataset if dataset is not None else main_app.refined_dataset
    prior = empirical_cause_distribution(name, targets=targets, dataset=ds)

    prior_causes = prior.get("causes", [])
    prior_top = prior_causes[0].get("cause") if prior_causes else None
    prior_top_generic = bool(prior_top) and str(prior_top).lower() in GENERIC_CAUSES
    cond_margin = float(cond.get("margin", 0.0) or 0.0)

    gate_fires = prior_top_generic and (cond_margin < margin_threshold)

    if gate_fires:
        chosen, path = prior, "prior"
        reason = (f"prior top cause is generic ('{prior_top}') and conditioned "
                  f"specific margin {cond_margin:.4f} < {margin_threshold} "
                  f"-> defaulting to the unconditioned prior")
    else:
        chosen, path = cond, "conditioned"
        reason = ("conditioned on the narrative" if not prior_top_generic
                  else f"prior is generic but conditioned specific margin "
                       f"{cond_margin:.4f} >= {margin_threshold} -> trust the narrative")

    causes = chosen.get("causes", [])[:top_n]
    return {
        "query": query,
        "outcome": name,
        "outcome_labels": sorted(targets),
        "outcome_count": chosen.get("outcome_count"),
        "mode": "gated",
        "gated": gate_fires,
        "path": path,
        "gate_reason": reason,
        "margin_threshold": margin_threshold,
        "conditioned_margin": cond_margin,
        "prior_top_cause": prior_top,
        "prior_top_generic": prior_top_generic,
        "conditioned_top_cause": (cond.get("causes") or [{}])[0].get("cause"),
        "causes": causes,
    }


def diagnose_retrieval(query: str, top_n_incidents: int = 200, top_n: int = 15,
                       calibrate: bool = None) -> dict:
    """Retrieval lane with ZHANG'S DENOMINATOR.

    Uses the engine's retrieval to surface incidents relevant to the query, keeps
    only those that actually had the outcome, then computes Zhang's
    count(cause & outcome) / (number of those outcome accidents).

    When retrieval is broad enough to surface every outcome accident, this
    reproduces Zhang's Table 7 RANKING exactly; with a tighter top_n_incidents it
    answers the same quantity restricted to the query-relevant subset (so the
    query's wording shapes the cause distribution).

    calibrate: apply temperature-scaling calibration (report §15) to the returned
    probability MAGNITUDES. Defaults to config.APPLY_CALIBRATION (ON). This is
    MONOTONIC -- it never changes the cause ranking -- so the confidence signal
    (top_prob / margin / top_cause) used by selective_diagnose / gated_diagnose is
    deliberately computed on the RAW distribution and left UNCHANGED, preserving
    their validated semantics. Set calibrate=False to recover the exact Zhang
    magnitudes (e.g. for Table-7 magnitude convergence checks)."""
    import main_app  # lazy: engine retrieval index (window by default)

    det = detect_outcome(query, dataset=main_app.refined_dataset)
    if det is None:
        return {"error": "no recognized outcome in query", "query": query}
    name, targets = det

    q_emb = main_app.get_embedding(query)
    _, matches = main_app.find_top_matches(q_emb)
    seen, ev_ids = set(), []
    for m in matches[:top_n_incidents]:
        if m.get("source") != "incident":
            continue
        ev = m.get("ev_id")
        if ev and ev not in seen:
            seen.add(ev)
            ev_ids.append(ev)

    res = empirical_cause_distribution(
        name, targets=targets, dataset=main_app.refined_dataset, restrict_ev_ids=ev_ids
    )
    res["query"] = query
    res["outcome"] = name
    res["outcome_labels"] = sorted(targets)
    res["mode"] = "retrieval"
    res["retrieved_incidents"] = len(ev_ids)

    # Confidence signal over the specific-mechanism ranking, computed on the RAW
    # (uncalibrated) distribution BEFORE truncation/calibration -- selective_diagnose
    # and gated_diagnose rely on these exact validated margins. (Computing over the
    # full vs the truncated list is equivalent: at most the two generic catch-alls
    # can outrank the top specific causes.) Additive keys only; existing keys
    # untouched so all current callers keep working.
    top_prob, margin, top_cause = _confidence_signal(res["causes"])

    # Temperature-scaling calibration of the conditioned distribution (report §15).
    # Applied over the FULL support (matching tests/calibration_analysis) BEFORE
    # truncation so the softmax denominator -- and thus the calibrated top-1
    # confidence -- reproduces the validated ECE. Monotonic: ranking is unchanged.
    if calibrate is None:
        calibrate = getattr(config, "APPLY_CALIBRATION", True)
    if calibrate:
        T = float(getattr(config, "CALIBRATION_TEMPERATURE", CALIBRATION_TEMPERATURE))
        res["causes"] = _temperature_scale(res["causes"], T)
        res["calibrated"] = True
        res["calibration_temperature"] = T
        res["calibrated_top_prob"] = (
            float(res["causes"][0]["probability"]) if res["causes"] else 0.0)
    else:
        res["calibrated"] = False

    res["causes"] = res["causes"][:top_n]
    res["top_prob"] = top_prob
    res["margin"] = margin
    res["top_cause"] = top_cause
    return res


def diagnose(query: str, top_n: int = 15, mode: str = "retrieval",
             top_n_incidents: int = 120, gated: bool = None) -> dict:
    """Engine entry point. Detects ANY outcome from a free-text query and returns
    P(cause | outcome).

    mode="retrieval" (default): query wording drives the answer -- diagnose over
        the query-relevant subset of outcome accidents (this is YOUR method). By
        DEFAULT this now routes through gated_diagnose (the validated safety gate,
        report §16) and applies temperature-scaling calibration (report §15) to the
        conditioned distribution -- both backward-compatible and configurable.
    mode="global": count over ALL outcome accidents (reproduces Zhang's Table 7
        for fire; the query only selects the outcome, not the subset). UNCHANGED.

    gated: route the retrieval lane through the safety gate. Defaults to
        config.GATE_DIAGNOSIS_BY_DEFAULT (ON). Set gated=False for the plain ungated
        conditioned lane (diagnose_retrieval) -- e.g. for reproduction/comparison.
        Ignored when mode="global".
    """
    if mode == "retrieval":
        if gated is None:
            gated = getattr(config, "GATE_DIAGNOSIS_BY_DEFAULT", True)
        if gated:
            return gated_diagnose(query, top_n=top_n, top_n_incidents=top_n_incidents)
        return diagnose_retrieval(query, top_n_incidents=top_n_incidents, top_n=top_n)

    det = detect_outcome(query)
    if det is None:
        return {"error": "no recognized outcome in query", "query": query}
    name, targets = det
    res = empirical_cause_distribution(name, targets=targets)
    res["query"] = query
    res["outcome"] = name
    res["outcome_labels"] = sorted(targets)
    res["mode"] = "global"
    res["causes"] = res["causes"][:top_n]
    return res


if __name__ == "__main__":
    import sys

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    q = args[0] if args else "What is the probability of fire?"
    mode = "global" if "--global" in flags else "retrieval"
    r = diagnose(q, top_n=15, mode=mode)
    print(f"QUERY: {q}")
    if r.get("error"):
        print("ERROR:", r["error"])
        sys.exit(0)
    print(f"outcome = {r['outcome']!r}   outcome accidents = {r['outcome_count']}")

    # Per-incident confidence indicator (retrieval lane only; population/global
    # diagnosis output is intentionally unchanged). The default retrieval path is
    # gated (mode="gated" in the result), so read the gating provenance + the raw
    # conditioned margin from whichever shape came back.
    if mode == "retrieval":
        if r.get("mode") == "gated":
            print(f"path = {r.get('path')}  (gated={r.get('gated')})  "
                  f"-> {r.get('gate_reason')}")
            margin = r.get("conditioned_margin", 0.0)
            top_cause = r.get("conditioned_top_cause")
        else:
            margin = r.get("margin", 0.0)
            top_cause = r.get("top_cause")
        thr = SELECTIVE_MARGIN_THRESHOLD
        if margin >= thr:
            print(f"confidence = HIGH  (margin {margin:.4f} >= {thr})  "
                  f"top cause -> {top_cause}")
        else:
            print(f"confidence = LOW   (margin {margin:.4f} < {thr})  "
                  f"-> multiple plausible causes; not committing to one")

    print(f"\n{'cause':62} {'n':>4} {'P(cause|outcome)':>16}")
    print("-" * 86)
    for c in r["causes"]:
        print(f"{c['cause'][:62]:62} {c['n']:>4} {c['probability']:>16.4f}")

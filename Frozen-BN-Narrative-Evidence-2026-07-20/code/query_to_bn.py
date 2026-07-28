"""Query -> BN evidence bridge.

Answers Maha's question "how do the narratives reach the Bayesian network?":
the narrative never enters the BN math -- it is parsed into STRUCTURED
evidence nodes (events, findings, persons), and those nodes are set as
evidence (hard or soft) before propagation. This module does the parsing.

    parse_query_to_bn_evidence(query, bn_names)
        -> {"evidence": [node, ...],
            "trace": [(node, how), ...],
            "confidence": {node: float}}   # 1.0 = deterministic parse

Deterministic passes (confidence 1.0), all against the network's own
node vocabulary:
  1. OUTCOME  -- zhang_diagnosis.detect_outcome (same parser the counting
     pipeline uses), intersected with the BN's node names.
  2. PERSONS  -- "person: ..." nodes via a small alias table ("pilot",
     "pilot error", "pilot in command" -> person: pilot-in-command) plus
     generic matching of the person description itself.
  3. MENTIONS -- any other event/finding node whose name appears in the
     normalized query as a whole-token phrase (longest match first, so
     "loss of engine power" wins over "engine power"). Multi-word cores
     only: a single common word ("monitoring") never becomes hard evidence.

All deterministic passes apply an ASSERTION guard: a mention inside a
negated ("no fire on board") or hypothetical ("worried about a loss of
engine power") clause is not evidence and is skipped.

Optional RETRIEVAL pass (soft evidence, confidence < 1.0): the SAME
embedding retrieval the counting pipeline uses ranks every accident by
similarity to the narrative; a fact strongly over-represented among the
most-similar accidents (weighted frequency f_q, lift over base rate f_0)
is added as soft evidence with confidence c = f_q: "given this narrative,
the fact holds with probability about f_q".

apply_evidence turns c into JEFFREY conditioning via Pearl's virtual
evidence: the likelihood ratio is chosen against the network's OWN prior
p0 for the node,
    LR = [c/(1-c)] / [p0/(1-p0)],
so after propagation the network believes the fact with probability ~c
(exactly c when it is the only soft fact). Suggestion, not assertion --
and this is how the narrative is exploited END TO END: text -> retrieval
-> facts + calibrated strengths -> joint propagation.
"""
from __future__ import annotations

import re

import zhang_diagnosis as zd

# person aliases checked longest-first; values are the node's person key
# (matched against the "person: <desc>" suffix, hyphens treated as spaces)
_PERSON_ALIASES = [
    ("pilot in command", "pilot-in-command"),
    ("pilot error", "pilot-in-command"),
    ("captain", "pilot-in-command"),
    ("copilot", "copilot/second pilot"),
    ("co-pilot", "copilot/second pilot"),
    ("first officer", "copilot/second pilot"),
    ("flight crew", "flightcrew"),
    ("flight attendant", "flight attendant"),
    ("mechanic", "mechanic"),
    ("pilot", "pilot-in-command"),
]

_STOP_CORES = {"other", "unknown", "miscellaneous"}
_STOPWORDS = {"of", "the", "a", "an", "in", "on", "to", "and", "or", "with"}

# --- assertion guard: a mentioned fact is only EVIDENCE when the clause
# asserts it -- not when it is negated ("no fire") or hypothetical
# ("worried about a loss of engine power"). Cues must PRECEDE the fact
# inside the same clause.
_NEG_CUES = re.compile(
    r"\b(no|not|never|without|absent|denied|nor|"
    r"did not|didn ?t|was not|wasn ?t|were not|weren ?t|"
    r"has not|hasn ?t|had not|hadn ?t)\b")
_IRREALIS_CUES = re.compile(
    r"\b(worried about|worried that|feared|fear of|afraid of|"
    r"concerned about|concern about|in case of|in case|anticipated|"
    r"anticipating|possibility of|risk of|chance of|potential for|"
    r"almost|nearly|would have|could have|might have|"
    r"precaution against|precautionary against|simulated|practice)\b")
_CLAUSE_SPLIT = re.compile(r"[,.;:!?()]|\bbut\b|\balthough\b|\bthough\b|"
                           r"\bhowever\b|\bwhile\b|\bwhereas\b|\band\b")


def _asserted(phrase: str, raw_query: str) -> bool:
    """True when `phrase` appears in the query as an ASSERTED fact, i.e. at
    least one occurrence is NOT preceded (in its own clause) by a negation or
    hypothetical cue. Fails open: if the phrase cannot be located in any
    clause (normalization mismatch), the mention is treated as asserted."""
    if not phrase:
        return True
    found = False
    for clause in _CLAUSE_SPLIT.split(raw_query.lower()):
        c = " ".join(re.sub(r"[^a-z0-9/ ]", " ", clause).split())
        m = re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", c)
        if not m:
            continue
        found = True
        before = c[:m.start()]
        if not (_NEG_CUES.search(before) or _IRREALIS_CUES.search(before)):
            return True
    return not found


def _norm(text: str) -> str:
    """Same normalization the outcome parser applies to the query."""
    return zd._normalize_query(text)


def _node_core(name: str) -> str:
    """Matchable core of a node name: drop parentheticals / ' - ' qualifiers,
    normalize punctuation to spaces (so 'fluid, oil grade' -> 'fluid oil grade')."""
    s = re.sub(r"\(.*?\)", " ", name.lower())
    s = s.split(" - ")[0]
    s = re.sub(r"[^a-z0-9/ ]", " ", s)
    return " ".join(s.split())


def _phrase_in(phrase: str, text: str) -> bool:
    """Whole-token phrase containment ('fire' does not match 'misfire')."""
    if not phrase:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


# ---------------------------------------------------------------------------
# Retrieval soft-fact extraction -- SAME embedding retrieval as the counting
# pipeline, turned into virtual-evidence strengths.
# ---------------------------------------------------------------------------
_SUPPORT_CACHE: dict[int, dict] = {}


def _label_support(dataset: dict) -> dict:
    """How many incidents contain each label (occurrence, finding, person) --
    used as an empirical prior to break near-ties among semantic matches."""
    key = id(dataset)
    if key in _SUPPORT_CACHE:
        return _SUPPORT_CACHE[key]
    from collections import Counter
    support: Counter = Counter()
    for inc in dataset.values():
        labs = set()
        for s in inc.get("sequence_of_events", []):
            d = str(s.get("Occurrence_Description") or "").strip().lower()
            if d:
                labs.add(d)
        for f in inc.get("findings", []):
            d = str(f.get("finding_description") or "").strip().lower()
            if d:
                labs.add(d)
            p = str(f.get("person_description") or "").strip().lower()
            if p and p not in ("nan", "none", "0", "(unspecified person)"):
                labs.add(f"person: {p}")
        support.update(labs)
    _SUPPORT_CACHE[key] = dict(support)
    return _SUPPORT_CACHE[key]


def _incident_bn_labels(inc: dict) -> set:
    """All labels of an incident that can be BN nodes (occurrences, findings,
    person findings)."""
    labs = set()
    for s in inc.get("sequence_of_events", []):
        d = str(s.get("Occurrence_Description") or "").strip().lower()
        if d:
            labs.add(d)
    for f in inc.get("findings", []):
        d = str(f.get("finding_description") or "").strip().lower()
        if d:
            labs.add(d)
        p = str(f.get("person_description") or "").strip().lower()
        if p and p not in ("nan", "none", "0", "(unspecified person)"):
            labs.add(f"person: {p}")
    return labs


def retrieval_facts(query: str, names, dataset: dict,
                    top_k: int = 100, min_fq: float = 0.15,
                    min_lift: float = 3.0, top_m: int = 3,
                    max_conf: float = 0.95):
    """Soft facts suggested by the narrative, via the SAME embedding retrieval
    the counting pipeline uses.

    The top_k most-similar accidents form a similarity-weighted pool. A label
    with weighted frequency f_q in the pool and base rate f_0 in the dataset
    is kept when f_q >= min_fq and its odds lift over f_0 >= min_lift.
    The soft-evidence confidence is c = f_q itself: "given this narrative,
    the fact holds with probability about f_q" -- an empirical, calibrated
    number (apply_evidence turns it into a likelihood ratio against the
    node's own prior, i.e. Jeffrey conditioning).

    Returns [(node, confidence, reason), ...] strongest first.
    """
    from collections import defaultdict
    import main_app

    names = set(names)
    support = _label_support(dataset)
    n_total = len(dataset)

    emb = main_app.get_embedding(query)
    scores, matches = main_app.find_top_matches(emb)
    pool, seen = [], set()
    for s, m in zip(scores, matches):
        if m.get("source") != "incident":
            continue
        ev = m.get("ev_id")
        if not ev or ev in seen or ev not in dataset:
            continue
        seen.add(ev)
        pool.append((float(s), dataset[ev]))
        if len(pool) >= top_k:
            break
    wsum = sum(w for w, _ in pool)
    if not wsum:
        return []

    mass = defaultdict(float)
    for w, inc in pool:
        for lab in _incident_bn_labels(inc) & names:
            # persons are only evidence when NAMED in the narrative (the
            # deterministic pass); crew-composition correlations are not facts
            if not lab.startswith("person: "):
                mass[lab] += w

    out = []
    for lab, m in mass.items():
        f_q = m / wsum
        f_0 = support.get(lab, 0) / n_total
        if f_q < min_fq or not f_0 or f_q >= 1.0:
            continue
        lift = (f_q / (1 - f_q)) / (f_0 / (1 - f_0))
        if lift < min_lift:
            continue
        conf = round(min(max_conf, f_q), 3)
        out.append((lab, conf,
                    f"suggested by the {len(pool)} most-similar accidents "
                    f"(seen in {f_q:.0%} of them vs {f_0:.1%} baseline, "
                    f"odds x{lift:.1f})"))
    out.sort(key=lambda t: -t[1])
    return out[:top_m]


# ---------------------------------------------------------------------------
# Severity statements -- NTSB factual narratives usually SAY the damage level
# ("the airplane sustained substantial damage") and often the injury level.
# A stated level is not gospel (writers say "minor damage" for accidents coded
# NONE, etc.), so it enters the network as VIRTUAL evidence on the multi-state
# severity node with likelihood L(j) = P(stated=i | coded=j), estimated from
# the TRAINING window's own narratives (no held-out data touches calibration).
# ---------------------------------------------------------------------------
DMG_CODES = ["DEST", "SUBS", "MINR", "NONE"]   # same order as bn_upgraded.DMG_STATES
INJ_CODES = ["FATL", "SERS", "MINR", "NONE"]   # same order as bn_upgraded.INJ_STATES

_DMG_STATED = [  # priority: worst level first
    ("DEST", re.compile(r"\bwas destroyed\b|\bdestroyed the (airplane|aircraft|helicopter)\b|\b(airplane|aircraft|helicopter)\b[^.]{0,60}\bdestroyed\b")),
    ("SUBS", re.compile(r"\bsubstantial(ly)? damage(d)?\b|\bsustained substantial\b")),
    ("MINR", re.compile(r"\bminor damage\b")),
    ("NONE", re.compile(r"\bwas not damaged\b|\bno damage\b|\bundamaged\b")),
]
_INJ_STATED = [
    ("FATL", re.compile(r"\bfatal(ly)? injur|\bwas killed\b|\bwere killed\b|\bfatalit")),
    ("SERS", re.compile(r"\bserious(ly)? injur|\bsustained serious\b")),
    ("MINR", re.compile(r"\bminor injur")),
    ("NONE", re.compile(r"\bwas not injured\b|\bwere not injured\b|\bno injur|\buninjured\b|\bnot injured\b")),
]


def severity_statements(query: str) -> dict:
    """Damage / injury level STATED in the narrative text (or None).
    Returns {"damage": code|None, "injury": code|None} with worst-first priority."""
    n = query.lower()

    def first(pats):
        for code, p in pats:
            if p.search(n):
                return code
        return None
    return {"damage": first(_DMG_STATED), "injury": first(_INJ_STATED)}


_SEV_CAL_CACHE: dict[int, dict] = {}


def severity_likelihoods(dataset: dict, alpha: float = 0.5) -> dict:
    """Calibrated likelihood matrices L[i][j] = P(stated=i | coded=j), estimated
    from the dataset's own narratives with Laplace smoothing alpha.

    Returns {"damage": {stated_code: [L over DMG_CODES]},
             "injury": {stated_code: [L over INJ_CODES]}}.
    The vectors follow the state order of the upgraded network's multi-state
    severity nodes, so they can be handed to pyAgrum addEvidence directly.
    """
    key = id(dataset)
    if key in _SEV_CAL_CACHE:
        return _SEV_CAL_CACHE[key]
    import prognosis as pg

    dmg_n = {i: {j: 0 for j in DMG_CODES} for i in DMG_CODES}
    inj_n = {i: {j: 0 for j in INJ_CODES} for i in INJ_CODES}
    dmg_tot = {j: 0 for j in DMG_CODES}
    inj_tot = {j: 0 for j in INJ_CODES}
    for inc in dataset.values():
        narr = str(inc.get("narr_accf") or "").strip()
        if len(narr) < 100:
            continue
        stated = severity_statements(narr)
        dmg = str(inc.get("damage") or "").upper()
        if dmg in DMG_CODES:
            dmg_tot[dmg] += 1
            if stated["damage"]:
                dmg_n[stated["damage"]][dmg] += 1
        inj = pg.zhang_injury_code(inc)
        inj_tot[inj] += 1
        if stated["injury"]:
            inj_n[stated["injury"]][inj] += 1

    def lik(counts, totals, codes):
        out = {}
        for i in codes:
            out[i] = [(counts[i][j] + alpha) / (totals[j] + 2 * alpha)
                      for j in codes]
        return out
    _SEV_CAL_CACHE[key] = {"damage": lik(dmg_n, dmg_tot, DMG_CODES),
                           "injury": lik(inj_n, inj_tot, INJ_CODES)}
    return _SEV_CAL_CACHE[key]


def severity_retrieval_distributions(query: str, dataset: dict,
                                     top_k: int = 100, alpha: float = 0.5):
    """Similarity-weighted injury/damage distributions among the top_k accidents
    most similar to the narrative -- the SAME retrieval pool the soft-fact pass
    uses, aggregated over the severity outcomes instead of the event labels.

    Returns {"injury": [4 floats over INJ_CODES], "damage": [4 over DMG_CODES]}
    (Laplace-smoothed), or {} when the pool is empty. Callers turn these into
    virtual-evidence likelihoods against the network's own severity priors:
        L(j) = f_q(j) / p0(j)
    so the posterior lands near f_q when this is the only severity evidence."""
    import main_app
    import prognosis as pg

    emb = main_app.get_embedding(query)
    scores, matches = main_app.find_top_matches(emb)
    pool, seen = [], set()
    for s, m in zip(scores, matches):
        if m.get("source") != "incident":
            continue
        ev = m.get("ev_id")
        if not ev or ev in seen or ev not in dataset:
            continue
        seen.add(ev)
        pool.append((float(s), dataset[ev]))
        if len(pool) >= top_k:
            break
    if not pool:
        return {}

    inj = [alpha] * 4
    dmg = [alpha] * 4
    inj_idx = {c: i for i, c in enumerate(INJ_CODES)}
    dmg_idx = {c: i for i, c in enumerate(DMG_CODES)}
    for w, inc in pool:
        inj[inj_idx[pg.zhang_injury_code(inc)]] += w
        d = str(inc.get("damage") or "").upper()
        if d in dmg_idx:
            dmg[dmg_idx[d]] += w
    si, sd = sum(inj), sum(dmg)
    return {"injury": [x / si for x in inj], "damage": [x / sd for x in dmg]}


def severity_virtual_evidence(query: str, dataset: dict) -> dict:
    """Virtual-evidence likelihood vectors for the severity nodes, from what the
    narrative STATES. Returns {} when nothing is stated, else a subset of
    {"injury": [4 floats], "damage": [4 floats]} in network state order."""
    stated = severity_statements(query)
    if not (stated["damage"] or stated["injury"]):
        return {}
    lik = severity_likelihoods(dataset)
    out = {}
    if stated["injury"]:
        out["injury"] = lik["injury"][stated["injury"]]
    if stated["damage"]:
        out["damage"] = lik["damage"][stated["damage"]]
    return out


def _node_priors(bn, nodes):
    """Marginal prior P(node='Yes') for each node (no evidence)."""
    import pyagrum as gum
    if not nodes:
        return {}
    ie = gum.LazyPropagation(bn)
    for n in nodes:
        ie.addTarget(n)
    ie.makeInference()
    out = {}
    for n in nodes:
        v = bn.variable(n)
        yes = [i for i in range(v.domainSize()) if v.label(i) == "Yes"][0]
        out[n] = float(ie.posterior(n)[yes])
    return out


def apply_evidence(ie, bn, confidence: dict):
    """Set parsed evidence on a pyAgrum inference engine.

    confidence 1.0  -> hard evidence (node clamped to 'Yes').
    confidence c<1  -> JEFFREY conditioning via Pearl virtual evidence: the
        narrative says the fact holds with probability ~c, so the likelihood
        ratio is set against the network's own prior p0,
            LR = [c/(1-c)] / [p0/(1-p0)],
        which makes the updated belief in the fact land at c (exactly, when
        it is the only soft fact). A plain [c, 1-c] likelihood would be
        swallowed by the per-flight priors (~1e-7) and move nothing.
    """
    soft = [n for n, c in confidence.items() if c < 0.999]
    priors = _node_priors(bn, soft)
    for node, c in confidence.items():
        v = bn.variable(node)
        if c >= 0.999:
            ie.addEvidence(node, "Yes")
            continue
        p0 = min(max(priors.get(node, 0.5), 1e-12), 1 - 1e-12)
        lr = (c / (1.0 - c)) / (p0 / (1.0 - p0))
        # likelihood vector proportional to [LR, 1]; scale to max 1 for pyAgrum
        l_yes, l_no = (1.0, 1.0 / lr) if lr >= 1.0 else (lr, 1.0)
        lik = [l_yes if v.label(i) == "Yes" else l_no
               for i in range(v.domainSize())]
        ie.addEvidence(node, lik)


def parse_query_to_bn_evidence(query: str, bn_names, dataset=None,
                               semantic: bool = False,
                               semantic_threshold: float = 0.45) -> dict:
    """Parse a free-text narrative into BN evidence nodes.

    bn_names : iterable of node names from the built network (bn.names()).
    semantic : when True and the deterministic passes leave gaps, embed each
               clause of the query and add the best-matching node as SOFT
               evidence (confidence < 1) -- same embedding model as retrieval.
    Returns {"evidence": [...], "trace": [(node, reason), ...],
             "confidence": {node: float}}; every node is guaranteed to be in
    bn_names; deterministic parses have confidence 1.0.
    """
    names = set(bn_names)
    nq = _norm(query)
    evidence: list[str] = []
    trace: list[tuple[str, str]] = []
    confidence: dict[str, float] = {}

    def add(node: str, reason: str, conf: float = 1.0):
        if node in names and node not in evidence:
            evidence.append(node)
            trace.append((node, reason))
            confidence[node] = conf

    # -- pass 1: the outcome family (same parser the counting pipeline uses) --
    # Guard against detect_outcome's fuzzy fallback: every content word of the
    # detected outcome must actually be spoken in the query ("engine instrument"
    # must NOT become the occurrence "engine tearaway" on one shared token).
    det = zd.detect_outcome(query, dataset=dataset)
    if det:
        name, targets = det
        qtok = set(nq.split())
        spoken = (set(_node_core(name).split()) - _STOPWORDS) <= qtok
        if spoken and _asserted(_node_core(name), query):
            if name in names:
                add(name, "outcome parsed from the narrative")
            else:
                for t in sorted(targets, key=len):   # shortest = most generic label
                    if t in names:
                        add(t, "outcome parsed from the narrative")
                        break

    # -- pass 2: person nodes ---------------------------------------------------
    person_nodes = {n for n in names if n.startswith("person: ")}

    def person_desc(n: str) -> str:
        return _node_core(n[len("person: "):].replace("-", " "))

    for alias, key in _PERSON_ALIASES:
        if not _phrase_in(alias, nq) or not _asserted(alias, query):
            continue
        want = key.replace("-", " ")
        # prefer the exact node ("pilot-in-command"), then the shortest
        # containing variant -- never the "..., certified flight instructor" one
        hits = [n for n in person_nodes
                if person_desc(n) == want or _phrase_in(want, person_desc(n))]
        hits.sort(key=lambda n: (person_desc(n) != want, len(person_desc(n)), n))
        if hits:
            add(hits[0], f"person mentioned ('{alias}')")
    # generic: a person description spoken verbatim ("flight attendant", ...)
    if not any(e.startswith("person: ") for e in evidence):
        for n in sorted(person_nodes):
            desc = person_desc(n)
            if len(desc) >= 5 and _phrase_in(desc, nq):
                add(n, "person mentioned")
                break

    # -- pass 3: other event/finding nodes named in the query -------------------
    # cores already covered by evidence from passes 1-2 (so 'loss of engine
    # power (total) - nonmechanical' is not re-added next to the outcome node)
    accepted_cores: list[str] = [_node_core(e) for e in evidence
                                 if not e.startswith("person: ")]
    candidates = []
    for n in names:
        if n.startswith("person: ") or n in evidence:
            continue
        core = _node_core(n)
        if not core or core in _STOP_CORES:
            continue
        # single common words are too weak to clamp a node to 100% ("the
        # gauges MONITORING the engine" must not assert the node
        # 'monitoring'); multi-word phrases only. Single-word outcomes
        # ("fire") still enter through the outcome parser above, and
        # everything else can arrive as soft evidence via retrieval.
        if " " not in core:
            continue
        if _phrase_in(core, nq) and _asserted(core, query):
            candidates.append((core, n))
    # longest core first; drop cores subsumed by an already-accepted longer one
    candidates.sort(key=lambda t: -len(t[0]))
    for core, n in candidates:
        if any(_phrase_in(core, prev) for prev in accepted_cores):
            continue
        add(n, "event/finding named in the narrative")
        accepted_cores.append(core)

    # -- pass 4 (optional): retrieval soft facts for paraphrases -----------------
    # The same embedding retrieval the counting pipeline uses suggests facts
    # that are strongly over-represented among the accidents most similar to
    # the narrative; they enter as VIRTUAL evidence with likelihood-ratio
    # derived confidence (Pearl's method) -- suggestion, not assertion.
    # It only fires when the narrative's EVENT content is ungrounded: once a
    # deterministic event/outcome parse exists, the narrative is already in
    # the network and adding correlated soft facts would double-count it.
    have_event_evidence = any(not e.startswith("person: ") for e in evidence)
    if semantic and dataset and not have_event_evidence:
        try:
            soft = retrieval_facts(query, names, dataset)
        except Exception:
            soft = []
        for node, conf, why in soft:
            if node in evidence:
                continue
            # skip facts whose core is already covered by hard evidence
            core = _node_core(node)
            if any(_phrase_in(core, prev) or _phrase_in(prev, core)
                   for prev in accepted_cores if prev):
                continue
            # the narrative SPEAKS this fact but only negated/hypothetically
            # ("no fire on board"): retrieval must not sneak it back in as a
            # soft suggestion (embeddings are blind to negation)
            if _phrase_in(core, nq) and not _asserted(core, query):
                continue
            add(node, why, conf)

    return {"evidence": evidence, "trace": trace, "confidence": confidence}

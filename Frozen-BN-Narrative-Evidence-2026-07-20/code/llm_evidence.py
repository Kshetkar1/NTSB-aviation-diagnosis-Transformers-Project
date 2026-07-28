"""LLM narrative -> BN evidence extraction (Maha's 'use an LLM' request).

DESIGN PRINCIPLE (the defensible architecture): the LLM never produces a
probability that we report. It performs exactly ONE job -- reading the
narrative and proposing the structured evidence dictionary
{node: confidence} -- and the Bayesian network computes every posterior.
That keeps all reported numbers auditable (same CPTs, same exact inference
validated against Zhang's 93 published values) while the LLM contributes
what it is actually good at: language understanding.

Guardrails applied to the LLM output:
  * Nodes must come verbatim from the network's vocabulary; anything else is
    dropped (and reported in the trace) -- the LLM cannot invent evidence.
  * "asserted" facts (explicitly stated) are forced to confidence 1.0.
  * "implied" facts keep the LLM's confidence, clamped to [0.05, 0.95];
    these enter the network as Jeffrey/virtual evidence, same machinery as
    the retrieval soft facts.
  * temperature=0 and a fixed seed for reproducibility.

Usage:
    from llm_evidence import llm_parse_evidence
    out = llm_parse_evidence(narrative, bn_node_names)
    # {"evidence": [...], "confidence": {node: c}, "trace": [(node, why)],
    #  "dropped": [...], "raw": {...}}
"""
from __future__ import annotations

import json

DEFAULT_MODEL = "gpt-4o-mini"

_SYSTEM = """\
You are an NTSB aviation-accident investigator. You will read an accident
narrative and identify which facts from an official vocabulary of event /
finding / person nodes are supported by the narrative.

Rules:
1. Only use node names EXACTLY as written in the vocabulary -- copy them
   character-for-character, including punctuation. Never invent, shorten, or
   paraphrase a node name. If the fact you want has no matching node, omit it.
1b. When both a GENERIC node and a more SPECIFIC node could match, choose the
   one at the same level of detail as the narrative: a narrative that says
   "an engine instrument" matches the generic 'engine instrument', not a
   specific gauge node, unless the specific gauge is named.
1c. Never ALSO select a generic/parent category node when you have selected a
   specific node of that category. NTSB coding uses the generic code (e.g.
   'landing gear', 'combustion assembly') ONLY when the specific component is
   unknown; if the narrative names 'landing gear, main gear strut', select
   that node alone -- do NOT add 'landing gear'.
1d. The narrative usually DESCRIBES facts instead of naming them. Map the
   description to the vocabulary node whose official meaning matches it,
   even when the words differ completely: "the burner section of the
   engine" is the combustion assembly; "the gauges monitoring the engine"
   are engine instruments; "came in too high and too fast" is an
   unstabilized approach. Mark such mappings "implied" with your
   confidence. Only omit a fact when NO vocabulary node means what the
   narrative describes.
2. Mark each selected node:
   - "asserted": the narrative explicitly states this fact.
   - "implied": not stated, but strongly suggested; give a probability
     "confidence" between 0.05 and 0.95 that the fact holds given the
     narrative.
3. Select only facts about THIS flight/accident. Do not select outcomes the
   narrative merely speculates about.
3b. NEVER select a fact the narrative NEGATES or rules out ("there was no
   fire", "the engine did not lose power", "no injuries") or mentions only
   hypothetically ("the pilot worried about a loss of engine power"). A
   negated or feared fact is NOT evidence that the fact occurred.
4. Be precise about word sense: e.g. metal fatigue in a component is NOT the
   node 'fatigue (flight and ground schedule)' (which means human/crew
   fatigue). Do not select a node whose official meaning differs from the
   narrative's meaning, even if the words overlap.
5. Person nodes (prefixed 'person: ') only when that person/role is
   explicitly mentioned as involved.
6. Prefer few, well-supported facts (typically 1-6) over many weak ones.

Return JSON only:
{"evidence": [{"node": "<verbatim node name>",
               "status": "asserted" | "implied",
               "confidence": <float, only for implied>,
               "why": "<short quote or reason>"}]}
"""


def _complete_openai(model: str, user: str, client=None) -> str:
    import main_app
    if client is None:
        client = main_app.get_client()
    kwargs = dict(
        model=model,
        temperature=0,
        seed=7,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": user}],
    )
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        # newer model families reject sampling params; retry without them
        if "temperature" in str(e) or "seed" in str(e) or "param" in str(e):
            kwargs.pop("temperature", None)
            kwargs.pop("seed", None)
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    return resp.choices[0].message.content


def _complete_anthropic(model: str, user: str) -> str:
    """Claude path. Needs ANTHROPIC_API_KEY in the environment (.env)."""
    import anthropic
    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
    kwargs = dict(
        model=model,
        max_tokens=2000,
        temperature=0,
        system=_SYSTEM + "\nReturn ONLY the JSON object, no prose, no "
                         "markdown fences.",
        messages=[{"role": "user", "content": user}],
    )
    try:
        resp = client.messages.create(**kwargs)
    except anthropic.BadRequestError as e:
        if "temperature" in str(e):   # deprecated on newest models
            kwargs.pop("temperature", None)
            resp = client.messages.create(**kwargs)
        else:
            raise
    # newest models may emit thinking blocks first; take the text block
    for block in resp.content:
        if getattr(block, "type", "") == "text":
            return block.text
    return ""


def _extract_json(text: str) -> dict:
    """Parse a JSON object out of a model reply (tolerates code fences)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        t = t[t.find("{"):]
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        t = t[start:end + 1]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return {"evidence": []}


def llm_parse_evidence(narrative: str, bn_names, model: str = DEFAULT_MODEL,
                       client=None) -> dict:
    """Extract BN evidence from a narrative with an LLM. See module docstring.

    model: any OpenAI chat model, or an Anthropic model (name starting with
    'claude', requires ANTHROPIC_API_KEY)."""
    names = list(bn_names)
    name_set = set(names)

    vocab = "\n".join(sorted(names))
    user = (f"VOCABULARY ({len(names)} nodes):\n{vocab}\n\n"
            f"NARRATIVE:\n{narrative.strip()}\n\n"
            "Extract the supported facts as JSON.")

    if model.startswith("claude"):
        raw_text = _complete_anthropic(model, user)
    else:
        raw_text = _complete_openai(model, user, client)
    raw = _extract_json(raw_text)

    def _core(s: str) -> set:
        import re
        s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
        return set(s.split())

    def repair(bad: str):
        """Map an off-vocabulary name to a vocab node ONLY when unambiguous:
        every token of the LLM's name appears in exactly one node's tokens."""
        toks = _core(bad)
        if not toks:
            return None
        hits = [n for n in names if toks <= _core(n)]
        return hits[0] if len(hits) == 1 else None

    evidence, confidence, trace, dropped = [], {}, [], []
    for item in raw.get("evidence", []):
        node = str(item.get("node", "")).strip()
        status = str(item.get("status", "")).strip().lower()
        why = str(item.get("why", ""))[:160]
        if node not in name_set:
            fixed = repair(node)
            if fixed is None:
                dropped.append(node)      # hallucination guard
                continue
            why = f"[name repaired from {node!r}] " + why
            node = fixed
        if node in confidence:
            continue
        if status == "asserted":
            c = 1.0
        else:
            try:
                c = float(item.get("confidence", 0.5))
            except (TypeError, ValueError):
                c = 0.5
            c = min(max(c, 0.05), 0.95)
        evidence.append(node)
        confidence[node] = c
        tag = "asserted" if c >= 0.999 else f"implied ({c:.0%})"
        trace.append((node, f"LLM {tag}: {why}"))

    return {"evidence": evidence, "confidence": confidence, "trace": trace,
            "dropped": dropped, "raw": raw, "model": model}


def combined_parse(query: str, bn_names, dataset: dict,
                   model: str = DEFAULT_MODEL) -> dict:
    """TIERED narrative -> evidence parser.

    Tier 1  deterministic parse (query_to_bn) WITHOUT the retrieval soft
            pass. When the narrative names facts, this lands on exactly the
            evidence a human would click -- the validated, reproducible
            path -- and the LLM is never called.
    Tier 2  when tier 1 finds no event/outcome evidence (paraphrased or
            vague wording), the LLM proposes facts and hybrid_confidence
            grounds every strength in the data (measured f_q, lift rescue
            for rare facts). The retrieval soft facts are deliberately NOT
            unioned in here: on adversarial text (negations, hypotheticals)
            the retrieval layer suggests the very facts the text rules out,
            while the prompted LLM excludes them. Falls back to tier 3 on
            any API failure.
    Tier 3  retrieval soft facts alone (the pre-LLM behaviour).

    Person evidence from tier 1 is kept in all tiers. Returns the same
    shape as parse_query_to_bn_evidence, plus "tier".
    """
    import query_to_bn as qb

    det = qb.parse_query_to_bn_evidence(query, bn_names, dataset=dataset,
                                        semantic=False)
    have_event = any(not e.startswith("person: ") for e in det["evidence"])
    if have_event:
        det["tier"] = 1
        return det

    conf = dict(det["confidence"])          # keep person hard evidence
    trace = list(det["trace"])
    tier = 2
    try:
        raw = llm_parse_evidence(query, bn_names, model=model)
        hyb = hybrid_confidence(query, raw, dataset)
        why = {n: w for n, w in raw["trace"]}
        for n, c in hyb.items():
            if n not in conf:
                conf[n] = c
                trace.append((n, why.get(n, "LLM + data-grounded strength")))
    except Exception:
        tier = 3
        soft = qb.parse_query_to_bn_evidence(query, bn_names, dataset=dataset,
                                             semantic=True)
        conf, trace = soft["confidence"], soft["trace"]
    return {"evidence": list(conf), "trace": trace, "confidence": conf,
            "tier": tier}


def measured_fq(query: str, nodes, dataset: dict, top_k: int = 100):
    """Empirical strength of each fact given the narrative: its similarity-
    weighted frequency among the top_k most-similar accidents (the SAME pool
    the retrieval soft-fact pass uses). This is the calibration source for
    hybrid evidence -- a measured number, not a model opinion."""
    import query_to_bn as qb
    import main_app

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
    wsum = sum(w for w, _ in pool) or 1.0
    out = {}
    for node in nodes:
        m = sum(w for w, inc in pool
                if node in qb._incident_bn_labels(inc))
        out[node] = m / wsum
    return out


def hybrid_confidence(query: str, llm_out: dict, dataset: dict,
                      floor: float = 0.05, cap: float = 0.95,
                      min_lift: float = 5.0, lift_floor: float = 0.005) -> dict:
    """HYBRID evidence: LLM chooses WHICH facts, the data decides HOW STRONGLY.

    * A fact stays HARD (1.0) only if it is verifiable in the narrative text
      (every content word of the node's core appears in the query) -- i.e. the
      narrative really asserts it; the LLM cannot up-grade an inference.
    * Every other fact becomes SOFT with confidence = measured f_q (its
      weighted frequency among the 100 accidents most similar to the
      narrative), clamped to cap. A soft fact is KEPT when either
        - f_q >= floor (common enough among similar accidents), or
        - f_q >= lift_floor AND its odds lift over the dataset base rate
          f_0 is >= min_lift (RARE facts, e.g. specific components, are
          legitimately low-frequency; over-representation is the signal --
          same criterion the retrieval soft-fact pass uses).
      Facts with no data support (f_q ~ 0) or no lift (generic labels that
      are frequent everywhere) are dropped.
    """
    import query_to_bn as qb

    nq = qb._norm(query)
    qtok = set(nq.split())
    conf, soft_nodes = {}, []
    for node in llm_out["evidence"]:
        core_words = set(qb._node_core(node).split()) - qb._STOPWORDS
        verbatim = bool(core_words) and core_words <= qtok
        if verbatim and llm_out["confidence"].get(node, 0) >= 0.999:
            conf[node] = 1.0
        else:
            soft_nodes.append(node)
    if soft_nodes:
        fq = measured_fq(query, soft_nodes, dataset)
        support = qb._label_support(dataset)
        n_total = len(dataset)
        for node in soft_nodes:
            c = fq.get(node, 0.0)
            if c <= 0.0 or c >= 1.0:
                continue
            f0 = support.get(node, 0) / n_total
            lift = ((c / (1 - c)) / (f0 / (1 - f0))) if f0 else float("inf")
            keep = c >= floor and (not f0 or lift >= 1.0)
            keep = keep or (c >= lift_floor and lift >= min_lift)
            if not keep:
                continue
            conf[node] = min(round(c, 3), cap)
    return conf

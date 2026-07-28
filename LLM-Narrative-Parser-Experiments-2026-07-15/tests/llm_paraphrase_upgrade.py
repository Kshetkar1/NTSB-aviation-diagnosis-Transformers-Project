#!/usr/bin/env python3
"""Can we IMPROVE paraphrase recovery without losing safety?

Two levers, tested against the baseline on the same 7 paraphrases and the
6 adversarial safety probes:

  PROMPT+    add an explicit paraphrase-mapping rule (the baseline rules
             push the model toward conservatism: it omits facts unless the
             node is nearly named).
  SHORTLIST  instead of the full ~1,900-node vocabulary, give the model a
             per-sentence candidate list: nodes carried by the 100 most
             similar accidents (retrieval) UNION nodes whose NAME embedding
             is close to the sentence embedding. The intended node must be
             findable in a list of ~60, not a haystack.

Both variants keep every guardrail: vocabulary-verbatim output, dropped
hallucinations, hybrid data-grounded confidences.

Run (needs OPENAI_API_KEY; uses gpt-4.1 and claude-sonnet-5 if key present):
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/llm_paraphrase_upgrade.py
Writes outputs/llm_paraphrase_upgrade.md.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

import prognosis as pg  # noqa: E402
import query_to_bn as qb  # noqa: E402
import llm_evidence as le  # noqa: E402
from llm_evidence import llm_parse_evidence, hybrid_confidence  # noqa: E402
from bn_upgraded import build_upgraded, INJ_NODE, DMG_NODE  # noqa: E402
from parser_flaws_and_paraphrases import FLAW_PROBES, PARAPHRASES  # noqa: E402
from llm_model_ladder import probe_safe  # noqa: E402

OUT = ROOT / "outputs" / "llm_paraphrase_upgrade.md"
NAME_EMB_CACHE = ROOT / "outputs" / "node_name_embeddings.json"

MODELS = ["gpt-4.1"]
if os.environ.get("ANTHROPIC_API_KEY"):
    MODELS.append("claude-sonnet-5")

_PARA_RULE = """\
1d. The narrative usually DESCRIBES facts instead of naming them. Map the
   description to the vocabulary node whose official meaning matches it,
   even when the words differ completely: "the burner section of the
   engine" is the combustion assembly; "the gauges monitoring the engine"
   are engine instruments; "came in too high and too fast" is an
   unstabilized approach. Mark such mappings "implied" with your
   confidence. Only omit a fact when NO vocabulary node means what the
   narrative describes.
"""

L: list[str] = []


def emit(s: str = "") -> None:
    print(s)
    L.append(s)


# ---------------- node-name embeddings (one batch call, cached) --------------
def name_embeddings(names):
    import main_app
    from config import EMBEDDING_MODEL  # same model as retrieval

    if NAME_EMB_CACHE.is_file():
        cached = json.loads(NAME_EMB_CACHE.read_text())
        if set(cached) >= set(names):
            return {n: np.array(cached[n]) for n in names}
    client = main_app.get_client()
    out = {}
    batch = 512
    todo = list(names)
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        resp = client.embeddings.create(input=chunk, model=EMBEDDING_MODEL)
        for n, d in zip(chunk, resp.data):
            out[n] = d.embedding
    NAME_EMB_CACHE.write_text(json.dumps(out))
    return {n: np.array(v) for n, v in out.items()}


def shortlist(sentence, names, dataset, name_embs, k_ret=30, k_emb=30,
              k_lift=20):
    """Candidate nodes: retrieval labels (by mass AND by lift) +
    name-embedding neighbours. Mass favours common facts; lift surfaces
    rare-but-over-represented ones (specific components)."""
    import main_app

    cands = {}
    # retrieval: labels carried by the most similar accidents
    emb = np.array(main_app.get_embedding(sentence))
    scores, matches = main_app.find_top_matches(emb.tolist())
    pool, seen = [], set()
    for s, m in zip(scores, matches):
        if m.get("source") != "incident":
            continue
        ev = m.get("ev_id")
        if not ev or ev in seen or ev not in dataset:
            continue
        seen.add(ev)
        pool.append((float(s), dataset[ev]))
        if len(pool) >= 100:
            break
    mass = {}
    for w, inc in pool:
        for lab in qb._incident_bn_labels(inc) & set(names):
            mass[lab] = mass.get(lab, 0.0) + w
    wsum = sum(w for w, _ in pool) or 1.0
    for lab in sorted(mass, key=lambda x: -mass[x])[:k_ret]:
        cands[lab] = "retrieval"
    support = qb._label_support(dataset)
    n_total = len(dataset)

    def lift(lab):
        f_q = mass[lab] / wsum
        f_0 = support.get(lab, 0) / n_total
        if not f_0 or f_q >= 1.0:
            return 0.0
        return (f_q / (1 - f_q)) / (f_0 / (1 - f_0))
    for lab in sorted(mass, key=lambda x: -lift(x))[:k_lift]:
        cands.setdefault(lab, "retrieval-lift")
    # name embeddings: nodes whose NAME is semantically near the sentence
    qv = emb / np.linalg.norm(emb)
    sims = []
    for n, v in name_embs.items():
        sims.append((float(qv @ (v / np.linalg.norm(v))), n))
    sims.sort(reverse=True)
    for _s, n in sims[:k_emb]:
        cands.setdefault(n, "name-similarity")
    return list(cands)


def run_variant(model, sentence, names, dataset, variant, name_embs):
    if variant == "baseline":
        raw = llm_parse_evidence(sentence, names, model=model)
    elif variant == "prompt+":
        old = le._SYSTEM
        le._SYSTEM = old.replace("2. Mark each selected node:",
                                 _PARA_RULE + "2. Mark each selected node:")
        try:
            raw = llm_parse_evidence(sentence, names, model=model)
        finally:
            le._SYSTEM = old
    else:  # shortlist (includes prompt+)
        cand = shortlist(sentence, names, dataset, name_embs)
        old = le._SYSTEM
        le._SYSTEM = old.replace("2. Mark each selected node:",
                                 _PARA_RULE + "2. Mark each selected node:")
        try:
            raw = llm_parse_evidence(sentence, cand, model=model)
        finally:
            le._SYSTEM = old
    return hybrid_confidence(sentence, raw, dataset)


def main():
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]
    name_embs = name_embeddings(names)

    emit("# Improving paraphrase recovery: prompt rule vs shortlist")
    emit()
    emit("| Model | Variant | Safety (6) | Paraphrase recovery (7) |")
    emit("|---|---|---|---|")

    detail: list[str] = []
    for model in MODELS:
        for variant in ["baseline", "prompt+", "shortlist"]:
            n_safe = n_rec = 0
            detail.append(f"\n## {model} / {variant}\n")
            for label, sentence in FLAW_PROBES:
                conf = run_variant(model, sentence, names, ds, variant,
                                   name_embs)
                ok = probe_safe(label, conf)
                n_safe += ok
                codes = "; ".join(f"`{n}` ({c:.2f})"
                                  for n, c in conf.items()) or "(none)"
                detail.append(f"- probe {label}: {codes} "
                              f"{'OK' if ok else '<-- UNSAFE'}")
            for lab, sentence, intended, _t, _s in PARAPHRASES:
                conf = run_variant(model, sentence, names, ds, variant,
                                   name_embs)
                got = intended in conf
                n_rec += got
                codes = "; ".join(f"`{n}` ({c:.2f})"
                                  for n, c in conf.items()) or "(none)"
                detail.append(f"- paraphrase {lab}: {codes} "
                              f"{'FOUND' if got else 'missed'}")
            emit(f"| {model} | {variant} | {n_safe}/6 | {n_rec}/7 |")

    emit()
    emit("Baseline reference: keyword parser 6/6 safety, 1/7 recovery; "
         "best frontier model with old prompt: 5/6, 3/7.")
    L.extend(detail)
    OUT.write_text("\n".join(L) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""THE decisive LLM experiment: re-code the ENTIRE dataset from narratives alone.

Zhang's analysis consumed the coded NTSB database (occurrence chains, findings,
damage / injury fields produced by investigators). Maha's question: could an
LLM do that work from the free-text narratives instead? This script answers it
literally: for every 1982-2006 accident that has a real narrative, gpt-4o-mini
reads ONLY the narrative and produces the same structured record investigators
produced --

    * ordered occurrence chain    (47-label occurrence vocabulary)
    * findings attached to an occurrence (694-label finding vocabulary)
    * damage level (DEST/SUBS/MINR/NONE) and highest injury (FATL/SERS/MINR/NONE)

Results are cached one JSON per accident in outputs/llm_recode/ so the run is
restartable. tests/llm_recode_analysis.py then rebuilds Table 7 / priors / the
full Boolean BN from these LLM-only records and compares against the coded
data and Zhang's published numbers.

Run (needs OPENAI_API_KEY + network):
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/llm_recode_dataset.py [--limit N] [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import prognosis as pg  # noqa: E402

CACHE = ROOT / "outputs" / "llm_recode"
MODEL = "gpt-4o-mini"

_SYSTEM = """\
You are an NTSB aviation-accident investigator. From an accident narrative you
will reconstruct the official coded record, using ONLY facts in the narrative.

You are given two vocabularies:
  OCCURRENCES: event types that form the accident's chronological chain.
  FINDINGS: factors/components/conditions attached to a specific occurrence.

Rules:
1. Copy names EXACTLY, character-for-character, from the vocabularies. Never
   invent, shorten, or paraphrase. If no vocabulary entry fits, omit the fact.
2. occurrences: the accident's event chain in CHRONOLOGICAL order (1-4 typical).
   Only events that happened to THIS flight.
3. findings: each finding attaches to one occurrence via "occ" = its 1-based
   index in your occurrences list. Only causes/factors/components the narrative
   supports. Never ALSO select a generic/parent category when you selected a
   specific entry of that category.
4. damage: the aircraft damage stated or clearly implied by the narrative --
   one of "DEST" (destroyed), "SUBS" (substantial), "MINR" (minor), "NONE".
   Use "UNK" only if the narrative says nothing about damage.
5. injury: the HIGHEST injury level among people, from the narrative --
   "FATL", "SERS", "MINR", or "NONE" (default "NONE" if uninjured/unstated).

Return JSON only:
{"occurrences": ["<name>", ...],
 "findings": [{"finding": "<name>", "occ": <int>}, ...],
 "damage": "<code>", "injury": "<code>"}
"""


def vocab_from_dataset(ds: dict):
    occ, fnd = set(), set()
    for inc in ds.values():
        descs, _ = pg._ordered_occurrences(inc)
        occ.update(descs)
        for f in inc.get("findings", []):
            d = pg._s(f.get("finding_description")).lower()
            if d:
                fnd.add(d)
    return sorted(occ), sorted(fnd)


def _core(s: str) -> set:
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return set(s.split())


def make_repair(names):
    cores = [(n, _core(n)) for n in names]

    def repair(bad: str):
        toks = _core(bad)
        if not toks:
            return None
        hits = [n for n, c in cores if toks <= c]
        return hits[0] if len(hits) == 1 else None
    return repair


def recode_one(client, narrative: str, occ_vocab, fnd_vocab,
               occ_repair, fnd_repair) -> dict:
    user = ("OCCURRENCES ({} labels):\n{}\n\nFINDINGS ({} labels):\n{}\n\n"
            "NARRATIVE:\n{}\n\nReconstruct the coded record as JSON.").format(
        len(occ_vocab), "\n".join(occ_vocab),
        len(fnd_vocab), "\n".join(fnd_vocab),
        narrative.strip()[:4000])
    resp = client.chat.completions.create(
        model=MODEL, temperature=0, seed=7,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": user}],
    )
    try:
        raw = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        raw = {}

    occ_set, fnd_set = set(occ_vocab), set(fnd_vocab)
    occs, dropped = [], []
    for o in raw.get("occurrences", []) or []:
        o = str(o).strip().lower()
        if o not in occ_set:
            fixed = occ_repair(o)
            if fixed is None:
                dropped.append(o)
                continue
            o = fixed
        occs.append(o)

    finds = []
    for f in raw.get("findings", []) or []:
        name = str(f.get("finding", "")).strip().lower()
        if name not in fnd_set:
            fixed = fnd_repair(name)
            if fixed is None:
                dropped.append(name)
                continue
            name = fixed
        try:
            k = int(f.get("occ", 1))
        except (TypeError, ValueError):
            k = 1
        k = min(max(k, 1), max(len(occs), 1))
        finds.append({"finding": name, "occ": k})

    dmg = str(raw.get("damage", "UNK")).strip().upper()
    if dmg not in ("DEST", "SUBS", "MINR", "NONE", "UNK"):
        dmg = "UNK"
    inj = str(raw.get("injury", "NONE")).strip().upper()
    if inj not in ("FATL", "SERS", "MINR", "NONE"):
        inj = "NONE"

    return {"occurrences": occs, "findings": finds,
            "damage": dmg, "injury": inj, "dropped": dropped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    import main_app
    client = main_app.get_client()

    ds = pg.load_dataset()
    occ_vocab, fnd_vocab = vocab_from_dataset(ds)
    occ_repair, fnd_repair = make_repair(occ_vocab), make_repair(fnd_vocab)
    print(f"vocab: {len(occ_vocab)} occurrences, {len(fnd_vocab)} findings")

    CACHE.mkdir(parents=True, exist_ok=True)
    todo = []
    for ev, inc in ds.items():
        narr = str(inc.get("narr_accf") or "").strip()
        if not narr or narr.lower() in ("nan", "none") or len(narr) < 80:
            continue
        if (CACHE / f"{ev}.json").exists():
            continue
        todo.append((ev, narr))
    if args.limit:
        todo = todo[:args.limit]
    n_cached = len(list(CACHE.glob("*.json")))
    print(f"already cached: {n_cached}; to run: {len(todo)}")

    lock = threading.Lock()
    done = [0]
    t0 = time.time()

    def work(ev, narr):
        for attempt in range(3):
            try:
                rec = recode_one(client, narr, occ_vocab, fnd_vocab,
                                 occ_repair, fnd_repair)
                (CACHE / f"{ev}.json").write_text(json.dumps(rec))
                return ev, None
            except Exception as e:
                if attempt == 2:
                    return ev, str(e)
                time.sleep(2 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, ev, narr) for ev, narr in todo]
        for fut in as_completed(futs):
            ev, err = fut.result()
            with lock:
                done[0] += 1
                if err:
                    print(f"  FAIL {ev}: {err[:100]}")
                if done[0] % 50 == 0:
                    rate = done[0] / (time.time() - t0)
                    print(f"  {done[0]}/{len(todo)} "
                          f"({rate:.1f}/s, ~{(len(todo)-done[0])/max(rate,0.01):.0f}s left)")

    print(f"finished: {len(list(CACHE.glob('*.json')))} records cached "
          f"in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

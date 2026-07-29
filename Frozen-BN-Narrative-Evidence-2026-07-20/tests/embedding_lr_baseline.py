#!/usr/bin/env python3
"""Strongest fair text baseline: logistic regression on the REDACTED
narrative embedding itself.

Reviewers will ask for this: if a linear model on the raw embedding beats
the BN chain, the parsed-evidence architecture must justify itself on
interpretability, not accuracy. We run it ourselves, first.

Protocol (identical split to the main eval):
  * Train: 1982-2006 window accidents with a factual narrative.
  * Test: the 2007-2019 held-out accidents.
  * Features: text-embedding-3-small of the LEAK-SAFE (redacted) narrative.
  * Labels: coded injury (Zhang derivation) / damage.

Embeddings are cached in outputs/emb_cache_redacted.npz keyed by ev_id and
a hash of the redacted text, so re-runs cost zero API calls.

Run:
  export PYTHONPATH="shared/code:Frozen-BN-Narrative-Evidence-2026-07-20/code"
  python Frozen-BN-Narrative-Evidence-2026-07-20/tests/embedding_lr_baseline.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT / "shared" / "code", FROZEN_DIR / "code"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

import prognosis as pg  # noqa: E402
import query_to_bn as qb  # noqa: E402

FULL = REPO_ROOT / "shared" / "data" / "processed" / "refined_dataset.json"
WINDOW = REPO_ROOT / "shared" / "data" / "processed" / "refined_dataset_1982_2006.json"
CACHE = FROZEN_DIR / "outputs" / "emb_cache_redacted.npz"
OUT = FROZEN_DIR / "outputs" / "emb_lr_baseline.json"
PER_ITEM = FROZEN_DIR / "outputs" / "emb_lr_per_item.json"

BATCH = 100


def _key(ev: str, text: str) -> str:
    return f"{ev}:{hashlib.sha1(text.encode()).hexdigest()[:12]}"


def embed_all(pop):
    """pop: [(ev_id, redacted_text)] -> np.ndarray, using/refreshing cache."""
    from main_app import get_client
    from config import EMBEDDING_MODEL

    cache = {}
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        cache = {k: z[k] for k in z.files}
    missing = [(ev, t) for ev, t in pop if _key(ev, t) not in cache]
    print(f"embeddings: {len(pop) - len(missing)} cached, {len(missing)} to fetch")
    client = get_client() if missing else None
    t0 = time.time()
    for i in range(0, len(missing), BATCH):
        chunk = missing[i:i + BATCH]
        resp = client.embeddings.create(
            input=[t.replace("\n", " ")[:8000] for _, t in chunk],
            model=EMBEDDING_MODEL)
        for (ev, t), d in zip(chunk, resp.data):
            cache[_key(ev, t)] = np.asarray(d.embedding, dtype=np.float32)
        done = i + len(chunk)
        print(f"  embedded {done}/{len(missing)} ({done/(time.time()-t0):.1f}/s)")
    if missing:
        np.savez_compressed(CACHE, **cache)
        print(f"cache updated: {CACHE.name} ({len(cache)} entries)")
    return np.stack([cache[_key(ev, t)] for ev, t in pop])


def main() -> int:
    full = json.loads(FULL.read_text())
    window_ids = set(json.loads(WINDOW.read_text()).keys())

    def rows(pred):
        out = []
        for k, inc in full.items():
            if pred(k) is False:
                continue
            narr = str(inc.get("narr_accf") or "").strip()
            if len(narr) < 100:
                continue
            out.append((k, inc, qb.redact_severity_phrases(narr[:4000])))
        out.sort(key=lambda t: t[0])
        return out

    train = rows(lambda k: k in window_ids)
    test = rows(lambda k: k not in window_ids)
    print(f"train {len(train)} (1982-2006)   test {len(test)} (2007-2019)")

    Xtr = embed_all([(k, t) for k, _, t in train])
    Xte = embed_all([(k, t) for k, _, t in test])

    inj = {"FATL": 0, "SERS": 1, "MINR": 2, "NONE": 3}
    dmg = {"DEST": 0, "SUBS": 1, "MINR": 2, "NONE": 3}
    yi_tr = np.array([inj[pg.zhang_injury_code(inc)] for _, inc, _ in train])
    yi_te = np.array([inj[pg.zhang_injury_code(inc)] for _, inc, _ in test])
    yd_tr = np.array([dmg.get(str(inc.get("damage") or "").upper(), -1)
                      for _, inc, _ in train])
    yd_te = np.array([dmg.get(str(inc.get("damage") or "").upper(), -1)
                      for _, inc, _ in test])

    results, per_item = {}, {}
    for tag, ytr, yte in (("injury", yi_tr, yi_te), ("damage", yd_tr, yd_te)):
        tr_ok, te_ok = ytr >= 0, yte >= 0
        clf = LogisticRegression(max_iter=5000, C=1.0)
        clf.fit(Xtr[tr_ok], ytr[tr_ok])
        proba = np.zeros((int(te_ok.sum()), 4))
        p = clf.predict_proba(Xte[te_ok])
        for j, cls in enumerate(clf.classes_):
            proba[:, int(cls)] = p[:, j]
        ys = yte[te_ok]
        acc = float(np.mean(proba.argmax(1) == ys))
        bri = float(np.mean([((proba[i] - np.eye(4)[ys[i]]) ** 2).sum()
                             for i in range(len(ys))]))
        results[tag] = {"acc": acc, "brier": bri, "n": int(te_ok.sum())}
        short = "inj" if tag == "injury" else "dmg"
        kept = [k for (k, _, _), ok in zip(test, te_ok) if ok]
        for i, ev in enumerate(kept):
            rec = per_item.setdefault(ev, {"id": ev})
            rec[f"{short}_true"] = int(ys[i])
            rec[f"emb-lr:{short}_pred"] = int(proba[i].argmax())
            rec[f"emb-lr:{short}_brier"] = float(
                ((proba[i] - np.eye(4)[ys[i]]) ** 2).sum())
            rec[f"emb-lr:{short}_probs"] = [round(float(x), 6) for x in proba[i]]
        print(f"emb-LR {tag}: acc {acc:.1%}   Brier {bri:.3f}   (n={te_ok.sum()})")

    OUT.write_text(json.dumps({"train_n": len(train), "test_n": len(test),
                               **results}, indent=2))
    PER_ITEM.write_text(json.dumps({"items": list(per_item.values())}, indent=1))
    print(f"wrote {OUT.name}, {PER_ITEM.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

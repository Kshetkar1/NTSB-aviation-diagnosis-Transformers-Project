#!/usr/bin/env python3
"""Supervised baseline for the DIAGNOSIS task: one-vs-rest logistic
regression on the redacted narrative embedding, predicting cause CATEGORIES.

Symmetric with the severity eval's emb-lr baseline: if a trained linear
model on the raw embedding beats the retrieval vote, the retrieval
architecture must justify itself on other grounds. We run it ourselves, first.

Protocol (identical split + rollup to tests/diagnosis_heldout_eval.py):
  * Train: 1982-2006 window accidents with narrative + >=1 mapped C/F
    category (legacy keyword rollup).
  * Test: the held-out 2007-2019 accidents from diagnosis_heldout_eval.json
    (CICTT top-level truth).
  * Features: text-embedding-3-small of the LEAK-SAFE (redacted) narrative,
    reusing outputs/emb_cache_redacted.npz (zero API calls when cached).
  * Ranking: per-category OvR probability; scored top-1-in-truth-set + MRR,
    McNemar against the unsupervised predictors.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/diagnosis_emb_lr_baseline.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT / "shared" / "code", FROZEN_DIR / "code",
          FROZEN_DIR / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

import prognosis as pg  # noqa: E402
import query_to_bn as qb  # noqa: E402
from diagnosis_heldout_eval import (CATS, categorize_legacy,  # noqa: E402
                                    mcnemar_exact, bootstrap_ci, score_ranking)
from embedding_lr_baseline import embed_all  # noqa: E402  (shared cache)

FULL = REPO_ROOT / "shared" / "data" / "processed" / "refined_dataset.json"
WINDOW = REPO_ROOT / "shared" / "data" / "processed" / "refined_dataset_1982_2006.json"
DIAG = FROZEN_DIR / "outputs" / "diagnosis_heldout_eval.json"
OUT_MD = FROZEN_DIR / "outputs" / "diagnosis_emb_lr.md"
OUT_JSON = FROZEN_DIR / "outputs" / "diagnosis_emb_lr.json"


def window_categories(inc: dict) -> set:
    out = set()
    for f in inc.get("findings") or []:
        if f.get("Cause_Factor") not in ("C", "F"):
            continue
        c = categorize_legacy(f.get("finding_description"),
                              f.get("person_description"))
        if c:
            out.add(c)
    return out


def main() -> int:
    full = json.loads(FULL.read_text())
    window_ids = set(json.loads(WINDOW.read_text()).keys())
    diag = json.loads(DIAG.read_text())
    per_item_prev = {r["id"]: r for r in diag["items"]}

    train = []
    for k, inc in full.items():
        if k not in window_ids:
            continue
        narr = str(inc.get("narr_accf") or "").strip()
        if len(narr) < 100:
            continue
        cats = window_categories(inc)
        if not cats:
            continue
        train.append((k, cats, qb.redact_severity_phrases(narr[:4000])))
    train.sort(key=lambda t: t[0])

    test = []
    for k, inc in full.items():
        if k not in per_item_prev:
            continue
        narr = str(inc.get("narr_accf") or "").strip()
        test.append((k, set(per_item_prev[k]["truth"]),
                     qb.redact_severity_phrases(narr[:4000])))
    test.sort(key=lambda t: t[0])
    print(f"train {len(train)} (window, mapped C/F)   test {len(test)} (held-out)")

    Xtr = embed_all([(k, t) for k, _, t in train])
    Xte = embed_all([(k, t) for k, _, t in test])

    # one-vs-rest per category
    prob = np.zeros((len(test), len(CATS)))
    for j, c in enumerate(CATS):
        y = np.array([1 if c in cats else 0 for _, cats, _ in train])
        clf = LogisticRegression(max_iter=5000, C=1.0)
        clf.fit(Xtr, y)
        prob[:, j] = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)]

    hits, mrrs = [], []
    for i, (k, truth, _) in enumerate(test):
        ranking = [CATS[j] for j in np.argsort(-prob[i])]
        hit, mrr = score_ranking(ranking, truth)
        hits.append(hit)
        mrrs.append(mrr)
        per_item_prev[k]["emb-lr:top1"] = hit
        per_item_prev[k]["emb-lr:mrr"] = mrr
        per_item_prev[k]["emb-lr:rank"] = ranking

    lo, hi = bootstrap_ci(hits)
    lines = ["# Diagnosis: supervised embedding-LR baseline", "",
             f"OvR logistic regression on redacted narrative embeddings, "
             f"trained on {len(train)} window accidents (mapped C/F "
             f"categories), tested on the same {len(test)} held-out "
             "accidents as `diagnosis_heldout_eval.md`.", "",
             "| Predictor | Top-1 | 95% CI | MRR | n |", "|---|---|---|---|---|"]
    lines.append(f"| emb-lr (supervised) | {100*np.mean(hits):.1f}% | "
                 f"[{100*lo:.1f}%, {100*hi:.1f}%] | {np.mean(mrrs):.3f} | "
                 f"{len(hits)} |")
    for p in ("retrieval", "bn-post", "freq"):
        ph = [r[f"{p}:top1"] for r in per_item_prev.values()]
        pm = [r[f"{p}:mrr"] for r in per_item_prev.values()]
        plo, phi = bootstrap_ci(ph)
        lines.append(f"| {p} | {100*np.mean(ph):.1f}% | "
                     f"[{100*plo:.1f}%, {100*phi:.1f}%] | {np.mean(pm):.3f} | "
                     f"{len(ph)} |")

    lines += ["", "## McNemar exact (top-1) vs emb-lr", "",
              "| A vs B | A only right | B only right | p |", "|---|---|---|---|"]
    for p in ("retrieval", "bn-post", "freq"):
        aw = sum(1 for r in per_item_prev.values()
                 if r[f"{p}:top1"] > r["emb-lr:top1"])
        bw = sum(1 for r in per_item_prev.values()
                 if r["emb-lr:top1"] > r[f"{p}:top1"])
        pv = mcnemar_exact(aw, bw)
        star = " *" if pv < 0.05 else ""
        lines.append(f"| {p} vs emb-lr | {aw} | {bw} | {pv:.4f}{star} |")

    OUT_MD.write_text("\n".join(lines) + "\n")
    OUT_JSON.write_text(json.dumps(
        {"n": len(test), "emb_lr_top1": float(np.mean(hits)),
         "emb_lr_mrr": float(np.mean(mrrs)), "ci": [lo, hi],
         "items": list(per_item_prev.values())}, indent=1))
    print("\n".join(lines))
    print(f"\nwrote {OUT_MD}\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

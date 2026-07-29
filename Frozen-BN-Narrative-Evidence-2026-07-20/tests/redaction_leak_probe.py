#!/usr/bin/env python3
"""Residual outcome-leakage probe for the severity redaction.

Jesse's concern, made measurable: after redact_severity_phrases strips
outcome wording, how much of the coded injury/damage label can a trivial
bag-of-words model still read off the text?

Method: TF-IDF + multinomial logistic regression, 5-fold CV on the 296
held-out narratives (this is a DIAGNOSTIC fit on the test narratives, not a
predictor -- it never touches the paper's pipeline). Compared on:
  * FULL text  (no redaction)
  * REDACTED text (what the leak-safe pipeline actually embeds)

Interpretation:
  * full >> redacted  -> the redaction is removing real outcome leakage.
  * redacted >> majority is EXPECTED: crash-mechanism wording (stall,
    terrain, forced landing) legitimately predicts severity. The check is
    the top-weight tokens: mechanism words are fine, outcome words
    ("died", "autopsy", "wreckage") mean the lexicon must grow.

Writes outputs/redaction_leak_probe.md.

Run from repo root:
  export PYTHONPATH="shared/code:Frozen-BN-Narrative-Evidence-2026-07-20/code"
  python Frozen-BN-Narrative-Evidence-2026-07-20/tests/redaction_leak_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT / "shared" / "code", FROZEN_DIR / "code"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import cross_val_predict  # noqa: E402

import prognosis as pg  # noqa: E402
import query_to_bn as qb  # noqa: E402

FULL = REPO_ROOT / "shared" / "data" / "processed" / "refined_dataset.json"
WINDOW = REPO_ROOT / "shared" / "data" / "processed" / "refined_dataset_1982_2006.json"
OUT = FROZEN_DIR / "outputs" / "redaction_leak_probe.md"

INJ_LAB = ["FATL", "SERS", "MINR", "NONE"]
DMG_LAB = ["DEST", "SUBS", "MINR", "NONE"]


def probe(texts, y, seed=42):
    """5-fold CV accuracy of TF-IDF + LR, plus top tokens per class."""
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                          sublinear_tf=True, min_df=2)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=3000, C=1.0)
    pred = cross_val_predict(clf, X, y, cv=5)
    acc = float(np.mean(pred == y))
    clf.fit(X, y)          # refit on all data only to inspect coefficients
    vocab = np.array(vec.get_feature_names_out())
    tops = {}
    for ci, cls in enumerate(clf.classes_):
        w = clf.coef_[ci] if len(clf.classes_) > 2 else clf.coef_[0]
        tops[int(cls)] = [str(t) for t in vocab[np.argsort(-w)[:12]]]
    return acc, tops


def main() -> int:
    full = json.loads(FULL.read_text())
    window_ids = set(json.loads(WINDOW.read_text()).keys())

    held = []
    for k, inc in full.items():
        if k in window_ids:
            continue
        narr = str(inc.get("narr_accf") or "").strip()
        if len(narr) < 100:
            continue
        held.append((k, inc, narr))
    held.sort(key=lambda t: t[0])
    print(f"held-out narratives: {len(held)}")

    texts_full = [narr[:4000] for _, _, narr in held]
    texts_red = [qb.redact_severity_phrases(t) for t in texts_full]
    n_changed = sum(a != b for a, b in zip(texts_full, texts_red))

    y_inj = np.array([{"FATL": 0, "SERS": 1, "MINR": 2, "NONE": 3}
                      [pg.zhang_injury_code(inc)] for _, inc, _ in held])
    dmg_map = {"DEST": 0, "SUBS": 1, "MINR": 2, "NONE": 3}
    y_dmg = np.array([dmg_map.get(str(inc.get("damage") or "").upper(), -1)
                      for _, inc, _ in held])
    dmg_ok = y_dmg >= 0

    lines = ["# Redaction residual-leak probe", "",
             f"n = {len(held)} held-out narratives (2007-2019); "
             f"{n_changed} changed by redaction.",
             "",
             "Probe: TF-IDF (1-2 grams) + logistic regression, 5-fold CV "
             "fit on the held-out narratives themselves (diagnostic only).",
             ""]

    for tgt, y, ok, labels in (("Injury", y_inj, np.ones(len(y_inj), bool), INJ_LAB),
                               ("Damage", y_dmg, dmg_ok, DMG_LAB)):
        yy = y[ok]
        tf = [t for t, o in zip(texts_full, ok) if o]
        tr = [t for t, o in zip(texts_red, ok) if o]
        maj = float(np.mean(yy == np.bincount(yy).argmax()))
        acc_f, tops_f = probe(tf, yy)
        acc_r, tops_r = probe(tr, yy)
        print(f"{tgt}: majority {maj:.1%}  full-text probe {acc_f:.1%}  "
              f"redacted probe {acc_r:.1%}")
        lines += [f"## {tgt}", "",
                  "| Text | Probe accuracy |", "|---|---|",
                  f"| majority class | {maj:.1%} |",
                  f"| full narrative | {acc_f:.1%} |",
                  f"| redacted narrative | {acc_r:.1%} |",
                  "",
                  "Top-weight tokens on REDACTED text (leak check -- these "
                  "must be mechanism words, not outcome words):", ""]
        for cls, toks in tops_r.items():
            lines.append(f"- **{labels[cls]}**: {', '.join(toks)}")
        lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

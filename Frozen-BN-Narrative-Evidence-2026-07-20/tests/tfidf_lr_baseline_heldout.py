#!/usr/bin/env python3
"""Raw-text supervised baseline: TF-IDF bag-of-words + logistic regression.

Provenance, stated honestly: the redaction leak probe
(tests/redaction_leak_probe.py) showed that this exact model family,
window-trained and scored on the redacted held-out narratives, reaches
~92% injury accuracy -- numerically the strongest injury predictor in the
repo. Hiding that inside a leak-audit file while the baseline table showed
only weaker baselines would be burying an inconvenient result, so this
script promotes it to a first-class baseline: same training cohort, same
redacted text, and per-item logging so it enters the McNemar/Holm family
in tests/heldout_significance.py.

Train: 1982-2006 window accidents with a factual narrative >= 100 chars
(the same 1,286 used by lr_baseline_heldout.py). Features: TF-IDF 1-2
grams on the REDACTED, 4000-char-truncated narrative (identical text
treatment to the main eval). Labels: coded injury (Zhang derivation) and
coded damage.

Test: the identical 296 held-out accidents (2007-2019).

Model: the probe's configuration, unchanged (TfidfVectorizer
max_features=20000, ngram 1-2, sublinear_tf, min_df=2;
LogisticRegression C=1.0). No tuning was performed on top of the probe
settings, precisely so this number cannot be accused of test-set fitting
beyond what the probe already reported.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/tfidf_lr_baseline_heldout.py
Writes outputs/tfidf_lr_baseline_heldout.json and
outputs/tfidf_lr_per_item.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "shared" / "code", FROZEN_DIR / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.path.insert(0, str(FROZEN_DIR / "tests"))

import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

import query_to_bn as qb  # noqa: E402
from frozenbn_heldout_narrative_bn_eval import truth_states, brier  # noqa: E402

FULL = REPO_ROOT / "shared" / "data" / "processed" / "refined_dataset.json"
WINDOW = (REPO_ROOT / "shared" / "data" / "processed" /
          "refined_dataset_1982_2006.json")
OUT_DIR = FROZEN_DIR / "outputs"


def main() -> int:
    full = json.loads(FULL.read_text())
    window_ids = set(json.loads(WINDOW.read_text()).keys())

    train_pop, test_pop = [], []
    for k, inc in full.items():
        narr = str(inc.get("narr_accf") or "").strip()
        if len(narr) < 100:          # same cohort filter as the main eval
            continue
        (train_pop if k in window_ids else test_pop).append((k, inc, narr))
    test_pop.sort(key=lambda t: t[0])
    print(f"train: {len(train_pop)} (1982-2006)   "
          f"test: {len(test_pop)} (2007-2019)")

    def prep(pop):
        texts = [qb.redact_severity_phrases(narr[:4000])
                 for _, _, narr in pop]
        yi, yd = [], []
        for _, inc, _ in pop:
            inj_t, dmg_t = truth_states(inc)
            yi.append(inj_t)
            yd.append(dmg_t if dmg_t is not None else -1)
        return texts, np.array(yi), np.array(yd)

    tr_texts, yi_tr, yd_tr = prep(train_pop)
    te_texts, yi_te, yd_te = prep(test_pop)
    kept_te = [k for k, _, _ in test_pop]

    results = {}
    per_item = {}
    for tag, ytr, yte in (("injury", yi_tr, yi_te), ("damage", yd_tr, yd_te)):
        tr_ok = ytr >= 0
        te_ok = yte >= 0
        # Probe configuration, verbatim (see module docstring).
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                              sublinear_tf=True, min_df=2)
        Xtr = vec.fit_transform([t for t, o in zip(tr_texts, tr_ok) if o])
        clf = LogisticRegression(max_iter=3000, C=1.0)
        clf.fit(Xtr, ytr[tr_ok])
        Xte = vec.transform([t for t, o in zip(te_texts, te_ok) if o])
        proba = np.zeros((int(te_ok.sum()), 4))
        p = clf.predict_proba(Xte)
        for j, cls in enumerate(clf.classes_):
            proba[:, cls] = p[:, j]
        ys = yte[te_ok]
        bri = float(np.mean([brier(proba[i], ys[i]) for i in range(len(ys))]))
        acc = float(np.mean(np.argmax(proba, axis=1) == ys))
        results[tag] = (bri, acc, int(te_ok.sum()))
        kept_ids = [kte for kte, ok in zip(kept_te, te_ok) if ok]
        short = "inj" if tag == "injury" else "dmg"
        for i, ev in enumerate(kept_ids):
            rec = per_item.setdefault(ev, {"id": ev})
            rec[f"{short}_true"] = int(ys[i])
            rec[f"tfidf-lr:{short}_pred"] = int(np.argmax(proba[i]))
            rec[f"tfidf-lr:{short}_brier"] = brier(proba[i], ys[i])
        print(f"TF-IDF LR {tag}: Brier {bri:.3f}   acc {acc:.1%}   "
              f"(n={int(te_ok.sum())})")

    (OUT_DIR / "tfidf_lr_baseline_heldout.json").write_text(json.dumps({
        "train_n": len(train_pop), "test_n": len(test_pop),
        "injury": {"brier": results["injury"][0],
                   "acc": results["injury"][1], "n": results["injury"][2]},
        "damage": {"brier": results["damage"][0],
                   "acc": results["damage"][1], "n": results["damage"][2]},
    }, indent=2))
    (OUT_DIR / "tfidf_lr_per_item.json").write_text(
        json.dumps({"items": list(per_item.values())}, indent=1))
    print(f"wrote {OUT_DIR / 'tfidf_lr_baseline_heldout.json'}")
    print(f"wrote {OUT_DIR / 'tfidf_lr_per_item.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

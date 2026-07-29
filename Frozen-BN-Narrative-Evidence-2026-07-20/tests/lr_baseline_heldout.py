#!/usr/bin/env python3
"""Classical baseline reviewers will ask for: multinomial logistic regression
on the SAME parsed-narrative evidence the BN chain consumes.

Train: 1982-2006 accidents with a factual narrative; features = the parsed
evidence vector (node confidence in [0,1], 0 when absent) from the SAME
deterministic + retrieval parser used by the BN pipeline; labels = coded
injury (Zhang derivation) / damage.

Test: the 2007-2019 held-out accidents (same population as
tests/heldout_narrative_bn_eval.py).

Note the asymmetry, stated honestly: the LR is SUPERVISED on severity labels;
the BN never fits severity from parsed evidence -- its CPTs come from the
Section 4 recipe. If the unsupervised-for-this-task BN chain is competitive
with a supervised discriminative model, the architecture costs little
accuracy while buying full posteriors, what-if reasoning, and auditability.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/lr_baseline_heldout.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
_SHARED = REPO_ROOT / "shared" / "code"
_FROZEN_CODE = FROZEN_DIR / "code"
for _p in (_SHARED, _FROZEN_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
ROOT = REPO_ROOT
sys.path.insert(0, str(FROZEN_DIR / "tests"))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

import prognosis as pg  # noqa: E402
import query_to_bn as qb  # noqa: E402
from frozenbn_heldout_narrative_bn_eval import truth_states, brier  # noqa: E402
from bn_upgraded import build_upgraded, INJ_NODE, DMG_NODE  # noqa: E402


def main():
    full = json.loads((ROOT / "shared" / "data" / "processed" /
                       "refined_dataset.json").read_text())
    window_ids = set(json.loads((ROOT / "shared" / "data" / "processed" /
                                 "refined_dataset_1982_2006.json").read_text()).keys())
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = sorted(n for n in bn.names() if n not in (INJ_NODE, DMG_NODE))
    idx = {n: i for i, n in enumerate(names)}

    def rows_for(pop):
        X, yi, yd, kept = [], [], [], []
        t0 = time.time()
        for k, (ev, inc, narr) in enumerate(pop):
            conf = qb.parse_query_to_bn_evidence(
                narr, names, dataset=ds, semantic=True)["confidence"]
            v = np.zeros(len(names))
            for n, c in conf.items():
                if n in idx:
                    v[idx[n]] = c
            inj_t, dmg_t = truth_states(inc)
            X.append(v)
            yi.append(inj_t)
            yd.append(dmg_t if dmg_t is not None else -1)
            kept.append(ev)
            if (k + 1) % 200 == 0:
                print(f"  parsed {k+1}/{len(pop)} "
                      f"({(k+1)/(time.time()-t0):.1f}/s)")
        return np.array(X), np.array(yi), np.array(yd), kept

    train_pop, test_pop = [], []
    for k, inc in full.items():
        narr = str(inc.get("narr_accf") or "").strip()
        if len(narr) < 200:
            continue
        (train_pop if k in window_ids else test_pop).append((k, inc, narr))
    print(f"train: {len(train_pop)} (1982-2006)   "
          f"test: {len(test_pop)} (2007-2019)")

    print("parsing train narratives...")
    Xtr, yi_tr, yd_tr, _ = rows_for(train_pop)
    print("parsing test narratives...")
    Xte, yi_te, yd_te, kept_te = rows_for(test_pop)

    results = {}
    per_item = {}                     # ev_id -> record for paired tests
    for tag, ytr, yte in (("injury", yi_tr, yi_te), ("damage", yd_tr, yd_te)):
        tr_ok = ytr >= 0
        te_ok = yte >= 0
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr[tr_ok], ytr[tr_ok])
        # probability matrix aligned to the 4 states
        proba = np.zeros((te_ok.sum(), 4))
        p = clf.predict_proba(Xte[te_ok])
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
            rec[f"lr:{short}_pred"] = int(np.argmax(proba[i]))
            rec[f"lr:{short}_brier"] = brier(proba[i], ys[i])
        print(f"\nLR {tag}: Brier {bri:.3f}   acc {acc:.0%}   "
              f"(n={te_ok.sum()})")

    out = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs" / "lr_baseline_heldout.json"
    out.write_text(json.dumps({
        "train_n": len(train_pop), "test_n": len(test_pop),
        "injury": {"brier": results["injury"][0], "acc": results["injury"][1],
                   "n": results["injury"][2]},
        "damage": {"brier": results["damage"][0], "acc": results["damage"][1],
                   "n": results["damage"][2]},
    }, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
    per_out = ROOT / "Frozen-BN-Narrative-Evidence-2026-07-20" / "outputs" / "lr_per_item.json"
    per_out.write_text(json.dumps({"items": list(per_item.values())},
                                  indent=1))
    print(f"wrote {per_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

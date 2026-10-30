#!/usr/bin/env python3
"""Would a Bayesian *neural* network (BNN) strengthen the severity results?

A BNN is not a Bayesian network. It is a neural classifier with a posterior
over its weights, so its selling point is calibrated predictive uncertainty,
not structure or interpretability. The only honest place it can enter this
paper is as one more supervised text baseline next to emb-LR and TF-IDF.
This script runs that baseline under the SAME protocol as
embedding_lr_baseline.py and scores it against bn-sev on the same 296 items.

Two standard BNN approximations (numpy + sklearn only, no torch):
  * deep-ensemble  : M bootstrap MLPs, predictive = mean softmax
                     (Lakshminarayanan et al. 2017)
  * laplace-blr    : Bayesian multinomial logistic regression, diagonal
                     Laplace posterior around the L2 MAP, MC predictive
                     (MacKay 1992)

Metrics: top-1 accuracy, Brier, NLL, ECE (10 bins), paired disagreement
vs bn-sev. Decision rule written into the output: the BNN earns a sentence
in Section 8 only if it beats bn-sev on Brier/ECE by a margin a reviewer
would notice; a tie or loss means the fixed-network path already carries the
uncertainty story and the BNN adds nothing but a naming collision.

Run:
  export PYTHONPATH="shared/code:Frozen-BN-Narrative-Evidence-2026-07-20/code"
  python Frozen-BN-Narrative-Evidence-2026-07-20/tests/bnn_baseline_heldout.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT / "shared" / "code", FROZEN_DIR / "code", FROZEN_DIR / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

import prognosis as pg  # noqa: E402
import query_to_bn as qb  # noqa: E402
from embedding_lr_baseline import FULL, WINDOW, embed_all  # noqa: E402

PER_ITEM_BN = FROZEN_DIR / "outputs" / "heldout_per_item.json"
OUT = FROZEN_DIR / "outputs" / "bnn_baseline_heldout.json"

SEED = 0
N_ENSEMBLE = 10
N_MC = 200
K = 4


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    hit = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(total)


def score(probs: np.ndarray, y: np.ndarray) -> dict:
    probs = probs / probs.sum(1, keepdims=True)
    onehot = np.eye(K)[y]
    return {
        "acc": float((probs.argmax(1) == y).mean()),
        "brier": float(((probs - onehot) ** 2).sum(1).mean()),
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-9, 1)).mean()),
        "ece": ece(probs, y),
        "n": int(len(y)),
    }


def full_proba(clf, X: np.ndarray) -> np.ndarray:
    """predict_proba padded to all K classes (a class can be absent in train)."""
    out = np.zeros((len(X), K))
    p = clf.predict_proba(X)
    for j, cls in enumerate(clf.classes_):
        out[:, int(cls)] = p[:, j]
    return out


def deep_ensemble(Xtr, ytr, Xte, rng) -> np.ndarray:
    preds = []
    for m in range(N_ENSEMBLE):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        clf = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3,
                            max_iter=500, early_stopping=True,
                            random_state=SEED + m)
        clf.fit(Xtr[idx], ytr[idx])
        preds.append(full_proba(clf, Xte))
    return np.mean(preds, axis=0)


def laplace_blr(Xtr, ytr, Xte, rng, prior_var: float = 1.0) -> np.ndarray:
    """Diagonal Laplace around the L2-MAP multinomial LR, MC predictive."""
    clf = LogisticRegression(max_iter=5000, C=prior_var)
    clf.fit(Xtr, ytr)
    classes = [int(c) for c in clf.classes_]
    W = clf.coef_                      # (C, D)
    b = clf.intercept_                 # (C,)
    P = clf.predict_proba(Xtr)         # (N, C)
    # diagonal Hessian of the negative log posterior, per class and feature
    curv = (P * (1 - P)).T @ (Xtr ** 2) + 1.0 / prior_var      # (C, D)
    curv_b = (P * (1 - P)).sum(0) + 1.0 / prior_var           # (C,)
    sd_W, sd_b = 1.0 / np.sqrt(curv), 1.0 / np.sqrt(curv_b)
    acc = np.zeros((len(Xte), len(classes)))
    for _ in range(N_MC):
        Ws = W + rng.standard_normal(W.shape) * sd_W
        bs = b + rng.standard_normal(b.shape) * sd_b
        logits = Xte @ Ws.T + bs
        logits -= logits.max(1, keepdims=True)
        e = np.exp(logits)
        acc += e / e.sum(1, keepdims=True)
    mc = acc / N_MC
    out = np.zeros((len(Xte), K))
    for j, cls in enumerate(classes):
        out[:, cls] = mc[:, j]
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
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
    labels = {
        "injury": (
            np.array([inj[pg.zhang_injury_code(inc)] for _, inc, _ in train]),
            np.array([inj[pg.zhang_injury_code(inc)] for _, inc, _ in test]),
            "inj"),
        "damage": (
            np.array([dmg.get(str(inc.get("damage") or "").upper(), -1) for _, inc, _ in train]),
            np.array([dmg.get(str(inc.get("damage") or "").upper(), -1) for _, inc, _ in test]),
            "dmg"),
    }

    bn_items = {it["id"]: it for it in json.loads(PER_ITEM_BN.read_text())["items"]}
    test_ids = [k for k, _, _ in test]

    results = {"train_n": len(train), "test_n": len(test),
               "n_ensemble": N_ENSEMBLE, "n_mc": N_MC}
    for tag, (ytr, yte, short) in labels.items():
        tr_ok, te_ok = ytr >= 0, yte >= 0
        ys = yte[te_ok]
        ids = [i for i, ok in zip(test_ids, te_ok) if ok]

        ens = deep_ensemble(Xtr[tr_ok], ytr[tr_ok], Xte[te_ok], rng)
        blr = laplace_blr(Xtr[tr_ok], ytr[tr_ok], Xte[te_ok], rng)

        bn = np.array([bn_items[i][f"bn-sev:{short}_probs"] for i in ids])
        bn = bn / bn.sum(1, keepdims=True)

        block = {
            "deep-ensemble": score(ens, ys),
            "laplace-blr": score(blr, ys),
            "bn-sev (same items)": score(bn, ys),
        }
        for name, probs in (("deep-ensemble", ens), ("laplace-blr", blr)):
            a, c = probs.argmax(1) == ys, bn.argmax(1) == ys
            block[name]["vs_bn_sev"] = {
                "bnn_right_bn_wrong": int((a & ~c).sum()),
                "bn_right_bnn_wrong": int((~a & c).sum()),
                "brier_delta": round(block[name]["brier"] - block["bn-sev (same items)"]["brier"], 4),
                "ece_delta": round(block[name]["ece"] - block["bn-sev (same items)"]["ece"], 4),
            }
        results[tag] = block
        print(f"\n{tag}")
        for name, r in block.items():
            print(f"  {name:22s} acc {r['acc']:.1%}  Brier {r['brier']:.3f}  "
                  f"NLL {r['nll']:.2f}  ECE {r['ece']:.3f}")

    worth_it = any(
        results[t][m]["vs_bn_sev"]["brier_delta"] < -0.02
        and results[t][m]["vs_bn_sev"]["ece_delta"] < -0.02
        for t in labels for m in ("deep-ensemble", "laplace-blr"))
    results["verdict"] = (
        "BNN improves calibration over bn-sev by a reviewer-visible margin; "
        "report one line in Section 8.1 as a supervised baseline."
        if worth_it else
        "BNN does not beat bn-sev on Brier and ECE together; the fixed-network "
        "path already carries the uncertainty story. Do not add a BNN.")
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nverdict: {results['verdict']}\nwrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

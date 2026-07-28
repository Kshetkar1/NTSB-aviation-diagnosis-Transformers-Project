"""FIX 1 - Calibration of the diagnosis output probabilities (READ-ONLY).

Question: when the diagnosis method states a probability of ~0.6 for its top
cause, is the top cause empirically correct ~60% of the time?

We reuse the validated LOO scaffolding (``tests/query_conditioning_validation.py``
via ``tests/qc_common.py``) and assess **confidence calibration** of the top-1
cause on the **leakage-free factual stratum (A-clean)**, for BOTH the conditioned
prediction and the unconditioned population prior:

  * confidence  = P-mass the method puts on its predicted top-1 cause,
  * correct     = the top-1 cause is in the incident's true-cause set,
  * reliability diagram (10 equal-width bins), Expected Calibration Error (ECE),
    Maximum Calibration Error (MCE), and Brier score of (confidence - correct).

If miscalibrated, we fit three standard post-hoc recalibrators on a **held-out
train split** and report the **test** ECE/Brier improvement (no overfitting):
  * temperature scaling on the cause distribution (Guo et al. 2017),
  * isotonic regression (confidence -> correctness),
  * Platt scaling (logistic confidence -> correctness).

The calibration is computed over the **actual engine output** = the full
all-cause distribution (generic catch-alls included), because that is the number
a user would see.

Outputs: docs/calibration_results.json, docs/figures/calibration.png
Run:  python3.11 tests/calibration_analysis.py            (offline, cached embeds)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "docs" / ".mplcache"))

import qc_common as qc  # noqa: E402

DOCS = ROOT / "docs"
FIG_DIR = DOCS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = DOCS / "calibration_results.json"
FIG_PATH = FIG_DIR / "calibration.png"

N_BINS = 10
SEED = 0


# --- calibration metrics ------------------------------------------------------
def reliability(confs, corrects, n_bins=N_BINS):
    confs = np.asarray(confs, float)
    corrects = np.asarray(corrects, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows, ece, mce, N = [], 0.0, 0.0, len(confs)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        nb = int(m.sum())
        if nb == 0:
            rows.append({"lo": float(lo), "hi": float(hi), "n": 0,
                         "conf": None, "acc": None})
            continue
        conf_b, acc_b = float(confs[m].mean()), float(corrects[m].mean())
        rows.append({"lo": float(lo), "hi": float(hi), "n": nb,
                     "conf": conf_b, "acc": acc_b})
        gap = abs(acc_b - conf_b)
        ece += nb / N * gap
        mce = max(mce, gap)
    return {"bins": rows, "ece": float(ece), "mce": float(mce)}


def brier(confs, corrects):
    confs = np.asarray(confs, float)
    corrects = np.asarray(corrects, float)
    return float(np.mean((confs - corrects) ** 2))


# --- recalibrators (fit on train, applied to test) ---------------------------
def fit_temperature(dists, trues):
    """Temperature scaling on the cause distribution; returns T minimizing NLL
    of the true-cause mass on the train split. conf' = max softmax(log p / T)."""
    from scipy.optimize import minimize_scalar

    supports = []
    for prob, true in zip(dists, trues):
        causes = list(prob)
        z = np.log(np.array([prob[c] for c in causes], float))
        true_mask = np.array([c in true for c in causes], bool)
        supports.append((z, true_mask))

    def nll(T):
        s = 0.0
        for z, tm in supports:
            w = z / T
            w = w - w.max()
            e = np.exp(w)
            sm = e / e.sum()
            m = sm[tm].sum() if tm.any() else 0.0
            s += -np.log(max(m, 1e-12))
        return s / len(supports)

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(res.x)


def apply_temperature(dists, T):
    out = []
    for prob in dists:
        causes = list(prob)
        z = np.log(np.array([prob[c] for c in causes], float))
        w = z / T
        w = w - w.max()
        e = np.exp(w)
        sm = e / e.sum()
        out.append(float(sm.max()))
    return np.array(out)


# --- build confidence/correctness arrays --------------------------------------
def arrays(records):
    """all-cause-space confidence + correctness for conditioned and prior."""
    out = {"cond": {"conf": [], "correct": [], "dist": [], "true": []},
           "unc": {"conf": [], "correct": [], "dist": [], "true": []}}
    for r in records:
        true = r["true_all"]
        for key, order, prob in (("cond", r["cond_order"], r["cond_prob"]),
                                  ("unc", r["unc_order"], r["unc_prob"])):
            if not order:
                continue
            out[key]["conf"].append(qc.top1_conf(order, prob))
            out[key]["correct"].append(1.0 if order[0] in true else 0.0)
            out[key]["dist"].append(prob)
            out[key]["true"].append(true)
    return out


def recalibrate(conf, correct, dist, true, name):
    """Train/test 50/50 split; fit 3 recalibrators on train, report test gains."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    n = len(conf)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    tr, te = perm[: n // 2], perm[n // 2:]

    def metrics(c, y):
        return {"ece": reliability(c, y)["ece"], "brier": brier(c, y),
                "mce": reliability(c, y)["mce"]}

    raw = metrics(conf[te], correct[te])

    # temperature scaling (uses full distributions)
    dist_tr = [dist[i] for i in tr]
    true_tr = [true[i] for i in tr]
    T = fit_temperature(dist_tr, true_tr)
    dist_te = [dist[i] for i in te]
    conf_temp = apply_temperature(dist_te, T)
    temp = metrics(conf_temp, correct[te])
    temp["T"] = T

    # isotonic
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(conf[tr], correct[tr])
    conf_iso = iso.predict(conf[te])
    iso_m = metrics(conf_iso, correct[te])

    # Platt (logistic)
    lr = LogisticRegression()
    lr.fit(conf[tr].reshape(-1, 1), correct[tr])
    conf_platt = lr.predict_proba(conf[te].reshape(-1, 1))[:, 1]
    platt_m = metrics(conf_platt, correct[te])

    return {
        "name": name, "n_train": int(len(tr)), "n_test": int(len(te)),
        "raw": raw, "temperature": temp, "isotonic": iso_m, "platt": platt_m,
        "_test_curves": {
            "raw": (conf[te], correct[te]),
            "isotonic": (conf_iso, correct[te]),
        },
    }


def main():
    records = qc.build_records(top_n_incidents=100)
    clean = [r for r in records if qc.is_clean_A(r)]
    print(f"A-clean (leakage-free factual) incidents: {len(clean)}", file=sys.stderr)

    arr = arrays(clean)
    summary = {"n_A_clean": len(clean), "n_bins": N_BINS, "seed": SEED,
               "space": "all-cause (actual engine output)",
               "stratum": "A-clean (factual, containment < 0.7)"}

    for key in ("cond", "unc"):
        a = arr[key]
        rel = reliability(a["conf"], a["correct"])
        summary[key] = {
            "n": len(a["conf"]),
            "mean_conf": float(np.mean(a["conf"])),
            "accuracy": float(np.mean(a["correct"])),
            "ece": rel["ece"], "mce": rel["mce"],
            "brier": brier(a["conf"], a["correct"]),
            "reliability": rel["bins"],
        }

    # recalibration (the method of interest = conditioned; report prior too)
    recal = {}
    for key in ("cond", "unc"):
        a = arr[key]
        recal[key] = recalibrate(a["conf"], a["correct"], a["dist"], a["true"], key)
    summary["recalibration"] = {
        k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
        for k, v in recal.items()
    }

    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"results -> {RESULTS_PATH}")
    _print_console(summary)
    try:
        _make_figure(summary, recal)
        print(f"figure  -> {FIG_PATH}")
    except Exception as e:  # noqa: BLE001
        print(f"(figure skipped: {e})", file=sys.stderr)


def _print_console(s):
    print("\n" + "=" * 78)
    print("CALIBRATION  (A-clean, all-cause space; confidence = P on top-1 cause)")
    print("=" * 78)
    for key in ("cond", "unc"):
        g = s[key]
        print(f"\n[{key}]  n={g['n']}  mean_conf={g['mean_conf']:.3f}  "
              f"acc={g['accuracy']:.3f}  ECE={g['ece']:.3f}  MCE={g['mce']:.3f}  "
              f"Brier={g['brier']:.3f}")
        r = s["recalibration"][key]
        print(f"   recal (test n={r['n_test']}): "
              f"raw ECE {r['raw']['ece']:.3f}/Brier {r['raw']['brier']:.3f} | "
              f"temp(T={r['temperature']['T']:.2f}) ECE {r['temperature']['ece']:.3f}/"
              f"Brier {r['temperature']['brier']:.3f} | "
              f"iso ECE {r['isotonic']['ece']:.3f}/Brier {r['isotonic']['brier']:.3f} | "
              f"platt ECE {r['platt']['ece']:.3f}/Brier {r['platt']['brier']:.3f}")


def _make_figure(s, recal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def plot_rel(ax, g, title):
        confs = [b["conf"] for b in g["reliability"] if b["n"] > 0]
        accs = [b["acc"] for b in g["reliability"] if b["n"] > 0]
        ns = [b["n"] for b in g["reliability"] if b["n"] > 0]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.plot(confs, accs, "o-", color="#1f77b4", label="empirical")
        for c, a, n in zip(confs, accs, ns):
            ax.annotate(str(n), (c, a), fontsize=7, color="#555",
                        xytext=(2, 3), textcoords="offset points")
        ax.set_xlabel("stated probability (top-1 cause)")
        ax.set_ylabel("empirical correctness")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"{title}\nECE={g['ece']:.3f}  Brier={g['brier']:.3f}")
        ax.legend(fontsize=8)

    plot_rel(axes[0], s["cond"], "Conditioned (narrative)")
    plot_rel(axes[1], s["unc"], "Unconditioned prior")

    # recalibrated conditioned (isotonic) reliability on the held-out test split
    ax = axes[2]
    c_raw, y = recal["cond"]["_test_curves"]["raw"]
    c_iso, _ = recal["cond"]["_test_curves"]["isotonic"]
    rel_raw = reliability(c_raw, y)
    rel_iso = reliability(c_iso, y)
    for rel, lab, col in ((rel_raw, "raw", "#1f77b4"), (rel_iso, "isotonic", "#2ca02c")):
        confs = [b["conf"] for b in rel["bins"] if b["n"] > 0]
        accs = [b["acc"] for b in rel["bins"] if b["n"] > 0]
        ax.plot(confs, accs, "o-", color=col,
                label=f"{lab} (ECE={rel['ece']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("stated probability (top-1 cause)")
    ax.set_ylabel("empirical correctness")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Conditioned: recalibration (held-out test)")
    ax.legend(fontsize=8)

    fig.suptitle("Diagnosis output calibration (leakage-free A-clean stratum)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_PATH, dpi=130)


if __name__ == "__main__":
    main()

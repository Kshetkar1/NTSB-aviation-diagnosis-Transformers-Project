"""
Reproduce Zhang's Table 7 calibration: fit alpha and beta of a Beta CDF
to the 8 observed conditional probabilities he uses to parameterize CPTs.

From his CPT_estimation.ipynb (cells 17, 18, 21):

    probs = [0.00980392, 0.01960784, 0.02941176, 0.03921569,
             0.04901961, 0.05882353, 0.08823529, 0.31372549]
    totalContr = 1.735294117647057
    lambda_ = probs / totalContr

    def obj_beta(x):
        return sum((scipy.stats.beta.cdf(lambda_, a=x[0], b=x[1]) - probs)**2)

    res = scipy.optimize.minimize(obj_beta, x_0=[2, 1],
                                  method='Nelder-Mead', tol=1e-6)

His published optimum: alpha=1.04645351, beta=2.02591394, residual=2.74e-6.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.stats
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "table7_alpha_beta.json"

ZHANG_ALPHA = 1.04645351
ZHANG_BETA = 2.02591394
ZHANG_RESIDUAL = 2.74086711e-6

PROBS = np.array([
    0.00980392156862745,
    0.0196078431372549,
    0.029411764705882353,
    0.0392156862745098,
    0.049019607843137254,
    0.058823529411764705,
    0.08823529411764706,
    0.3137254901960784,
])
TOTAL_CONTR = 1.735294117647057
LAMBDA = PROBS / TOTAL_CONTR


def obj_beta(x):
    diff = scipy.stats.beta.cdf(LAMBDA, a=x[0], b=x[1]) - PROBS
    return float(np.sum(diff ** 2))


def main() -> None:
    print("Reproducing Zhang Table 7 (CPT_estimation.ipynb cell 21)")
    print(f"  observations    : {len(PROBS)} probabilities")
    print(f"  totalContr      : {TOTAL_CONTR}")
    print(f"  initial x_0     : [2, 1]")
    print(f"  method          : Nelder-Mead, tol=1e-6")
    print()

    res = minimize(obj_beta, x0=[2.0, 1.0], method="Nelder-Mead", tol=1e-6)

    a_ours, b_ours = float(res.x[0]), float(res.x[1])
    fun_ours = float(res.fun)

    print(f"Reproduced:")
    print(f"  alpha           = {a_ours:.8f}    (Zhang: {ZHANG_ALPHA:.8f})")
    print(f"  beta            = {b_ours:.8f}    (Zhang: {ZHANG_BETA:.8f})")
    print(f"  residual SSE    = {fun_ours:.4e}     (Zhang: {ZHANG_RESIDUAL:.4e})")
    print()

    da = a_ours - ZHANG_ALPHA
    db = b_ours - ZHANG_BETA
    print(f"  delta(alpha)    = {da:+.2e}")
    print(f"  delta(beta)     = {db:+.2e}")

    if abs(da) < 1e-5 and abs(db) < 1e-5:
        verdict = "EXACT MATCH (within 1e-5)"
    elif abs(da) < 1e-3 and abs(db) < 1e-3:
        verdict = "OK (within 1e-3, expected for Nelder-Mead simplex)"
    else:
        verdict = "FAIL"
    print(f"\nVerdict: {verdict}")

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({
            "method": "Nelder-Mead",
            "x_0": [2.0, 1.0],
            "tol": 1e-6,
            "alpha_ours": a_ours,
            "beta_ours": b_ours,
            "residual_ours": fun_ours,
            "alpha_zhang": ZHANG_ALPHA,
            "beta_zhang": ZHANG_BETA,
            "residual_zhang": ZHANG_RESIDUAL,
            "delta_alpha": da,
            "delta_beta": db,
            "verdict": verdict,
            "n_iter": int(res.nit),
            "n_fev": int(res.nfev),
            "scipy_version": scipy.__version__,
            "numpy_version": np.__version__,
        }, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

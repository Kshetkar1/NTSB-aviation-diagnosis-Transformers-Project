"""Apply Zhang's ACTUAL conditional-probability method (paper Sec 4.3 / 5.1)
to the Table-4 fire causes, to see what his method really produces.

Method (paper Eqs 8, 9, 15, 16; calibration Eq 17):
  - single-cause contribution  P(fire|e_i) = count(e_i & fire) / 102
  - lambda = sum(contributions present) / sum(ALL contributions = 1.735)
  - P(fire | causes) = BetaCDF(lambda; alpha, beta)
  - alpha, beta calibrated by Nelder-Mead -> 1.04645, 2.02591
"""
from __future__ import annotations

import scipy.stats

ALPHA = 1.04645351
BETA = 2.02591394
TOTAL_CONTRIB = 1.735294117647057  # paper's "1.735" (sum of all fire contributions)

# Single-cause contributions from Table 7 (paper p.12)
BRAKE = 0.01960784          # "Brakes (normal)"
WIRING = 0.08823529         # "Electrical system, electric wiring"
AIRFRAME = 0.31372549       # "Airframe/component/system failure/malfunction" (biggest)
LOEP_MECH = 0.08823529      # "Loss of engine power (total) - mechanical"


def cpt(*contribs: float) -> float:
    lam = sum(contribs) / TOTAL_CONTRIB
    return lam, float(scipy.stats.beta.cdf(lam, a=ALPHA, b=BETA))


def main() -> None:
    print(f"alpha={ALPHA}, beta={BETA}, total contribution={TOTAL_CONTRIB:.4f}\n")
    print("Zhang's METHOD applied to the Table-4 fire scenario:")
    for name, contribs in [
        ("brake wear only            (Table4 cell yes,no)", (BRAKE,)),
        ("wiring overheat only       (Table4 cell no,yes)", (WIRING,)),
        ("brake AND wiring overheat  (Table4 cell yes,yes)", (BRAKE, WIRING)),
    ]:
        lam, p = cpt(*contribs)
        print(f"  {name:48} lambda={lam:.4f}  P(fire)={p:.4f}")

    print("\nTable-4 says these should be ~0.93 / 0.95 / 0.99 (assumed demo values).")

    print("\nWhat lambda would Zhang's method need to reach ~0.90?")
    for target in (0.50, 0.90, 0.99):
        lam = scipy.stats.beta.ppf(target, a=ALPHA, b=BETA)
        contrib_needed = lam * TOTAL_CONTRIB
        print(f"  P(fire)={target:.2f}  needs lambda={lam:.4f}  "
              f"= {contrib_needed:.3f} of 1.735 in contributions present")

    print("\nSanity: a single big cause (airframe malfunction, 0.314):")
    lam, p = cpt(AIRFRAME)
    print(f"  airframe only              lambda={lam:.4f}  P(fire)={p:.4f}")
    lam, p = cpt(AIRFRAME, WIRING, LOEP_MECH, BRAKE)
    print(f"  airframe+wiring+LOEP+brake lambda={lam:.4f}  P(fire)={p:.4f}")


if __name__ == "__main__":
    main()

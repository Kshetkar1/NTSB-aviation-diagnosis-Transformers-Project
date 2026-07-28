"""See the 184,517,128 'total flights' for yourself.

This number is NOT in the NTSB accident data. It comes from the Bureau of
Transportation Statistics (BTS) departures-by-year table that Zhang cites
(paper Table 6, ref [45]), vendored at:
    Zhang's Approach 2026/data/table_01_37_061019.xlsx

Zhang's recipe (paper Sec 4.2):
  1. read total *performed* aircraft departures per year (only some years given)
  2. linear-interpolate to fill missing years
  3. sum the interpolated departures over 1982..2006  -> total flights
That sum is the denominator he uses instead of the 2,243 accidents.
"""
from __future__ import annotations

from pathlib import Path

from scipy.interpolate import interp1d
import pandas as pd

BTS_XLSX = (
    Path(__file__).resolve().parents[1]
    / "Zhang's Approach 2026" / "data" / "table_01_37_061019.xlsx"
)


def main() -> None:
    dep = pd.read_excel(BTS_XLSX)
    dep.drop(dep.columns[0], axis=1, inplace=True)
    years = [int(y) for y in dep.iloc[0].index]
    flights = dep.iloc[0].values

    print("STEP 1 — raw BTS 'total performed departures' (only these years exist):")
    for y, fl in zip(years, flights):
        print(f"    {y}: {fl:>12,.0f}")

    f = interp1d(years, flights)  # linear interpolation across the gaps

    print("\nSTEP 2 — interpolate every year 1982..2006 and add them up:")
    running = 0.0
    for y in range(1982, 2007):
        v = float(f(y))
        running += v
        print(f"    {y}: {v:>12,.0f}    running total: {running:>15,.0f}")

    total = round(running)
    print(f"\nSTEP 3 — TOTAL FLIGHTS (1982-2006) = {total:,}")
    print(f"    (paper Table 6 / Sec 4.2 reports 184,517,128)")
    print(f"\nCompare denominators for P(fire) = 102 / denominator:")
    print(f"    your old way (2,243 accidents): {102/2243:.6f}  (= {102/2243*100:.2f}%)")
    print(f"    Zhang's way   ({total:,} flights): {102/total:.3e}")
    print(f"    ratio of denominators: {total/2243:,.0f}x larger")


if __name__ == "__main__":
    main()

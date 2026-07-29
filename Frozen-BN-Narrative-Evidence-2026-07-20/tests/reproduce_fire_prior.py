"""Reproduce Zhang's fire prior: totalFlights and P(fire) = 102 / totalFlights.

Mirrors Zhang's main.py cells In[279]-In[288]:
  - load table_01_37_061019.xlsx (BTS departures-by-year, NOT NTSB data)
  - linear-interpolate departures across years
  - sum interpolated departures over 1982..2006 -> totalFlights
  - P(fire) = totalOccurrences / totalFlights   (totalOccurrences = 102 per Zhang)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

DEPARTURES_XLSX = (
    Path(__file__).resolve().parents[2]
    / "Zhang-Replication-Foundation-2026-06-04" / "reference" / "data"
    / "table_01_37_061019.xlsx"
)

# 184,517,128 is the correct interpolated sum. (The figure "184,572,128"
# sometimes quoted verbally is a digit transposition of this value.)
ZHANG_TOTAL_FLIGHTS = 184517128
ZHANG_TOTAL_OCCURRENCES = 102
ZHANG_PRIOR = 5.527942099770814e-07


def main() -> None:
    departures = pd.read_excel(DEPARTURES_XLSX)
    departures.drop(departures.columns[0], axis=1, inplace=True)

    years = [int(i) for i in departures.iloc[0].index]
    flights = departures.iloc[0].values

    print("Raw departures-by-year (from table_01_37_061019.xlsx):")
    for y, fl in zip(years, flights):
        print(f"  {y}: {fl:,.0f}")
    print()

    f = interp1d(years, flights)
    total_flights = round(sum(float(f(i)) for i in range(1982, 2007)))

    prob_yes = ZHANG_TOTAL_OCCURRENCES / total_flights

    print(f"totalFlights (1982-2006, interpolated sum) = {total_flights:,}")
    print(f"  Zhang's published value                  = {ZHANG_TOTAL_FLIGHTS:,}")
    print(f"  match: {total_flights == ZHANG_TOTAL_FLIGHTS}")
    print()
    print(f"totalOccurrences of Fire = {ZHANG_TOTAL_OCCURRENCES}")
    print(f"P(fire) = {ZHANG_TOTAL_OCCURRENCES} / {total_flights:,} = {prob_yes:.6e}")
    print(f"  Zhang's published prior = {ZHANG_PRIOR:.6e}")
    print(f"  match (to 1e-12): {abs(prob_yes - ZHANG_PRIOR) < 1e-12}")


if __name__ == "__main__":
    main()

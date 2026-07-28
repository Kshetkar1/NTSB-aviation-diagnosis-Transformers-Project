"""Compare raw-count vs Zhang Beta-CDF vs YOUR semantic smoothing on the forward
conditional P(fire | cause), for single causes and the Table-4 combo."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
_SHARED = REPO_ROOT / "shared" / "code"
_FROZEN_CODE = FROZEN_DIR / "code"
for _p in (_SHARED, _FROZEN_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
ROOT = REPO_ROOT
import main_app  # noqa: E402
import sparse_cpt as sc  # noqa: E402

OUTCOME = "fire"


def pick(count, *subs):
    """first label in count containing any substring (case-insensitive)."""
    for c in count:
        cl = c.lower()
        if all(s.lower() in cl for s in subs):
            return c
    return None


def main() -> None:
    ds = main_app.refined_dataset
    count, total, n_out = sc._outcome_cause_counts(OUTCOME, ds)
    print(f"fire accidents (with cause edge): {n_out}   total contribution units: {total}")
    print("\nTop fire causes (label -- count):")
    for c, n in sorted(count.items(), key=lambda x: -x[1])[:12]:
        print(f"  {n:3}  {c}")

    wiring = pick(count, "wiring") or pick(count, "electrical")
    airframe = pick(count, "airframe")
    brakes = pick(count, "brake")
    fuel = pick(count, "fuel")
    print("\nSelected labels:")
    print(f"  wiring   = {wiring}")
    print(f"  airframe = {airframe}")
    print(f"  brakes   = {brakes}")
    print(f"  fuel     = {fuel}")

    scenarios = [
        ("wiring (electrical)", [wiring], "The aircraft electrical wiring failed and overheated, causing a short circuit."),
        ("airframe", [airframe], "The aircraft airframe component or system failed or malfunctioned during flight."),
        ("fuel", [fuel], "A fuel system problem occurred; fuel leaked and ignited on the aircraft."),
        ("brakes (normal)", [brakes], "The landing gear normal brake system was worn out and overheated."),
        ("brakes + wiring (Table-4 combo)", [brakes, wiring],
         "The landing gear normal brake system is worn out and the electrical wiring is overheating."),
    ]

    print("\n" + "=" * 92)
    print(f"P(fire | cause)  --  three estimators")
    print("=" * 92)
    print(f"{'scenario':34} {'raw-count':>16} {'Beta-CDF':>16} {'semantic(K=50)':>18}")
    print("-" * 92)
    for name, labels, query in scenarios:
        labels = [l for l in labels if l]
        if not labels:
            print(f"{name:34}  (label not found in data)")
            continue
        rc, rco, rcn = sc.raw_count_cpt(labels, OUTCOME, ds)
        contrib, bc = sc.beta_cdf_cpt(labels, count, total)
        sm, smo, smn = sc.semantic_cpt(query, OUTCOME, k=50, main_app=main_app)
        print(f"{name:34} {rc:>7.3f}({rco:>2}/{rcn:>3}) {bc:>10.3f}[c={contrib:.3f}] {sm:>9.3f}({smo:>2}/{smn:>2})")


if __name__ == "__main__":
    main()

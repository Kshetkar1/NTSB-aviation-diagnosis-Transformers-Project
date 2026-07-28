"""Offline check: faithful mode (full_population=True + cause_factor_only=True)
reproduces Zhang's published Table-7 level-1 edges (airframe = 32/102 = 0.31373,
denominator = 102). No network/embeddings needed because full_population skips
retrieve_pool."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
import trees


class _Shim:
    """Minimal main_app stand-in: only refined_dataset is used in faithful mode."""
    def __init__(self, ds):
        self.refined_dataset = ds

    def get_embedding(self, *a, **k):
        raise RuntimeError("get_embedding must NOT be called in full_population mode")

    def find_top_matches(self, *a, **k):
        raise RuntimeError("find_top_matches must NOT be called in full_population mode")


def main():
    ds = json.loads(Path(config.ACTIVE_INCIDENT_DATA_PATH).read_text())
    shim = _Shim(ds)

    out = trees.build_diagnosis_tree(
        "cargo compartment fire on the aircraft",
        branching=6, depth=1, min_prob=0.0, drop_generic=False,
        dataset=ds, main_app=shim,
        cause_factor_only=True, full_population=True,
    )
    meta = out["meta"]
    root = out["tree"]
    print("outcome:", meta["outcome"])
    print("denominator (should be 102):", meta["outcome_count_in_pool"])
    print("population_mode:", meta["population_mode"])
    print("\nlevel-1 edges:")
    air = None
    for c in root["children"]:
        print(f"  {c['label']:<40} p={c['edge_prob']:.5f}  n={c['n']}/{c['denom']}")
        if "airframe" in c["label"].lower():
            air = c
    print("\nairframe edge:", air["edge_prob"] if air else "NOT FOUND",
          "(Zhang Table 7 = 0.31373, 32/102)")

    ok = (meta["outcome_count_in_pool"] == 102 and air is not None
          and abs(air["edge_prob"] - 0.31373) < 0.001 and air["n"] == 32)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Catch-all decomposition: can NTSB narratives recover the specific causes that
the coarse structured *occurrence* codes bury inside generic catch-all labels?

Context
-------
Zhang's Bayesian network reasons over two structured layers (see
docs/ZHANG_REPRODUCTION_REPORT.md Section 5):
  1. the *occurrence* event chain  (``sequence_of_events[*].Occurrence_Description``)
  2. the *findings* subject codes  (``findings[*].finding_description`` + modifier)

The single most frequent occurrence label in the 1982-2006 window is the
catch-all "Airframe/component/system failure/malfunction" (and "Miscellaneous/
other"). When that catch-all is the coded event, the occurrence node says nothing
about *what* actually failed. The question this script answers, with hard numbers:

  * How big is the catch-all burden (overall + fire subset)?               [Stage A]
  * If we read the NARRATIVE, can we assign a SPECIFIC mechanism?           [Stage B]
  * Crucially -- does the narrative recover anything the structured FINDINGS
    layer (which Zhang ALSO uses) does not already encode?                  [Stage C]

The honest control here is the findings layer. If a catch-all incident already
carries a specific physical-system finding, then the specific cause is NOT
"buried" in the structured record -- it is just absent from the occurrence node.
Stage C measures narrative-vs-findings agreement and isolates the genuinely
"buried" subset (occurrence catch-all AND no specific structured finding).

Stages
------
  A  (no network)  burden counts + structured-findings resolution + ground truth
  B  (network)     blind LLM mechanism extraction from narratives (cached to JSON)
  C  (no network)  resolution gain, taxonomy distribution, narrative<->findings
                   agreement (precision), buried-subset, verbatim examples

Usage
-----
  python3.11 tests/catchall_decomposition.py --stage A
  python3.11 tests/catchall_decomposition.py --stage B   # needs OPENAI_API_KEY + net
  python3.11 tests/catchall_decomposition.py --stage C
  python3.11 tests/catchall_decomposition.py --all       # A, then C if cache exists

All catch-all labels and the mechanism taxonomy are derived/configurable here, not
buried as magic constants in the engine.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

DATA_PATH = Path(getattr(config, "ACTIVE_INCIDENT_DATA_PATH",
                         ROOT / "data" / "processed" / "refined_dataset_1982_2006.json"))
OUT_DIR = ROOT / "docs"
CACHE_PATH = OUT_DIR / "catchall_llm_cache.json"
RESULTS_PATH = OUT_DIR / "catchall_decomposition_results.json"

# --- Catch-all (generic occurrence) labels -----------------------------------
# Seeded from zhang_diagnosis.GENERIC_CAUSES, then any occurrence label that is
# high-frequency yet semantically uninformative (names no specific system /
# mechanism). Kept here (derived + explicit) rather than hard-coded in the engine.
SEED_CATCHALL = {
    "airframe/component/system failure/malfunction",
    "miscellaneous/other",
}
# Regexes that flag a label as a generic catch-all (uninformative about mechanism).
CATCHALL_PATTERNS = [
    r"^airframe/component/system failure/malfunction$",
    r"^miscellaneous/other$",
    r"^unknown/not determined$",
    r"^reason for occurrence undetermined$",
]


def is_catchall_occ(label: str) -> bool:
    s = " ".join((label or "").lower().split())
    if s in SEED_CATCHALL:
        return True
    return any(re.match(p, s) for p in CATCHALL_PATTERNS)


# --- Mechanism taxonomy --------------------------------------------------------
# Specific physical-system / mechanism buckets (1-9) plus human-cause and
# residual buckets. The LLM (Stage B) is constrained to exactly these labels, and
# the deterministic findings mapper (Stage A) uses the same labels so narrative
# and structured assignments are directly comparable.
PHYSICAL_MECHANISMS = [
    "fuel_system",
    "electrical_wiring",
    "engine_powerplant",
    "landing_gear_brakes",
    "hydraulic_system",
    "flight_controls",
    "airframe_structure",
    "propeller_rotor",
    "other_aircraft_system",
]
HUMAN_CAUSES = ["maintenance_error", "pilot_operational"]
RESIDUAL = ["environmental", "undetermined"]
TAXONOMY = PHYSICAL_MECHANISMS + HUMAN_CAUSES + RESIDUAL

# Ordered keyword rules mapping a structured finding_description (lowercased) to a
# taxonomy bucket. First match wins, so more specific rules come first. Physical
# systems are checked before human/residual so an incident that names a system
# resolves to that system.
FINDING_RULES: list[tuple[str, str]] = [
    # fuel
    (r"fuel system|fluid, fuel|\bfuel\b", "fuel_system"),
    # electrical
    (r"electrical system|electric wiring|electric relay|circuit breaker|"
     r"\bgenerator\b|\bbattery\b|\balternator\b|\bbus\b", "electrical_wiring"),
    # hydraulic
    (r"hydraulic", "hydraulic_system"),
    # landing gear / brakes / wheels / tires
    (r"landing gear|drag brace|\bstrut\b|\bwheel\b|\btire\b|\bbrake\b|\baxle\b",
     "landing_gear_brakes"),
    # flight controls
    (r"flight control|aileron|elevator|rudder|spoiler|\bflap\b|trim/tab|"
     r"horizontal stabilizer|control cable", "flight_controls"),
    # propeller / rotor
    (r"propeller|\brotor\b|main rotor|tail rotor|\bblade\b", "propeller_rotor"),
    # engine / powerplant mechanical
    (r"engine assembly|engine compartment|\bcylinder\b|\bcrankshaft\b|"
     r"\bbearing\b|compressor|turbine|combust|\bmagneto\b|carburet|"
     r"\bpiston\b|oil system|\bexhaust\b|induction|\bcamshaft\b|\bvalve\b|"
     r"\bnacelle\b|fan blade|rotor disc|reduction gear", "engine_powerplant"),
    # other aircraft systems (pressurization, fire system, doors, furnishings, etc.)
    (r"air conditioning|heating|pressuriz|fire warning|fire extinguish|"
     r"\bdoor\b|furnishings|galley|\bwindow\b|antenna|instrument|pump|"
     r"miscellaneous equipment|equipment, other|seat", "other_aircraft_system"),
    # structure
    (r"\bwing\b|fuselage|\bspar\b|\bskin\b|empennage|airframe|structure|"
     r"attachment|\bfitting\b|\bbolt\b|fastener|material defect|"
     r"inadequate design", "airframe_structure"),
    # human: maintenance
    (r"maintenance|\boverhaul\b|\binstallation\b|\binspection\b|quality control|"
     r"service bulletin|airworthiness", "maintenance_error"),
    # human: pilot / operational / procedural
    (r"aircraft control|aircraft handling|in-flight planning|flight/nav|"
     r"crew/group|procedure|directive|checklist|wrong runway|emergency procedure|"
     r"decision|lookout|clearance|supervision|remedial action|instructions",
     "pilot_operational"),
    # environmental
    (r"weather|terrain|\bobject\b|light condition|\bbird\b|\bwind\b|gust|"
     r"\bice\b|icing", "environmental"),
    # residual / undetermined
    (r"undetermined|not determined|^miscellaneous$|condition.*insufficient",
     "undetermined"),
]


def map_finding_to_bucket(label: str) -> str | None:
    s = " ".join((label or "").lower().split())
    if not s:
        return None
    for pat, bucket in FINDING_RULES:
        if re.search(pat, s):
            return bucket
    return None


# --- Dataset helpers -----------------------------------------------------------
def load_ds() -> dict:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def occ_labels(inc: dict) -> list[str]:
    return [(s.get("Occurrence_Description") or "").strip()
            for s in inc.get("sequence_of_events", [])
            if (s.get("Occurrence_Description") or "").strip()]


def is_fire(inc: dict) -> bool:
    return any(o.lower() == "fire" for o in occ_labels(inc))


def has_catchall(inc: dict) -> bool:
    return any(is_catchall_occ(o) for o in occ_labels(inc))


def usable_narrative(inc: dict) -> str:
    """Prefer the probable-cause narrative, then the factual; >=80 chars."""
    for k in ("narr_cause", "narr_accp", "narr_accf"):
        t = (inc.get(k) or "").strip()
        if len(t) >= 80:
            return t
    return ""


def cause_factor_findings(inc: dict) -> list[str]:
    """finding_descriptions flagged as Cause (C) or Factor (F)."""
    out = []
    for f in inc.get("findings", []):
        if str(f.get("Cause_Factor") or "").strip() in ("C", "F"):
            d = (f.get("finding_description") or "").strip()
            if d:
                out.append(d)
    return out


def all_findings(inc: dict) -> list[str]:
    return [(f.get("finding_description") or "").strip()
            for f in inc.get("findings", [])
            if (f.get("finding_description") or "").strip()]


def findings_buckets(inc: dict, cause_factor_only: bool = True) -> set[str]:
    labels = cause_factor_findings(inc) if cause_factor_only else all_findings(inc)
    buckets = set()
    for lab in labels:
        b = map_finding_to_bucket(lab)
        if b:
            buckets.add(b)
    return buckets


def structured_physical_bucket(inc: dict) -> set[str]:
    """Physical-mechanism buckets present in the structured findings (C/F or any)."""
    b = {x for x in findings_buckets(inc, cause_factor_only=True)
         if x in PHYSICAL_MECHANISMS}
    if not b:
        b = {x for x in findings_buckets(inc, cause_factor_only=False)
             if x in PHYSICAL_MECHANISMS}
    return b


# =============================================================================
# Stage A: burden + structured ground truth
# =============================================================================
def target_ev_ids(ds: dict) -> list[str]:
    """Catch-all occurrence + usable narrative (the decomposition population)."""
    return [ev for ev, inc in ds.items()
            if has_catchall(inc) and usable_narrative(inc)]


def stage_a(ds: dict) -> dict:
    N = len(ds)
    fires = [ev for ev, inc in ds.items() if is_fire(inc)]
    n_fire = len(fires)

    n_catchall = sum(1 for inc in ds.values() if has_catchall(inc))
    n_fire_catchall = sum(1 for ev in fires if has_catchall(ds[ev]))

    # incidents whose non-fire occurrence chain is ENTIRELY catch-all
    n_chain_only_catchall = 0
    for inc in ds.values():
        nonfire = [o for o in occ_labels(inc) if o.lower() != "fire"]
        if nonfire and all(is_catchall_occ(o) for o in nonfire):
            n_chain_only_catchall += 1

    targets = target_ev_ids(ds)
    targets_fire = [ev for ev in targets if is_fire(ds[ev])]

    # structured-findings resolution within the target set
    n_struct_physical = 0   # findings already name a specific physical system
    n_struct_none = 0       # no specific physical finding -> narrative is sole source
    buried = []             # ev_ids: catch-all occ AND no specific physical finding
    for ev in targets:
        inc = ds[ev]
        if structured_physical_bucket(inc):
            n_struct_physical += 1
        else:
            n_struct_none += 1
            buried.append(ev)

    # occurrence-label frequency (to justify the catch-all flag empirically)
    occ_freq = Counter()
    for inc in ds.values():
        for o in occ_labels(inc):
            occ_freq[o] += 1

    return {
        "N": N,
        "n_fire": n_fire,
        "catchall_labels": sorted(SEED_CATCHALL),
        "n_incidents_with_catchall_occ": n_catchall,
        "pct_incidents_with_catchall_occ": n_catchall / N,
        "n_fire_with_catchall_occ": n_fire_catchall,
        "pct_fire_with_catchall_occ": n_fire_catchall / n_fire if n_fire else 0.0,
        "n_chain_only_catchall": n_chain_only_catchall,
        "n_target": len(targets),
        "n_target_fire": len(targets_fire),
        "n_catchall_no_narrative": n_catchall - len(targets),
        "n_target_with_specific_structured_finding": n_struct_physical,
        "n_target_buried_no_specific_finding": n_struct_none,
        "buried_ev_ids": buried,
        "top_occurrence_labels": occ_freq.most_common(15),
        "target_ev_ids": targets,
    }


# =============================================================================
# Stage B: blind LLM mechanism extraction from narratives
# =============================================================================
EXTRACT_SYSTEM = (
    "You are an aviation safety analyst. You read an NTSB accident narrative and "
    "identify the single most specific PHYSICAL failure mechanism or causal factor "
    "that initiated the accident. You must choose exactly one label from a fixed "
    "list. Answer with ONLY the label, nothing else."
)
EXTRACT_INSTRUCTIONS = (
    "Choose the ONE label that best names the specific mechanism/cause described:\n"
    "- fuel_system: fuel line/tank/pump/selector leak, fuel starvation/exhaustion/contamination\n"
    "- electrical_wiring: wiring, relay, circuit breaker, generator, battery, short/arc\n"
    "- engine_powerplant: engine/cylinder/bearing/crankshaft/turbine/compressor/oil mechanical failure\n"
    "- landing_gear_brakes: landing gear, strut, drag brace, wheel, tire, brake, axle\n"
    "- hydraulic_system: hydraulic line/pump/fluid failure\n"
    "- flight_controls: aileron/elevator/rudder/spoiler/flap/trim/control-cable failure\n"
    "- airframe_structure: wing/fuselage/spar/skin/door/structural/fastener failure\n"
    "- propeller_rotor: propeller or (helicopter) rotor/blade failure\n"
    "- other_aircraft_system: pressurization, fire system, instruments, furnishings, other system\n"
    "- maintenance_error: improper maintenance/installation/inspection/overhaul caused it\n"
    "- pilot_operational: pilot handling/decision/procedure error, crew coordination\n"
    "- environmental: weather, terrain, bird, object, icing as the primary cause\n"
    "- undetermined: the narrative does not name any specific mechanism/cause\n"
    "Prefer a specific PHYSICAL mechanism (first 9 labels) when the narrative names a "
    "failed part/system, even if maintenance contributed. Use undetermined only when "
    "truly no mechanism is stated.\n\nNARRATIVE:\n"
)


def stage_b(ds: dict, target_ids: list[str], max_workers: int = 10,
            limit: int | None = None) -> dict:
    import concurrent.futures
    import main_app

    cache = {}
    if CACHE_PATH.is_file():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    todo = [ev for ev in target_ids if ev not in cache]
    if limit:
        todo = todo[:limit]
    print(f"Stage B: {len(target_ids)} targets, {len(cache)} cached, "
          f"{len(todo)} to extract", file=sys.stderr)

    client = main_app.get_client()

    def extract(ev: str) -> tuple[str, str]:
        narr = usable_narrative(ds[ev])[:1500]
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": EXTRACT_SYSTEM},
                    {"role": "user", "content": EXTRACT_INSTRUCTIONS + narr},
                ],
                temperature=0,
                max_tokens=8,
            )
            label = " ".join(resp.choices[0].message.content.strip().lower().split())
            label = re.sub(r"[^a-z_]", "", label.split()[0]) if label else "undetermined"
            if label not in TAXONOMY:
                label = "undetermined"
            return ev, label
        except Exception as e:  # noqa: BLE001
            print(f"  ! {ev}: {e}", file=sys.stderr)
            return ev, "ERROR"

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(extract, ev): ev for ev in todo}
        for fut in concurrent.futures.as_completed(futs):
            ev, label = fut.result()
            if label != "ERROR":
                cache[ev] = label
            done += 1
            if done % 25 == 0:
                print(f"  ...{done}/{len(todo)}", file=sys.stderr)
                CACHE_PATH.write_text(json.dumps(cache, indent=1), encoding="utf-8")

    CACHE_PATH.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    print(f"Stage B done: {len(cache)} cached labels -> {CACHE_PATH}", file=sys.stderr)
    return cache


# =============================================================================
# Stage C: resolution gain, agreement, buried subset, examples
# =============================================================================
def stage_c(ds: dict, stage_a_res: dict, cache: dict) -> dict:
    targets = stage_a_res["target_ev_ids"]
    have = [ev for ev in targets if ev in cache]

    llm_dist = Counter(cache[ev] for ev in have)
    n = len(have)

    n_specific_physical = sum(1 for ev in have if cache[ev] in PHYSICAL_MECHANISMS)
    n_non_undetermined = sum(1 for ev in have if cache[ev] != "undetermined")

    # narrative <-> structured findings agreement (precision of narrative recovery)
    # restricted to incidents where the structured findings DO name a physical
    # bucket (so there is a ground truth to check against).
    agree = 0
    checked = 0
    confusion = defaultdict(Counter)
    for ev in have:
        truth = structured_physical_bucket(ds[ev])
        if not truth:
            continue
        checked += 1
        pred = cache[ev]
        confusion[next(iter(sorted(truth)))][pred] += 1
        if pred in truth:
            agree += 1

    # buried subset: occurrence catch-all AND no specific structured physical
    # finding, yet the narrative recovered a specific physical mechanism.
    buried_ids = set(stage_a_res["buried_ev_ids"])
    buried_recovered = [ev for ev in have
                        if ev in buried_ids and cache[ev] in PHYSICAL_MECHANISMS]

    # verbatim examples: catch-all occurrence, show structured occ + finding + narr
    def example(ev: str) -> dict:
        inc = ds[ev]
        return {
            "ev_id": ev,
            "occurrence_chain": occ_labels(inc),
            "cause_factor_findings": cause_factor_findings(inc)[:6],
            "structured_physical_bucket": sorted(structured_physical_bucket(inc)),
            "narrative_recovered_mechanism": cache.get(ev),
            "narrative_snippet": usable_narrative(inc)[:320],
        }

    examples_buried = [example(ev) for ev in buried_recovered[:6]]
    # examples where narrative AND findings agree on a specific mechanism (shows the
    # specific cause is real and matches structured findings)
    examples_agree = []
    for ev in have:
        truth = structured_physical_bucket(ds[ev])
        if truth and cache[ev] in truth:
            examples_agree.append(example(ev))
        if len(examples_agree) >= 5:
            break

    return {
        "n_target_with_llm_label": n,
        "llm_mechanism_distribution": dict(llm_dist.most_common()),
        "resolution_gain_physical": n_specific_physical / n if n else 0.0,
        "resolution_gain_any_specific": n_non_undetermined / n if n else 0.0,
        "n_resolution_physical": n_specific_physical,
        "n_resolution_undetermined": n - n_non_undetermined,
        "narrative_vs_findings_checked": checked,
        "narrative_vs_findings_agree": agree,
        "narrative_vs_findings_agreement_rate": agree / checked if checked else 0.0,
        "n_buried_no_specific_finding": len(buried_ids),
        "n_buried_recovered_by_narrative": len(buried_recovered),
        "examples_buried_recovered": examples_buried,
        "examples_narrative_findings_agree": examples_agree,
    }


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B", "C"], default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap Stage B extractions (debug)")
    args = ap.parse_args()

    ds = load_ds()
    print(f"dataset: {DATA_PATH.name}  ({len(ds)} incidents)", file=sys.stderr)

    a = stage_a(ds)

    if args.stage == "A" or args.all:
        print("\n===== STAGE A: catch-all burden =====")
        print(f"N={a['N']}  fire={a['n_fire']}")
        print(f"catch-all labels: {a['catchall_labels']}")
        print(f"incidents w/ catch-all occurrence: {a['n_incidents_with_catchall_occ']}"
              f" ({a['pct_incidents_with_catchall_occ']:.1%})")
        print(f"  fire subset: {a['n_fire_with_catchall_occ']}"
              f" ({a['pct_fire_with_catchall_occ']:.1%} of fires)")
        print(f"chain entirely catch-all (non-fire): {a['n_chain_only_catchall']}")
        print(f"TARGET (catch-all + narrative): {a['n_target']} "
              f"(fire {a['n_target_fire']}); no-narrative {a['n_catchall_no_narrative']}")
        print(f"  target w/ specific structured finding: "
              f"{a['n_target_with_specific_structured_finding']}")
        print(f"  target BURIED (no specific finding): "
              f"{a['n_target_buried_no_specific_finding']}")

    cache = {}
    if args.stage == "B":
        cache = stage_b(ds, a["target_ev_ids"], limit=args.limit)

    if CACHE_PATH.is_file():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    if (args.stage == "C" or args.all) and cache:
        c = stage_c(ds, a, cache)
        print("\n===== STAGE C: narrative decomposition =====")
        print(f"targets with LLM label: {c['n_target_with_llm_label']}")
        print(f"resolution gain (specific physical mechanism): "
              f"{c['resolution_gain_physical']:.1%} "
              f"({c['n_resolution_physical']})")
        print(f"resolution gain (any non-undetermined cause): "
              f"{c['resolution_gain_any_specific']:.1%}")
        print(f"undetermined from narrative: {c['n_resolution_undetermined']}")
        print("mechanism distribution:")
        for k, v in c["llm_mechanism_distribution"].items():
            print(f"   {v:4}  {k}")
        print(f"narrative<->findings agreement: "
              f"{c['narrative_vs_findings_agree']}/{c['narrative_vs_findings_checked']}"
              f" = {c['narrative_vs_findings_agreement_rate']:.1%}")
        print(f"BURIED (no specific structured finding): "
              f"{c['n_buried_no_specific_finding']}; "
              f"recovered by narrative: {c['n_buried_recovered_by_narrative']}")

        out = {"stage_a": {k: v for k, v in a.items()
                           if k not in ("target_ev_ids", "buried_ev_ids")},
               "stage_c": {k: v for k, v in c.items()
                           if not k.startswith("examples_")},
               "examples_buried_recovered": c["examples_buried_recovered"],
               "examples_narrative_findings_agree": c["examples_narrative_findings_agree"]}
        RESULTS_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nresults -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()

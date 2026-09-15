"""Zhang published reference numbers for live side-by-side comparison."""

from __future__ import annotations

# Table 9 evidence columns (Zhang RESS 2021 §5.4) — our BN node names
TABLE9_COLUMNS = [
    {
        "label": "Inoperative engine instruments",
        "nodes": frozenset({"engine instrument"}),
    },
    {
        "label": "Combustion liner failure",
        "nodes": frozenset({"combustion assembly, combustion liner"}),
    },
    {
        "label": "Improper oil usage",
        "nodes": frozenset({"fluid, oil grade"}),
    },
    {
        "label": "Engine instruments & improper oil",
        "nodes": frozenset({"engine instrument", "fluid, oil grade"}),
    },
    {
        "label": "Loss of engine power",
        "nodes": frozenset({"loss of engine power"}),
    },
]

# target display name, BN node (or injury/damage state key), values per column
TABLE9_ROWS = [
    ("Loss of engine power", "loss of engine power", "event",
     [0.95, 0.50, 0.95, 0.99, 1.0]),
    ("Forced landing", "forced landing", "event",
     [0.1357, 0.0714, 0.1357, 0.1471, 0.1429]),
    ("Ditching", "ditching", "event", [0.00437, 0.00230, 0.00437, 0.00457, 0.00461]),
    ("Gear collapsed", "gear collapsed", "event",
     [0.00960, 0.00230, 0.00437, 0.00982, 0.00518]),
    ("Destroyed aircraft", "destroyed aircraft", "damage",
     [0.0133, 0.00230, 0.00437, 0.0135, 0.00559]),
    ("Substantial aircraft damage", "substantial damage", "damage",
     [0.0460, 0.00363, 0.00609, 0.0463, 0.0166]),
    ("Minor aircraft damage", "minor damage", "damage",
     [0.00934, 0.00154, 0.00292, 0.00947, 0.00378]),
    ("Serious injury", "serious injury", "injury",
     [0.0623, 0.000768, 0.00146, 0.0623, 0.00822]),
    ("No injury", "no injury", "injury",
     [0.9431, 0.9978, 0.9958, 0.9429, 0.9899]),
]

# Fig 12 / Table 9 LOEP-column injury+damage (upgraded network worked example)
FIG12_BY_EVIDENCE = {
    frozenset({"person: pilot-in-command"}): {
        "no injury": 0.97,
        "substantial damage": 0.0458,
        "unstabilized approach": 0.00484,
        "dragged wing, rotor, pod, float or tail/skid": 0.023,
    },
    frozenset({"loss of engine power"}): {
        "no injury": 0.9899,
        "serious injury": 0.00822,
        "destroyed aircraft": 0.00559,
        "substantial damage": 0.0166,
        "minor damage": 0.00378,
    },
}


def match_table9_column(hard_evidence: set[str]) -> tuple[int | None, str]:
    """Return (column_index, column_label) for exact evidence-set match."""
    ev = {e.strip().lower() for e in hard_evidence if e}
    for i, col in enumerate(TABLE9_COLUMNS):
        if ev == col["nodes"]:
            return i, col["label"]
    # subset: pick column whose nodes are contained in evidence (largest match)
    best_i, best_n = None, -1
    for i, col in enumerate(TABLE9_COLUMNS):
        if col["nodes"] <= ev and len(col["nodes"]) > best_n:
            best_i, best_n = i, len(col["nodes"])
    if best_i is not None:
        return best_i, TABLE9_COLUMNS[best_i]["label"] + " (partial match)"
    return None, ""


def table9_zhang_value(col_idx: int | None, row_idx: int) -> float | None:
    if col_idx is None:
        return None
    return TABLE9_ROWS[row_idx][3][col_idx]


def fig12_zhang_value(hard_evidence: set[str], state: str) -> float | None:
    ev = frozenset(e.strip().lower() for e in hard_evidence if e)
    for key, vals in FIG12_BY_EVIDENCE.items():
        if key <= ev:
            return vals.get(state)
    return None


def comparison_row(
    target: str,
    ours: float | None,
    zhang: float | None,
    method: str,
) -> dict:
    delta = None
    if ours is not None and zhang is not None:
        delta = round(ours - zhang, 4)
    return {
        "Target": target,
        "Ours": None if ours is None else round(ours, 4),
        "Zhang (published)": None if zhang is None else round(zhang, 4),
        "Δ (ours − Zhang)": delta if delta is not None else "N/A",
        "Method": method,
    }

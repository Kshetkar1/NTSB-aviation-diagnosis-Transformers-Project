"""Shared per-incident record builder for the diagnosis-hardening analyses.

This reuses the *validated* LOO harness (``tests/query_conditioning_validation.py``)
so calibration (FIX 1), auto-gating (FIX 2) and realistic-query robustness (FIX 3)
all rest on the exact same scaffolding:

  * the same eval-set construction (``build_eval_set``),
  * the same leave-self-out unconditioned prior ``P(cause | outcome)``
    (``_dist_excluding``),
  * the same conditioned ``P(cause | outcome, narrative-neighbours)``
    (``neighbor_ev_ids`` + ``_dist_restricted``),
  * the same GENERIC_CAUSES split and A/B leakage stratification.

Unlike the harness's ``evaluate`` (which only keeps scalar metrics), this returns
the *full per-incident distributions* (ordered cause lists + probability maps for
both the conditioned and the unconditioned prior) so the downstream analyses can
compute calibration, gating decisions, and degraded-query lifts.

READ-ONLY: this module does not modify any engine code. It runs offline from the
cached query embeddings (``docs/qc_embed_*.{npy,json}``) by default.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))

import main_app  # noqa: E402
import query_conditioning_validation as qcv  # noqa: E402

GENERIC = qcv.GENERIC  # lowercased generic catch-all labels


def _ensure_ds():
    """Bind the harness's module-global dataset (normally set in its main())."""
    if getattr(qcv, "_DS", None) is None:
        qcv._DS = main_app.refined_dataset
    return qcv._DS


def _det_order(prob: dict) -> list:
    """Deterministic cause ranking: by -probability, then cause label.

    The harness derives the order from a set-iteration whose tie-break depends on
    PYTHONHASHSEED; re-sorting with an explicit alphabetical secondary key makes
    every downstream analysis reproducible regardless of the hash seed.
    """
    return sorted(prob, key=lambda c: (-prob[c], c))


def build_records(top_n_incidents: int = 100, cache: dict | None = None,
                  limit: int | None = None) -> list[dict]:
    """Per-incident records with full conditioned & unconditioned distributions.

    Each record:
      ev, stratum ('A'/'B'), containment, generic_only (bool),
      true_all (set), true_spec (set),
      cond_order (list), cond_prob (dict), pool_cond (int),
      unc_order  (list), unc_prob  (dict), pool_uncond (int).

    Only incidents whose query embedding is cached are returned (offline-safe).
    """
    _ensure_ds()
    if cache is None:
        cache = qcv.load_emb_cache()
    items = qcv.build_eval_set(limit=limit)
    records = []
    for it in items:
        vec = cache.get(it["key"])
        if vec is None:
            continue
        pop = qcv._population(it["targets"])
        ev = it["ev"]
        neighbors = qcv.neighbor_ev_ids(vec, ev, top_n_incidents)
        u_order, u_prob = qcv._dist_excluding(pop, ev)
        c_order, c_prob, pool_n = qcv._dist_restricted(pop, neighbors, ev)
        u_order, c_order = _det_order(u_prob), _det_order(c_prob)
        pool_uncond = pop["total"] - (1 if ev in pop["ev_causes"] else 0)
        records.append({
            "ev": ev,
            "stratum": it["stratum"],
            "containment": it["containment"],
            "generic_only": it["generic_only"],
            "true_all": set(it["true_all"]),
            "true_spec": set(it["true_spec"]),
            "cond_order": c_order,
            "cond_prob": c_prob,
            "pool_cond": pool_n,
            "unc_order": u_order,
            "unc_prob": u_prob,
            "pool_uncond": pool_uncond,
        })
    return records


# --- small metric helpers (shared) -------------------------------------------
def top1(order, true_set):
    return bool(order) and order[0] in true_set


def mrr(order, true_set):
    for i, p in enumerate(order, 1):
        if p in true_set:
            return 1.0 / i
    return 0.0


def topk(order, true_set, k):
    return any(p in true_set for p in order[:k])


def top1_conf(order, prob):
    """Confidence = probability mass on the predicted top-1 cause."""
    if not order:
        return 0.0
    return float(prob.get(order[0], 0.0))


def is_clean_A(rec) -> bool:
    """Leakage-free factual stratum (A-clean)."""
    return rec["stratum"] == "A" and rec["containment"] < qcv.LEAK_CONTAINMENT


def is_A(rec) -> bool:
    return rec["stratum"] == "A"

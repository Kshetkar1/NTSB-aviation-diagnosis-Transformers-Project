"""FIX 3 - Realistic-query robustness of the conditioning lift (READ-ONLY).

The §14 validation conditioned on each incident's CLEAN factual NTSB narrative.
Real users type rough/partial descriptions. This script simulates degraded
queries and re-measures the conditioning LIFT over the unconditioned prior, to
bound real-world performance. We reuse the validated LOO scaffolding (self
excluded; same leave-self-out prior; same GENERIC split; leakage-free A-clean
stratum; specific-cause space = the headline contribution).

Degradations (per incident, derived from its factual narrative):
  (a) first_sentence : truncate to the first sentence only,
  (b) keyword        : content words only (stopwords stripped) -- a keyword query,
  (c) paraphrase     : LLM (gpt-4o-mini) short lay rewrite, instructed NOT to name
                       the cause (so we don't inject leakage),
  (d) noise          : the clean narrative + an appended block of irrelevant text.

For each variant we embed the degraded query, retrieve (self excluded), build the
conditioned P(cause|outcome, neighbours), and compute top-1 / MRR / mass lift vs
the SAME leave-self-out prior, on the SAME subsample as the clean baseline. We
report how much of the clean lift SURVIVES each degradation.

Degraded-query embeddings (and paraphrases) are cached to
docs/qc_robust_*.{npy,json} so re-runs are offline.

Outputs: docs/query_robustness_results.json, docs/figures/query_robustness.png
Run:  python3.11 tests/query_robustness.py --n 500           (network: embed + LLM)
      python3.11 tests/query_robustness.py --no-llm          (skip paraphrase)
      python3.11 tests/query_robustness.py --no-embed        (offline, cache only)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "docs" / ".mplcache"))

import main_app  # noqa: E402
import qc_common as qc  # noqa: E402
import query_conditioning_validation as qcv  # noqa: E402

DOCS = ROOT / "docs"
FIG_DIR = DOCS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = DOCS / "query_robustness_results.json"
FIG_PATH = FIG_DIR / "query_robustness.png"
EMB_VECS = DOCS / "qc_robust_embed_vecs.npy"
EMB_KEYS = DOCS / "qc_robust_embed_keys.json"
PARA_PATH = DOCS / "qc_robust_paraphrase.json"

SEED = 0
TOP_N = 100
VARIANTS = ["clean", "first_sentence", "keyword", "paraphrase", "noise"]

_NOISE = (" Routine administrative notes followed. The aircraft was registered "
          "several years prior and paperwork was on file. Weather briefing "
          "materials and unrelated scheduling memos were attached to the folder. "
          "The reviewer logged the case number and filed the standard cover sheet.")


# --- degradations -------------------------------------------------------------
def first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = parts[0] if parts else text
    # if the first sentence is very short, include the next one
    if len(out) < 40 and len(parts) > 1:
        out = out + " " + parts[1]
    return out


def keyword_only(text: str) -> str:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    seen, out = set(), []
    for w in toks:
        if w in qcv._STOP or len(w) <= 2 or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return " ".join(out)


def add_noise(text: str) -> str:
    return text.strip() + _NOISE


def degrade(variant: str, narrative: str) -> str:
    if variant == "clean":
        return narrative
    if variant == "first_sentence":
        return first_sentence(narrative)
    if variant == "keyword":
        return keyword_only(narrative)
    if variant == "noise":
        return add_noise(narrative)
    return narrative  # paraphrase handled separately (LLM)


# --- caches -------------------------------------------------------------------
def load_emb():
    if EMB_VECS.is_file() and EMB_KEYS.is_file():
        keys = json.loads(EMB_KEYS.read_text())
        vecs = np.load(EMB_VECS)
        return {k: vecs[i] for i, k in enumerate(keys)}
    return {}


def save_emb(cache):
    if not cache:
        return
    keys = list(cache)
    np.save(EMB_VECS, np.vstack([cache[k] for k in keys]).astype(np.float32))
    EMB_KEYS.write_text(json.dumps(keys))


def load_para():
    return json.loads(PARA_PATH.read_text()) if PARA_PATH.is_file() else {}


def save_para(d):
    PARA_PATH.write_text(json.dumps(d, indent=2))


# --- LLM paraphrase -----------------------------------------------------------
def paraphrase_missing(need: dict, cache: dict):
    """need: {ev: factual_narrative}; fill cache[ev] with a short lay rewrite."""
    import concurrent.futures
    from config import LLM_MODEL

    todo = [(ev, t) for ev, t in need.items() if ev not in cache]
    if not todo:
        return
    client = main_app.get_client()
    print(f"paraphrasing {len(todo)} narratives (model={LLM_MODEL})...",
          file=sys.stderr)

    def one(item):
        ev, txt = item
        prompt = (
            "Rewrite the following aviation accident report as a SHORT (1-2 "
            "sentence) plain-language description of what happened, as a layperson "
            "with no aviation training might type it into a search box. Describe "
            "only the observable events; do NOT state or speculate about the cause "
            "or any findings. Report:\n\n" + txt[:4000])
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=80, timeout=30)
            return ev, resp.choices[0].message.content.strip()
        except Exception as e:  # noqa: BLE001
            print(f"  paraphrase err {ev}: {e}", file=sys.stderr)
            return ev, None

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(one, it): it for it in todo}
        for fut in concurrent.futures.as_completed(futures):
            ev, out = fut.result()
            if out:
                cache[ev] = out
            done += 1
            if done % 100 == 0:
                print(f"  ...{done}/{len(todo)}", file=sys.stderr)
                save_para(cache)
    save_para(cache)


def embed_missing(need: dict, cache: dict, batch=128):
    """need: {key: text}; fill cache for missing keys (network)."""
    from config import EMBEDDING_MODEL
    todo = [(k, t) for k, t in need.items() if k not in cache and t]
    if not todo:
        return
    client = main_app.get_client()
    print(f"embedding {len(todo)} degraded queries...", file=sys.stderr)
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        resp = client.embeddings.create(
            input=[t[:qcv.MAX_QUERY_CHARS].replace("\n", " ") for _, t in chunk],
            model=EMBEDDING_MODEL)
        for (k, _), d in zip(chunk, resp.data):
            cache[k] = np.asarray(d.embedding, dtype=np.float32)
        if (i // batch) % 4 == 0:
            save_emb(cache)
    save_emb(cache)


# --- metrics over the conditioned distribution for a given query vector -------
def _spec(seq):
    return [c for c in seq if c not in qc.GENERIC]


def cond_metrics(vec, ev, pop, true_spec):
    neighbors = qcv.neighbor_ev_ids(vec, ev, TOP_N)
    c_order, c_prob, _ = qcv._dist_restricted(pop, neighbors, ev)
    c_order = qc._det_order(c_prob)
    so = _spec(c_order)
    return {
        "top1": 1.0 if qc.top1(so, true_spec) else 0.0,
        "mrr": qc.mrr(so, true_spec),
        "mass": sum(c_prob.get(c, 0.0) for c in true_spec),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="subsample size (A-clean, specific-true)")
    ap.add_argument("--no-llm", action="store_true", help="skip paraphrase variant")
    ap.add_argument("--no-embed", action="store_true", help="use cached embeddings only")
    args = ap.parse_args()

    qcv._DS = main_app.refined_dataset
    ds = main_app.refined_dataset

    # build the eval universe (A-clean, specific-true) reusing the harness
    items = qcv.build_eval_set()
    indexed = {c.get("ev_id") for c in main_app.embeddings_map}
    universe = []
    for it in items:
        if it["stratum"] != "A" or it["containment"] >= qcv.LEAK_CONTAINMENT:
            continue
        if not it["true_spec"] or it["ev"] not in indexed:
            continue
        universe.append(it)
    rng = np.random.default_rng(SEED)
    if args.n and args.n < len(universe):
        idx = rng.choice(len(universe), size=args.n, replace=False)
        sample = [universe[i] for i in sorted(idx)]
    else:
        sample = universe
    print(f"robustness subsample: {len(sample)} A-clean specific-true incidents",
          file=sys.stderr)

    variants = [v for v in VARIANTS if not (v == "paraphrase" and args.no_llm)]

    # paraphrases (LLM)
    para = load_para()
    if "paraphrase" in variants and not args.no_embed and not args.no_llm:
        paraphrase_missing({it["ev"]: it["query"] for it in sample}, para)

    # build degraded query texts
    texts = {}  # key f"{ev}|{variant}" -> text
    for it in sample:
        ev, narr = it["ev"], it["query"]
        for v in variants:
            if v == "paraphrase":
                t = para.get(ev)
            else:
                t = degrade(v, narr)
            if t:
                texts[f"{ev}|{v}"] = t

    cache = load_emb()
    if not args.no_embed:
        embed_missing(texts, cache)

    # evaluate
    per_variant = {v: {"top1": [], "mrr": [], "mass": [],
                       "u_top1": [], "u_mrr": [], "u_mass": [], "n": 0}
                   for v in variants}
    for it in sample:
        ev = it["ev"]
        pop = qcv._population(it["targets"])
        u_order, u_prob = qcv._dist_excluding(pop, ev)
        u_order = qc._det_order(u_prob)
        uso = _spec(u_order)
        true_spec = it["true_spec"]
        u_top1 = 1.0 if qc.top1(uso, true_spec) else 0.0
        u_mrr = qc.mrr(uso, true_spec)
        u_mass = sum(u_prob.get(c, 0.0) for c in true_spec)
        for v in variants:
            vec = cache.get(f"{ev}|{v}")
            if vec is None:
                continue
            m = cond_metrics(vec, ev, pop, true_spec)
            pv = per_variant[v]
            pv["top1"].append(m["top1"]); pv["mrr"].append(m["mrr"]); pv["mass"].append(m["mass"])
            pv["u_top1"].append(u_top1); pv["u_mrr"].append(u_mrr); pv["u_mass"].append(u_mass)
            pv["n"] += 1

    # aggregate lifts (paired vs the SAME prior, on the matched subset)
    summary = {"seed": SEED, "top_n_incidents": TOP_N, "subsample_n": len(sample),
               "stratum": "A-clean specific-true", "variants": {}}
    clean_lift = None
    for v in variants:
        pv = per_variant[v]
        if pv["n"] == 0:
            continue
        cond_top1, unc_top1 = np.mean(pv["top1"]), np.mean(pv["u_top1"])
        cond_mrr, unc_mrr = np.mean(pv["mrr"]), np.mean(pv["u_mrr"])
        cond_mass, unc_mass = np.mean(pv["mass"]), np.mean(pv["u_mass"])
        row = {
            "n": pv["n"],
            "cond_top1": float(cond_top1), "unc_top1": float(unc_top1),
            "lift_top1": float(cond_top1 - unc_top1),
            "cond_mrr": float(cond_mrr), "unc_mrr": float(unc_mrr),
            "lift_mrr": float(cond_mrr - unc_mrr),
            "cond_mass": float(cond_mass), "unc_mass": float(unc_mass),
            "lift_mass": float(cond_mass - unc_mass),
        }
        if v == "clean":
            clean_lift = row
        summary["variants"][v] = row

    # % of clean lift surviving each degradation
    if clean_lift:
        for v, row in summary["variants"].items():
            for metric in ("top1", "mrr", "mass"):
                cl = clean_lift[f"lift_{metric}"]
                row[f"survive_{metric}"] = (row[f"lift_{metric}"] / cl) if cl else None

    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"results -> {RESULTS_PATH}")
    _print_console(summary)
    try:
        _make_figure(summary)
        print(f"figure  -> {FIG_PATH}")
    except Exception as e:  # noqa: BLE001
        print(f"(figure skipped: {e})", file=sys.stderr)


def _print_console(s):
    print("\n" + "=" * 86)
    print(f"REALISTIC-QUERY ROBUSTNESS  (A-clean specific-true, n~{s['subsample_n']}; "
          f"lift = conditioned - prior)")
    print("=" * 86)
    print(f"{'variant':16} {'n':>5} {'cond_t1':>8} {'unc_t1':>7} {'lift_t1':>8} "
          f"{'lift_mrr':>9} {'%surv_t1':>9} {'%surv_mrr':>10}")
    print("-" * 86)
    for v in VARIANTS:
        r = s["variants"].get(v)
        if not r:
            continue
        st1 = r.get("survive_top1"); smrr = r.get("survive_mrr")
        st1s = f"{st1*100:7.0f}%" if st1 is not None else "    n/a"
        smrrs = f"{smrr*100:8.0f}%" if smrr is not None else "     n/a"
        print(f"{v:16} {r['n']:>5} {r['cond_top1']:8.3f} {r['unc_top1']:7.3f} "
              f"{r['lift_top1']:+8.3f} {r['lift_mrr']:+9.3f} {st1s:>9} {smrrs:>10}")


def _make_figure(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vs = [v for v in VARIANTS if v in s["variants"]]
    lift_t1 = [s["variants"][v]["lift_top1"] for v in vs]
    lift_mrr = [s["variants"][v]["lift_mrr"] for v in vs]
    x = np.arange(len(vs)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, [v * 100 for v in lift_t1], w, label="top-1 lift (pp)", color="#1f77b4")
    ax.bar(x + w / 2, [v * 1000 for v in lift_mrr], w, label="MRR lift (x1000)", color="#ff7f0e")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(vs, fontsize=9)
    ax.set_ylabel("conditioning lift over prior\n(top-1 in pp / MRR x1000)")
    ax.set_title("How much of the conditioning lift survives degraded queries?\n"
                 "(A-clean specific-true; 'clean' = validated +5.2pp/+0.052 baseline)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=130)


if __name__ == "__main__":
    main()

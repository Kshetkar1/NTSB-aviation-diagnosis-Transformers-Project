#!/usr/bin/env python3
"""Does a larger embedding model raise the headline retrieval numbers?

The severity headline (90.9% injury / 77.4% damage) is a k-NN readout, so the
only lever on it is retrieval quality. `text-embedding-3-small` was used
throughout and `text-embedding-3-large` was never tried. This scores both on
the same cohort, the same k, the same aggregation.

Both indexes are rebuilt here from one explicit text rule rather than reusing
the shipped index, for two reasons: the shipped index holds 1,703 vectors
while only 1,363 window accidents currently have any narrative at all, so its
provenance no longer matches the dataset; and an A/B on the model has to hold
the text fixed. The shipped-index result is reported alongside as a reference
point, not as one of the two arms.

Cost: roughly 1,300 narratives x 2 models, a few cents at current prices.

Requires OPENAI_API_KEY (in .env or the environment).

Run:
  # validate the scoring path with zero API calls, using the shipped index
  python3.11 tests/embedding_model_ablation.py --shipped-only

  # the real comparison
  python3.11 tests/embedding_model_ablation.py \
      --models text-embedding-3-small,text-embedding-3-large
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.pop("NTSB_FULL_CORPUS", None)
os.environ.pop("NTSB_USE_TRAIN_INDEX", None)

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "shared" / "code", FROZEN_DIR / "code",
           FROZEN_DIR / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import prognosis as pg  # noqa: E402
import query_to_bn as qb  # noqa: E402
from bn_upgraded import DMG_BY_CODE  # noqa: E402

FULL = REPO_ROOT / "shared" / "data" / "processed" / "refined_dataset.json"
WINDOW = REPO_ROOT / "shared" / "data" / "processed" / "refined_dataset_1982_2006.json"
PROC = REPO_ROOT / "shared" / "data" / "processed"
OUTDIR = FROZEN_DIR / "outputs"

INJ_LAB = ["fatal", "serious", "minor", "none"]
DMG_LAB = ["destroyed", "substantial", "minor", "none"]
TEXT_FIELD = "narr_accf"


def truth_states(inc):
    inj = pg.zhang_injury_code(inc)
    inj_i = {"FATL": 0, "SERS": 1, "MINR": 2, "NONE": 3}[inj]
    d = str(inc.get("damage") or "").upper()
    return inj_i, (DMG_BY_CODE.get(d) if d in DMG_BY_CODE else None)


def embed_batch(texts, model, batch=64):
    """Embed with retry; returns an (n, dim) L2-normalised float32 array."""
    import main_app
    client = main_app.get_client()
    out = []
    for i in range(0, len(texts), batch):
        chunk = [t.replace("\n", " ") for t in texts[i:i + batch]]
        for attempt in range(5):
            try:
                r = client.embeddings.create(input=chunk, model=model)
                out.extend([d.embedding for d in r.data])
                break
            except Exception as exc:
                if attempt == 4:
                    raise
                print(f"    retry {attempt + 1} after {exc}")
                time.sleep(2 ** attempt)
        print(f"    embedded {min(i + batch, len(texts))}/{len(texts)}")
    a = np.asarray(out, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def build_index(model, window_ds):
    """Build (or load) a window index for `model` from TEXT_FIELD."""
    tag = model.replace("/", "_")
    npy = PROC / f"embeddings_1982_2006__{tag}.npy"
    jsn = PROC / f"embeddings_map_1982_2006__{tag}.json"
    if npy.exists() and jsn.exists():
        print(f"  reusing cached index for {model}")
        return np.load(npy), json.loads(jsn.read_text())
    ids, texts = [], []
    for ev, inc in sorted(window_ds.items()):
        t = str(inc.get(TEXT_FIELD) or "").strip()
        if len(t) >= 100:
            ids.append(ev)
            texts.append(t)
    print(f"  embedding {len(texts)} window narratives with {model} ...")
    emb = embed_batch(texts, model)
    emap = [{"source": "incident", "ev_id": e, "type": "narrative"}
            for e in ids]
    np.save(npy, emb)
    jsn.write_text(json.dumps(emap))
    print(f"  wrote {npy.name} {emb.shape}")
    return emb, emap


def score(model, held, ds, use_shipped=False):
    """Severity argmax from the k-NN readout, under the given index."""
    import main_app
    import config
    if not use_shipped:
        emb, emap = build_index(model, ds)
        main_app.embeddings = emb
        main_app.embeddings_map = emap
        main_app.DATA_LOADED = True
        config.EMBEDDING_MODEL = model
        main_app.EMBEDDING_MODEL = model
    pi, pd_, ti, td = [], [], [], []
    for n, (k, inc, narr) in enumerate(held):
        t = qb.redact_severity_phrases(narr[:4000])
        yi, yd = truth_states(inc)
        rd = qb.severity_retrieval_distributions(t, ds, top_k=100)
        if not rd:
            continue
        pi.append(int(np.argmax(rd["injury"])))
        pd_.append(int(np.argmax(rd["damage"])))
        ti.append(yi)
        td.append(yd)
        if (n + 1) % 50 == 0:
            print(f"    scored {n + 1}/{len(held)}")
    return pi, pd_, ti, td


def macro_f1(p, y, n_cls):
    f1 = []
    for c in range(n_cls):
        tp = sum(1 for a, b in zip(p, y) if a == c and b == c)
        fp = sum(1 for a, b in zip(p, y) if a == c and b != c)
        fn = sum(1 for a, b in zip(p, y) if a != c and b == c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(f1) / n_cls


def main():
    models = ["text-embedding-3-small", "text-embedding-3-large"]
    if "--models" in sys.argv:
        models = sys.argv[sys.argv.index("--models") + 1].split(",")
    shipped_only = "--shipped-only" in sys.argv

    full = json.loads(FULL.read_text())
    window_ids = set(json.loads(WINDOW.read_text()).keys())
    ds = pg.load_dataset()
    import main_app  # noqa: F401

    held = []
    for k, inc in full.items():
        if k in window_ids:
            continue
        narr = str(inc.get("narr_accf") or "").strip()
        if len(narr) < 100:
            continue
        held.append((k, inc, narr))
    held.sort(key=lambda t: t[0])
    print(f"cohort: {len(held)}")

    results = {}
    print("\nshipped index (reference point)")
    results["shipped index (3-small)"] = score(None, held, ds, use_shipped=True)
    if not shipped_only:
        for m in models:
            print(f"\n{m}")
            results[m] = score(m, held, ds)

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit()
    emit("# Embedding model ablation on the severity readout")
    emit()
    emit(f"k = 100, Laplace alpha = 0.5, identical aggregation. Rebuilt "
         f"indexes use `{TEXT_FIELD}` with a 100-character floor; the shipped "
         "index row is a reference point only (different pool).")
    emit()
    for tgt, lab, ix in (("Injury", INJ_LAB, 0), ("Damage", DMG_LAB, 1)):
        emit(f"## {tgt}")
        emit()
        emit("| index | n | Accuracy | Macro-F1 | "
             + " | ".join(f"{s} recall" for s in lab) + " |")
        emit("|---|---|---|---|" + "---|" * len(lab))
        for name, (pi, pd_, ti, td) in results.items():
            p = pi if ix == 0 else pd_
            y = ti if ix == 0 else td
            pairs = [(a, b) for a, b in zip(p, y) if b is not None]
            p2 = [a for a, _ in pairs]
            y2 = [b for _, b in pairs]
            acc = sum(1 for a, b in zip(p2, y2) if a == b) / len(p2)
            rec = []
            for c in range(len(lab)):
                n = sum(1 for b in y2 if b == c)
                h = sum(1 for a, b in zip(p2, y2) if a == c and b == c)
                rec.append(f"{h}/{n}" if n else "-")
            emit(f"| {name} | {len(p2)} | {100*acc:.1f}% | "
                 f"{macro_f1(p2, y2, len(lab)):.3f} | " + " | ".join(rec) + " |")
        emit()

    out = OUTDIR / "embedding_model_ablation.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

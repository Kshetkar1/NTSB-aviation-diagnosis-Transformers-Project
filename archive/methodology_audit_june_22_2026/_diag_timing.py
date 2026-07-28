import os, sys, time

os.environ["NTSB_USE_TRAIN_INDEX"] = "1"
os.environ.setdefault("OPENAI_API_KEY", "offline-dummy")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

t = time.time()
import main_app

print("import+load", round(time.time() - t, 1), flush=True)


class D:
    def __getattr__(self, n):
        raise RuntimeError("offline")


main_app.get_client = lambda: D()

from run_leakfree_eval import load_full_index, query_vectors_from_full

emb, fmap = load_full_index()
qv = query_vectors_from_full(emb, fmap)
test = [
    x
    for x in open(
        os.path.join(REPO, "data/Testing_Data_Metrics/splits/test_ev_ids.txt")
    ).read().split()
    if x
]
ev = test[0]
q = qv[ev]
print("ev", ev, "train_emb_shape", main_app.embeddings.shape, flush=True)

t = time.time()
ts, tm = main_app.find_top_matches(q, exclude_ev_ids={ev})
print("find_top_matches", round(time.time() - t, 2), "n=", len(ts), flush=True)

t = time.time()
cl = main_app.cluster_incidents_by_type(ts, tm, 50)
print("cluster", round(time.time() - t, 2), "clusters=", len(cl), flush=True)

t = time.time()
an = main_app.calculate_cause_probabilities_per_cluster(cl)
print("cause_probs", round(time.time() - t, 2), flush=True)

t = time.time()
res = main_app.calculate_chain_rule_diagnosis(cl, an)
print("ltp", round(time.time() - t, 2), flush=True)

wc = res.get("weighted_causes", [])
print(
    "n_causes",
    len(wc),
    "top_pct",
    round(wc[0]["probability"] * 100, 1) if wc else None,
    flush=True,
)

import json
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import random

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import REFINED_DATA_PATH, EMBEDDINGS_PATH, EMBEDDINGS_MAP_PATH, OPENAI_API_KEY

def simple_kmeans(data, k, max_iters=20):
    """
    A lightweight implementation of K-Means using only NumPy.
    Does not use scikit-learn to avoid sandbox locking issues.
    """
    n_samples, n_features = data.shape
    
    # 1. Initialize centroids randomly
    # We use a fixed seed for reproducibility
    rng = np.random.RandomState(42)
    random_indices = rng.permutation(n_samples)[:k]
    centroids = data[random_indices]
    
    labels = np.zeros(n_samples, dtype=int)
    
    print(f"   Simple K-Means: Clustering {n_samples} items into {k} clusters...")
    
    for i in range(max_iters):
        # 2. Assign steps: Calculate distances to all centroids
        # Compute Euclidean distance
        # dist = sqrt(sum((x - c)^2))
        
        # We process in chunks to be memory efficient if needed, but for 2000 items it's fine
        # Vectorized distance calculation: ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
        # But straightforward broadcasting is clearer for this size:
        
        # Calculate distances (N x K matrix)
        distances = np.sqrt(((data[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        
        # Assign to closest centroid
        new_labels = np.argmin(distances, axis=1)
        
        # Check convergence
        if np.all(labels == new_labels):
            print(f"   Converged at iteration {i}")
            break
            
        labels = new_labels
        
        # 3. Update step: Recompute centroids
        for c_idx in range(k):
            # Get points in this cluster
            mask = (labels == c_idx)
            if np.any(mask):
                centroids[c_idx] = data[mask].mean(axis=0)
                
    return labels, centroids

def precompute_clusters():
    print("🚀 Starting Pre-computation of Incident Clusters (Lightweight Version)...", flush=True)
    
    # 1. Load Data
    print("   Loading dataset and embeddings...", flush=True)
    with open(REFINED_DATA_PATH, 'r') as f:
        refined_data = json.load(f)
        
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(EMBEDDINGS_MAP_PATH, 'r') as f:
        embeddings_map = json.load(f)
        
    print(f"   Loaded {len(refined_data)} incidents and {len(embeddings)} embeddings.", flush=True)

    # 2. Filter Embeddings
    target_indices = []
    target_ev_ids = []
    
    for idx, meta in enumerate(embeddings_map):
        if meta.get('source') != 'incident': continue
        if 'ev_id' not in meta: continue
        
        # CHANGED: Accept ANY incident embedding, not just narratives
        # We prefer narratives, but findings/causes are better than nothing
        target_indices.append(idx)
        target_ev_ids.append(meta['ev_id'])
            
    print(f"   Found {len(target_indices)} embeddings matching criteria.", flush=True)
    
    # Deduplicate (keep first embedding per ev_id)
    unique_ev_ids = set()
    final_indices = []
    final_ev_ids = []
    
    for idx, ev_id in zip(target_indices, target_ev_ids):
        if ev_id not in unique_ev_ids:
            unique_ev_ids.add(ev_id)
            final_indices.append(idx)
            final_ev_ids.append(ev_id)
            
    clustering_embeddings = embeddings[final_indices]
    print(f"   Clustering {len(clustering_embeddings)} unique incidents.", flush=True)
    
    # 3. Perform Lightweight K-Means
    NUM_CLUSTERS = 50
    labels, centroids = simple_kmeans(clustering_embeddings, NUM_CLUSTERS)
    
    # 4. Name the Clusters using LLM
    print("   Naming clusters using GPT-4o-mini...", flush=True)
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    cluster_names = {}
    used_names = {}  # track name -> count so duplicates get a suffix

    # Find representative incidents (closest to centroid)
    distances = np.sqrt(((clustering_embeddings[:, np.newaxis] - centroids) ** 2).sum(axis=2))
    closest_indices = np.argmin(distances, axis=0)  # shape (k,)

    # We will name all 50 clusters
    for cluster_id in range(NUM_CLUSTERS):
        rep_idx = np.where(labels == cluster_id)[0]
        if len(rep_idx) == 0:
            cluster_names[cluster_id] = f"Cluster {cluster_id}"
            continue

        # Pick up to 5 incidents closest to the centroid for richer context
        cluster_distances = distances[rep_idx, cluster_id]
        sorted_within = rep_idx[np.argsort(cluster_distances)]
        top5_indices = sorted_within[:5]

        narratives = []
        for idx in top5_indices:
            ev_id = final_ev_ids[idx]
            incident = refined_data.get(ev_id, {})
            narr = (incident.get('narr_cause') or incident.get('narr_accp')
                    or incident.get('narr_accf') or "")
            if narr and narr.strip().upper() not in ("NO NARRATIVE", "UNKNOWN", ""):
                narratives.append(narr[:300])

        if not narratives:
            cluster_names[cluster_id] = f"Cluster {cluster_id}"
            continue

        numbered = "\n".join(f"{i+1}. {n}" for i, n in enumerate(narratives))

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": (
                        "You are labeling clusters of aviation incident reports for a safety analysis system.\n\n"
                        "Below are up to 5 representative incident narratives from the SAME cluster.\n"
                        "Provide a SPECIFIC 2-5 word label that captures what makes this cluster DISTINCT "
                        "from other aviation incidents (e.g., not just 'Engine Fire' if these are specifically "
                        "'Inflight Engine Fire from Fuel Leak', or not just 'Pilot Error' if these are "
                        "specifically 'Approach Airspeed Deviation').\n\n"
                        f"Narratives:\n{numbered}\n\n"
                        "Respond with ONLY the label, nothing else."
                    )
                }],
                max_tokens=15
            )
            name = response.choices[0].message.content.strip().replace('"', '').replace("'", "")

            # Deduplicate: if this name was used before, append a number
            base = name
            if base in used_names:
                used_names[base] += 1
                name = f"{base} {used_names[base]}"
            else:
                used_names[base] = 1

            cluster_names[cluster_id] = name
            print(f"   Named Cluster {cluster_id}: {name}", flush=True)

        except Exception as e:
            print(f"Error naming cluster {cluster_id}: {e}", flush=True)
            cluster_names[cluster_id] = f"Cluster {cluster_id}"

    # 5. Save Labels to Refined Dataset
    print("   Saving cluster labels to dataset...", flush=True)
    
    # Map: ev_id -> cluster_name
    ev_id_to_label = {ev_id: cluster_names[label] for ev_id, label in zip(final_ev_ids, labels)}
    
    count = 0
    for ev_id, label in ev_id_to_label.items():
        if ev_id in refined_data:
            refined_data[ev_id]['cluster_label'] = label
            count += 1
            
    with open(REFINED_DATA_PATH, 'w') as f:
        json.dump(refined_data, f, indent=4)
        
    print(f"✅ Successfully updated {count} incidents with cluster labels.", flush=True)
    print(f"✅ Saved to: {REFINED_DATA_PATH}", flush=True)

if __name__ == "__main__":
    precompute_clusters()

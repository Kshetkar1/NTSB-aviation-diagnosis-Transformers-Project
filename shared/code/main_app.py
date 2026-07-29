import json
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from config import (
    get_openai_api_key,
    EMBEDDING_MODEL,
    USE_TRAIN_INDEX,
    ACTIVE_INCIDENT_DATA_PATH,
    ACTIVE_EMBEDDINGS_PATH,
    ACTIVE_EMBEDDINGS_MAP_PATH,
    ACTIVE_INDEX_LABEL,
    OUTPUT_DIR,
)

# --- Configuration ---
# All configuration is loaded from config.py (which uses environment variables)

# --- Data Loading ---
# Only load data if files exist (for testing purposes)
# active_backend: which embedding index is loaded ("openai" default paths from config).
active_backend = None
try:
    print(f"🧠 Loading knowledge base ({ACTIVE_INDEX_LABEL})...")
    with open(ACTIVE_INCIDENT_DATA_PATH, 'r') as f:
        refined_dataset = json.load(f)

    embeddings = np.load(ACTIVE_EMBEDDINGS_PATH)
    with open(ACTIVE_EMBEDDINGS_MAP_PATH, 'r') as f:
        embeddings_map = json.load(f)
    print("✅ Knowledge base loaded successfully!")
    DATA_LOADED = True
    active_backend = "openai"
except FileNotFoundError as e:
    print(f"⚠️  Data files not found: {e}")
    print("   This is OK if you're just testing imports. Run data processing scripts to generate data.")
    refined_dataset = {}
    embeddings = None
    embeddings_map = []
    DATA_LOADED = False
    active_backend = None


# --- Core Functions ---
# Initialize client lazily when API key is actually needed
client = None

def get_client():
    """Get OpenAI client, initializing it if needed."""
    global client
    if client is None:
        client = OpenAI(api_key=get_openai_api_key())
    return client

_QUERY_EMB_CACHE_PATH = (Path(__file__).resolve().parents[1] / "data" /
                         "processed" / "query_emb_cache.npz")
_query_emb_cache = None


def _emb_cache():
    global _query_emb_cache
    if _query_emb_cache is None:
        _query_emb_cache = {}
        if _QUERY_EMB_CACHE_PATH.exists():
            z = np.load(_QUERY_EMB_CACHE_PATH, allow_pickle=False)
            _query_emb_cache = {k: z[k] for k in z.files}
    return _query_emb_cache


def get_embedding(text, use_cache=True):
    """Embedding for a text via the OpenAI API, with a persistent disk cache.

    The cache (keyed by model + sha1 of the normalized text) makes eval
    re-runs deterministic and free: the API is only hit for texts never seen
    before. Pass use_cache=False to force a fresh API call.
    """
    import hashlib
    text = text.replace("\n", " ")
    key = f"{EMBEDDING_MODEL}_{hashlib.sha1(text.encode()).hexdigest()}"
    if use_cache:
        cached = _emb_cache().get(key)
        if cached is not None:
            return cached.tolist()
    response = get_client().embeddings.create(input=[text], model=EMBEDDING_MODEL)
    emb = response.data[0].embedding
    if use_cache:
        _emb_cache()[key] = np.asarray(emb, dtype=np.float32)
        try:
            _QUERY_EMB_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(_QUERY_EMB_CACHE_PATH, **_emb_cache())
        except OSError:
            pass                       # read-only environment: cache stays in memory
    return emb

def find_top_matches(query_embedding, exclude_ev_ids=None):
    """Finds the top N most similar embeddings from the knowledge base.

    Args:
        query_embedding: query vector (already L2-normalized like OpenAI embeddings).
        exclude_ev_ids: optional iterable of ev_id strings to drop from the results
            BEFORE ranking. This is the exclude-self / no-leakage guard: pass the
            query incident's own ev_id (and any known near-duplicate ev_ids) so the
            engine can never retrieve the case it is being asked about as its own
            evidence. Defaults to None (legacy behavior: nothing excluded).
    """
    if not DATA_LOADED:
        raise RuntimeError("Data files not loaded. Please run data processing scripts first.")
    
    # Calculate cosine similarity using pure NumPy (faster/safer in sandbox than sklearn)
    # Note: OpenAI embeddings are already normalized, so dot product == cosine similarity
    # We transpose embeddings (matrix) to shape (features, n_samples) for dot product
    # query_embedding is (features,)
    
    # Ensure query is array
    q = np.array(query_embedding)
    
    # Dot product: (n_samples, features) dot (features,) -> (n_samples,)
    similarities = np.dot(embeddings, q)
    
    # Get the indices of all similarities, sorted from highest to lowest
    all_sorted_indices = np.argsort(similarities)[::-1]

    # Exclude-self / leakage guard: drop any rows whose ev_id is in exclude_ev_ids.
    exclude = set(exclude_ev_ids) if exclude_ev_ids else None

    # Get the corresponding scores and mapping info
    if exclude:
        top_scores = []
        top_matches_info = []
        for i in all_sorted_indices:
            info = embeddings_map[i]
            if info.get('ev_id') in exclude:
                continue
            top_scores.append(similarities[i])
            top_matches_info.append(info)
    else:
        top_scores = [similarities[i] for i in all_sorted_indices]
        top_matches_info = [embeddings_map[i] for i in all_sorted_indices]
    
    return top_scores, top_matches_info

def get_causal_chains(ev_id_list):
    """
    Extracts the causal histories (sequences of events) for a given list of event IDs.
    """
    chains = {}
    for ev_id in ev_id_list:
        if ev_id in refined_dataset:
            # Ensure we're accessing the correct structure
            incident_data = refined_dataset[ev_id]
            sequence = incident_data.get('sequence_of_events', [])
            if sequence:
                chains[ev_id] = sequence
    return chains

def build_network_chart(chains):
    """Generates MermaidJS code for a network diagram from the causal chains."""
    if not chains:
        return "No causal chains found to build a chart."

    mermaid_string = "graph TD;\n"
    # Use subgraphs to visually group events by their incident ID
    for ev_id, sequence in chains.items():
        mermaid_string += f"    subgraph {ev_id}\n"
        for i in range(len(sequence) - 1):
            event_a = sequence[i].get('Occurrence_Description', 'Unknown Event')
            event_b = sequence[i+1].get('Occurrence_Description', 'Unknown Event')
            # Sanitize text for MermaidJS and create unique node IDs
            event_a_id = f'"{ev_id}_{i}"'
            event_b_id = f'"{ev_id}_{i+1}"'
            mermaid_string += f"        {event_a_id}[\"{event_a}\"] --> {event_b_id}[\"{event_b}\"];\n"
        mermaid_string += "    end\n"
            
    return mermaid_string


# --- Similarity-Based Diagnosis Functions ---

def calculate_similarity_weighted_diagnosis(top_scores, top_matches, top_n_incidents=50):
    """
    Implements similarity-weighted diagnosis using historical case frequencies.
    
    Takes similar incidents with cosine similarity scores and calculates
    weighted probabilities for root causes based on similarity-weighted evidence.
    
    Formula: P_weighted(cause) = Σ(similarity × presence) / Σ(similarity)
    
    Note: This is NOT Bayesian inference - it's weighted averaging of historical
    case frequencies using similarity scores as weights.
    
    Args:
        top_scores: List of cosine similarity scores
        top_matches: List of match metadata from embeddings_map
        top_n_incidents: Number of top incidents to consider (default: 50)
    
    Returns:
        Dictionary with:
            - 'weighted_causes': List of (cause, probability, supporting_incidents)
            - 'total_incidents_analyzed': Number of incidents used
            - 'methodology': Description of the approach
    """
    from collections import defaultdict
    
    # Track causes and their weighted evidence
    cause_evidence = defaultdict(list)  # cause -> [(similarity, ev_id), ...]
    incidents_analyzed = 0
    
    # Process top N similar incidents
    for score, match in zip(top_scores[:top_n_incidents], top_matches[:top_n_incidents]):
        # Only process incident entries (not dictionary entries)
        if match.get('source') != 'incident':
            continue
            
        ev_id = match.get('ev_id')
        if not ev_id:
            continue
            
        # Get diagnostic data (backward compatible with old 'bayesian_data' field name)
        diagnostic_data = match.get('diagnostic_data') or match.get('bayesian_data', {})
        if not diagnostic_data.get('has_diagnostic_data'):
            continue
        
        incidents_analyzed += 1
        
        # Get causes from diagnostic data (contains both structured findings and narratives)
        all_causes = diagnostic_data.get('all_causes', [])
        
        # Record evidence for each cause (from both sources)
        # Normalize to lowercase+stripped so "IMPROPER MAINTENANCE" and
        # "Improper Maintenance" are counted as the same cause.
        for cause in all_causes:
            normalized = cause.strip().lower()
            cause_evidence[normalized].append((score, ev_id))
    
    # Calculate weighted probabilities for each cause
    weighted_causes = []
    total_similarity_sum = sum(score for score, _ in 
                               [item for sublist in cause_evidence.values() for item in sublist])
    
    for cause, evidence_list in cause_evidence.items():
        # Calculate weighted probability
        # P(cause) = sum(similarity scores where cause is present) / sum(all similarities)
        cause_similarity_sum = sum(score for score, _ in evidence_list)
        weighted_probability = cause_similarity_sum / total_similarity_sum if total_similarity_sum > 0 else 0
        
        # Get supporting incident IDs
        supporting_incidents = [ev_id for _, ev_id in evidence_list]
        
        weighted_causes.append({
            'cause': cause,
            'probability': weighted_probability,
            'raw_score': cause_similarity_sum,
            'num_incidents': len(supporting_incidents),
            'supporting_incidents': supporting_incidents[:5]  # Top 5 for display
        })
    
    # Sort by probability (descending)
    weighted_causes.sort(key=lambda x: x['probability'], reverse=True)
    
    return {
        'weighted_causes': weighted_causes,
        'total_incidents_analyzed': incidents_analyzed,
        'methodology': 'Similarity-weighted historical case analysis',
        'total_similarity_weight': total_similarity_sum
    }


def clean_ntsb_text(text):
    """
    Cleans raw NTSB text to make it readable for the dashboard.
    Removes codes, simplifies hierarchical categories, and shortens sentences.
    """
    if not text: return "Unknown"
    
    text = str(text).strip()
    
    # 1. Remove NTSB Hierarchical Prefixes
    prefixes_to_strip = [
        "Personnel issues-",
        "Organizational issues-",
        "Aircraft-Aircraft power plant-",
        "Aircraft-Aircraft systems-",
        "Aircraft-",
        "Environmental issues-",
    ]
    
    for prefix in prefixes_to_strip:
        if prefix in text:
            # Keep the last meaningful part
            parts = text.split('-')
            # Filter out generic terms
            meaningful = [p for p in parts if len(p) > 2 and p.lower() not in ['general', 'miscellaneous', 'other']]
            if meaningful:
                text = meaningful[-1].strip()
                # If the last part is a single letter or code, take the one before it
                if len(text) < 3 and len(meaningful) > 1:
                    text = meaningful[-2].strip()
    
    # 2. Simplifications for specific common patterns
    if "Takeoff-rejected takeoff" in text:
        text = "Rejected Takeoff"
    elif "Standing-engine(s) start-up" in text:
        text = "Engine Start-up"
    elif "Landing-landing roll" in text:
        text = "Landing Roll"
    elif "approach-VFR pattern" in text:
        text = "VFR Approach"
        
    # 3. Remove "which resulted in..." clauses (often redundant)
    if ", which resulted in" in text:
        text = text.split(", which resulted in")[0]
        
    # 4. Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
        
    return text

def generate_diagnosis_tree(cluster_name, causes_with_probs):
    """
    Generates a Mermaid chart visualizing Diagnosis (Effect <- Causes).
    Structure: Cluster (Effect) <-- [Prob] -- Cause
    """
    # Use LR layout for horizontal tree (Causes on Left -> Cluster on Right)
    # This matches the user's preference for horizontal flow, 
    # but logically arrows point towards the Effect (Forward Causal) 
    # or towards the Cause (Diagnostic Inference)?
    # Diagnostic Inference usually: Observed Effect -> Inferred Cause.
    # But Causal Graph usually: Cause -> Effect.
    # Dr. Maha's drawing had Q -> Cluster -> Cause (which is the inference direction).
    # Let's match Dr. Maha's drawing: Cluster -> Cause.
    
    mermaid = ["graph TD"]
    mermaid.append("    %% Styles")
    mermaid.append("    classDef effectNode fill:#e74c3c,stroke:#c0392b,stroke-width:2px,color:white;")
    mermaid.append("    classDef causeNode fill:#3498db,stroke:#2980b9,stroke-width:2px,color:white;")
    mermaid.append("    linkStyle default stroke:#7f8c8d,stroke-width:2px;")
    
    # Root Node (The Cluster / Observed Effect)
    cluster_id = "root"
    safe_cluster = cluster_name.replace('"', "'").title()
    mermaid.append(f'    {cluster_id}("{safe_cluster}"):::effectNode')
    
    # Causes
    import textwrap
    for i, item in enumerate(causes_with_probs[:8]): # Top 8 to keep it readable
        cause_text = item['cause']
        prob = item.get('probability', 0)
        
        # Wrap text - looser wrapping since we have zoom now
        if len(cause_text) > 50:
            cause_text = "<br/>".join(textwrap.wrap(cause_text, width=50))
        
        cause_id = f"c{i}"
        safe_cause = cause_text.replace('"', "'")
        
        mermaid.append(f'    {cause_id}("{safe_cause}"):::causeNode')
        
        # Edge: Cluster -> Cause (Inference Direction per Dr. Maha's drawing)
        # Label: Probability
        mermaid.append(f'    {cluster_id} -->|"{prob:.1%}"| {cause_id}')
        
    return "\n".join(mermaid)

def generate_probabilistic_sequence_diagram(incident_list, min_edge_prob=0.10):
    """
    Generates a Mermaid flowchart showing the most common event sequences
    and their transition probabilities.
    
    Args:
        incident_list: List of incident dictionaries (containing 'sequence_of_events')
        min_edge_prob: Hide transitions that happen less than this % of the time (to reduce clutter)
    """
    from collections import defaultdict, Counter
    
    # 1. Count transitions: 
    # (Finding/Cause) -> (Occurrence)
    # (Occurrence A) -> (Occurrence B)
    transitions = defaultdict(Counter) # {StartEvent: {EndEvent: count}}
    
    total_sequences = 0
    
    for inc in incident_list:
        if not inc: continue
        
        seq = inc.get('sequence_of_events', [])
        findings = inc.get('findings', [])
        
        if not seq: continue
        total_sequences += 1
        
        # --- Map Findings to Occurrences ---
        # Findings often have an 'Occurrence_No' linking them to a specific event in the sequence
        # We want to draw: Cause -> Occurrence
        
        # Create a map of Occurrence_No -> Occurrence_Description
        occ_map = {}
        for ev in seq:
            occ_no = ev.get('Occurrence_No')
            desc = ev.get('Occurrence_Description', 'Unknown').strip()
            if occ_no:
                occ_map[occ_no] = desc
        
        # Link Causes to Occurrences
        for f in findings:
            cause_text = f.get('finding_description', '').strip()
            # cause_text = clean_ntsb_text(raw_cause) # REVERTED: Keep raw text for now per user request
            
            # Text Wrap for readability (no truncation)
            import textwrap
            # Aggressive wrapping for chart visibility
            if len(cause_text) > 50:
                cause_text = "<br/>".join(textwrap.wrap(cause_text, width=50))
                
            if seq:
                first_event = seq[0].get('Occurrence_Description', 'Unknown').strip()
                # Wrap first event too
                if len(first_event) > 50:
                    first_event = "<br/>".join(textwrap.wrap(first_event, width=50))
                transitions[cause_text][first_event] += 1

        # --- Link Sequence of Events ---
        # (Occurrence A) -> (Occurrence B)
        clean_events = []
        import textwrap # Ensure imported
        for e in seq:
            desc = e.get('Occurrence_Description', 'Unknown').strip()
            
            # Wrap event text for cleaner chart layout
            if len(desc) > 50:
                desc = "<br/>".join(textwrap.wrap(desc, width=50))
                
            if not clean_events or clean_events[-1] != desc:
                clean_events.append(desc)
        
        for i in range(len(clean_events) - 1):
            start = clean_events[i]
            end = clean_events[i+1]
            transitions[start][end] += 1
            
        # --- Link Last Event to Outcome ---
        # (Last Occurrence) -> (Damage Level / Injury)
        if clean_events:
            last_event = clean_events[-1]
            
            # Damage
            damage = inc.get('damage', 'NONE')
            if damage and damage != 'NONE':
                damage_node = f"Damage: {damage}"
                transitions[last_event][damage_node] += 1
                
            # Injury (Highest)
            injury = inc.get('ev_highest_injury', 'NONE')
            if injury and injury != 'NONE':
                injury_node = f"Injury: {injury}"
                transitions[last_event][injury_node] += 1


    if total_sequences == 0:
        return None

    # 2. Build Mermaid String
    mermaid = ["graph LR"]
    mermaid.append("    %% Styles")
    mermaid.append("    classDef startNode fill:#3498db,stroke:#2980b9,stroke-width:2px,color:white;")
    mermaid.append("    classDef endNode fill:#e74c3c,stroke:#c0392b,stroke-width:2px,color:white;")
    mermaid.append("    classDef midNode fill:#3498db,stroke:#2980b9,stroke-width:1px,color:white;") 
    mermaid.append("    classDef defaultNode fill:#95a5a6,stroke:#7f8c8d,stroke-width:1px,color:white;")
    
    # Track created nodes
    node_id_map = {}
    node_types = {} 
    next_id = 0
    node_defined = set() # To prevent duplicate node definitions
    
    def get_id(name):
        nonlocal next_id
        if name not in node_id_map:
            safe_name = name.replace('"', '').replace("'", "")
            node_id_map[name] = f"node{next_id}"
            next_id += 1
        return node_id_map[name]

    # Identify targets for node typing
    all_targets = set()
    for targets in transitions.values():
        all_targets.update(targets.keys())
        
    link_count = 0
    link_styles = []

    # Generate Edges
    for start_event, targets in transitions.items():
        start_id = get_id(start_event)
        
        # Determine node type
        if start_event not in all_targets:
            node_types[start_id] = 'start'
        elif start_id not in node_types:
            node_types[start_id] = 'mid'
            
        total_out = sum(targets.values())
        # Identify Golden Path (most frequent transition from this node)
        primary_target = max(targets, key=targets.get) if targets else None
        
        for end_event, count in targets.items():
            prob = count / total_out
            
            if prob >= min_edge_prob:
                end_id = get_id(end_event)
                prob_pct = f"{prob:.0%}"
                
                # Determine end node type
                if end_event not in transitions:
                    node_types[end_id] = 'end'
                elif end_id not in node_types:
                    node_types[end_id] = 'mid'
                
                safe_start = start_event.replace('"', "'").replace('(', '[').replace(')', ']').replace('#', '')
                safe_end = end_event.replace('"', "'").replace('(', '[').replace(')', ']').replace('#', '')
                
                # Add nodes (if not already added)
                if start_id not in node_defined:
                    mermaid.append(f'    {start_id}("{safe_start}")')
                    node_defined.add(start_id)
                if end_id not in node_defined:
                    mermaid.append(f'    {end_id}("{safe_end}")')
                    node_defined.add(end_id)
                
                # Add Edge
                mermaid.append(f'    {start_id} -->|"{prob_pct}"| {end_id}')
                
                # Highlight Golden Path (Primary transition > 20% prob)
                # We highlight the "Main Artery" of the accident flow
                is_golden = (end_event == primary_target and prob > 0.20)
                
                if is_golden:
                    # Thick Orange for Golden Path
                    link_styles.append(f"    linkStyle {link_count} stroke:#f39c12,stroke-width:4px;")
                else:
                    # Thin Grey for Alternative Paths
                    link_styles.append(f"    linkStyle {link_count} stroke:#bdc3c7,stroke-width:1px;")
                
                link_count += 1

    # Apply Classes
    for nid, ntype in node_types.items():
        if ntype == 'start':
            mermaid.append(f'    class {nid} startNode;')
        elif ntype == 'end':
            mermaid.append(f'    class {nid} endNode;')
        else:
            mermaid.append(f'    class {nid} defaultNode;')
            
    # Add link styles at the very end
    mermaid.extend(link_styles)

    return "\n".join(mermaid)


def diagnose_root_causes(query, top_n=10):
    """
    High-level diagnostic interface using similarity-weighted historical analysis.
    
    Args:
        query: User's incident description
        top_n: Number of top causes to return (default: 10)
    
    Returns:
        Dictionary with:
            - 'top_causes': Top N most likely root causes
            - 'all_causes': All identified causes
            - 'diagnosis_metadata': Information about the analysis
    """
    print(f"\n🔍 Diagnosing root causes for: '{query}'")
    
    # 1. Get embedding for query
    query_embedding = get_embedding(query)
    
    # 2. Find similar incidents
    top_scores, top_matches = find_top_matches(query_embedding)
    
    # 3. Calculate weighted diagnosis
    diagnosis_results = calculate_similarity_weighted_diagnosis(top_scores, top_matches)
    
    # 4. Extract top N causes
    all_causes = diagnosis_results['weighted_causes']
    top_causes = all_causes[:top_n]
    
    print(f"✅ Analyzed {diagnosis_results['total_incidents_analyzed']} similar incidents")
    print(f"   Found {len(all_causes)} potential root causes")
    
    return {
        'top_causes': top_causes,
        'all_causes': all_causes,
        'diagnosis_metadata': {
            'incidents_analyzed': diagnosis_results['total_incidents_analyzed'],
            'total_causes_found': len(all_causes),
            'methodology': diagnosis_results['methodology']
        }
    }


# --- Conditional Probability Diagnosis (Chain Rule Approach) ---

def cluster_incidents_by_type(top_scores, top_matches, top_n_incidents=50):
    """
    Step 1: Cluster top N similar incidents into incident types.
    
    Uses pre-computed clusters if available (fast), otherwise falls back to LLM (slow).
    
    Args:
        top_scores: List of cosine similarity scores
        top_matches: List of match metadata from embeddings_map
        top_n_incidents: Number of top incidents to cluster (default: 50)
    
    Returns:
        Dictionary mapping cluster_type -> list of (score, ev_id, match_data)
    """
    print(f"\n🔄 Clustering top {top_n_incidents} incidents by type...")
    
    from collections import defaultdict
    import concurrent.futures
    
    clusters = defaultdict(list)
    client_instance = get_client()
    
    # Prepare list of incidents to process
    incidents_to_process = []
    
    for score, match in zip(top_scores[:top_n_incidents], top_matches[:top_n_incidents]):
        # Only process incident entries (not dictionary entries)
        if match.get('source') != 'incident':
            continue
            
        ev_id = match.get('ev_id')
        if not ev_id or ev_id not in refined_dataset:
            continue
        
        incident_data = refined_dataset[ev_id]
        
        # CHECK FOR PRE-COMPUTED CLUSTER LABEL
        pre_label = incident_data.get('cluster_label')
        if pre_label:
            # Normalize
            pre_label = ' '.join(pre_label.lower().split())
            
            # Add to cluster immediately
            clusters[pre_label].append({
                'score': score,
                'ev_id': ev_id,
                'match': match
            })
            continue # Skip LLM processing for this one
        
        # If no label, fallback to on-the-fly LLM classification
        # Get narrative for classification
        narrative = incident_data.get('narr_accp', '') or incident_data.get('narr_cause', '')
        if not narrative or len(narrative) < 20:
            # Fallback to findings if no narrative
            findings = incident_data.get('findings', [])
            if findings:
                narrative = '; '.join([f.get('finding_description', '') for f in findings[:3]])
        
        if not narrative or len(narrative) < 20:
            # Skip incidents with insufficient data
            continue
        
        # Truncate very long narratives to save tokens
        narrative_excerpt = narrative[:500] if len(narrative) > 500 else narrative
        
        incidents_to_process.append({
            'score': score,
            'ev_id': ev_id,
            'match': match,
            'narrative_excerpt': narrative_excerpt
        })
        
    # Function to classify a single incident (Fallback)
    def classify_incident(incident_info):
        try:
            prompt = f"""Classify this aviation incident into ONE specific category/type. 
Be concise (2-5 words max). Focus on the PRIMARY incident type.

Examples of good categories: "Engine failure", "Fuel system malfunction", "Landing gear collapse", "Bird strike", "Pilot error - landing"

Incident narrative: {incident_info['narrative_excerpt']}

Category:"""
            
            response = client_instance.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=20
            )
            
            incident_type = response.choices[0].message.content.strip()
            
            # Normalize the type (lowercase, remove extra spaces)
            incident_type = ' '.join(incident_type.lower().split())
            
            return incident_type, incident_info
            
        except Exception as e:
            print(f"⚠️  Error classifying incident {incident_info['ev_id']}: {e}")
            return 'uncategorized', incident_info
            
    # Execute in parallel only for unlabelled incidents
    if incidents_to_process:
        print(f"   Processing {len(incidents_to_process)} incidents via LLM (fallback)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_incident = {executor.submit(classify_incident, inc): inc for inc in incidents_to_process}
            
            for future in concurrent.futures.as_completed(future_to_incident):
                incident_type, incident_info = future.result()
                clusters[incident_type].append(incident_info)
    
    total_clustered = sum(len(v) for v in clusters.values())
    print(f"✅ Clustered {total_clustered} incidents into {len(clusters)} types")
    for cluster_type, incidents in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"   - '{cluster_type}': {len(incidents)} incidents")
    
    return dict(clusters)


def calculate_cause_probabilities_per_cluster(clusters):
    """
    Step 2: For each cluster, calculate P(Cause|Cluster) from historical data.
    
    This is the MISSING piece from the whiteboard - calculating the probability
    of each cause given the incident type.
    
    Args:
        clusters: Dictionary from cluster_incidents_by_type()
    
    Returns:
        Dictionary mapping cluster_type -> {
            'causes': {cause: probability},
            'total_incidents': count,
            'avg_similarity': average score
        }
    """
    print(f"\n🧮 Calculating P(Cause|Cluster) for each cluster...")
    
    from collections import Counter
    cluster_analysis = {}
    
    for cluster_type, incidents in clusters.items():
        # Extract all causes from incidents in this cluster
        all_causes_in_cluster = []
        total_similarity = 0
        
        for incident in incidents:
            match = incident['match']
            total_similarity += incident['score']
            
            # Get causes from diagnostic_data (backward compatible with 'bayesian_data')
            diagnostic_data = match.get('diagnostic_data') or match.get('bayesian_data', {})
            if diagnostic_data.get('has_diagnostic_data'):
                causes = list(diagnostic_data.get('all_causes', [])) # Make copy
                
                # REPAIR TRUNCATED TEXT:
                # refined_dataset sometimes truncates causes to 100 chars in 'all_causes'.
                # We try to find the full text in 'narr_cause' and use that instead.
                narr_cause = diagnostic_data.get('narr_cause')
                if narr_cause and len(narr_cause) > 100:
                    # Filter out the truncated version (which matches the start of full text)
                    # The truncation is usually exact 100 chars
                    causes = [c for c in causes if not (len(c) == 100 and narr_cause.startswith(c))]
                    # Add the full version if it wasn't already there (it might not be in all_causes)
                    if narr_cause not in causes:
                        causes.append(narr_cause)
                
                all_causes_in_cluster.extend(
                    c.strip().lower() for c in causes if c and c.strip()
                )
        
        # Count frequency of each cause
        if all_causes_in_cluster:
            cause_counts = Counter(all_causes_in_cluster)
            total_cause_occurrences = len(all_causes_in_cluster)
            
            # Calculate P(Cause|Cluster) = Count(Cause) / Total_causes_in_cluster
            cause_probabilities = {
                cause: count / total_cause_occurrences
                for cause, count in cause_counts.items()
            }
        else:
            cause_probabilities = {}
        
        avg_similarity = total_similarity / len(incidents) if incidents else 0
        
        cluster_analysis[cluster_type] = {
            'causes': cause_probabilities,
            'total_incidents': len(incidents),
            'avg_similarity': avg_similarity,
            'incident_ids': [inc['ev_id'] for inc in incidents]
        }
    
    print(f"✅ Calculated cause probabilities for {len(cluster_analysis)} clusters")
    
    return cluster_analysis


def calculate_chain_rule_diagnosis(clusters, cluster_analysis):
    """
    Apply the Law of Total Probability to compute P(Cause | Query).

    Implements: P(C|Q) = Σ_K P(C|K) · P(K|Q)

    Where:
    - P(C|K) comes from calculate_cause_probabilities_per_cluster()
    - P(K|Q) = (avg_sim_K × n_K) / Σ_{K' active} (avg_sim_{K'} × n_{K'})

    The partition {K} is restricted to clusters that have at least one recorded
    cause. Clusters whose member incidents have no recorded causes contribute
    zero to P(C|Q) and would otherwise dilute the denominator, causing Σ P(C|Q)
    to fall below 1. (Function name is preserved for backward compatibility;
    the math here is LTP, not the chain rule P(A,B) = P(A|B)·P(B).)

    Args:
        clusters: Dictionary from cluster_incidents_by_type()
        cluster_analysis: Dictionary from calculate_cause_probabilities_per_cluster()

    Returns:
        Dictionary with final cause probabilities and evidence
    """
    print(f"\n⚗️  Applying law of total probability: P(Cause|Query) = Σ P(Cause|Cluster) × P(Cluster|Query)")

    from collections import defaultdict

    # Restrict the partition {K} to clusters with at least one recorded cause.
    active_analysis = {
        k: v for k, v in cluster_analysis.items()
        if (v.get('causes') or {})
    }
    skipped_clusters = [k for k in cluster_analysis if k not in active_analysis]
    if skipped_clusters:
        print(
            f"   Skipped {len(skipped_clusters)} cluster(s) with no recorded "
            f"causes: {skipped_clusters}"
        )

    # Aggregate probabilities across all clusters
    cause_aggregate = defaultdict(lambda: {'probability': 0, 'clusters': [], 'incidents': []})

    # 1. Calculate the Denominator: Total Similarity Mass across ACTIVE clusters
    total_similarity_all_clusters = sum(
        analysis['avg_similarity'] * analysis['total_incidents']
        for analysis in active_analysis.values()
    )

    print(f"   Total Similarity Mass (Denominator): {total_similarity_all_clusters:.4f}")

    # Set P(K|Q) = 0 for any skipped cluster so downstream consumers see consistent state.
    for k in skipped_clusters:
        cluster_analysis[k]['p_cluster_query'] = 0.0

    for cluster_type, analysis in active_analysis.items():
        # 2. Calculate P(Cluster|Query) - The "Slice" for this cluster
        # Weight = Average Similarity * Count
        cluster_total_weight = analysis['avg_similarity'] * analysis['total_incidents']

        # P(Cluster|Query) = Cluster Weight / Total Weight
        p_cluster_query = cluster_total_weight / total_similarity_all_clusters if total_similarity_all_clusters > 0 else 0

        # Store for display/debugging
        analysis['p_cluster_query'] = p_cluster_query

        # 3. Apply LTP for each cause in this cluster
        for cause, p_cause_given_cluster in analysis['causes'].items():
            # Chain Rule: P(Cause|Query) += P(Cause|Cluster) × P(Cluster|Query)
            contribution = p_cause_given_cluster * p_cluster_query
            
            cause_aggregate[cause]['probability'] += contribution
            
            # Store breakdown for the UI (so we can explain the math to the user)
            cause_aggregate[cause]['clusters'].append({
                'type': cluster_type,
                'p_cluster_query': p_cluster_query,       # The Cluster Probability
                'p_cause_cluster': p_cause_given_cluster, # The Conditional Cause Probability
                'contribution': contribution              # The product
            })
            cause_aggregate[cause]['incidents'].extend(analysis['incident_ids'])
    
    # Convert to sorted list
    final_causes = []
    for cause, data in cause_aggregate.items():
        # Sort the contributing clusters by their contribution (descending)
        sorted_clusters = sorted(data['clusters'], key=lambda x: x['contribution'], reverse=True)
        
        final_causes.append({
            'cause': cause,
            'probability': data['probability'],
            'num_clusters': len(data['clusters']),
            'num_incidents': len(set(data['incidents'])),
            'supporting_incidents': list(set(data['incidents']))[:5],
            'cluster_breakdown': sorted_clusters  # Keep all clusters for detailed breakdown
        })
    
    # Sort by probability
    final_causes.sort(key=lambda x: x['probability'], reverse=True)
    
    print(f"✅ Calculated final probabilities for {len(final_causes)} causes")
    
    return {
        'weighted_causes': final_causes,
        'total_incidents_analyzed': sum(a['total_incidents'] for a in cluster_analysis.values()),
        'total_clusters': len(clusters),
        'methodology': "Conditional probability with chain rule: P(Cause|Query) = Σ P(Cause|Cluster) × P(Cluster|Query)"
    }


import time


# NTSB finding-taxonomy top-level categories (the decision-useful aggregation level).
NTSB_TOP_CATEGORIES = (
    "personnel issues",
    "aircraft",
    "environmental issues",
    "organizational issues",
    "not determined",
)


def _cause_to_category(cause: str, level: int = 1):
    """Map one free-text cause to its NTSB finding category (top-level or level-2).

    NTSB finding descriptions are hyphen-delimited taxonomy paths, e.g.
    "Personnel issues-Action/decision-Info processing/decision-...". We reduce a
    cause string to its first ``level`` taxonomy segments. Narrative-prose causes
    that do not start with a known top-level category return None (they are not
    forced into a bucket). This is the granularity fix for the "3% per unique
    sentence" problem: many distinct sentences collapse onto one decision-level
    category so their probability mass adds up.
    """
    c = " ".join(str(cause or "").lower().split())
    segs = [s.strip() for s in c.split("-") if s.strip()]
    if not segs:
        return None
    seg0 = segs[0]
    if not any(seg0 == t or seg0.startswith(t) for t in NTSB_TOP_CATEGORIES):
        return None
    return "-".join(segs[:level])


def aggregate_causes_to_categories(weighted_causes, level: int = 1):
    """Roll per-cause P(Cause|Query) up to decision-level NTSB categories.

    Turns the engine's fragmented per-sentence diagnosis (each cause ~few %) into
    an actionable category distribution (top category often 40-70%). Probabilities
    are summed per category, then renormalized over the categorized mass so the
    returned distribution sums to 1.

    Args:
        weighted_causes: the ``weighted_causes`` list from
            calculate_chain_rule_diagnosis (each item has 'cause' and 'probability').
        level: 1 = top-level category (e.g. "personnel issues"),
               2 = sub-category (e.g. "personnel issues-action/decision").

    Returns:
        dict with:
          'categories': sorted list of {category, probability, cumulative}
          'top_category', 'top_probability'
          'categorized_mass': fraction of original probability that mapped to a
                              category (the rest was narrative prose, left out)
    """
    from collections import defaultdict

    agg = defaultdict(float)
    categorized = 0.0
    total = 0.0
    for wc in weighted_causes or []:
        p = float(wc.get("probability", 0.0))
        total += p
        cat = _cause_to_category(wc.get("cause", ""), level)
        if cat is None:
            continue
        agg[cat] += p
        categorized += p

    mass = sum(agg.values())
    rows = []
    cum = 0.0
    for cat, p in sorted(agg.items(), key=lambda x: x[1], reverse=True):
        prob = p / mass if mass > 0 else 0.0
        cum += prob
        rows.append({"category": cat, "probability": prob, "cumulative": cum})

    return {
        "categories": rows,
        "top_category": rows[0]["category"] if rows else None,
        "top_probability": rows[0]["probability"] if rows else 0.0,
        "categorized_mass": (categorized / total) if total > 0 else 0.0,
        "level": level,
    }

def diagnose_with_conditional_probabilities(
    query, top_n=10, top_n_incidents=50, score_adjust_fn=None, exclude_ev_ids=None
):
    """
    Diagnosis using conditional probabilities and chain rule.

    score_adjust_fn: optional callable (cosine_score, match_dict) -> float.
    When set, retrieval scores passed into clustering / chain rule are replaced
    by this value (e.g. structural reweighting). Non-incident rows are unchanged.

    exclude_ev_ids: optional iterable of ev_id strings to exclude from retrieval
    (exclude-self / no-leakage guard). Pass the query incident's own ev_id during
    evaluation so the engine cannot retrieve the case it is diagnosing.
    """
    start_t = time.time()
    print(f"\n[{time.time()-start_t:.2f}s] 🎓 START Diagnosis", flush=True)
    
    # Step 1: Get similar incidents
    print(f"[{time.time()-start_t:.2f}s]    DEBUG: Calling OpenAI Embedding...", flush=True)
    query_embedding = get_embedding(query)
    print(f"[{time.time()-start_t:.2f}s]    DEBUG: Embedding done. Finding matches...", flush=True)
    
    top_scores, top_matches = find_top_matches(query_embedding, exclude_ev_ids=exclude_ev_ids)
    if score_adjust_fn is not None:
        top_scores = [
            float(score_adjust_fn(s, m)) for s, m in zip(top_scores, top_matches)
        ]
    print(f"[{time.time()-start_t:.2f}s]    DEBUG: Matches found.", flush=True)
    
    # Step 2: Cluster incidents by type
    print(f"[{time.time()-start_t:.2f}s]    DEBUG: Clustering incidents...", flush=True)
    clusters = cluster_incidents_by_type(top_scores, top_matches, top_n_incidents)
    print(f"[{time.time()-start_t:.2f}s]    DEBUG: Clustering done. {len(clusters)} clusters.", flush=True)
    
    if not clusters:
        return {
            'error': 'No incidents could be clustered',
            'weighted_causes': [],
            'total_incidents_analyzed': 0
        }
    
    # Step 3: Calculate P(Cause|Cluster) for each cluster
    print(f"[{time.time()-start_t:.2f}s]    DEBUG: Calculating probabilities...", flush=True)
    cluster_analysis = calculate_cause_probabilities_per_cluster(clusters)
    
    # Step 4: Apply chain rule to get final P(Cause|Query)
    diagnosis_results = calculate_chain_rule_diagnosis(clusters, cluster_analysis)
    
    # Add cluster details to results
    diagnosis_results['clusters'] = cluster_analysis
    
    print(f"[{time.time()-start_t:.2f}s] ✅ DONE Diagnosis", flush=True)
    return diagnosis_results


def _normalize_occurrence_text(text) -> str:
    if text is None:
        return ""
    return " ".join(str(text).lower().split())


def _sequence_prefix_matches(subsequent: list, prefix: list) -> bool:
    """True if subsequent[:len(prefix)] matches prefix (text-normalized)."""
    if len(subsequent) < len(prefix):
        return False
    for a, b in zip(subsequent, prefix):
        if _normalize_occurrence_text(a) != _normalize_occurrence_text(b):
            return False
    return True


def _collect_weighted_sequence_contributions(
    query_embedding,
    top_scores,
    top_matches,
    top_n_incidents,
    min_event_match_score=None,
    score_adjust_fn=None,
):
    """
    For each similar incident, align the user query to one step in sequence_of_events;
    return weighted rows for downstream empirical transitions.
    Weight = retrieval similarity (incident match score), not cluster probabilities.

    Alignment: pick the step whose Occurrence_Description has highest cosine similarity
    to the query (no minimum score by default). Single-event sequences (or a match on the
    last step) still appear in matched_incidents with has_downstream=False; only rows with
    a non-empty tail are added to contributions for transition weights.
    Optional min_event_match_score (float) enforces a cosine floor.
    score_adjust_fn: optional (cosine_score, match_dict) -> float for incident weights.
    """
    event_embeddings_cache = {}
    client_instance = get_client()
    contributions = []
    matched_incidents = []

    def get_cached_embedding(text):
        if text not in event_embeddings_cache:
            event_embeddings_cache[text] = get_embedding(text)
        return event_embeddings_cache[text]

    for score, match in zip(top_scores[:top_n_incidents], top_matches[:top_n_incidents]):
        if match.get("source") != "incident":
            continue
        ev_id = match.get("ev_id")
        if not ev_id or ev_id not in refined_dataset:
            continue
        adj_score = float(score_adjust_fn(score, match)) if score_adjust_fn else float(score)
        incident = refined_dataset[ev_id]
        sequence = incident.get("sequence_of_events", [])
        if not sequence:
            continue
        descriptions = [e.get("Occurrence_Description", "") for e in sequence]
        if not descriptions or not any(str(d).strip() for d in descriptions):
            continue

        unique_descs = [d for d in descriptions if d and d not in event_embeddings_cache]
        if unique_descs:
            try:
                resp = client_instance.embeddings.create(input=unique_descs, model=EMBEDDING_MODEL)
                for d, data in zip(unique_descs, resp.data):
                    event_embeddings_cache[d] = data.embedding
            except Exception as e:
                print(f"Error embedding events: {e}")
                continue

        best_match_idx = -1
        best_match_score = -1.0
        for idx, desc in enumerate(descriptions):
            if not desc:
                continue
            emb = get_cached_embedding(desc)
            sim = float(np.dot(query_embedding, emb))
            if sim > best_match_score:
                best_match_score = sim
                best_match_idx = idx

        if best_match_idx < 0:
            continue
        if min_event_match_score is not None and best_match_score <= min_event_match_score:
            continue
        subsequent_events = descriptions[best_match_idx + 1 :]
        has_downstream = len(subsequent_events) > 0

        matched_incidents.append(
            {
                "ev_id": ev_id,
                "matched_event": descriptions[best_match_idx],
                "subsequent_events": subsequent_events,
                "incident_similarity": adj_score,
                "event_match_score": best_match_score,
                "has_downstream": has_downstream,
            }
        )

        if not subsequent_events:
            continue

        w = max(adj_score, 0.0)
        contributions.append(
            {
                "ev_id": ev_id,
                "weight": w,
                "matched_event": descriptions[best_match_idx],
                "subsequent_events": subsequent_events,
                "event_match_score": best_match_score,
                "incident_similarity": adj_score,
            }
        )

    return contributions, matched_incidents


def _weighted_next_event_distribution(contributions, prefix_events):
    """
    P_hat(next event | prefix_events, query) via query-weighted counts over incidents.
    prefix_events: list of raw occurrence strings already observed after the anchor; length k.
    Next token is subsequent_events[k].
    """
    from collections import defaultdict

    k = len(prefix_events)
    mass_by_event = defaultdict(float)
    evidence_by_event = defaultdict(list)
    total_w = 0.0

    for c in contributions:
        sub = c["subsequent_events"]
        if len(sub) <= k:
            continue
        if k > 0 and not _sequence_prefix_matches(sub[:k], prefix_events):
            continue
        nxt = sub[k]
        w = c["weight"]
        mass_by_event[nxt] += w
        total_w += w
        evidence_by_event[nxt].append((c["ev_id"], w))

    return mass_by_event, total_w, evidence_by_event


def _distribution_to_future_events(mass_by_event, total_w, evidence_by_event, top_evidence=5):
    """Sort by probability; attach evidence ev_ids (highest-weight incidents first)."""
    if total_w <= 0:
        return []
    rows = []
    for event, mass in mass_by_event.items():
        prob = mass / total_w
        # One row per ev_id (max weight) so evidence lists don't repeat the same case
        by_eid: dict = {}
        for eid, w in evidence_by_event.get(event, []):
            by_eid[eid] = max(by_eid.get(eid, 0.0), float(w))
        pairs = sorted(by_eid.items(), key=lambda x: -x[1])
        ev_ids = [p[0] for p in pairs[:top_evidence]]
        rows.append(
            {
                "event": event,
                "probability": prob,
                "weighted_mass": mass,
                "likelihood_score": prob * 100.0,
                "evidence": ev_ids,
            }
        )
    rows.sort(key=lambda x: x["probability"], reverse=True)
    return rows


def _defining_event_index(sequence):
    """Return index of the NTSB defining event (Defining_ev == 1), or None."""
    for i, step in enumerate(sequence or []):
        flag = step.get("Defining_ev")
        if flag in (1, "1", True):
            return i
    return None


def _immediate_next_after_defining(incident):
    """Event description immediately after the defining event, or None."""
    sequence = incident.get("sequence_of_events") or []
    idx = _defining_event_index(sequence)
    if idx is None or idx + 1 >= len(sequence):
        return None
    text = str(sequence[idx + 1].get("Occurrence_Description") or "").strip()
    return text or None


def _defining_event_description(incident):
    sequence = incident.get("sequence_of_events") or []
    idx = _defining_event_index(sequence)
    if idx is None:
        return None
    text = str(sequence[idx].get("Occurrence_Description") or "").strip()
    return text or None


def calculate_next_event_probabilities_per_cluster(clusters):
    """
    P(Next Event | Cluster): frequency of the immediate post-defining event per cluster.
    """
    from collections import Counter

    print("\n🧮 Calculating P(Next Event|Cluster) for each cluster...")
    cluster_analysis = {}

    for cluster_type, incidents in clusters.items():
        next_events = []
        total_similarity = 0.0
        incident_ids = []

        for incident in incidents:
            match = incident["match"]
            ev_id = incident["ev_id"]
            total_similarity += incident["score"]
            incident_ids.append(ev_id)
            if ev_id not in refined_dataset:
                continue
            nxt = _immediate_next_after_defining(refined_dataset[ev_id])
            if nxt:
                next_events.append(nxt)

        if next_events:
            counts = Counter(next_events)
            total = len(next_events)
            event_probs = {ev: cnt / total for ev, cnt in counts.items()}
        else:
            event_probs = {}

        avg_similarity = total_similarity / len(incidents) if incidents else 0.0
        cluster_analysis[cluster_type] = {
            "events": event_probs,
            "total_incidents": len(incidents),
            "sequences_with_next": len(next_events),
            "avg_similarity": avg_similarity,
            "incident_ids": incident_ids,
        }

    print(f"✅ Calculated next-event probabilities for {len(cluster_analysis)} clusters")
    return cluster_analysis


def calculate_ltp_prognosis(clusters, cluster_analysis):
    """
    Law of total probability for prognosis:
    P(Next Event | Query) = Σ_K P(Next Event | K) · P(K | Query)
    """
    from collections import defaultdict

    print(
        "\n⚗️  Applying law of total probability: "
        "P(Next Event|Query) = Σ P(Next Event|Cluster) × P(Cluster|Query)"
    )

    active_analysis = {
        k: v for k, v in cluster_analysis.items() if (v.get("events") or {})
    }
    skipped_clusters = [k for k in cluster_analysis if k not in active_analysis]
    if skipped_clusters:
        print(
            f"   Skipped {len(skipped_clusters)} cluster(s) with no post-defining next events: "
            f"{skipped_clusters}"
        )

    event_aggregate = defaultdict(lambda: {"probability": 0.0, "clusters": [], "incidents": []})
    total_weight = sum(
        a["avg_similarity"] * a["total_incidents"] for a in active_analysis.values()
    )
    print(f"   Total Similarity Mass (Denominator): {total_weight:.4f}")

    for k in skipped_clusters:
        cluster_analysis[k]["p_cluster_query"] = 0.0

    for cluster_type, analysis in active_analysis.items():
        cluster_weight = analysis["avg_similarity"] * analysis["total_incidents"]
        p_k_q = cluster_weight / total_weight if total_weight > 0 else 0.0
        analysis["p_cluster_query"] = p_k_q

        for event, p_next_k in analysis["events"].items():
            contribution = p_next_k * p_k_q
            event_aggregate[event]["probability"] += contribution
            event_aggregate[event]["clusters"].append(
                {
                    "type": cluster_type,
                    "p_cluster_query": p_k_q,
                    "p_next_cluster": p_next_k,
                    "contribution": contribution,
                }
            )
            event_aggregate[event]["incidents"].extend(analysis["incident_ids"])

    final_events = []
    for event, data in event_aggregate.items():
        final_events.append(
            {
                "event": event,
                "probability": data["probability"],
                "num_clusters": len(data["clusters"]),
                "cluster_breakdown": sorted(
                    data["clusters"], key=lambda x: x["contribution"], reverse=True
                ),
            }
        )
    final_events.sort(key=lambda x: x["probability"], reverse=True)
    print(f"✅ Calculated final probabilities for {len(final_events)} next events")
    return {
        "weighted_events": final_events,
        "skipped_clusters": skipped_clusters,
        "total_weight": total_weight,
        "methodology": (
            "Law of total probability: P(Next|Query) = Σ P(Next|Cluster) × P(Cluster|Query)"
        ),
    }


def _transition_probabilities(top_scores, top_matches, top_n_incidents):
    """P(event_b | event_a) from consecutive pairs in retrieved incident timelines."""
    from collections import Counter, defaultdict

    pair_counts = Counter()
    from_counts = Counter()
    for score, match in zip(top_scores[:top_n_incidents], top_matches[:top_n_incidents]):
        if match.get("source") != "incident":
            continue
        ev_id = match.get("ev_id")
        if not ev_id or ev_id not in refined_dataset:
            continue
        sequence = refined_dataset[ev_id].get("sequence_of_events") or []
        descs = [
            str(e.get("Occurrence_Description") or "").strip().lower()
            for e in sequence
            if str(e.get("Occurrence_Description") or "").strip()
        ]
        for i in range(len(descs) - 1):
            pair_counts[(descs[i], descs[i + 1])] += 1
            from_counts[descs[i]] += 1

    trans = defaultdict(dict)
    for (a, b), cnt in pair_counts.items():
        trans[a][b] = cnt / from_counts[a] if from_counts[a] else 0.0
    return trans


def _ltp_multistep_from_transitions(step1_events, transition_probs, max_chain_steps):
    """Steps 2+ via P(Event_t|Q) = Σ_e P(Event_t|e) P(e|Q) (paper Eq. multi_step)."""
    from collections import defaultdict

    if max_chain_steps is None or int(max_chain_steps) <= 1:
        return []

    multi = []
    prev_dist = {e["event"]: e["probability"] for e in step1_events}
    conditioned_on = []

    for step_idx in range(2, int(max_chain_steps) + 1):
        next_dist = defaultdict(float)
        for prev_event, p_prev in prev_dist.items():
            for nxt, p_nxt_given_prev in transition_probs.get(prev_event, {}).items():
                next_dist[nxt] += p_nxt_given_prev * p_prev
        if not next_dist:
            break
        total = sum(next_dist.values())
        dist = [
            {
                "event": ev,
                "probability": mass / total if total > 0 else 0.0,
                "weighted_mass": mass,
                "likelihood_score": (mass / total * 100.0) if total > 0 else 0.0,
                "evidence": [],
            }
            for ev, mass in sorted(next_dist.items(), key=lambda x: -x[1])
        ]
        multi.append(
            {
                "step": step_idx,
                "conditioned_on": list(conditioned_on),
                "total_weight": total,
                "distribution": dist,
            }
        )
        prev_dist = {row["event"]: row["probability"] for row in dist}
        if dist:
            conditioned_on.append(dist[0]["event"])
    return multi


def predict_future_events_lotp(
    query,
    top_n_incidents=50,
    max_chain_steps=3,
    score_adjust_fn=None,
):
    """
    Prognosis via law of total probability over failure-theme clusters (paper Section 7.2).

    1. Retrieve top-N similar incidents.
    2. Cluster by failure theme (same as diagnosis).
    3. P(Next Event | Cluster) from post-defining-event frequencies.
    4. P(Next Event | Query) = Σ_K P(Next Event | K) · P(K | Query).
    5. Optional multi-step chain via transition probabilities.
    """
    print(f"\n🔮 Predicting future events (LTP over clusters) for: '{query}'")

    query_embedding = get_embedding(query)
    top_scores, top_matches = find_top_matches(query_embedding)
    if score_adjust_fn is not None:
        top_scores = [
            float(score_adjust_fn(s, m)) for s, m in zip(top_scores, top_matches)
        ]

    clusters = cluster_incidents_by_type(top_scores, top_matches, top_n_incidents)
    if not clusters:
        return {
            "error": "No incidents could be clustered",
            "future_events": [],
            "multi_step": [],
            "aligned_incidents": [],
            "methodology": "Law of total probability over clusters",
        }

    cluster_analysis = calculate_next_event_probabilities_per_cluster(clusters)
    prognosis = calculate_ltp_prognosis(clusters, cluster_analysis)
    future_events = [
        {
            "event": row["event"],
            "probability": row["probability"],
            "weighted_mass": row["probability"],
            "likelihood_score": row["probability"] * 100.0,
            "evidence": [],
        }
        for row in prognosis["weighted_events"]
    ]

    transitions = _transition_probabilities(top_scores, top_matches, top_n_incidents)
    multi_step = []
    if future_events:
        multi_step.append(
            {
                "step": 1,
                "conditioned_on": [],
                "total_weight": sum(e["probability"] for e in future_events),
                "distribution": future_events,
            }
        )
        multi_step.extend(
            _ltp_multistep_from_transitions(future_events, transitions, max_chain_steps)
        )

    defining_rows = []
    for score, match in zip(top_scores[:top_n_incidents], top_matches[:top_n_incidents]):
        if match.get("source") != "incident":
            continue
        ev_id = match.get("ev_id")
        if not ev_id or ev_id not in refined_dataset:
            continue
        inc = refined_dataset[ev_id]
        defining_rows.append(
            {
                "ev_id": ev_id,
                "incident_similarity": float(score),
                "defining_event": _defining_event_description(inc),
                "next_event": _immediate_next_after_defining(inc),
                "has_downstream": _immediate_next_after_defining(inc) is not None,
            }
        )

    sequences_analyzed = sum(
        a.get("sequences_with_next", 0) for a in cluster_analysis.values()
    )
    print(
        f"✅ Prognosis LTP: {sequences_analyzed} sequences with post-defining next events, "
        f"{len(future_events)} distinct next-event labels"
    )

    return {
        "future_events": future_events,
        "multi_step": multi_step,
        "sequences_analyzed": sequences_analyzed,
        "aligned_incidents": defining_rows,
        "cluster_analysis": cluster_analysis,
        "clusters": clusters,
        "skipped_clusters": prognosis.get("skipped_clusters") or [],
        "ltp_sum": sum(e["probability"] for e in future_events),
        "methodology": prognosis["methodology"],
    }


def predict_future_events_aligned(
    query,
    top_n_incidents=50,
    max_chain_steps=None,
    min_event_match_score=None,
    score_adjust_fn=None,
):
    """
    Alternate prognosis: query-aligned timeline steps + similarity weights (no cluster LTP).

    Kept for experiments; the paper worked examples use predict_future_events_lotp().
    """
    print(f"\n🔮 Predicting future events (query-weighted transitions) for: '{query}'")

    query_embedding = get_embedding(query)
    top_scores, top_matches = find_top_matches(query_embedding)

    contributions, matched_incidents = _collect_weighted_sequence_contributions(
        query_embedding,
        top_scores,
        top_matches,
        top_n_incidents,
        min_event_match_score=min_event_match_score,
        score_adjust_fn=score_adjust_fn,
    )

    if not matched_incidents:
        return {
            "error": "No matching event sequences found",
            "future_events": [],
            "multi_step": [],
            "aligned_incidents": [],
            "methodology": "Query-weighted empirical transitions",
        }

    aligned_incidents = list(matched_incidents)
    n_downstream = len(contributions)
    n_terminal = sum(1 for m in aligned_incidents if not m.get("has_downstream", True))

    if not contributions:
        print(
            f"✅ Prognosis: {len(aligned_incidents)} aligned sequence(s), "
            f"0 with downstream steps ({n_terminal} terminal / single-step only)"
        )
        return {
            "future_events": [],
            "multi_step": [],
            "sequences_analyzed": 0,
            "downstream_incident_count": 0,
            "total_incident_weight": 0.0,
            "aligned_incidents": aligned_incidents,
            "matched_incidents": aligned_incidents[:20],
            "terminal_only_count": n_terminal,
            "methodology": (
                "Query-weighted empirical transitions: similar incidents align to a sequence step, "
                "but none have additional events after that step (single-event sequence or match on last event)."
            ),
        }

    total_incident_weight = sum(c["weight"] for c in contributions)
    sequences_analyzed = len(contributions)

    # How many "next event" positions exist at least once in the weighted set
    max_depth = max(len(c["subsequent_events"]) for c in contributions)
    if max_chain_steps is not None:
        max_depth = min(max_depth, int(max_chain_steps))

    multi_step = []
    greedy_prefix = []

    for step_idx in range(max_depth):
        masses, step_total_w, ev_map = _weighted_next_event_distribution(contributions, greedy_prefix)
        if step_total_w <= 0:
            break
        dist = _distribution_to_future_events(masses, step_total_w, ev_map)
        multi_step.append(
            {
                "step": step_idx + 1,
                "conditioned_on": list(greedy_prefix),
                "total_weight": step_total_w,
                "distribution": dist,
            }
        )
        if not dist:
            break
        greedy_prefix.append(dist[0]["event"])

    future_events = multi_step[0]["distribution"] if multi_step else []

    print(
        f"✅ Prognosis: {sequences_analyzed} with downstream data, "
        f"{len(aligned_incidents)} aligned total, {len(multi_step)} prognosis step(s)"
    )

    return {
        "future_events": future_events,
        "sequences_analyzed": sequences_analyzed,
        "downstream_incident_count": n_downstream,
        "total_incident_weight": total_incident_weight,
        "aligned_incidents": aligned_incidents,
        "matched_incidents": aligned_incidents[:20],
        "terminal_only_count": n_terminal,
        "multi_step": multi_step,
        "methodology": (
            "Query-weighted empirical transitions: each retrieved incident contributes "
            "its observed next event with weight = cosine similarity to the query incident. "
            "Single-step sequences appear under aligned incidents only (no downstream steps)."
        ),
    }


def predict_future_events(
    query,
    top_n_incidents=50,
    max_chain_steps=3,
    score_adjust_fn=None,
    **kwargs,
):
    """
    Prognosis for paper worked examples: LTP over failure-theme clusters.

    See predict_future_events_lotp() and predict_future_events_aligned() for details.
    """
    kwargs.pop("min_event_match_score", None)
    return predict_future_events_lotp(
        query,
        top_n_incidents=top_n_incidents,
        max_chain_steps=max_chain_steps,
        score_adjust_fn=score_adjust_fn,
    )


# --- Prognosis Mode (Survival Analysis) ---

def cluster_failures_by_mode(top_scores, top_matches, query, top_n_incidents=50):
    """
    Cluster top incidents by FAILURE MODE using LLM.
    Similar to diagnosis clustering but focused on HOW it failed.
    """
    print(f"\n🔄 Clustering top {top_n_incidents} incidents by failure mode...")
    
    from collections import defaultdict
    import concurrent.futures
    
    clusters = defaultdict(list)
    client_instance = get_client()
    
    # Prepare list of incidents to process
    incidents_to_process = []
    
    for score, match in zip(top_scores[:top_n_incidents], top_matches[:top_n_incidents]):
        if match.get('source') != 'incident':
            continue
            
        ev_id = match.get('ev_id')
        if not ev_id or ev_id not in refined_dataset:
            continue
            
        incident_data = refined_dataset[ev_id]
        
        # Must have flight hours data
        afm_hrs = incident_data.get('afm_hrs')
        if not afm_hrs or afm_hrs <= 0:
            continue
            
        narrative = incident_data.get('narr_cause', '') or incident_data.get('narr_accp', '')
        if not narrative:
            continue
            
        narrative_excerpt = narrative[:500] if len(narrative) > 500 else narrative
        
        incidents_to_process.append({
            'score': score,
            'ev_id': ev_id,
            'hours_at_failure': afm_hrs,
            'narrative': narrative_excerpt
        })
        
    # Function to classify a single incident
    def classify_failure_mode(incident_info):
        try:
            prompt = f"""Given a query about: '{query}'
            And this incident narrative: {incident_info['narrative']}
            
            Classify the FAILURE MODE or MECHANISM into one specific category (2-4 words).
            Focus on the physical reason for failure (e.g., "Fatigue cracking", "Thermal stress", "Corrosion").
            
            Category:"""
            
            response = client_instance.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=20
            )
            
            failure_mode = response.choices[0].message.content.strip()
            failure_mode = ' '.join(failure_mode.lower().split())
            
            return failure_mode, incident_info
            
        except Exception as e:
            print(f"⚠️  Error classifying: {e}")
            return 'uncategorized', incident_info

    # Execute in parallel
    print(f"   Processing {len(incidents_to_process)} incidents in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_incident = {executor.submit(classify_failure_mode, inc): inc for inc in incidents_to_process}
        
        for future in concurrent.futures.as_completed(future_to_incident):
            failure_mode, incident_info = future.result()
            clusters[failure_mode].append(incident_info)
    
    return dict(clusters)


def calculate_cluster_survival_distributions(clusters):
    """
    Calculate survival statistics for each cluster.
    """
    cluster_distributions = {}
    
    for cluster_type, incidents in clusters.items():
        if not incidents:
            continue
            
        failure_times = [inc['hours_at_failure'] for inc in incidents]
        total_similarity = sum(inc['score'] for inc in incidents)
        avg_similarity = total_similarity / len(incidents)
        
        cluster_distributions[cluster_type] = {
            'mean_ttf': np.mean(failure_times),
            'std_ttf': np.std(failure_times) if len(failure_times) > 1 else 0,
            'percentiles': {
                '10%': np.percentile(failure_times, 10),
                '50%': np.percentile(failure_times, 50),
                '90%': np.percentile(failure_times, 90)
            },
            'n_incidents': len(incidents),
            'avg_similarity': avg_similarity,
            'incidents': incidents
        }
    
    return cluster_distributions


def prognosis_with_chain_rule(query, current_hours=0, top_n_incidents=50):
    """
    Prognosis using Chain Rule:
    P(Failure at t|Query) = Σ P(Failure at t|Cluster) × P(Cluster|Query)
    """
    print(f"\n🔮 Running Chain Rule Prognosis for: '{query}'")
    
    # 1. Get similar incidents
    query_embedding = get_embedding(query)
    top_scores, top_matches = find_top_matches(query_embedding)
    
    # 2. Cluster by failure mode
    clusters = cluster_failures_by_mode(top_scores, top_matches, query, top_n_incidents)
    
    if not clusters:
        return {'error': 'No suitable incidents found for clustering'}
        
    # 3. Calculate distributions per cluster
    cluster_dists = calculate_cluster_survival_distributions(clusters)
    
    # 4. Apply Chain Rule (Weighted Aggregation)
    total_weight_score = sum(d['avg_similarity'] * d['n_incidents'] for d in cluster_dists.values())
    
    weighted_mean = 0
    weighted_percentiles = {'10%': 0, '50%': 0, '90%': 0}
    
    # Prepare for risk curve calculation
    time_points = [100, 500, 1000, 2000, 5000]
    weighted_risk_probs = {tp: 0.0 for tp in time_points}
    
    cluster_breakdown = []
    
    for cluster_type, dist in cluster_dists.items():
        # P(Cluster|Query) weight
        weight = (dist['avg_similarity'] * dist['n_incidents']) / total_weight_score if total_weight_score > 0 else 0
        
        # Contribution to mean
        weighted_mean += weight * dist['mean_ttf']
        
        # Contribution to percentiles
        for k in weighted_percentiles:
            weighted_percentiles[k] += weight * dist['percentiles'][k]
            
        # Contribution to risk curve
        # P(Failure <= t | Query) += P(Failure <= t | Cluster) * P(Cluster | Query)
        incidents = dist['incidents']
        hours_list = [inc['hours_at_failure'] for inc in incidents]
        if hours_list:
            for tp in time_points:
                 target_hours = current_hours + tp
                 failures_before = sum(1 for h in hours_list if h <= target_hours)
                 prob_failure_given_cluster = failures_before / len(hours_list)
                 weighted_risk_probs[tp] += weight * prob_failure_given_cluster
            
        cluster_breakdown.append({
            'type': cluster_type,
            'weight': weight,
            'mean_ttf': dist['mean_ttf'],
            'n_incidents': dist['n_incidents']
        })
        
    # Construct final risk_curve list
    risk_curve = []
    for tp in time_points:
        risk_pct = weighted_risk_probs[tp] * 100
        risk_curve.append({
            'additional_hours': tp,
            'total_hours': current_hours + tp,
            'failure_probability': risk_pct,
            'risk_level': 'LOW' if risk_pct < 25 else 'MODERATE' if risk_pct < 50 else 'HIGH' if risk_pct < 75 else 'CRITICAL'
        })
        
    # Sort breakdown by weight
    cluster_breakdown.sort(key=lambda x: x['weight'], reverse=True)
    
    # 5. Calculate remaining time
    remaining_mean = max(0, weighted_mean - current_hours)
    remaining_10th = max(0, weighted_percentiles['10%'] - current_hours)
    
    # Calculate range for evidence
    all_hours = []
    for dist in cluster_dists.values():
         all_hours.extend([inc['hours_at_failure'] for inc in dist['incidents']])
    
    evidence = {
        'earliest_failure': min(all_hours) if all_hours else 0,
        'latest_failure': max(all_hours) if all_hours else 0,
        'sample_incidents': [] # Not used for chain rule display
    }
    
    return {
        'estimated_remaining_hours': remaining_mean,
        'conservative_estimate': remaining_10th,
        'median_estimate': max(0, weighted_percentiles['50%'] - current_hours),
        'average_failure_time': weighted_mean,
        'std_deviation': 0, # Simplified for aggregation
        'percentiles': weighted_percentiles,
        'current_component_hours': current_hours,
        'similar_incidents': sum(d['n_incidents'] for d in cluster_dists.values()),
        'cluster_breakdown': cluster_breakdown,
        'clusters': cluster_dists, # Full details
        'methodology': 'Conditional probability (Chain Rule) on failure modes',
        'risk_curve': risk_curve,
        'evidence': evidence,
        
        # Recommendations (same logic as simple method)
        'recommendations': {
            'inspect_within_hours': max(0, remaining_10th * 0.5),
            'repair_by_hours': max(0, remaining_10th * 0.8),
            'critical_threshold_hours': max(0, weighted_percentiles['50%'] - current_hours),
            'urgency': 'IMMEDIATE' if remaining_10th < 100 else 'HIGH' if remaining_10th < 500 else 'MODERATE'
        }
    }


def predict_time_to_failure(query, current_hours=0):
    """
    Prognosis mode: Predict when a detected issue will become critical.
    
    Uses survival analysis on historical similar incidents to estimate:
    - Average time to failure
    - Probability distribution of failure times
    - Recommended action timeline
    
    Args:
        query: Description of current condition/defect (e.g., "crack in turbine blade")
        current_hours: Current flight hours of the component (default: 0 = assume new)
    
    Returns:
        Dictionary with prognosis results
    """
    print(f"\n🔮 Running Prognosis Analysis for: '{query}'")
    print(f"   Current component hours: {current_hours}")
    
    # Step 1: Find similar historical incidents
    query_embedding = get_embedding(query)
    top_scores, top_matches = find_top_matches(query_embedding)
    
    # Step 2: Extract time-to-failure data from similar incidents
    failure_times = []
    incident_details_list = []
    
    for score, match in zip(top_scores[:50], top_matches[:50]):
        if match.get('source') != 'incident':
            continue
            
        ev_id = match.get('ev_id')
        if not ev_id or ev_id not in refined_dataset:
            continue
            
        incident = refined_dataset[ev_id]
        
        # Skip if incident data is None or invalid
        if not incident:
            continue
            
        afm_hrs = incident.get('afm_hrs')
        
        # Only include incidents with flight hours data
        if afm_hrs and afm_hrs > 0:
            narrative = incident.get('narr_cause', '') or ''  # Handle None
            failure_times.append({
                'ev_id': ev_id,
                'hours_at_failure': afm_hrs,
                'similarity': score,
                'narrative': narrative[:200] if narrative else ''
            })
            incident_details_list.append(incident)
    
    if not failure_times:
        return {
            'error': 'No similar incidents with flight hours data found',
            'recommendation': 'Insufficient historical data for prognosis'
        }
    
    # Step 3: Calculate survival statistics
    import numpy as np
    
    hours_array = np.array([f['hours_at_failure'] for f in failure_times])
    similarities = np.array([f['similarity'] for f in failure_times])
    
    # Weighted statistics (weight by similarity)
    weighted_mean = np.average(hours_array, weights=similarities)
    weighted_std = np.sqrt(np.average((hours_array - weighted_mean)**2, weights=similarities))
    
    # Calculate percentiles
    percentiles = {
        '10%': np.percentile(hours_array, 10),
        '25%': np.percentile(hours_array, 25),
        '50%': np.percentile(hours_array, 50),
        '75%': np.percentile(hours_array, 75),
        '90%': np.percentile(hours_array, 90)
    }
    
    # Step 4: Calculate remaining time estimates
    if current_hours > 0:
        remaining_mean = max(0, weighted_mean - current_hours)
        remaining_10th = max(0, percentiles['10%'] - current_hours)
        remaining_median = max(0, percentiles['50%'] - current_hours)
    else:
        remaining_mean = weighted_mean
        remaining_10th = percentiles['10%']
        remaining_median = percentiles['50%']
    
    # Step 5: Generate recommendations
    # Conservative approach: recommend action before 10th percentile
    inspect_by = remaining_10th * 0.5  # Inspect at 50% of 10th percentile
    repair_by = remaining_10th * 0.8   # Repair by 80% of 10th percentile
    critical_at = remaining_median     # 50% of similar cases failed by this point
    
    # Calculate risk levels at different time points
    time_points = [100, 500, 1000, 2000, 5000]
    risk_curve = []
    
    for hours in time_points:
        target_hours = current_hours + hours
        # Count how many failures occurred before this point
        failures_before = sum(1 for h in hours_array if h <= target_hours)
        risk_pct = (failures_before / len(hours_array)) * 100
        risk_curve.append({
            'additional_hours': hours,
            'total_hours': target_hours,
            'failure_probability': risk_pct,
            'risk_level': 'LOW' if risk_pct < 25 else 'MODERATE' if risk_pct < 50 else 'HIGH' if risk_pct < 75 else 'CRITICAL'
        })
    
    return {
        'similar_incidents': len(failure_times),
        'current_component_hours': current_hours,
        
        # Statistical estimates
        'average_failure_time': weighted_mean,
        'std_deviation': weighted_std,
        'percentiles': percentiles,
        
        # Remaining time estimates
        'estimated_remaining_hours': remaining_mean,
        'conservative_estimate': remaining_10th,
        'median_estimate': remaining_median,
        
        # Recommendations
        'recommendations': {
            'inspect_within_hours': max(0, inspect_by),
            'repair_by_hours': max(0, repair_by),
            'critical_threshold_hours': max(0, critical_at),
            'urgency': 'IMMEDIATE' if remaining_10th < 100 else 'HIGH' if remaining_10th < 500 else 'MODERATE' if remaining_10th < 2000 else 'LOW'
        },
        
        # Risk timeline
        'risk_curve': risk_curve,
        
        # Supporting evidence
        'evidence': {
            'earliest_failure': min(hours_array),
            'latest_failure': max(hours_array),
            'sample_incidents': failure_times[:5]  # Top 5 most similar
        },
        
        'methodology': 'Survival analysis based on historical incident data'
    }


def prognosis_with_details(query, current_hours=0):
    """
    High-level prognosis interface that returns both predictions and incident details.
    
    Args:
        query: Description of condition/defect
        current_hours: Current flight hours
        
    Returns:
        Tuple of (prognosis_results, incident_details)
    """
    prognosis_results = predict_time_to_failure(query, current_hours)
    
    # Get incident details for evidence
    if 'evidence' in prognosis_results and 'sample_incidents' in prognosis_results['evidence']:
        ev_ids = [inc['ev_id'] for inc in prognosis_results['evidence']['sample_incidents']]
        incident_details = get_full_incident_details(ev_ids)
    else:
        incident_details = {}
    
    return prognosis_results, incident_details


# --- Main Logic Flow ---
def diagnose_incident_and_get_details(user_query):
    """
    The main function to handle the diagnosis process.
    This version returns the full details, scores, and root cause diagnosis for use in a UI.
    
    Returns:
        Tuple of (incident_scores, incident_details, diagnosis_results)
    """
    print(f"\n🔍 Analyzing query: '{user_query}'")
    
    # 1. Get embedding for the user's query
    query_embedding = get_embedding(user_query)
    
    # 2. Find all matches from the knowledge base, sorted by similarity
    top_scores, top_matches = find_top_matches(query_embedding)
    
    if not top_matches:
        return None, "Could not find any relevant matches in the knowledge base.", None
        
    # 3. Collect the highest score for each unique incident
    incident_scores = {}
    unique_ids_in_order = []
    for score, match in zip(top_scores, top_matches):
        if 'ev_id' in match:
            ev_id = match['ev_id']
            if ev_id not in incident_scores:
                incident_scores[ev_id] = score
                unique_ids_in_order.append(ev_id)

    if not unique_ids_in_order:
        # This can happen if matches are only from the dictionary
        if top_matches[0]['source'] == 'dictionary':
            event_text = top_matches[0].get('text', 'Unknown Event')
            return None, f"Your query is conceptually similar to the dictionary event: '{event_text}'. However, no specific historical incidents were found.", None
        return None, "Could not find any incidents related to the query.", None

    print(f"✅ Found {len(unique_ids_in_order)} unique incidents. Fetching full details...")
    
    # 4. Get the full details for all identified incidents
    incident_details = get_full_incident_details(unique_ids_in_order)
    
    # 5. Calculate similarity-weighted diagnosis
    print(f"🧠 Calculating similarity-weighted root cause diagnosis...")
    diagnosis_results = calculate_similarity_weighted_diagnosis(top_scores, top_matches)
    
    # 6. Return the scores, detailed data, and diagnosis
    return incident_scores, incident_details, diagnosis_results


def get_full_incident_details(ev_id_list):
    """
    Extracts all relevant details for a given list of event IDs,
    including narratives and the sequence of events.
    """
    details = {}
    for ev_id in ev_id_list:
        if ev_id in refined_dataset:
            details[ev_id] = refined_dataset[ev_id]
    return details


def verify_math_for_one_pair(text_a, text_b):
    """
    A standalone function to transparently verify the cosine similarity calculation
    between two specific pieces of text.
    """
    print("\n" + "="*50)
    print(f"VERIFYING MATH: '{text_a}' vs '{text_b}'")
    print("="*50)

    # 1. Get embeddings for both texts
    embedding_a = np.array(get_embedding(text_a))
    embedding_b = np.array(get_embedding(text_b))
    print(f"Vector for A loaded (shape: {embedding_a.shape})")
    print(f"Vector for B loaded (shape: {embedding_b.shape})")

    # 2. Manual Cosine Similarity Calculation
    # dot_product = sum(a * b)
    dot_product = np.dot(embedding_a, embedding_b)
    # magnitude = sqrt(sum(a^2))
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)
    # similarity = dot_product / (magnitude_a * magnitude_b)
    manual_score = dot_product / (norm_a * norm_b)
    print(f"\nMANUAL CALCULATION:")
    print(f"  - Dot Product: {dot_product:.4f}")
    print(f"  - Vector A Magnitude: {norm_a:.4f}")
    print(f"  - Vector B Magnitude: {norm_b:.4f}")
    print(f"  - Manual Score: {manual_score:.6f}")

    # 3. Scikit-learn's Calculation (for comparison)
    sklearn_score = cosine_similarity([embedding_a], [embedding_b])[0][0]
    print(f"\nSKLEARN'S CALCULATION:")
    print(f"  - Sklearn Score: {sklearn_score:.6f}")

    # 4. Final Comparison
    difference = abs(manual_score - sklearn_score)
    print(f"\nDIFFERENCE: {difference:.10f}")
    if difference < 1e-6:
        print("✅ SUCCESS: The manual calculation matches sklearn's result.")
    else:
        print("❌ WARNING: Discrepancy found between manual and sklearn calculation.")
    print("="*50)


def create_incidents_json_for_query(user_query, output_path=None):
    """
    Generates a structured JSON file of the top 50 incidents for a given query.
    
    Args:
        user_query: The incident query string
        output_path: Optional path for output file. If None, uses OUTPUT_DIR from config.
    """
    if output_path is None:
        output_path = OUTPUT_DIR / 'top_50_incidents.json'
    else:
        output_path = Path(output_path)
    """
    Generates a structured JSON file of the top 50 incidents for a given query.
    
    For each incident, it records the EVID, the findings cause, the narrative cause,
    and uses the cosine similarity score as its probability score.
    """
    print(f"\n🚀 Generating incident report for query: '{user_query}'")
    
    # 1. Get scores and details for the query
    incident_scores, incident_details, _ = diagnose_incident_and_get_details(user_query)
    
    if not incident_scores:
        print(f"No incidents found for query: '{user_query}'")
        return

    # 2. Sort incidents by score and take the top 50
    sorted_incidents = sorted(incident_scores.items(), key=lambda item: item[1], reverse=True)
    top_50_incidents = sorted_incidents[:50]
    
    # 3. Build the structured list
    output_data = []
    for ev_id, score in top_50_incidents:
        details = incident_details.get(ev_id, {})
        incident_info = {
            'EVID': ev_id,
            # Using .get() to avoid errors if keys are missing
            'findings_cause': details.get('findings_cause', 'N/A'),
            'narrative_cause': details.get('narrative_cause', 'N/A'),
            'cosine_similarity_score': score
        }
        output_data.append(incident_info)
        
    # 4. Write to JSON file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if needed
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"✅ Successfully created JSON file with top {len(output_data)} incidents at: {output_path}")


# --- Example Usage ---
if __name__ == "__main__":
    # This block is now for simple terminal testing of the main logic.
    test_query = "engine fire during takeoff"
    scores, details, diagnosis = diagnose_incident_and_get_details(test_query)

    if scores:
        print("\n--- Top 5 Similar Incidents ---")
        sorted_incidents = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        for ev_id, score in sorted_incidents[:5]:
            print(f"  - Incident: {ev_id}, Score: {score:.4f}")
        
        print("\n--- Root Cause Diagnosis ---")
        if diagnosis and diagnosis['weighted_causes']:
            print(f"  Analyzed {diagnosis['total_incidents_analyzed']} incidents")
            print(f"  Top 5 Most Likely Causes:")
            for i, cause_info in enumerate(diagnosis['weighted_causes'][:5], 1):
                prob_pct = cause_info['probability'] * 100
                cause_text = cause_info['cause'][:70]
                print(f"  {i}. {prob_pct:.1f}% - {cause_text}")
                print(f"     (Found in {cause_info['num_incidents']} similar incidents)")
        else:
            print("  No diagnostic data available")
    else:
        print(details) # Print the error/info message
    
    # Generate the JSON file for the test query
    create_incidents_json_for_query(test_query)

    print("------------------------")

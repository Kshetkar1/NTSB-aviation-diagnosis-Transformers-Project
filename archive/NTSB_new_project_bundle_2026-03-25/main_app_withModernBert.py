import json
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from config import (
    get_openai_api_key,
    EMBEDDING_MODEL,
    REFINED_DATA_PATH,
    EMBEDDINGS_PATH,
    EMBEDDINGS_MAP_PATH,
    OUTPUT_DIR
)

# --- Configuration ---
# All configuration is loaded from config.py (which uses environment variables)

# --- Data Loading ---
# Only load data if files exist (for testing purposes)
try:
    print("🧠 Loading knowledge base...")
    # Load the master dataset
    with open(REFINED_DATA_PATH, 'r') as f:
        refined_dataset = json.load(f)

    # Load the embeddings and their map
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(EMBEDDINGS_MAP_PATH, 'r') as f:
        embeddings_map = json.load(f)
    print("✅ Knowledge base loaded successfully!")
    DATA_LOADED = True
except FileNotFoundError as e:
    print(f"⚠️  Data files not found: {e}")
    print("   This is OK if you're just testing imports. Run data processing scripts to generate data.")
    refined_dataset = {}
    embeddings = None
    embeddings_map = []
    DATA_LOADED = False


# --- Core Functions ---
# Initialize client lazily when API key is actually needed
client = None

def get_client():
    """Get OpenAI client, initializing it if needed."""
    global client
    if client is None:
        client = OpenAI(api_key=get_openai_api_key())
    return client

def get_embedding(text):
    """Generates an embedding for a given text using the OpenAI API."""
    text = text.replace("\n", " ")
    response = get_client().embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return response.data[0].embedding

def find_top_matches(query_embedding):
    """Finds the top N most similar embeddings from the knowledge base."""
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
    
    # Get the corresponding scores and mapping info
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
        for cause in all_causes:
            cause_evidence[cause].append((score, ev_id))
    
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
                
                all_causes_in_cluster.extend(causes)
        
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
    Step 3: Apply chain rule to calculate P(Cause|Query).
    
    Implements: P(C|Q) = Σ P(C|Cluster_i) × P(Cluster_i|Q)
    
    Where:
    - P(C|Cluster_i) comes from calculate_cause_probabilities_per_cluster()
    - P(Cluster_i|Q) is the average similarity score of incidents in that cluster
    
    Args:
        clusters: Dictionary from cluster_incidents_by_type()
        cluster_analysis: Dictionary from calculate_cause_probabilities_per_cluster()
    
    Returns:
        Dictionary with final cause probabilities and evidence
    """
    print(f"\n⚗️  Applying chain rule: P(Cause|Query) = P(Cause|Cluster) × P(Cluster|Query)")
    
    from collections import defaultdict
    
    # Aggregate probabilities across all clusters
    cause_aggregate = defaultdict(lambda: {'probability': 0, 'clusters': [], 'incidents': []})
    
    # 1. Calculate the Denominator: Total Similarity Mass across ALL clusters
    # This represents the "Whole Pizza"
    total_similarity_all_clusters = sum(
        analysis['avg_similarity'] * analysis['total_incidents']
        for analysis in cluster_analysis.values()
    )
    
    print(f"   Total Similarity Mass (Denominator): {total_similarity_all_clusters:.4f}")
    
    for cluster_type, analysis in cluster_analysis.items():
        # 2. Calculate P(Cluster|Query) - The "Slice" for this cluster
        # Weight = Average Similarity * Count
        cluster_total_weight = analysis['avg_similarity'] * analysis['total_incidents']
        
        # P(Cluster|Query) = Cluster Weight / Total Weight
        p_cluster_query = cluster_total_weight / total_similarity_all_clusters if total_similarity_all_clusters > 0 else 0
        
        # Store for display/debugging
        analysis['p_cluster_query'] = p_cluster_query
        
        # 3. Apply Chain Rule for each cause in this cluster
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

def diagnose_with_conditional_probabilities(query, top_n=10, top_n_incidents=50):
    """
    Diagnosis using conditional probabilities and chain rule.
    """
    start_t = time.time()
    print(f"\n[{time.time()-start_t:.2f}s] 🎓 START Diagnosis", flush=True)
    
    # Step 1: Get similar incidents
    print(f"[{time.time()-start_t:.2f}s]    DEBUG: Calling OpenAI Embedding...", flush=True)
    query_embedding = get_embedding(query)
    print(f"[{time.time()-start_t:.2f}s]    DEBUG: Embedding done. Finding matches...", flush=True)
    
    top_scores, top_matches = find_top_matches(query_embedding)
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


def predict_future_events(query, top_n_incidents=50):
    """
    Prognosis: Predicts future events based on the sequence of events in similar incidents.
    
    Implements the logic:
    1. Find similar incidents to the query.
    2. In each incident's sequence of events, find the event that best matches the query.
    3. Identify all DOWNSTREAM (subsequent) events.
    4. Calculate the probability of each future event occurring.
    
    Args:
        query: User's description of the current situation (e.g., "engine fire")
        top_n_incidents: Number of incidents to analyze
        
    Returns:
        Dictionary with:
            - 'future_events': List of (event, probability, evidence)
            - 'total_sequences_analyzed': Count
    """
    print(f"\n🔮 Predicting future events for: '{query}'")
    
    # 1. Find similar incidents
    query_embedding = get_embedding(query)
    top_scores, top_matches = find_top_matches(query_embedding)
    
    from collections import Counter, defaultdict
    import numpy as np
    
    next_events = []
    sequences_analyzed = 0
    matched_incidents = []
    
    # Optimization: Cache embeddings for frequent event descriptions to avoid re-embedding
    # In a real production system, these should be pre-computed.
    # For now, we'll embed unique descriptions on the fly.
    event_embeddings_cache = {}
    
    client_instance = get_client()
    
    def get_cached_embedding(text):
        if text not in event_embeddings_cache:
            event_embeddings_cache[text] = get_embedding(text)
        return event_embeddings_cache[text]

    print(f"   Analyzing sequences from top {top_n_incidents} incidents...")
    
    for score, match in zip(top_scores[:top_n_incidents], top_matches[:top_n_incidents]):
        if match.get('source') != 'incident':
            continue
            
        ev_id = match.get('ev_id')
        if not ev_id or ev_id not in refined_dataset:
            continue
            
        incident = refined_dataset[ev_id]
        sequence = incident.get('sequence_of_events', [])
        
        if not sequence:
            continue
            
        # 2. Find the "current" event in this sequence
        # We look for the event that best matches the user query
        best_match_idx = -1
        best_match_score = -1
        
        # Extract descriptions
        descriptions = [e.get('Occurrence_Description', '') for e in sequence]
        
        # Calculate similarity for each event in the sequence
        # This can be slow if we do it for every event.
        # Heuristic: Only check if the sequence has at least 2 events
        if len(descriptions) < 2:
            continue
            
        # Embed all descriptions in this sequence (batching would be better but this is simple)
        # To speed up, we can assume the query matches one of the text descriptions
        # directly using substring matching first, then embedding if needed.
        
        # Let's use a simpler approach first: Substring/Keyword matching
        # If that fails, we could use embeddings, but let's try to be efficient.
        # Actually, for "Engine Fire", exact match might fail. Embedding is safer.
        
        # Batch embedding for the sequence
        unique_descs = [d for d in descriptions if d and d not in event_embeddings_cache]
        if unique_descs:
            try:
                # small batch
                resp = client_instance.embeddings.create(input=unique_descs, model=EMBEDDING_MODEL)
                for d, data in zip(unique_descs, resp.data):
                    event_embeddings_cache[d] = data.embedding
            except Exception as e:
                print(f"Error embedding events: {e}")
                continue
        
        # Find best match
        for idx, desc in enumerate(descriptions):
            if not desc: continue
            emb = get_cached_embedding(desc)
            sim = np.dot(query_embedding, emb) # Cosine sim (assuming normalized)
            
            if sim > best_match_score:
                best_match_score = sim
                best_match_idx = idx
        
        # Threshold: If the best match is too weak, maybe this incident isn't relevant 
        # in the way we think. But 'score' (incident similarity) is already high.
        # Let's use a loose threshold.
        if best_match_score > 0.75:
            # 3. Identify downstream events
            # Get all events strictly AFTER the match
            subsequent_events = descriptions[best_match_idx+1:]
            
            if subsequent_events:
                sequences_analyzed += 1
                matched_incidents.append({
                    'ev_id': ev_id,
                    'matched_event': descriptions[best_match_idx],
                    'subsequent_events': subsequent_events,
                    'incident_similarity': score
                })
                
                # Add immediate next event
                if len(subsequent_events) > 0:
                    next_events.append(subsequent_events[0])
                    
    # 4. Calculate probabilities
    if sequences_analyzed == 0:
        return {
            'error': 'No matching event sequences found',
            'future_events': []
        }
        
    # Count next events
    next_event_counts = Counter(next_events)
    total_next = sum(next_event_counts.values())
    
    # Calculate probabilities
    future_events = []
    for event, count in next_event_counts.items():
        prob = count / sequences_analyzed  # P(FutureEvent | CurrentEvent)
        # Note: We divide by total sequences analyzed (the condition), 
        # not just total next events (which would be P(Next=X | ThereIsANext))
        
        future_events.append({
            'event': event,
            'probability': prob,
            'count': count,
            'likelihood_score': prob * 100
        })
        
    future_events.sort(key=lambda x: x['probability'], reverse=True)
    
    # Find evidence (full chains) for the top future events
    for fe in future_events:
        fe['evidence'] = []
        target = fe['event']
        for inc in matched_incidents:
            if inc['subsequent_events'] and inc['subsequent_events'][0] == target:
                fe['evidence'].append(inc['ev_id'])
                if len(fe['evidence']) >= 5: break
    
    print(f"✅ Predicted future events based on {sequences_analyzed} sequences")
    
    return {
        'future_events': future_events,
        'sequences_analyzed': sequences_analyzed,
        'matched_incidents': matched_incidents[:5] # Debug info
    }

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

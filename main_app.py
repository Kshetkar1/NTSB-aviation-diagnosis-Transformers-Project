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
    # Calculate cosine similarity between the query and all stored embeddings
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    
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


# --- Dr. Smith's Weighted Bayesian Diagnosis Functions ---

def calculate_weighted_diagnosis(top_scores, top_matches, top_n_incidents=50):
    """
    Implements Dr. Smith's "quick and dirty" weighted diagnosis approach.
    
    Takes similar incidents with cosine similarity scores and calculates
    weighted probabilities for root causes based on similarity-weighted evidence.
    
    Formula: P_weighted(cause) = Σ(similarity × presence) / Σ(similarity)
    
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
            
        # Get bayesian diagnostic data
        bayesian_data = match.get('bayesian_data', {})
        if not bayesian_data.get('has_diagnostic_data'):
            continue
            
        incidents_analyzed += 1
        
        # The 'all_causes' field contains the union of structured findings and narrative-extracted causes
        all_causes = bayesian_data.get('all_causes', [])
        
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
        'methodology': 'Similarity-weighted empirical Bayesian diagnosis',
        'total_similarity_weight': total_similarity_sum
    }


def diagnose_root_causes(query, top_n=10):
    """
    High-level diagnostic interface using Dr. Smith's weighted approach.
    
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
    diagnosis_results = calculate_weighted_diagnosis(top_scores, top_matches)
    
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
    
    # 5. Calculate weighted diagnosis (Dr. Smith's approach)
    print(f"🧠 Calculating weighted root cause diagnosis...")
    diagnosis_results = calculate_weighted_diagnosis(top_scores, top_matches)
    
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

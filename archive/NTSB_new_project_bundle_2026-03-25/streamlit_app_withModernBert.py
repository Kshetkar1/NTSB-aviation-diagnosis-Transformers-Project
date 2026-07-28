import os
# Force single-threaded execution for NumPy/Scikit-learn to prevent hanging in sandbox
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import streamlit as st

# Configure page to use wide layout for better chart visibility
st.set_page_config(page_title="NTSB Analysis", layout="wide")

from main_app import (
    diagnose_incident_and_get_details, 
    build_network_chart,
    diagnose_with_conditional_probabilities,
    prognosis_with_details,
    prognosis_with_chain_rule,
    predict_future_events,  # Imported new function
    generate_probabilistic_sequence_diagram, # Imported new function
    generate_diagnosis_tree, # Imported new function
    clean_ntsb_text, # Imported new function
    get_embedding,
    find_top_matches,
    get_full_incident_details
)
import streamlit.components.v1 as components

def render_mermaid_html(mermaid_code):
    """
    Generates HTML to render a Mermaid diagram using the CDN library.
    This ensures the visual chart is displayed even if st.markdown fails.
    """
    return f"""
    <div style="width: 100%; overflow: auto;">
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{
                startOnLoad:true, 
                securityLevel:'loose', 
                theme:'neutral',
                flowchart: {{ useMaxWidth: true, htmlLabels: true }} 
            }});
        </script>
        <div class="mermaid">
            {mermaid_code}
        </div>
    </div>
    """

def format_flight_time(hours_float):
    """Convert flight hours to human readable Days, Hours, Minutes"""
    if hours_float is None:
        return "N/A"
    
    # Handle negative or zero
    if hours_float <= 0:
        return "0h"
        
    total_seconds = int(hours_float * 3600)
    days = total_seconds // 86400
    remaining = total_seconds % 86400
    hours = remaining // 3600
    remaining %= 3600
    minutes = remaining // 60
    
    # Build string parts
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
        
    # For very short durations, show seconds
    if not parts:
        return "< 1m"
        
    return " ".join(parts)

# --- Streamlit App Layout ---

st.set_page_config(layout="wide")

st.title("✈️ Aviation Incident Analysis Engine")

# Mode selection
analysis_mode = st.radio(
    "Select Analysis Mode:",
    ["🔍 Diagnosis Mode", "🔮 Prognosis Mode"],
    horizontal=True,
    help="**Diagnosis**: Find root causes of incidents\n\n**Prognosis**: Predict when issues will become critical"
)

if analysis_mode == "🔍 Diagnosis Mode":
    st.write(
        "Enter a description of an aviation incident or observation below. "
        "The system will analyze it against a database of historical NTSB reports "
        "to find related incidents and display their causal event chains."
    )
    
    # Method selection for diagnosis
    st.markdown("### Diagnosis Method")
    diagnosis_method = st.radio(
        "Choose diagnosis approach:",
        ["Similarity-Weighted (Current)", "Conditional Probability with Chain Rule (Professor's Method)"],
        help="**Similarity-Weighted**: Weights causes by similarity scores\n\n**Conditional Probability**: Clusters incidents by type, calculates P(Cause|Type), applies chain rule"
    )
else:
    st.write(
        "Enter a description of a detected issue or defect. "
        "The system will analyze similar historical incidents to predict when this issue "
        "is likely to become critical and provide maintenance recommendations."
    )
    st.info("💡 Example queries: 'small crack in turbine blade', 'corrosion detected on fuselage', 'fuel pump showing signs of wear'")
    
    # Method selection for prognosis
    st.markdown("### Prognosis Method")
    prognosis_method = st.radio(
        "Choose prognosis approach:",
        ["Similarity-Weighted (Standard)", "Chain Rule (Cluster-Based)"],
        help="**Similarity-Weighted**: Weighted average of all similar incidents\n\n**Chain Rule**: Clusters failure modes first, then aggregates (Better for heterogeneous failures)"
    )

# User input
if analysis_mode == "🔍 Diagnosis Mode":
    user_query = st.text_area(
        "Enter Incident Description:", 
        "engine fire during takeoff",
        height=100
    )
    current_hours = None
    button_label = "Diagnose Incident"
else:  # Prognosis Mode
    user_query = st.text_area(
        "Enter Defect/Issue Description:",
        "crack detected in turbine blade",
        height=100
    )
    current_hours = st.number_input(
        "Current Component Flight Hours:",
        min_value=0,
        max_value=100000,
        value=10000,
        step=100,
        help="Enter the current flight hours of the component with the issue"
    )
    button_label = "Predict Time to Failure"

if st.button(button_label):
    print("DEBUG: Button clicked!", flush=True)
    if user_query:
        # Different processing for diagnosis vs prognosis
        if analysis_mode == "🔍 Diagnosis Mode":
            # REMOVED SPINNER to debug hanging
            if True:
                
                # DEBUG LOG
                print("DEBUG: Starting Diagnosis...", flush=True)
                
                # Choose method based on user selection
                if diagnosis_method == "Conditional Probability with Chain Rule (Professor's Method)":
                    # Professor's method
                    print("DEBUG: Calling diagnose_with_conditional_probabilities", flush=True)
                    diagnosis_results = diagnose_with_conditional_probabilities(user_query, top_n=15, top_n_incidents=50)
                    print(f"DEBUG: Results received. Keys: {diagnosis_results.keys()}", flush=True)
                    
                    if 'error' in diagnosis_results:
                        st.error(f"Error in diagnosis: {diagnosis_results['error']}")
                        st.stop()
                    
                    # Get incident details for display
                    print("DEBUG: Getting embeddings for display...", flush=True)
                    query_embedding = get_embedding(user_query)
                    top_scores, top_matches = find_top_matches(query_embedding)
                    
                    # Collect unique incident IDs
                    incident_scores = {}
                    for score, match in zip(top_scores, top_matches):
                        if 'ev_id' in match:
                            ev_id = match['ev_id']
                            if ev_id not in incident_scores:
                                incident_scores[ev_id] = score
                    
                    print(f"DEBUG: Fetching details for {len(incident_scores)} incidents", flush=True)
                    incident_details = get_full_incident_details(list(incident_scores.keys()))
                else:
                    # Current similarity-weighted method
                    print("DEBUG: Calling diagnose_incident_and_get_details", flush=True)
                    incident_scores, incident_details, diagnosis_results = diagnose_incident_and_get_details(user_query)
            
            st.subheader("Diagnostic Results")

            if not incident_scores:
                st.warning(incident_details if isinstance(incident_details, str) else "No incidents found") # Display the info/error message
            else:
                # ============ CLUSTER ANALYSIS (Professor's Method Only) ============
                if diagnosis_method == "Conditional Probability with Chain Rule (Professor's Method)" and diagnosis_results.get('clusters'):
                    st.markdown("---")
                    st.markdown("## 📊 Incident Clustering Analysis")
                    st.markdown(f"*Clustered {diagnosis_results.get('total_incidents_analyzed', 0)} incidents into {diagnosis_results.get('total_clusters', 0)} types*")
                    
                    # Show cluster breakdown
                    clusters = diagnosis_results.get('clusters', {})
                    cluster_data = []
                    for cluster_type, analysis in sorted(clusters.items(), key=lambda x: x[1]['total_incidents'], reverse=True):
                        cluster_data.append({
                            'Cluster Type': cluster_type,
                            'Incidents': analysis['total_incidents'],
                            'Avg Similarity': f"{analysis['avg_similarity']:.3f}",
                            'Unique Causes': len(analysis['causes'])
                        })
                    
                    if cluster_data:
                        st.dataframe(cluster_data, use_container_width=True)
                        
                        # Show detailed cluster info in expander
                        with st.expander("📋 View Detailed Cluster Breakdown"):
                            for cluster_type, analysis in sorted(clusters.items(), key=lambda x: x[1]['total_incidents'], reverse=True):
                                st.markdown(f"**{cluster_type}** ({analysis['total_incidents']} incidents)")
                                
                                # Show top causes in this cluster
                                top_causes = sorted(analysis['causes'].items(), key=lambda x: x[1], reverse=True)[:3]
                                if top_causes:
                                    st.markdown("Top causes in this cluster:")
                                    for cause, prob in top_causes:
                                        cause_text = cause[:80] + "..." if len(cause) > 80 else cause
                                        st.markdown(f"  - {prob*100:.1f}%: {cause_text}")
                                st.markdown("---")
                
                # ============ ROOT CAUSE DIAGNOSIS ============
                st.markdown("---")
                st.markdown("## 🔍 Root Cause Diagnosis")
                
                # Show methodology info
                method_name = diagnosis_results.get('methodology', 'Unknown method')
                st.markdown(f"*Method: {method_name}*")
                st.markdown(f"*Analyzed {diagnosis_results.get('total_incidents_analyzed', 0)} similar historical NTSB incidents*")
                
                if diagnosis_results and diagnosis_results.get('weighted_causes'):
                    
                    # --- PREPARE DATA ---
                    from collections import defaultdict
                    cluster_groups = defaultdict(list)
                    cluster_probs = {}
                    
                    for cause in diagnosis_results['weighted_causes']:
                        if cause.get('cluster_breakdown'):
                            top_cluster = cause['cluster_breakdown'][0]
                            c_name = top_cluster['type']
                            c_prob = top_cluster['p_cluster_query']
                            cluster_groups[c_name].append(cause)
                            cluster_probs[c_name] = c_prob
                    
                    sorted_clusters = sorted(cluster_probs.items(), key=lambda x: x[1], reverse=True)

                    # --- STEP 1: CLUSTER IDENTIFICATION (P(I|Q)) ---
                    st.markdown("### Step 1: Identify Incident Types (Clusters)")
                    st.markdown(r"First, we calculate the probability of each incident type given the query: $P(Cluster|Query)$")
                    
                    cluster_table = []
                    for c_name, c_prob in sorted_clusters[:5]:
                        cluster_table.append({
                            "Incident Type (Cluster)": c_name.title(),
                            "Probability P(I|Q)": f"{c_prob:.1%}"
                        })
                    st.dataframe(cluster_table, hide_index=True)
                    
                    # --- STEP 2: CONDITIONAL CAUSES (P(C|I)) ---
                    st.markdown("---")
                    st.markdown("### Step 2: Analyze Causes within Clusters")
                    st.markdown(r"For each cluster, we identify the conditional probability of causes: $P(Cause|Cluster)$")
                    
                    for c_name, c_prob in sorted_clusters[:5]:
                        with st.expander(f"📂 {c_name.title()} (P={c_prob:.1%})", expanded=False):
                            st.markdown(f"**Conditional Probabilities for {c_name.title()}:**")
                            
                            causes_in_group = cluster_groups[c_name]
                            causes_in_group.sort(key=lambda x: x['probability'], reverse=True)
                            
                            # Show more causes (Top 10) to avoid hiding information
                            for cause_info in causes_in_group[:10]:
                                # Back-calculate conditional prob: P(C|I) = P(C|Q) / P(I|Q)
                                p_cause_query = cause_info['probability']
                                p_cond = p_cause_query / c_prob if c_prob > 0 else 0
                                
                                cause_text = cause_info['cause'] # Keep raw text
                                
                                c1, c2 = st.columns([4, 1])
                                c1.markdown(f"- {cause_text}")
                                c2.markdown(f"**{p_cond:.1%}**")
                            
                            # Mermaid Sequence (Causal Event Chain)
                            if 'clusters' in diagnosis_results:
                                cluster_data = diagnosis_results['clusters'].get(c_name)
                                if cluster_data:
                                    inc_ids = cluster_data.get('incident_ids', [])
                                    if inc_ids:
                                        st.markdown("---")
                                        st.caption("Analysis of Precursors & Event Sequence:")
                                        
                                        full_incidents_map = get_full_incident_details(inc_ids[:20])
                                        full_incidents_list = list(full_incidents_map.values())
                                        
                                        mermaid_code = generate_probabilistic_sequence_diagram(full_incidents_list)
                                        
                                        if mermaid_code:
                                            with st.expander("👁️ View Precursors & Causal Chain", expanded=True):
                                                components.html(render_mermaid_html(mermaid_code), height=600, scrolling=True)

                    # --- STEP 3: FINAL AGGREGATION (P(C|Q)) ---
                    st.markdown("---")
                    st.markdown("### Step 3: Final Root Cause Diagnosis")
                    st.markdown(r"Aggregated probability using Total Probability Theorem: $P(C|Q) = \sum P(C|I) \times P(I|Q)$")
                    
                    # Deduplicate Causes (Group by raw text)
                    aggregated_causes = defaultdict(lambda: {
                        'prob': 0.0, 
                        'breakdown': [],
                        'count': 0
                    })
                    
                    for cause in diagnosis_results['weighted_causes']:
                        key = cause['cause'] 
                        aggregated_causes[key]['prob'] += cause['probability']
                        aggregated_causes[key]['count'] += 1
                        if cause.get('cluster_breakdown'):
                            aggregated_causes[key]['breakdown'].extend(cause['cluster_breakdown'])
                    
                    # Convert to list and sort
                    final_causes_list = [
                        {'cause': k, 'prob': v['prob'], 'breakdown': v['breakdown']}
                        for k, v in aggregated_causes.items()
                    ]
                    final_causes_list.sort(key=lambda x: x['prob'], reverse=True)
                    
                    # Display Top 10 (Expanded from 5)
                    for i, cause_item in enumerate(final_causes_list[:10], 1):
                        cause_text = cause_item['cause']
                        prob = cause_item['prob']
                        
                        # Build Detailed Calculation String
                        sorted_bd = sorted(cause_item['breakdown'], key=lambda x: x['contribution'], reverse=True)
                        
                        calc_parts = []
                        # Show ALL components (no limit) to be fully transparent
                        for comp in sorted_bd: 
                             p_cause = comp['p_cause_cluster']
                             p_cluster = comp['p_cluster_query']
                             # c_name = comp['type'] 
                             # Use more precision to show small numbers
                             calc_parts.append(f"({p_cause:.4f} × {p_cluster:.4f})")
                        
                        calc_str = " + ".join(calc_parts)
                        
                        # Only show full detail if it's not empty
                        if calc_str:
                            final_calc_display = f"{calc_str} = {prob:.4f} ({prob:.1%})"
                        else:
                            final_calc_display = f"{prob:.1%}"

                        with st.container():
                            c1, c2 = st.columns([4, 1])
                            c1.markdown(f"**{i}. {cause_text}**")
                            c2.metric("Probability", f"{prob:.1%}")
                            st.caption(f"**Calculation:** {final_calc_display}")
                        st.divider()
                        
                else:
                    st.info("No diagnostic data available for this query.")
                
                # ============ SIMILAR INCIDENTS SECTION ============
                st.markdown("## 📋 Similar Historical Incidents")
                st.markdown(f"*Showing top 5 of {len(incident_scores)} similar incidents found. Click to expand and view details.*")
                
                # Sort incidents by score, highest first
                sorted_incidents = sorted(incident_scores.items(), key=lambda item: item[1], reverse=True)

                # Display only top 5 incidents in an expandable section (a dropdown)
                for ev_id, score in sorted_incidents[:5]:
                    if ev_id in incident_details:
                        details = incident_details[ev_id]
                        # 1. Format score as percentage
                        score_percent = f"{score * 100:.2f}%"
                        
                        # 2. Create the dropdown
                        with st.expander(f"**Incident: {ev_id}** (Similarity: {score_percent})"):
                            
                            # 3. Display all info inside the dropdown
                            
                            # --- Display Narratives ---
                            st.markdown("---")
                            st.markdown("#### Narratives")
                            
                            # Factual Narrative
                            narr_accf = details.get('narr_accf')
                            if narr_accf:
                                st.markdown(f"**Factual Narrative (`narr_accf`):**\n> {narr_accf}")
                            
                            # Causal Narrative
                            narr_cause = details.get('narr_cause')
                            if narr_cause:
                                st.markdown(f"**Causal Narrative (`narr_cause`):**\n> {narr_cause}")
                            
                            # Probable Cause Narrative (Fallback)
                            narr_accp = details.get('narr_accp')
                            if narr_accp:
                                st.markdown(f"**Probable Cause Narrative (`narr_accp`):**\n> {narr_accp}")

                            # --- Display Horizontal Flow Chart ---
                            st.markdown("---")
                            st.markdown("#### Sequence of Events")
                            sequence = details.get('sequence_of_events', [])
                            
                            if sequence:
                                flow_html = "<div style='display: flex; align-items: center; flex-wrap: wrap; gap: 10px; padding: 10px;'>"
                                for i, event in enumerate(sequence):
                                    event_desc = event.get('Occurrence_Description', 'Unknown Event')
                                    flow_html += f"<div style='background-color: #3498db; color: white; padding: 10px 15px; border-radius: 8px; font-size: 14px; text-align: center;'>{event_desc}</div>"
                                    if i < len(sequence) - 1:
                                        flow_html += "<div style='font-size: 24px; color: #e74c3c; font-weight: bold; margin: 0 10px;'>→</div>"
                                flow_html += "</div>"
                                
                                components.html(flow_html, height=150, scrolling=True)
                            else:
                                st.write("No sequence of events data available for this incident.")
        
        else:  # Prognosis Mode
            with st.spinner("🔮 Analyzing historical failure patterns..."):
                if prognosis_method == "Chain Rule (Cluster-Based)":
                    # --- NEW LOGIC: Future Event Sequence Prediction ---
                    prognosis_results = predict_future_events(user_query, top_n_incidents=50)
                    incident_details = {} 
                else:
                    # --- OLD LOGIC: Time-to-Failure ---
                    prognosis_results, incident_details = prognosis_with_details(user_query, current_hours)
            
            st.subheader("Prognosis Results")
            
            # --- DISPLAY FOR FUTURE EVENT PREDICTION (NEW) ---
            if prognosis_method == "Chain Rule (Cluster-Based)":
                if 'future_events' in prognosis_results and prognosis_results['future_events']:
                    st.markdown("### 🔮 Predicted Downstream Events")
                    st.markdown(f"*Based on {prognosis_results['sequences_analyzed']} similar historical event sequences where this occurred.*")
                    
                    # Create columns for the top events
                    events = prognosis_results['future_events']
                    
                    for event in events[:5]:
                        st.markdown("---")
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{event['event']}**")
                            st.progress(event['probability'])
                            if 'evidence' in event and event['evidence']:
                                with st.expander("Show supporting cases"):
                                    st.write(f"Occurred in incidents: {', '.join(event['evidence'])}")
                        with col2:
                            st.metric("Probability", f"{event['likelihood_score']:.1f}%")
                            
                    if not events:
                        st.info("No downstream events found in the similar incidents.")
                else:
                    st.warning("Could not find enough matching event sequences to make a prediction.")
                    
            # --- DISPLAY FOR TIME-TO-FAILURE (OLD) ---
            else:
                if prognosis_results.get('error'):
                    st.warning(prognosis_results['error'])
                    if 'recommendation' in prognosis_results:
                        st.info(prognosis_results['recommendation'])
                    st.stop()
                
                # ============ CLUSTER ANALYSIS (Chain Rule Method Only) ============
                if prognosis_method == "Chain Rule (Cluster-Based)" and prognosis_results.get('cluster_breakdown'):
                    st.markdown("---")
                    st.markdown("## 📊 Failure Mode Clustering")
                    st.markdown(f"*Identified {len(prognosis_results['cluster_breakdown'])} distinct failure modes from {prognosis_results['similar_incidents']} incidents*")
                    
                    # Create breakdown table
                    breakdown_data = []
                    for cluster in prognosis_results['cluster_breakdown']:
                        breakdown_data.append({
                            'Failure Mode': cluster['type'].title(),
                            'Weight': f"{cluster['weight']*100:.1f}%",
                            'Avg Failure Time': f"{format_flight_time(cluster['mean_ttf'])} ({cluster['mean_ttf']:.0f}h)",
                            'Incidents': cluster['n_incidents']
                        })
                    
                    st.dataframe(breakdown_data, use_container_width=True)
                    
                    with st.expander("📋 View Cluster Details"):
                        for cluster in prognosis_results['cluster_breakdown']:
                            st.markdown(f"**{cluster['type'].title()}**")
                            st.markdown(f"- Contribution: {cluster['weight']*100:.1f}%")
                            st.markdown(f"- Mean TTF: {format_flight_time(cluster['mean_ttf'])} ({cluster['mean_ttf']:.0f}h)")
                            st.markdown(f"- Incidents: {cluster['n_incidents']}")
                            st.markdown("---")

                # ============ PROGNOSIS SUMMARY ============
                st.markdown("---")
                st.markdown("## ⏱️ Time-to-Failure Prediction")
                
                col1, col2, col3 = st.columns(3)
                
                # Helper to display formatted time with hours in smaller text
                def display_time_metric(label, hours, help_text):
                    formatted = format_flight_time(hours)
                    st.metric(
                        label,
                        formatted,
                        delta=f"{hours:,.0f} flight hours",
                        delta_color="off",
                        help=help_text
                    )
                
                with col1:
                    display_time_metric(
                        "Estimated Remaining",
                        prognosis_results['estimated_remaining_hours'],
                        "Weighted average based on similar incidents"
                    )
                
                with col2:
                    display_time_metric(
                        "Conservative (10th %)",
                        prognosis_results['conservative_estimate'],
                        "10% of similar cases failed before this point"
                    )
                
                with col3:
                    display_time_metric(
                        "Median (50th %)",
                        prognosis_results['median_estimate'],
                        "Half of similar cases failed before this point"
                    )
                
                # ============ RECOMMENDATIONS ============
                st.markdown("---")
                st.markdown("## 🔧 Maintenance Recommendations")
                
                recommendations = prognosis_results['recommendations']
                urgency = recommendations['urgency']
                
                # Color code by urgency
                urgency_colors = {
                    'IMMEDIATE': '🔴',
                    'HIGH': '🟠',
                    'MODERATE': '🟡',
                    'LOW': '🟢'
                }
                
                st.markdown(f"### Urgency Level: {urgency_colors.get(urgency, '⚪')} {urgency}")
                
                rec_col1, rec_col2, rec_col3 = st.columns(3)
                
                with rec_col1:
                    hrs = recommendations['inspect_within_hours']
                    st.info(f"**Inspect Within**\n\n### {format_flight_time(hrs)}\n({hrs:.0f} hours)")
                
                with rec_col2:
                    hrs = recommendations['repair_by_hours']
                    st.warning(f"**Repair By**\n\n### {format_flight_time(hrs)}\n({hrs:.0f} hours)")
                
                with rec_col3:
                    hrs = recommendations['critical_threshold_hours']
                    st.error(f"**Critical Threshold**\n\n### {format_flight_time(hrs)}\n({hrs:.0f} hours)")
                
                # ============ RISK TIMELINE ============
                st.markdown("---")
                st.markdown("## 📈 Risk Timeline")
                st.markdown("*Probability of failure at different time points*")
                
                risk_data = prognosis_results['risk_curve']
                
                for risk_point in risk_data:
                    risk_level = risk_point['risk_level']
                    risk_pct = risk_point['failure_probability']
                    add_hours = risk_point['additional_hours']
                    total_hours = risk_point['total_hours']
                    
                    # Color by risk level
                    if risk_level == 'CRITICAL':
                        color = 'red'
                    elif risk_level == 'HIGH':
                        color = 'orange'
                    elif risk_level == 'MODERATE':
                        color = 'yellow'
                    else:
                        color = 'green'
                    
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"**+{add_hours:,} hours** (Total: {total_hours:,} hrs)")
                        st.progress(risk_pct / 100)
                    with col_b:
                        st.markdown(f"<span style='color:{color}; font-weight:bold'>{risk_pct:.1f}% - {risk_level}</span>", unsafe_allow_html=True)
                
                # ============ STATISTICAL SUMMARY ============
                st.markdown("---")
                st.markdown("## 📊 Statistical Analysis")
                
                stat_col1, stat_col2 = st.columns(2)
                
                with stat_col1:
                    st.markdown("### Distribution")
                    percentiles = prognosis_results['percentiles']
                    for pct, value in percentiles.items():
                        st.markdown(f"- **{pct}**: {value:.0f} hours")
                
                with stat_col2:
                    st.markdown("### Analysis Details")
                    st.markdown(f"- **Similar incidents**: {prognosis_results['similar_incidents']}")
                    st.markdown(f"- **Current component hours**: {prognosis_results['current_component_hours']:,.0f}")
                    st.markdown(f"- **Average failure time**: {prognosis_results['average_failure_time']:.0f} ± {prognosis_results['std_deviation']:.0f} hrs")
                    evidence = prognosis_results['evidence']
                    st.markdown(f"- **Range**: {evidence['earliest_failure']:.0f} - {evidence['latest_failure']:.0f} hrs")
                
                # ============ SUPPORTING EVIDENCE ============
                st.markdown("---")
                st.markdown("## 📋 Supporting Historical Evidence")
                
                # Handle evidence for both methods
                if 'evidence' in prognosis_results and prognosis_results['evidence'].get('sample_incidents'):
                    st.markdown("*Top 5 most similar incidents*")
                    sample_incidents = prognosis_results['evidence']['sample_incidents']
                    
                    for i, inc in enumerate(sample_incidents, 1):
                        with st.expander(f"Incident {i}: {inc['ev_id']} (Similarity: {inc['similarity']*100:.1f}%, Failed at {inc['hours_at_failure']:.0f} hrs)"):
                            st.markdown(f"**Failure Time**: {inc['hours_at_failure']:,.0f} flight hours")
                            st.markdown(f"**Similarity Score**: {inc['similarity']:.3f}")
                            if inc['narrative']:
                                st.markdown(f"**Narrative**: {inc['narrative']}")
                            
                            # Show full details if available
                            if inc['ev_id'] in incident_details:
                                details = incident_details[inc['ev_id']]
                                
                                # --- Display Narratives ---
                                st.markdown("---")
                                st.markdown("#### Narratives")
                                
                                # Factual Narrative
                                narr_accf = details.get('narr_accf')
                                if narr_accf:
                                    st.markdown(f"**Factual Narrative (`narr_accf`):**\n> {narr_accf}")
                                
                                # Causal Narrative
                                narr_cause = details.get('narr_cause')
                                if narr_cause:
                                    st.markdown(f"**Causal Narrative (`narr_cause`):**\n> {narr_cause}")
                                
                                # Probable Cause Narrative (Fallback)
                                narr_accp = details.get('narr_accp')
                                if narr_accp:
                                    st.markdown(f"**Probable Cause Narrative (`narr_accp`):**\n> {narr_accp}")

                                # --- Display Sequence (Mermaid) ---
                                st.markdown("---")
                                st.markdown("#### Sequence of Events")
                                
                                # Use the shared Mermaid generator for consistency
                                # We pass [details] as a list of 1 incident to generate a linear path
                                mermaid_code = generate_probabilistic_sequence_diagram([details])
                                
                                if mermaid_code:
                                    # Use Left-to-Right layout for single linear sequences to save vertical space
                                    mermaid_code = mermaid_code.replace("graph TD", "graph LR")
                                    components.html(render_mermaid_html(mermaid_code), height=200, scrolling=True)
                                else:
                                    st.caption("No sequence data available.")
                
                elif 'cluster_breakdown' in prognosis_results:
                     # For chain rule method, show examples from top clusters
                     st.markdown("*Sample incidents from top failure mode clusters*")
                     clusters = prognosis_results['clusters']
                     
                     for cluster in prognosis_results['cluster_breakdown'][:3]: # Top 3 clusters
                         cluster_type = cluster['type']
                         st.markdown(f"### {cluster_type.title()}")
                         
                         cluster_incidents = clusters[cluster_type]['incidents']
                         # Sort by score
                         cluster_incidents.sort(key=lambda x: x['score'], reverse=True)
                         
                         for i, inc in enumerate(cluster_incidents[:2], 1): # Top 2 per cluster
                             with st.expander(f"{inc['ev_id']} ({inc['hours_at_failure']:.0f} hrs)"):
                                 st.markdown(f"**Failure Time**: {inc['hours_at_failure']:,.0f} flight hours")
                                 st.markdown(f"**Similarity**: {inc['score']:.3f}")
                                 st.markdown(f"**Narrative**: {inc['narrative']}")

    else:
        st.error(f"Please enter a {'defect/issue' if analysis_mode == '🔮 Prognosis Mode' else 'incident'} description.")

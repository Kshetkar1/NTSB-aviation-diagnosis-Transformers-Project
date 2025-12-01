import streamlit as st
from main_app import diagnose_incident_and_get_details, build_network_chart
import streamlit.components.v1 as components

# --- Streamlit App Layout ---

st.set_page_config(layout="wide")

st.title("✈️ Aviation Incident Diagnosis Engine")
st.write(
    "Enter a description of an aviation incident or observation below. "
    "The system will analyze it against a database of historical NTSB reports "
    "to find related incidents and display their causal event chains."
)

# User input
user_query = st.text_area(
    "Enter Incident Description:", 
    "engine fire during takeoff",
    height=100
)

if st.button("Diagnose Incident"):
    if user_query:
        with st.spinner("🧠 Analyzing historical data... This may take a moment."):
            
            # Call backend with new weighted diagnosis capability
            incident_scores, incident_details, diagnosis_results = diagnose_incident_and_get_details(user_query)
            
            st.subheader("Diagnostic Results")

            if not incident_scores:
                st.warning(incident_details) # Display the info/error message
            else:
                # ============ NEW SECTION: ROOT CAUSE DIAGNOSIS ============
                st.markdown("---")
                st.markdown("## 🔍 Root Cause Diagnosis")
                st.markdown(f"*Based on similarity-weighted analysis of {diagnosis_results.get('total_incidents_analyzed', 0)} similar historical NTSB incidents*")
                
                if diagnosis_results and diagnosis_results.get('weighted_causes'):
                    st.markdown("### Most Likely Root Causes")
                    
                    # Display top 10 causes
                    for i, cause_info in enumerate(diagnosis_results['weighted_causes'][:10], 1):
                        prob = cause_info['probability']
                        prob_pct = prob * 100
                        cause_text = cause_info['cause']
                        num_incidents = cause_info['num_incidents']
                        supporting_incident_ids = cause_info.get('supporting_incidents', [])
                        
                        # Create columns for better layout
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**{i}. {cause_text}**")
                            st.progress(prob)
                            st.caption(f"Found in {num_incidents} of the similar incidents")
                        
                        with col2:
                            st.metric(label="Probability", value=f"{prob_pct:.1f}%")
                        
                        # Add Mermaid chart showing event sequences that led to this cause
                        if supporting_incident_ids:
                            st.markdown(f"**Event Sequences Leading to This Cause:**")
                            
                            # Extract sequence_of_events for incidents that had this cause
                            causal_chains = {}
                            for ev_id in supporting_incident_ids[:5]:  # Show up to 5 incidents
                                if ev_id in incident_details:
                                    sequence = incident_details[ev_id].get('sequence_of_events', [])
                                    if sequence:
                                        causal_chains[ev_id] = sequence
                            
                            # Generate and display Mermaid chart
                            if causal_chains:
                                mermaid_code = build_network_chart(causal_chains)
                                st.code(mermaid_code, language="mermaid")
                            else:
                                st.caption("_No event sequence data available for these incidents_")
                        
                        st.markdown("---")
                else:
                    st.info("No diagnostic data available for this query. The similar incidents may not have documented root causes.")
                
                # ============ SIMILAR INCIDENTS SECTION ============
                st.markdown("## 📋 Similar Historical Incidents")
                st.markdown("*Click to expand and view details*")
                
                # Sort incidents by score, highest first
                sorted_incidents = sorted(incident_scores.items(), key=lambda item: item[1], reverse=True)

                # Display each incident in an expandable section (a dropdown)
                for ev_id, score in sorted_incidents:
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

    else:
        st.error("Please enter an incident description.")

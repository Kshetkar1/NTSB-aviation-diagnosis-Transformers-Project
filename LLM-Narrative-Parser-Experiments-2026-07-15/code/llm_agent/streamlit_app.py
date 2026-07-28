import streamlit as st
from agent import run_diagnosis_agent
import streamlit.components.v1 as components

# --- Streamlit App Layout ---

st.set_page_config(layout="wide")

st.title("✈️ Aviation Incident Diagnosis Engine (LLM Agent)")
st.write(
    "Enter a description of an aviation incident. An LLM agent will first generate a "
    "hypothetical incident report, then use a diagnostic tool to analyze it against "
    "a database of historical NTSB reports to find probable root causes."
)

# User input
user_query = st.text_area(
    "Enter a brief incident description:",
    "engine fire during takeoff",
    height=100
)

if st.button("Run Diagnostic Agent"):
    if user_query:
        with st.spinner("🤖 LLM Agent is running... This may take a moment."):
            
            # Call the new agent function
            agent_results = run_diagnosis_agent(user_query)

            st.markdown("---")
            st.subheader("Agent Execution Log")

            # 1. Display the LLM-generated hypothetical report
            st.markdown("### 1. LLM-Generated Incident Synopsis")
            st.info(agent_results['generated_report'])
            
            # 2. Display the final summary from the LLM
            st.markdown("### 2. LLM's Final Diagnosis Summary")
            st.success(agent_results['final_summary'])

            # 3. Display the structured data from the tool call
            st.markdown("---")
            st.subheader("🔬 Detailed Diagnostic Data (from Tool)")
            
            diagnosis_results = agent_results.get('tool_results')
            
            if diagnosis_results and diagnosis_results.get('top_causes'):
                st.markdown(
                    f"*Analysis based on {diagnosis_results['diagnosis_metadata'].get('incidents_analyzed', 0)} "
                    "similar historical incidents*"
                )
                
                # Display top 10 causes from the tool's raw output
                for i, cause_info in enumerate(diagnosis_results['top_causes'], 1):
                    prob = cause_info['probability']
                    prob_pct = prob * 100
                    cause_text = cause_info['cause']
                    num_incidents = cause_info['num_incidents']
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**{i}. {cause_text}**")
                        st.progress(prob)
                        st.caption(f"Found in {num_incidents} of the similar incidents")
                    
                    with col2:
                        st.metric(label="Probability", value=f"{prob_pct:.1f}%")
                    
                    st.markdown("---")
            else:
                st.warning("The diagnostic tool did not return any structured data.")

    else:
        st.error("Please enter an incident description.")

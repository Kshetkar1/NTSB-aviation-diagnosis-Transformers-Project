import json
from openai import OpenAI
from main_app import diagnose_root_causes
from config import get_openai_api_key, LLM_MODEL

# --- Configuration ---
# API key and model are loaded from config.py (which uses environment variables)

# Initialize client lazily
_client = None

def get_client():
    """Get OpenAI client, initializing it if needed."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=get_openai_api_key())
    return _client

# --- Tool Definition ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "diagnose_root_causes",
            "description": "Diagnoses the probable root causes of an aviation incident based on a descriptive text query. Returns a list of the most likely causes and their weighted probabilities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A detailed text description of the incident, e.g., 'engine fire during takeoff' or 'loss of power during landing approach'."
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "The number of top causes to return. Defaults to 10."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# A mapping from function names to the actual Python functions
available_functions = {
    "diagnose_root_causes": diagnose_root_causes,
}

# --- Agent Logic ---
def run_diagnosis_agent(user_query):
    """
    Runs the full diagnosis process using a multi-step LLM agent.
    
    Returns:
        A dictionary containing the generated synopsis, tool results, and final summary.
    """
    print("🚀 Starting diagnosis agent...")
    
    # --- Step 1: Generate a detailed, hypothetical incident report ---
    print("   Step 1: Generating hypothetical incident report...")
    report_generation_prompt = f"""
    Based on the following brief user query, please generate a detailed, hypothetical NTSB-style incident synopsis. 
    This synopsis should be a plausible, factual-sounding report that can be used for further analysis. 
    Include details such as aircraft type, flight phase, a summary of events, and the immediate outcome.
    
    User Query: "{user_query}"
    """
    
    report_response = get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": report_generation_prompt}],
    )
    generated_report = report_response.choices[0].message.content
    print(f"   ✅ Report generated: {generated_report[:100]}...")

    # --- Step 2: Call the diagnosis tool with the generated report ---
    print("   Step 2: Analyzing report with diagnostic tools...")
    analysis_prompt = f"""
    Based on the following incident synopsis, please use the available tools to diagnose the most likely root causes.

    Synopsis:
    {generated_report}
    """

    messages = [{"role": "user", "content": analysis_prompt}]
    
    # This loop handles the tool-calling logic
    tool_call_response = get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    
    response_message = tool_call_response.choices[0].message
    tool_calls = response_message.tool_calls
    
    if not tool_calls:
        # If the model decides not to use a tool, handle it gracefully
        final_summary = "The model did not find a reason to call the diagnostic tool for this query."
        tool_results = None
    else:
        # A tool was called, so we execute it
        messages.append(response_message)
        
        # In this case, we expect only one tool call, but we loop for robustness
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"      - Executing tool: {function_name}({function_args.get('query')[:50]}...)")
            
            function_response = function_to_call(
                query=function_args.get("query"),
                top_n=function_args.get("top_n", 10)
            )
            
            tool_results = function_response  # Store the raw tool output
            
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(tool_results), 
                }
            )

        # --- Step 3: Synthesize a final answer ---
        print("   Step 3: Synthesizing final summary...")
        final_response = get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
        )
        final_summary = final_response.choices[0].message.content
        print("   ✅ Agent run complete!")

    return {
        "generated_report": generated_report,
        "tool_results": tool_results,
        "final_summary": final_summary
    }

# Aviation Incident Diagnosis Engine: Project Plan (V2)

This document outlines the complete step-by-step process for building the Aviation Incident Diagnosis Engine. This version includes the strategy of creating a unified embedding memory from both historical data and the master event dictionary.

---

### **Phase 1: Setup and Data Consolidation**

**Goal:** To combine all separate data files into a single, clean, and structured master JSON file. This is the foundation for everything that follows.

**Tools:** Python, `pandas` library.

**Steps:**

1. **Project Setup:**

   - Create a new folder for your project.
   - Create a Python script named `1_create_refined_dataset.py`.
   - Place all your source data files (CSVs, etc.) in a subfolder named `Data`.
   - Install pandas: `pip install pandas openpyxl`.

2. **Load All Data F iles:**

   - In your script, import pandas and load each of your data files into its own DataFrame. Prioritize using `.xlsx` files over `.txt` when available.

3. **Combine Core Incident Data:**

   - Merge the primary DataFrames (like narratives and aircraft info) on their common `ev_id` column.

4. **Structure the Event Sequences:**

   - Group all events for a single incident together into a list. Ensure they are sorted chronologically first. This may involve merging data from multiple sequence files.

5. **Create the Final JSON Object:**

   - Merge the grouped event sequences with your main data.
   - Convert the final DataFrame into a dictionary, using `ev_id` as the primary key.

6. **Save the Master File:**

   - Save your final dictionary to a `refined_dataset.json` file. This completes the data consolidation phase.

   **Result:** A single `refined_dataset.json` file containing all your structured data.

---

### **Phase 2: Generate and Save Combined Embeddings**

**Goal:** To convert all relevant text from **both the historical incidents and the master event dictionary** into numerical embeddings and save them as a single, searchable "memory".

**Tools:** Python, `openai` library, `numpy`, `tqdm`.

**Steps:**

1. **Create a New Script:**

   - Create a file named `2_generate_embeddings.py`.
   - Install required libraries: `pip install openai numpy tqdm`.

2. **Load Your Processed Data Files:**

   - Load the `refined_dataset.json` file created in Phase 1.
   - Load your master event dictionary file (e.g., `ct_seqevt.txt`). Make sure to load it correctly (e.g., as a CSV).

3. **Collect All Text for Embedding:**

   - You will now create two sets of texts and mapping information.

   ```python
   # Part A: Collect texts from real incidents
   incident_texts = []
   incident_mapping_info = []
   for ev_id, incident_data in refined_data.items():
       # Add the main narrative text
       incident_texts.append(incident_data['narrative_text_column'])
       incident_mapping_info.append({'source': 'incident', 'ev_id': ev_id, 'type': 'narrative'})

       # Add each event text in the sequence
       for event in incident_data.get('sequence_of_events', []):
           incident_texts.append(event['event_text_column'])
           incident_mapping_info.append({'source': 'incident', 'ev_id': ev_id, 'seq_no': event['sequence_number_column'], 'type': 'event'})

   # Part B: Collect texts from the master dictionary
   # Assuming df_dictionary has columns 'code' and 'meaning'
   dictionary_texts = df_dictionary['meaning'].tolist()
   dictionary_mapping_info = [{'source': 'dictionary', 'code': row['code'], 'text': row['meaning']} for index, row in df_dictionary.iterrows()]
   ```

4. **Combine Text and Mapping Lists:**

   - Combine the two lists into a single master list for each.

   ```python
   all_texts = incident_texts + dictionary_texts
   all_mapping_info = incident_mapping_info + dictionary_mapping_info
   ```

5. **Embed in Batches:**

   - Send the combined `all_texts` list to the embedding API. This is the long process that creates the vectors for your entire knowledge base.

   ```python
   from openai import OpenAI
   from tqdm import tqdm

   client = OpenAI(api_key="YOUR_API_KEY")
   all_embeddings = []
   batch_size = 100

   for i in tqdm(range(0, len(all_texts), batch_size)):
       batch = all_texts[i:i + batch_size]
       response = client.embeddings.create(input=batch, model="text-embedding-3-small")
       embeddings = [item.embedding for item in response.data]
       all_embeddings.extend(embeddings)
   ```

6. **Save the Results:**

   - Save the final combined embeddings array and the combined mapping list.

   ```python
   import numpy as np
   import json

   np.save('embeddings.npy', np.array(all_embeddings))

   with open('embeddings_map.json', 'w') as f:
       json.dump(all_mapping_info, f)
   ```

   **Result:** Two files that represent your entire knowledge base: `embeddings.npy` (the vectors) and `embeddings_map.json` (the metadata).

---

### **Phase 3: Build the Diagnosis Application**

**Goal:** To create the main program that takes a user query and produces the visual diagnostic chart by searching the combined embedding memory.

**Tools:** Python, `numpy`, `scikit-learn`.

**Steps:**

1. **Create Your Main Script:**

   - Create a file named `main_app.py`.
   - Install scikit-learn: `pip install scikit-learn`.

2. **Load All Processed Data:**

   - At the start of the script, load the three files: `refined_dataset.json`, `embeddings.npy`, and `embeddings_map.json`.

3. **Create Core Functions:**

   - `get_embedding(text)`: Gets a single embedding for the user's query text.
   - `find_top_matches(query_embedding, top_n=50)`: Uses `cosine_similarity` to find the top N matches from `embeddings.npy`. It should return both the scores and the original mapping info for these matches.
   - `get_causal_chains(top_incident_events)`: Takes a list of top **incident** events and extracts their causal histories from `refined_dataset.json`.
   - `build_network_chart(chains)`: Generates the MermaidJS code from the causal chains.

4. **Implement the Main Logic Flow:**

   - Prompt the user for an incident description.
   - Get the embedding for the user's text.
   - Find the top matches (e.g., top 50) from your combined memory.
   - **Implement the Source and Confidence Check:**

     1. Get the metadata and score for the **single best match** from your search results.
     2. First, check the similarity score. If it's too low (e.g., `< 0.7`), trigger the fallback logic regardless of the source.
     3. Next, check the `source` field of the best match's metadata.
     4. **If `source` is `'incident'**: You have relevant historical data! Proceed with the diagnosis:
        - Filter your top 50 matches to keep only those with `source == 'incident'`.
        - Pass these to `get_causal_chains()`.
        - Pass the chains to `build_network_chart()`.
        - Present the MermaidJS chart.
     5. **If `source` is `'dictionary'**: You understand the concept but have no direct historical data. Trigger the fallback logic:
        - Inform the user: "Your query matches the standard event '[event text]'. While no high-confidence historical incidents match this directly, here is an analysis based on general aviation knowledge."
        - Call an LLM with the specialized fallback prompt.

   - Print the final output (either the MermaidJS string or the fallback analysis).

---

### **Phase 4: Visualization**

**Goal:** To render the visual output from your program.

**Steps:**

1. **Copy the Output:**

   - Run `python main_app.py` and provide an input.
   - Copy the entire block of MermaidJS code that it prints.

2. **Paste into a Viewer:**

   - Go to a free online editor like `mermaid.live`.
   - Paste the code into the editor to see your rendered Bayesian network diagram.

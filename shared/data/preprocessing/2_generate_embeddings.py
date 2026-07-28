import argparse
import sys
from pathlib import Path

# Parse CLI before heavy scientific / API imports so `--help` stays responsive.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    DATA_DIR,
    REFINED_DATA_PATH,
    TRAIN_MERGED_DATA_PATH,
    EMBEDDINGS_PATH,
    EMBEDDINGS_MAP_PATH,
    EMBEDDINGS_TRAIN_PATH,
    EMBEDDINGS_MAP_TRAIN_PATH,
    EMBEDDINGS_CHECKPOINT_TRAIN_PATH,
)

_parser = argparse.ArgumentParser(description="Embed merged incidents + ct_seqevt dictionary rows.")
_parser.add_argument(
    "--train",
    action="store_true",
    help="Read train-only merged JSON and write embeddings_train.npy / embeddings_map_train.json (never overwrites full-corpus outputs).",
)
_cli = _parser.parse_args()

import json
import os
import time

import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm

API_KEY = OPENAI_API_KEY
if not API_KEY:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

BATCH_SIZE = 100
CHECKPOINT_INTERVAL = 500  # Save checkpoint every 500 embeddings
MAX_RETRIES = 3  # Retry failed requests up to 3 times
RETRY_DELAY = 5  # Wait 5 seconds between retries

# --- File Paths ---
_merged_in = TRAIN_MERGED_DATA_PATH if _cli.train else REFINED_DATA_PATH
if _cli.train:
    EMBEDDINGS_OUTPUT_PATH = EMBEDDINGS_TRAIN_PATH
    MAP_OUTPUT_PATH = EMBEDDINGS_MAP_TRAIN_PATH
    CHECKPOINT_PATH = EMBEDDINGS_CHECKPOINT_TRAIN_PATH
else:
    EMBEDDINGS_OUTPUT_PATH = EMBEDDINGS_PATH
    MAP_OUTPUT_PATH = EMBEDDINGS_MAP_PATH
    CHECKPOINT_PATH = DATA_DIR / "embeddings_checkpoint.npz"


# --- Main Script ---
print(
    "Starting Phase 2 (Revised): Causal Embedding Generation"
    + (" — TRAIN (train-only outputs)" if _cli.train else "")
)

# Step 1: Load Your Processed Data Files
print(f"Loading merged dataset from {_merged_in}...")
with open(_merged_in, 'r') as f:
    refined_data = json.load(f)

# Step 2: Collect All Text for Causal Embedding
print("Collecting texts for causal embedding...")
all_texts = []
all_mapping_info = []

# Helper function to clean text
def clean_text(text):
    """Clean text for embedding - remove null characters and ensure valid string"""
    if text is None or text == 'null' or text == 'None':
        return None
    text = str(text).strip()
    # Remove null bytes and other problematic characters
    text = text.replace('\x00', '').replace('\0', '')
    return text if len(text) > 0 else None

# Part A: Collect causal texts from real incidents
for ev_id, incident_data in tqdm(refined_data.items(), desc="Processing Incidents"):
    
    # Tiered Fallback Logic for Narratives
    narr_cause = clean_text(incident_data.get('narr_cause'))
    narr_accf = clean_text(incident_data.get('narr_accf'))
    narr_accp = clean_text(incident_data.get('narr_accp'))

    # 1. Embed narr_cause if it exists
    if narr_cause:
        all_texts.append(narr_cause)
        all_mapping_info.append({'source': 'incident', 'ev_id': ev_id, 'type': 'causal_narrative'})

    # 2. Embed narr_accf if it exists
    if narr_accf:
        all_texts.append(narr_accf)
        all_mapping_info.append({'source': 'incident', 'ev_id': ev_id, 'type': 'factual_narrative'})
    
    # 3. Fallback to narr_accp if both primary narratives are missing
    if not narr_cause and not narr_accf and narr_accp:
        all_texts.append(narr_accp)
        all_mapping_info.append({'source': 'incident', 'ev_id': ev_id, 'type': 'fallback_narrative'})

    # 4. Embed each individual finding
    for finding in incident_data.get('findings', []):
        finding_text = clean_text(finding.get('finding_description'))
        if finding_text:
            all_texts.append(finding_text)
            all_mapping_info.append({
                'source': 'incident', 
                'ev_id': ev_id, 
                'type': 'finding',
                'finding_no': finding.get('finding_no')
            })

# Part B: Collect texts from the master dictionary
print("Re-including master dictionary for completeness...")
_ct_seqevt = DATA_DIR / "ct_seqevt.txt"
if not _ct_seqevt.is_file():
    _ct_seqevt = DATA_DIR.parent / "raw" / "ct_seqevt.txt"
df_dictionary = pd.read_csv(_ct_seqevt, sep=",", header=None, names=['code', 'meaning'])
dictionary_texts = [clean_text(meaning) for meaning in df_dictionary['meaning'].tolist() if clean_text(meaning)]
dictionary_mapping_info = [
    {'source': 'dictionary', 'code': row['code'], 'text': clean_text(row['meaning'])} 
    for index, row in df_dictionary.iterrows() 
    if clean_text(row['meaning'])
]

# Combine all lists
all_texts.extend(dictionary_texts)
all_mapping_info.extend(dictionary_mapping_info)

print(f"Total causal texts collected for embedding: {len(all_texts)}")


# Step 3: Embed in Batches with Checkpoint/Resume
print("Generating causal embeddings in batches...")
client = OpenAI(api_key=API_KEY)

# Check if checkpoint exists (resume from previous run)
if os.path.exists(CHECKPOINT_PATH):
    print("📂 Found checkpoint! Resuming from previous run...")
    checkpoint = np.load(CHECKPOINT_PATH, allow_pickle=True)
    all_embeddings = checkpoint['embeddings'].tolist()
    start_index = int(checkpoint['last_index'])
    print(f"✅ Resuming from index {start_index} ({len(all_embeddings)} embeddings already done)")
else:
    all_embeddings = []
    start_index = 0
    print("🆕 Starting fresh embedding generation...")

# Function to save checkpoint
def save_checkpoint(embeddings, last_idx):
    """Save current progress to checkpoint file"""
    np.savez_compressed(CHECKPOINT_PATH, 
                       embeddings=np.array(embeddings),
                       last_index=last_idx)

# Embedding loop with error handling and checkpoints
try:
    for i in tqdm(range(start_index, len(all_texts), BATCH_SIZE), desc="Generating Embeddings"):
        batch = all_texts[i:i + BATCH_SIZE]
        
        # Retry logic for API failures
        for attempt in range(MAX_RETRIES):
            try:
                response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)
                break  # Success, exit retry loop
                
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"\n⚠️  Error at batch {i} (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                    print(f"   Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"\n❌ Failed after {MAX_RETRIES} attempts at batch {i}")
                    print(f"   Error: {e}")
                    print("   Saving checkpoint before exiting...")
                    save_checkpoint(all_embeddings, i)
                    raise
        
        # Save checkpoint periodically
        if len(all_embeddings) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(all_embeddings, i + BATCH_SIZE)
            
except KeyboardInterrupt:
    print("\n\n⚠️  Process interrupted by user (Ctrl+C)")
    print("   Saving checkpoint...")
    save_checkpoint(all_embeddings, i)
    print(f"   ✅ Checkpoint saved. Run script again to resume from {len(all_embeddings)} embeddings.")
    exit(0)


# Step 4: Save the Final Results
print("\n💾 Saving the embeddings and mapping file...")
np.save(EMBEDDINGS_OUTPUT_PATH, np.array(all_embeddings))
with open(MAP_OUTPUT_PATH, 'w') as f:
    json.dump(all_mapping_info, f, indent=4)

# Clean up checkpoint file after successful completion
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
    print("✅ Checkpoint file cleaned up")

print(f"\n🎉 Embedding generation complete!")
print(f"📊 Total embeddings generated: {len(all_embeddings)}")
print(f"📁 Embeddings saved to: {EMBEDDINGS_OUTPUT_PATH}")
print(f"📁 Mapping info saved to: {MAP_OUTPUT_PATH}")
print("✅ Phase 2 is complete.")

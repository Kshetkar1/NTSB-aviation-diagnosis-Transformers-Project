
import json
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDINGS_MAP_PATH

print(f"Reading {EMBEDDINGS_MAP_PATH}...")
with open(EMBEDDINGS_MAP_PATH, 'r') as f:
    # Read first 1000 chars to avoid loading whole file
    # Actually, let's load it, it's just JSON.
    data = json.load(f)

print(f"Total entries: {len(data)}")
print("First 5 entries:")
for i, item in enumerate(data[:5]):
    print(f"[{i}] Source: {item.get('source')} | Type: {item.get('type')} | ID: {item.get('ev_id')}")

print("\nChecking for 'narrative' types:")
count = 0
for item in data:
    if item.get('source') == 'incident' and 'narrative' in item.get('type', ''):
        count += 1
print(f"Found {count} entries matching filter criteria.")

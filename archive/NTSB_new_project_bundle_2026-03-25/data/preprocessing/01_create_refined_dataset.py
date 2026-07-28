# import libraries
import pandas as pd
import json
import numpy as np
from pathlib import Path
import sys

# Import config from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

print("Starting Data loading process now! ")

# --- STEP 1: LOAD THE DATA ---
# It's good practice to wrap this in a try/except block to catch file errors
try:
    # Excel Files are generally reliable to load
    df_aircraft = pd.read_excel(DATA_DIR / "aircraft.xlsx")
    df_engines = pd.read_excel(DATA_DIR / "engines.xlsx")
    df_events = pd.read_excel(DATA_DIR / "events.xlsx")
    df_findings = pd.read_excel(DATA_DIR / "findings.xlsx")
    df_injury = pd.read_excel(DATA_DIR / "injury.xlsx")
    df_narratives = pd.read_excel(DATA_DIR / "narratives.xlsx")

    # Text files need specific separators. Most NTSB files use TABS (`\t`).
    # The dictionary file `ct_seqevt.txt` is an exception and uses a COMMA.
    df_ct_seqevt = pd.read_csv(DATA_DIR / "ct_seqevt.txt", sep=",", low_memory=False, quotechar='"')
    # Correcting file names from .csv to .txt and using tab separator
    df_Events_sequences = pd.read_csv(DATA_DIR / "Events_Sequence.txt", sep='\t', low_memory=False, quotechar='"')
    df_occurences = pd.read_csv(DATA_DIR / "Occurrences.txt", sep='\t', low_memory=False, quotechar='"')
    df_seq_of_events = pd.read_csv(DATA_DIR / "seq_of_events.txt", sep='\t', low_memory=False, quotechar='"')
    print("\n✅ All files loaded successfully.")

except FileNotFoundError as e:
    print(f"\n❌ ERROR: Could not find a file. Please check the path.")
    print(e)
except Exception as e:
    print(f"\n❌ An unexpected error occurred during file loading: {e}")


# --- STEP 2: SELECT AND CLEAN IMPORTANT COLUMNS ---
print("\n--- Cleaning and selecting important columns ---")

# Define the comprehensive list of columns we want to keep
events_cols = ['ev_id', 'ev_date', 'ev_time', 'ev_city', 'ev_state', 'ev_country', 'light_cond', 'sky_cond_nonceil', 'vis_sm', 'wx_temp', 'wind_dir_deg', 'wind_vel_kts', 'gust_kts', 'wx_cond_basic', 'ev_highest_injury']
aircraft_cols = ['ev_id', 'damage', 'acft_fire', 'acft_expl', 'acft_make', 'acft_model', 'total_seats', 'num_eng', 'type_last_insp', 'afm_hrs']
narratives_cols = ['ev_id', 'narr_accp', 'narr_accf', 'narr_cause']
engines_cols = ['ev_id', 'eng_no', 'eng_type', 'eng_mfgr', 'eng_model', 'power_units', 'propeller_type']
injury_cols = ['ev_id', 'inj_person_category', 'injury_level', 'inj_person_count']
findings_cols = ['ev_id', 'finding_no', 'finding_description', 'Cause_Factor']
sequence_cols = [col for col in df_Events_sequences.columns if 'lchg' not in col] # Keep all except metadata
dictionary_cols = ['code', 'meaning']

# Create new, clean DataFrames. Using .reindex() is a safe way to select columns that may not exist.
df_events_clean = df_events.reindex(columns=events_cols).copy()
df_aircraft_clean = df_aircraft.reindex(columns=aircraft_cols).copy()
df_narratives_clean = df_narratives.reindex(columns=narratives_cols).copy()
df_engines_clean = df_engines.reindex(columns=engines_cols).copy()
df_injury_clean = df_injury.reindex(columns=injury_cols).copy()
df_findings_clean = df_findings.reindex(columns=findings_cols).copy()
df_sequences_clean = df_Events_sequences.reindex(columns=sequence_cols).copy()
df_dictionary_clean = df_ct_seqevt.reindex(columns=dictionary_cols).copy()

# --- STEP 3: HANDLE MISSING DATA ('fillna') ---
print("\n--- Handling missing values ---")
# For text columns in the AIRCRAFT dataframe, fill with "Unknown"
for col in ['acft_make', 'acft_model']:
    df_aircraft_clean[col].fillna("Unknown", inplace=True)

# For text columns in the EVENTS dataframe, fill with "Unknown"
for col in ['light_cond', 'wx_cond_basic']:
    df_events_clean[col].fillna("Unknown", inplace=True)

# For numeric columns, fill with np.nan which is the standard
for col in ['afm_hrs', 'total_seats', 'num_eng']:
     df_aircraft_clean[col].fillna(np.nan, inplace=True)

# --- STEP 4: GROUP ALL ONE-TO-MANY DATA BY EV_ID ---
print("\n--- Grouping detailed data by incident ---")
def group_to_list(df, group_col, sort_by=None, result_name=None):
    if sort_by:
        df = df.sort_values(by=sort_by)
    result = df.groupby(group_col).apply(lambda x: x.to_dict('records')).reset_index()
    if result_name:
        result = result.rename(columns={0: result_name})
    return result

grouped_sequences = group_to_list(df_sequences_clean, 'ev_id', sort_by=['ev_id', 'Occurrence_No'], result_name='sequence_of_events')
grouped_findings = group_to_list(df_findings_clean, 'ev_id', sort_by=['ev_id', 'finding_no'], result_name='findings')
grouped_injuries = group_to_list(df_injury_clean, 'ev_id', result_name='injuries')
grouped_engines = group_to_list(df_engines_clean, 'ev_id', sort_by=['ev_id', 'eng_no'], result_name='engines')

# --- STEP 5: COMBINE ALL DATA INTO A FINAL STRUCTURE ---
print("\n--- Merging all data into the final structure ---")
# Start with event details as the base, since it's the most central file
base_df = df_events_clean.drop_duplicates(subset=['ev_id']).copy()

# List of all dataframes to merge
data_to_merge = [
    df_narratives_clean.drop_duplicates(subset=['ev_id']),
    df_aircraft_clean.drop_duplicates(subset=['ev_id']),
    grouped_sequences,
    grouped_findings,
    grouped_injuries,
    grouped_engines
]

final_df = base_df
for df in data_to_merge:
    final_df = pd.merge(final_df, df, on='ev_id', how='left')

# If an incident was missing a category, fill it with an empty list for consistency
for col in ['sequence_of_events', 'findings', 'injuries', 'engines']:
    final_df[col] = final_df[col].apply(lambda x: x if isinstance(x, list) else [])

# --- STEP 6: CREATE AND SAVE THE FINAL JSON FILE ---
print("\n--- Creating and saving the final JSON file ---")

# Convert timestamps and clean NaN for JSON compatibility
def clean_for_json(obj):
    """
    Converts timestamps to strings and NaN to None (null in JSON)
    This is REQUIRED for valid JSON format
    """
    if hasattr(obj, 'isoformat'):  # pandas Timestamp objects
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: clean_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif pd.isna(obj):  # Handles np.nan, pd.NaT, etc. - REQUIRED for JSON
        return None
    elif isinstance(obj, str) and obj.lower() == 'nan':  # String "nan"
        return None
    else:
        return obj

final_df.set_index('ev_id', inplace=True)
final_data_dict = final_df.to_dict(orient='index')

# Clean data for JSON compatibility (timestamps and NaN)
print("--- Converting timestamps and NaN for JSON compatibility ---")
final_data_dict = clean_for_json(final_data_dict)

output_filename = DATA_DIR / 'refined_dataset.json'
with open(output_filename, 'w') as f:
    json.dump(final_data_dict, f, indent=4)

print(f"\n✅ Success! Your refined dataset has been saved to '{output_filename}'")


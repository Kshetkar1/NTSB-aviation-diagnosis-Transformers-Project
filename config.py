"""
Configuration file for the Aviation Incident Diagnosis Engine.

This file centralizes all configuration settings including API keys,
file paths, and model settings. API keys should be set via environment
variables for security.
"""

import os
from pathlib import Path

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, that's okay - user can use environment variables
    pass

# --- Project Root Directory ---
# Automatically detect project root (directory containing this config.py file)
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "Data"

# --- OpenAI API Configuration ---
# Get API key from environment variable (recommended)
# Note: Validation happens when API is actually used, not on import
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_openai_api_key():
    """Get OpenAI API key with validation. Call this when actually using the API."""
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it using: export OPENAI_API_KEY='your-key-here' "
            "or create a .env file (see .env.example)"
        )
    return OPENAI_API_KEY

# --- Model Configuration ---
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # Default to gpt-4o-mini
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # Default embedding model

# --- Data File Paths ---
# All paths are relative to PROJECT_ROOT
REFINED_DATA_PATH = DATA_DIR / "refined_dataset.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
EMBEDDINGS_MAP_PATH = DATA_DIR / "embeddings_map.json"
BAYESIAN_STATS_PATH = DATA_DIR / "bayesian_cause_statistics.json"

# --- Output Paths ---
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)  # Create output directory if it doesn't exist

# --- Validation ---
def validate_paths():
    """Validate that required data files exist."""
    required_files = [
        REFINED_DATA_PATH,
        EMBEDDINGS_PATH,
        EMBEDDINGS_MAP_PATH,
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Required data files not found: {[str(f) for f in missing_files]}. "
            f"Please run the data processing scripts first."
        )

# Optional: Validate paths on import (comment out if data files don't exist yet)
# validate_paths()


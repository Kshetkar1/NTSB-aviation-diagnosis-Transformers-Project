# Security & Configuration Changes

This document explains the changes made to improve security and configuration management.

## Changes Made

### 1. Centralized Configuration (`config.py`)
- Created a single `config.py` file that manages all configuration
- All API keys now loaded from environment variables
- All file paths are now relative to project root (no hardcoded absolute paths)

### 2. Environment Variables for API Keys
- **Before:** API keys were hardcoded in `agent.py` and `main_app.py`
- **After:** API keys are loaded from `OPENAI_API_KEY` environment variable
- Supports `.env` file (via python-dotenv) or system environment variables

### 3. Relative File Paths
- **Before:** Some scripts used absolute paths like `/Users/kanushetkar/Desktop/...`
- **After:** All paths are relative to project root using `Path` objects
- Paths automatically resolve based on where `config.py` is located

### 4. Files Updated
- ✅ `config.py` - New centralized configuration file
- ✅ `agent.py` - Now imports from config
- ✅ `main_app.py` - Now imports from config
- ✅ `Data/2_generate_embeddings.py` - Uses config paths
- ✅ `Data/01_create_refined_dataset.py` - Uses relative paths from config
- ✅ `Data/2b_precompute_bayesian_data.py` - Uses config paths
- ✅ `.gitignore` - Added to prevent committing `.env` file
- ✅ `.env.example` - Template for environment variables
- ✅ `requirements.txt` - Added `python-dotenv` for .env support

## Setup Instructions

### Option 1: Using .env file (Recommended)

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API key:
   ```
   OPENAI_API_KEY=your-actual-api-key-here
   ```

3. The `.env` file is automatically loaded (and ignored by git)

### Option 2: Using Environment Variables

Set the environment variable in your terminal:

**Linux/Mac:**
```bash
export OPENAI_API_KEY='your-actual-api-key-here'
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY='your-actual-api-key-here'
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=your-actual-api-key-here
```

## Benefits

1. **Security:** API keys are no longer exposed in code files
2. **Portability:** Works on any machine without editing code
3. **Version Control Safe:** `.env` is gitignored, so keys won't be committed
4. **Flexibility:** Easy to switch between different API keys or models
5. **Best Practices:** Follows industry standards for configuration management

## Troubleshooting

**Error: "OPENAI_API_KEY environment variable not set"**
- Make sure you've set the environment variable or created a `.env` file
- Check that `python-dotenv` is installed: `pip install python-dotenv`
- Verify your `.env` file is in the project root directory

**Error: "Required data files not found"**
- This is normal if you haven't run the data processing scripts yet
- The validation in `config.py` is commented out by default
- Uncomment `validate_paths()` if you want strict validation

## Migration Notes

If you had hardcoded API keys before:
1. Remove the old hardcoded keys from your code (already done)
2. Set up your `.env` file or environment variable
3. Test that everything still works

All existing functionality remains the same - only the configuration method has changed.


#!/bin/bash
# Run the Frozen-BN diagnosis + prognosis Streamlit demo (current paper path).
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY311=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11

if [ -z "$OPENAI_API_KEY" ]; then
  echo "⚠️  OPENAI_API_KEY is not set — query-first retrieval (embeddings) will fail."
fi

export PYTHONPATH="${ROOT}/shared/code:${ROOT}/Frozen-BN-Narrative-Evidence-2026-07-20/code:${PYTHONPATH:-}"

APP="${ROOT}/Frozen-BN-Narrative-Evidence-2026-07-20/apps/frozenbn_streamlit_diagnosis_prognosis_demo.py"
if [ ! -f "$APP" ]; then
  APP="${ROOT}/Frozen-BN-Narrative-Evidence-2026-07-20/apps/streamlit_tree_app.py"
fi

if [ "$1" == "legacy" ]; then
  echo "🚀 Legacy embedding Streamlit app..."
  exec "$PY311" -m streamlit run "${ROOT}/Embedding-Similarity-Counting-Path-2026-03-25/apps/embed_streamlit_similarity_diagnosis_app.py" 2>/dev/null \
    || exec "$PY311" -m streamlit run "${ROOT}/Embedding-Similarity-Counting-Path-2026-03-25/apps/streamlit_app.py"
fi

echo "🌳 Frozen-BN diagnosis + prognosis demo..."
exec "$PY311" -m streamlit run "$APP"

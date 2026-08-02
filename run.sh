#!/usr/bin/env bash
# Launch FastAPI + Streamlit together. LLM calls go to the Hugging Face
# Inference API by default, so there's no local model server to start.
set -e
source .venv/bin/activate 2>/dev/null || true

# 1) FastAPI backend
echo "==> Starting FastAPI on :8000"
uvicorn backend.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
BACK_PID=$!
sleep 3

# 2) Streamlit frontend
echo "==> Starting Streamlit on :8501"
streamlit run frontend/app.py --server.port 8501

kill $BACK_PID 2>/dev/null || true

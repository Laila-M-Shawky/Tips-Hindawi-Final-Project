#!/usr/bin/env bash
# Launch Ollama + FastAPI + Streamlit together.
set -e
source .venv/bin/activate 2>/dev/null || true

# 1) Ollama server (skip if already running)
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "==> Starting Ollama"
  ollama serve > ollama.log 2>&1 &
  sleep 3
fi

# 2) FastAPI backend
echo "==> Starting FastAPI on :8000"
uvicorn backend.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
BACK_PID=$!
sleep 3

# 3) Streamlit frontend
echo "==> Starting Streamlit on :8501"
streamlit run frontend/app.py --server.port 8501

kill $BACK_PID 2>/dev/null || true

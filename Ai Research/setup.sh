#!/usr/bin/env bash
# One-shot setup for ResearchAI.
set -e

echo "==> Creating virtual environment"
python3 -m venv .venv
source .venv/bin/activate

echo "==> Installing Python dependencies"
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Preparing .env"
[ -f .env ] || cp .env.example .env

echo "==> Checking Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama not found. Install from https://ollama.com/download, then rerun."
else
  echo "Pulling models (this may take a while)…"
  ollama pull llama3.1:8b || echo "Pull llama3.1:8b manually if this failed."
  # ollama pull qwen2.5:7b   # uncomment to also grab Qwen
fi

echo ""
echo "✅ Setup complete."
echo "   Start everything with:  ./run.sh"

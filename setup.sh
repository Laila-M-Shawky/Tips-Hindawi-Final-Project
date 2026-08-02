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

echo ""
echo "⚠️  Set HF_TOKEN in .env — get a free token at https://huggingface.co/settings/tokens"
echo "   LLM_BACKEND=hf_api by default, so the model runs on Hugging Face's"
echo "   servers, not this machine — no GPU or multi-GB download required."
echo ""
echo "✅ Setup complete."
echo "   Start everything with:  ./run.sh"

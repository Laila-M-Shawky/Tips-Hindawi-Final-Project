"""
Expose ResearchAI publicly with ngrok (great for demoing from Colab or a laptop).

Usage:
    python scripts/ngrok_deploy.py --port 8501         # tunnel Streamlit
    python scripts/ngrok_deploy.py --port 8000         # tunnel the API

Set NGROK_AUTHTOKEN in your .env (get one free at https://dashboard.ngrok.com).

Typical Colab demo flow:
    1. ollama serve &                      (start the model server)
    2. uvicorn backend.main:app --port 8000 &
    3. streamlit run frontend/app.py --server.port 8501 &
    4. python scripts/ngrok_deploy.py --port 8501
"""
from __future__ import annotations

import argparse
import time

from pyngrok import conf, ngrok

from config.settings import settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--proto", default="http")
    args = ap.parse_args()

    if settings.ngrok_authtoken:
        conf.get_default().auth_token = settings.ngrok_authtoken

    tunnel = ngrok.connect(args.port, args.proto)
    print("=" * 60)
    print(f"🌍 Public URL: {tunnel.public_url}")
    print(f"➡️  Forwarding to http://localhost:{args.port}")
    print("=" * 60)
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ngrok.disconnect(tunnel.public_url)
        ngrok.kill()
        print("Tunnel closed.")


if __name__ == "__main__":
    main()

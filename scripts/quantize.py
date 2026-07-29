"""
Model quantization / optimization for local inference via Ollama.

Ollama serves already-quantized GGUF weights. This script:
  1. Pulls a quantization level you choose (Q4_K_M is the sweet spot for 8B on
     consumer GPUs / CPU; Q8_0 if you have headroom and want max quality).
  2. Writes a Modelfile that fixes the context window and sampling params so the
     app's behavior is reproducible.
  3. Builds a named model (e.g. researchai-llama) you point LLM_MODEL at.

Usage:
    python scripts/quantize.py --base llama3.1:8b --quant q4_K_M --name researchai-llama
    python scripts/quantize.py --base qwen2.5:7b  --quant q4_K_M --name researchai-qwen

Notes on quantization levels (approx, 8B model):
    q8_0   ~8.5GB  highest quality, needs a real GPU
    q5_K_M ~5.7GB  strong quality
    q4_K_M ~4.9GB  recommended default (quality/size balance)
    q3_K_M ~4.0GB  runs on modest hardware, some quality loss
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

MODELFILE_TEMPLATE = """FROM {tag}

# Reproducible sampling for a research assistant (low temperature, long context)
PARAMETER temperature 0.1
PARAMETER num_ctx {num_ctx}
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

SYSTEM \"\"\"You are ResearchAI, a rigorous scientific assistant that answers
strictly from provided context and always cites sources.\"\"\"
"""


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="llama3.1:8b",
                    help="base model, e.g. llama3.1:8b or qwen2.5:7b")
    ap.add_argument("--quant", default="q4_K_M",
                    help="quantization suffix, e.g. q4_K_M, q5_K_M, q8_0")
    ap.add_argument("--name", default="researchai-llama")
    ap.add_argument("--num_ctx", type=int, default=8192)
    args = ap.parse_args()

    # Many Ollama models publish quantized tags like "llama3.1:8b-instruct-q4_K_M".
    family = args.base.split(":")[0]
    size = args.base.split(":")[1] if ":" in args.base else "8b"
    tag = f"{family}:{size}-instruct-{args.quant}"

    print(f"Pulling quantized tag: {tag}")
    try:
        run(["ollama", "pull", tag])
    except subprocess.CalledProcessError:
        print(f"⚠️  '{tag}' not found; falling back to base '{args.base}'.")
        run(["ollama", "pull", args.base])
        tag = args.base

    modelfile = MODELFILE_TEMPLATE.format(tag=tag, num_ctx=args.num_ctx)
    with tempfile.TemporaryDirectory() as tmp:
        mf = Path(tmp) / "Modelfile"
        mf.write_text(modelfile)
        run(["ollama", "create", args.name, "-f", str(mf)])

    print("=" * 60)
    print(f"✅ Built '{args.name}'.  Set LLM_MODEL={args.name} in your .env")
    print("=" * 60)


if __name__ == "__main__":
    main()

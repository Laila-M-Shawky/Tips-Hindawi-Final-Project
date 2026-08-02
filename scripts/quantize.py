"""
Local model quantization — ADVANCED / OPTIONAL.

Only relevant if you set LLM_BACKEND=hf_local (running the LLM on your own GPU
instead of the default Hugging Face Inference API). This downloads a base model
in 4-bit (bitsandbytes NF4) and saves the quantized weights locally so future
loads are faster and use far less VRAM.

Usage:
    python scripts/quantize.py --base Qwen/Qwen2.5-7B-Instruct --out models/researchai-qwen-4bit
    python scripts/quantize.py --base meta-llama/Llama-3.1-8B-Instruct --out models/researchai-llama-4bit

Then in .env:
    LLM_BACKEND=hf_local
    HF_MODEL=models/researchai-qwen-4bit
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct",
                    help="Hugging Face model id to quantize")
    ap.add_argument("--out", default="models/researchai-quantized",
                    help="local directory to save the quantized model to")
    args = ap.parse_args()

    # Imports are local so the default hf_api install doesn't need these heavy deps.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading '{args.base}' in 4-bit (NF4) ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb_config, device_map="auto"
    )

    print(f"Saving quantized weights to '{args.out}' ...")
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    print("=" * 60)
    print(f"✅ Saved quantized model to '{args.out}'.")
    print(f"   Set HF_MODEL={args.out} and LLM_BACKEND=hf_local in your .env")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Optional: LoRA / QLoRA fine-tuning of the base model on a domain corpus.

This is an ADVANCED module — the system works fully without it. Use it only if
you want to specialize the model on, say, biomedical or NLP-paper style Q&A and
have a GPU (Colab T4/A100 is enough for QLoRA on a 7-8B model).

Data format: a JSONL file where each line is
    {"instruction": "...", "input": "...", "output": "..."}

Pipeline (QLoRA):
    - load base in 4-bit (bitsandbytes)
    - attach LoRA adapters to attention/MLP projections
    - train with TRL's SFTTrainer
    - merge adapters, export to GGUF, and register with Ollama

Usage:
    python scripts/finetune_lora.py --base meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --data data/train.jsonl --out out/researchai-lora --qlora
"""
from __future__ import annotations

import argparse


def build_prompt(ex: dict) -> str:
    inp = f"\n\nContext:\n{ex['input']}" if ex.get("input") else ""
    return (f"<|system|>You are ResearchAI, cite your sources.\n"
            f"<|user|>{ex['instruction']}{inp}\n<|assistant|>{ex['output']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True, help="JSONL of instruction/input/output")
    ap.add_argument("--out", default="out/researchai-lora")
    ap.add_argument("--qlora", action="store_true", help="4-bit QLoRA (saves VRAM)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    # Imports are local so the rest of the project doesn't require these heavy deps.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from trl import SFTConfig, SFTTrainer

    quant_cfg = None
    if args.qlora:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=quant_cfg,
        device_map="auto", torch_dtype=torch.bfloat16,
    )

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.map(lambda ex: {"text": build_prompt(ex)})

    trainer = SFTTrainer(
        model=model, train_dataset=ds, peft_config=lora,
        processing_class=tok,
        args=SFTConfig(output_dir=args.out, num_train_epochs=args.epochs,
                       per_device_train_batch_size=2, gradient_accumulation_steps=4,
                       learning_rate=args.lr, logging_steps=10, bf16=True,
                       save_strategy="epoch", max_length=1024),
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"✅ Adapters saved to {args.out}")
    print(
        "\nNext — merge + export to Ollama:\n"
        "  1. Merge adapters:  model.merge_and_unload() then save_pretrained()\n"
        "  2. Convert to GGUF: python llama.cpp/convert_hf_to_gguf.py <merged>\n"
        "  3. Quantize:        ./llama-quantize model.gguf model.Q4_K_M.gguf Q4_K_M\n"
        "  4. Register:        ollama create researchai-ft -f Modelfile "
        "(FROM ./model.Q4_K_M.gguf)\n"
        "  5. Set LLM_MODEL=researchai-ft in .env"
    )


if __name__ == "__main__":
    main()

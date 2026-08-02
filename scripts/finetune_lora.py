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
    - merge adapters and save locally, or push to the Hugging Face Hub, then
      point HF_MODEL (with LLM_BACKEND=hf_local, or hf_api if pushed publicly/
      with an access token) at the result

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
        "\nNext — merge and use the fine-tuned model:\n"
        "  1. Merge adapters: model = model.merge_and_unload()\n"
        f"  2. Save it:        model.save_pretrained('{args.out}/merged')\n"
        "  3. Local use:       set LLM_BACKEND=hf_local and "
        f"HF_MODEL={args.out}/merged in .env\n"
        "  4. (optional) push to the Hub: model.push_to_hub('you/researchai-ft') "
        "then set HF_MODEL=you/researchai-ft"
    )


if __name__ == "__main__":
    main()

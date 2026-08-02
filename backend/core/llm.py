"""
LLM access (remote APIs + optional local) + shared generation helpers.

Three interchangeable backends, picked by `settings.llm_backend`:
  - "groq"     (default) calls the Groq API — free, and fast even for 70B
    models. Recommended if Hugging Face's Inference Providers free credits
    run out (a common snag — see README troubleshooting).
  - "hf_api"   calls the Hugging Face Inference API instead.
  - "hf_local" loads the model with `transformers` on your own GPU (optionally
    4-bit via bitsandbytes). Opt-in only, for machines that can handle it.

All three run remotely except hf_local, so nothing heavy loads on this machine
by default — the fix for a laptop that lags or freezes trying to serve an 8B
model locally (e.g. via Ollama).

Every feature module talks to the LLM through `chat` / `structured_json` /
`grounded_answer`, so swapping backends here doesn't touch the rest of the app.
"""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from typing import Any, Callable

from config.settings import settings
from backend.core.retrieval import Retrieved

ChatFn = Callable[[str, str, float], str]


def _retry_wait_seconds(resp, default: float = 2.0, cap: float = 30.0) -> float:
    """Groq's 429s carry a Retry-After header or a 'try again in Xs' message —
    prefer either over guessing, since the free tier's per-minute token limit
    resets on a rolling window, not a fixed clock tick."""
    ra = resp.headers.get("retry-after")
    if ra:
        try:
            return min(float(ra), cap)
        except ValueError:
            pass
    m = re.search(r"try again in ([\d.]+)s", resp.text)
    if m:
        return min(float(m.group(1)), cap)
    return default


def _groq_backend() -> ChatFn:
    import requests

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {settings.groq_api_key}",
               "Content-Type": "application/json"}
    max_retries = 3  # a single /ask chains several LLM calls (expand, grade,
                      # generate, verify) and can bump into the free-tier's
                      # per-minute token cap mid-pipeline — retry through it
                      # rather than failing the whole request.

    def call(system: str, user: str, temperature: float) -> str:
        payload = {
            "model": settings.groq_model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": max(temperature, 0.01),
            "max_tokens": settings.llm_max_tokens,
        }
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                r = requests.post(url, headers=headers, json=payload,
                                   timeout=settings.llm_timeout)
                if r.status_code == 429 and attempt < max_retries:
                    time.sleep(_retry_wait_seconds(r))
                    continue
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
            except requests.exceptions.RequestException as e:
                last_error = e
                if getattr(e, "response", None) is not None and e.response.status_code == 429 \
                        and attempt < max_retries:
                    time.sleep(_retry_wait_seconds(e.response))
                    continue
                break
        body = last_error.response.text[:300] if getattr(last_error, "response", None) is not None else ""
        raise RuntimeError(
            f"Groq API call failed for model '{settings.groq_model}': {last_error}\n{body}\n"
            "Check GROQ_API_KEY is set (https://console.groq.com/keys) and that "
            "GROQ_MODEL is a currently supported model id."
        ) from last_error

    return call


def _hf_api_backend() -> ChatFn:
    from huggingface_hub import InferenceClient

    client = InferenceClient(model=settings.hf_model, token=settings.hf_token or None,
                              timeout=settings.llm_timeout)

    def call(system: str, user: str, temperature: float) -> str:
        try:
            completion = client.chat_completion(
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=settings.llm_max_tokens,
                temperature=max(temperature, 0.01),  # 0 is rejected by some providers
            )
        except Exception as e:
            # HF routes chat_completion through Inference Providers, and not every
            # model is hosted by a provider your account has enabled — this is the
            # single most common setup failure, so surface it clearly instead of
            # letting a bare 500 reach the frontend.
            raise RuntimeError(
                f"Hugging Face Inference API call failed for model "
                f"'{settings.hf_model}': {e}\n"
                "Likely causes: the model isn't available on any Inference Provider "
                "enabled for your token, it's gated and you haven't accepted its "
                "license on the model page, or HF_TOKEN is missing/invalid. Try a "
                "different HF_MODEL (e.g. meta-llama/Llama-3.1-8B-Instruct) in .env."
            ) from e
        return completion.choices[0].message.content.strip()

    return call


def _hf_local_backend() -> ChatFn:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    quant_kwargs: dict[str, Any] = {}
    if settings.hf_local_4bit:
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    use_cuda = settings.embed_device == "cuda" and torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(settings.hf_model)
    model = AutoModelForCausalLM.from_pretrained(
        settings.hf_model,
        device_map="auto" if use_cuda else None,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        **quant_kwargs,
    )
    if not use_cuda:
        model.to("cpu")

    def call(system: str, user: str, temperature: float) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=settings.llm_max_tokens,
            do_sample=temperature > 0.01,
            temperature=max(temperature, 0.01),
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return call


@lru_cache(maxsize=1)
def get_llm() -> ChatFn:
    if settings.llm_backend == "hf_local":
        return _hf_local_backend()
    if settings.llm_backend == "hf_api":
        return _hf_api_backend()
    return _groq_backend()


def chat(system: str, user: str, temperature: float | None = None) -> str:
    call = get_llm()
    return call(system, user, settings.llm_temperature if temperature is None else temperature)


def format_context(chunks: list[Retrieved]) -> str:
    """Number the chunks so the model can cite them as [1], [2], ... and we can
    map those back to page-level citations."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        tag = f"[{i}] ({c.citation}, section: {c.metadata.get('section') or 'n/a'})"
        blocks.append(f"{tag}\n{c.text}")
    return "\n\n".join(blocks)


GROUNDED_SYSTEM = (
    "You are ResearchAI, a rigorous scientific assistant. Answer ONLY from the "
    "provided context. Every factual sentence must end with a bracket citation "
    "like [1] or [2] referring to the numbered context blocks. If the context is "
    "insufficient, say so explicitly instead of guessing. Never invent citations."
)


def grounded_answer(question: str, chunks: list[Retrieved]) -> dict[str, Any]:
    context = format_context(chunks)
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Write a precise, well-structured answer. Cite with [n] after each claim."
    )
    answer = chat(GROUNDED_SYSTEM, user)
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    citations = []
    for n in used:
        if 1 <= n <= len(chunks):
            c = chunks[n - 1]
            citations.append(
                {"n": n, "source": c.metadata.get("filename"),
                 "page": c.metadata.get("page"), "snippet": c.text[:220]}
            )
    return {"answer": answer, "citations": citations}


def structured_json(system: str, user: str, retries: int = 1) -> Any:
    """Ask the model for JSON and parse it robustly (models love code fences)."""
    for _ in range(retries + 1):
        raw = chat(system + "\nRespond with valid JSON only, no prose, no code fences.",
                   user)
        parsed = _extract_json(raw)
        if parsed is not None:
            return parsed
    return None


def _extract_json(text: str) -> Any:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                return None
    return None

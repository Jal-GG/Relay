"""Correctness check: compare single-machine vs split-pipeline output.

Run as a script::

    python -m tests.correctness --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

Or as a pytest test (skipped unless RUN_CORRECTNESS=1 env var is set)::

    RUN_CORRECTNESS=1 python -m pytest tests/correctness.py -v -s

The script loads a model, splits it into two InferenceRunners, generates
text greedily from both the full model and the split pipeline, and asserts
the output tokens match exactly.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPT = "The capital of France is"
DEFAULT_MAX_TOKENS = 10


def run_correctness_check(
    model_name: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = DEFAULT_MAX_TOKENS,
    split_at: int | None = None,
) -> dict:
    """Run the correctness check and return a summary dict.

    Raises ``AssertionError`` if tokens don't match.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from worker.inference import InferenceRunner

    print(f"Loading {model_name} …")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_layers = model.config.num_hidden_layers

    if split_at is None:
        split_at = num_layers // 2
    print(f"Model has {num_layers} layers, splitting at {split_at}")

    # Build two runners that share the loaded model's components.
    runner_a = InferenceRunner.__new__(InferenceRunner)
    runner_a.torch = torch
    runner_a.model_name = model_name
    runner_a.start_layer = 0
    runner_a.end_layer = split_at
    runner_a.is_first = True
    runner_a.is_last = False
    runner_a.config = model.config
    runner_a.embed_tokens = model.model.embed_tokens
    runner_a.layers = model.model.layers[0:split_at]
    runner_a.rotary_emb = getattr(model.model, "rotary_emb", None)
    runner_a.norm = None
    runner_a.lm_head = None

    runner_b = InferenceRunner.__new__(InferenceRunner)
    runner_b.torch = torch
    runner_b.model_name = model_name
    runner_b.start_layer = split_at
    runner_b.end_layer = num_layers
    runner_b.is_first = False
    runner_b.is_last = True
    runner_b.config = model.config
    runner_b.embed_tokens = None
    runner_b.layers = model.model.layers[split_at:num_layers]
    runner_b.rotary_emb = getattr(model.model, "rotary_emb", None)
    runner_b.norm = model.model.norm
    runner_b.lm_head = model.lm_head

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    print(f"Prompt: {prompt!r}  ({len(input_ids)} tokens)")

    # ── Full model generation ────────────────────────────────────────
    full_tokens: list[int] = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(torch.tensor([input_ids + full_tokens])).logits
            next_tok = int(torch.argmax(logits[0, -1, :]).item())
            if next_tok == tokenizer.eos_token_id:
                break
            full_tokens.append(next_tok)
    full_time = time.perf_counter() - t0
    full_text = tokenizer.decode(full_tokens, skip_special_tokens=True)

    # ── Split pipeline generation ────────────────────────────────────
    split_tokens: list[int] = []
    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        token_ids = input_ids + split_tokens
        hidden = runner_a.forward_from_tokens(token_ids)
        logits = runner_b.forward_from_activations(hidden, is_last=True)
        next_tok = int(torch.argmax(logits[0, -1, :]).item())
        if next_tok == tokenizer.eos_token_id:
            break
        split_tokens.append(next_tok)
    split_time = time.perf_counter() - t0
    split_text = tokenizer.decode(split_tokens, skip_special_tokens=True)

    # ── Compare ──────────────────────────────────────────────────────
    match = full_tokens == split_tokens
    print()
    print(f"Full model tokens:  {full_tokens}")
    print(f"Split tokens:       {split_tokens}")
    print(f"Full model text:    {full_text!r}")
    print(f"Split text:         {split_text!r}")
    print(f"Match: {'YES ✓' if match else 'NO ✗'}")
    print(f"Full time:  {full_time:.2f}s")
    print(f"Split time: {split_time:.2f}s")

    if not match:
        raise AssertionError(
            f"Token mismatch!\n  Full:  {full_tokens}\n  Split: {split_tokens}"
        )

    return {
        "match": match,
        "full_tokens": full_tokens,
        "split_tokens": split_tokens,
        "full_text": full_text,
        "split_text": split_text,
        "full_time_s": round(full_time, 2),
        "split_time_s": round(split_time, 2),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Correctness check: full vs split model")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--split-at", type=int, default=None, help="Layer index to split at")
    args = p.parse_args(argv)

    run_correctness_check(
        model_name=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        split_at=args.split_at,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

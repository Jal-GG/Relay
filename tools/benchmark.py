"""Benchmark: single-machine (CPU) vs single-machine (GPU) vs split (GPU+CPU).

Measures tokens/sec for each configuration and prints a comparison table.
Document the overhead honestly — split will likely be slower than single-GPU
due to network latency.  That's fine — the point is enabling models too large
for one machine.

Usage::

    python -m tools.benchmark \\
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --prompt "The capital of France is" \\
        --max-tokens 20 \\
        --split-at 11

This runs three benchmarks:
  1. Full model on CPU (single machine)
  2. Full model on GPU (single machine, if CUDA available)
  3. Split model: GPU (layers 0–10) + CPU (layers 11–21)

Results are printed as a table and saved to benchmark_results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPT = "The capital of France is"
DEFAULT_MAX_TOKENS = 20


def _generate_full(model, tokenizer, input_ids, max_tokens, device_str):
    """Run full-model generation and return (tokens, elapsed_s)."""
    import torch

    model = model.to(device_str)
    model.eval()
    generated = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_tokens):
            ids = torch.tensor([input_ids + generated]).to(device_str)
            logits = model(ids).logits
            next_tok = int(torch.argmax(logits[0, -1, :]).item())
            if next_tok == tokenizer.eos_token_id:
                break
            generated.append(next_tok)
    elapsed = time.perf_counter() - t0
    return generated, elapsed


def _generate_split(runner_a, runner_b, input_ids, max_tokens):
    """Run split-pipeline generation and return (tokens, elapsed_s)."""
    generated = []
    t0 = time.perf_counter()
    for _ in range(max_tokens):
        token_ids = input_ids + generated
        hidden = runner_a.forward_from_tokens(token_ids)
        logits = runner_b.forward_from_activations(hidden, is_last=True)
        import torch
        next_tok = int(torch.argmax(logits[0, -1, :]).item())
        if next_tok == tokenizer.eos_token_id if (tokenizer := globals().get("_tokenizer")) else False:
            break
        generated.append(next_tok)
    elapsed = time.perf_counter() - t0
    return generated, elapsed


def run_benchmark(
    model_name: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    split_at: int | None = None,
    output_file: str = "benchmark_results.json",
) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from worker.inference import InferenceRunner

    print(f"Loading {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
    ).eval()
    num_layers = model.config.num_hidden_layers

    if split_at is None:
        split_at = num_layers // 2

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    print(f"Prompt: {prompt!r}  ({len(input_ids)} tokens)")
    print(f"Model: {num_layers} layers, splitting at {split_at}")
    print()

    results = {}

    # ── 1. Full model on CPU ──────────────────────────────────────────
    print("Benchmark 1/3: Full model on CPU …")
    tokens_cpu, time_cpu = _generate_full(model, tokenizer, input_ids, max_tokens, "cpu")
    text_cpu = tokenizer.decode(tokens_cpu, skip_special_tokens=True)
    tps_cpu = len(tokens_cpu) / time_cpu if time_cpu > 0 else 0
    results["cpu_full"] = {
        "tokens": tokens_cpu,
        "text": text_cpu,
        "time_s": round(time_cpu, 3),
        "tokens_per_sec": round(tps_cpu, 2),
    }
    print(f"  {len(tokens_cpu)} tokens in {time_cpu:.2f}s → {tps_cpu:.2f} tok/s")
    print(f"  Output: {text_cpu!r}")
    print()

    # ── 2. Full model on GPU (if available) ───────────────────────────
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print("Benchmark 2/3: Full model on GPU …")
        tokens_gpu, time_gpu = _generate_full(model, tokenizer, input_ids, max_tokens, "cuda")
        text_gpu = tokenizer.decode(tokens_gpu, skip_special_tokens=True)
        tps_gpu = len(tokens_gpu) / time_gpu if time_gpu > 0 else 0
        results["gpu_full"] = {
            "tokens": tokens_gpu,
            "text": text_gpu,
            "time_s": round(time_gpu, 3),
            "tokens_per_sec": round(tps_gpu, 2),
        }
        print(f"  {len(tokens_gpu)} tokens in {time_gpu:.2f}s → {tps_gpu:.2f} tok/s")
        print(f"  Output: {text_gpu!r}")
    else:
        print("Benchmark 2/3: Skipped (no CUDA available)")
        results["gpu_full"] = None
    print()

    # ── 3. Split pipeline: GPU + CPU ─────────────────────────────────
    device_a = "cuda" if has_cuda else "cpu"
    print(f"Benchmark 3/3: Split pipeline ({device_a.upper()} layers 0-{split_at-1} + CPU layers {split_at}-{num_layers-1}) …")

    runner_a = InferenceRunner.__new__(InferenceRunner)
    runner_a.torch = torch
    runner_a.model_name = model_name
    runner_a.start_layer = 0
    runner_a.end_layer = split_at
    runner_a.is_first = True
    runner_a.is_last = False
    runner_a.device = device_a
    from worker.backends import create_backend
    runner_a.backend = create_backend(device_a)
    runner_a.config = model.config
    runner_a.embed_tokens = model.model.embed_tokens
    runner_a.layers = torch.nn.ModuleList(list(model.model.layers[0:split_at]))
    runner_a.rotary_emb = getattr(model.model, "rotary_emb", None)
    runner_a.norm = None
    runner_a.lm_head = None
    runner_a._move_to_device()

    runner_b = InferenceRunner.__new__(InferenceRunner)
    runner_b.torch = torch
    runner_b.model_name = model_name
    runner_b.start_layer = split_at
    runner_b.end_layer = num_layers
    runner_b.is_first = False
    runner_b.is_last = True
    runner_b.device = "cpu"
    runner_b.backend = create_backend("cpu")
    runner_b.config = model.config
    runner_b.embed_tokens = None
    runner_b.layers = torch.nn.ModuleList(list(model.model.layers[split_at:num_layers]))
    runner_b.rotary_emb = getattr(model.model, "rotary_emb", None)
    runner_b.norm = model.model.norm
    runner_b.lm_head = model.lm_head
    runner_b._move_to_device()

    split_tokens = []
    t0 = time.perf_counter()
    for _ in range(max_tokens):
        token_ids = input_ids + split_tokens
        hidden = runner_a.forward_from_tokens(token_ids)
        logits = runner_b.forward_from_activations(hidden, is_last=True)
        next_tok = int(torch.argmax(logits[0, -1, :]).item())
        if next_tok == tokenizer.eos_token_id:
            break
        split_tokens.append(next_tok)
    time_split = time.perf_counter() - t0
    text_split = tokenizer.decode(split_tokens, skip_special_tokens=True)
    tps_split = len(split_tokens) / time_split if time_split > 0 else 0
    results["split_gpu_cpu"] = {
        "tokens": split_tokens,
        "text": text_split,
        "time_s": round(time_split, 3),
        "tokens_per_sec": round(tps_split, 2),
        "device_a": device_a,
        "device_b": "cpu",
        "split_at": split_at,
    }
    print(f"  {len(split_tokens)} tokens in {time_split:.2f}s → {tps_split:.2f} tok/s")
    print(f"  Output: {text_split!r}")
    print()

    # ── Summary table ────────────────────────────────────────────────
    print("┌─────────────────────────┬──────────┬──────────┬───────────┐")
    print("│ Configuration           │ Time (s) │ Tokens   │ Tok/s     │")
    print("├─────────────────────────┼──────────┼──────────┼───────────┤")
    print(f"│ CPU (full model)        │ {time_cpu:8.2f} │ {len(tokens_cpu):8d} │ {tps_cpu:9.2f} │")
    if has_cuda:
        print(f"│ GPU (full model)        │ {time_gpu:8.2f} │ {len(tokens_gpu):8d} │ {tps_gpu:9.2f} │")
    else:
        print(f"│ GPU (full model)        │   N/A    │      N/A │       N/A │")
    print(f"│ Split ({device_a.upper()}+CPU)        │ {time_split:8.2f} │ {len(split_tokens):8d} │ {tps_split:9.2f} │")
    print("└─────────────────────────┴──────────┴──────────┴───────────┘")

    # ── Correctness check ────────────────────────────────────────────
    match = tokens_cpu == split_tokens
    print(f"\nCorrectness: {'PASS ✓' if match else 'FAIL ✗'}")
    if not match:
        print(f"  CPU tokens:  {tokens_cpu}")
        print(f"  Split tokens: {split_tokens}")

    results["correctness"] = match
    results["model"] = model_name
    results["num_layers"] = num_layers
    results["prompt"] = prompt

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Benchmark: CPU vs GPU vs split inference")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--split-at", type=int, default=None, help="Layer index to split at")
    p.add_argument("--output", default="benchmark_results.json")
    args = p.parse_args(argv)

    run_benchmark(
        model_name=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        split_at=args.split_at,
        output_file=args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

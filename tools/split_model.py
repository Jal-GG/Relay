"""Offline tool that splits a HuggingFace model's state dict by layer ranges.

Usage::

    python -m tools.split_model \\
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --splits 0:11,11:22 \\
        --output ./model_splits

    # Any HuggingFace CausalLM works — Llama, Gemma, Mistral, Phi, Qwen …
    python -m tools.split_model \\
        --model google/gemma-2b \\
        --splits 0:9,9:18 \\
        --output ./gemma_splits

This produces ``split_0.pt`` (embed layer + layers 0-N) and ``split_1.pt``
(layers N+1-M + final norm + lm_head) inside the output directory.

The tool auto-detects the correct state-dict key prefixes for any architecture
(``model.embed_tokens``, ``model.layers``, ``model.norm`` for Llama-style;
``transformer.word_embeddings``, ``transformer.h``, ``transformer.ln_f`` for
GPT-style, etc.) so no changes are needed when switching models.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Defer heavy imports so --help is fast.


def _parse_splits(spec: str) -> list[tuple[int, int]]:
    """Parse '0:11,11:22' → [(0, 11), (11, 22)]."""
    parts = spec.split(",")
    splits: list[tuple[int, int]] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        start_s, end_s = p.split(":")
        splits.append((int(start_s), int(end_s)))
    if not splits:
        raise ValueError("no splits parsed from spec")
    # Validate contiguous, non-overlapping ranges.
    for i in range(1, len(splits)):
        prev_end = splits[i - 1][1]
        curr_start = splits[i][0]
        if curr_start < prev_end:
            raise ValueError(
                f"overlapping splits: split {i - 1} ends at {prev_end} "
                f"but split {i} starts at {curr_start}"
            )
        if curr_start > prev_end:
            raise ValueError(
                f"gap in splits: split {i - 1} ends at {prev_end} "
                f"but split {i} starts at {curr_start}"
            )
    return splits


# ── Architecture-agnostic key detection ───────────────────────────────────────

def _find_prefix(sd: dict, candidates: list[str]) -> str | None:
    """Return the first candidate prefix that appears in the state dict."""
    for candidate in candidates:
        if any(k.startswith(candidate) for k in sd):
            return candidate
    return None


def _detect_structure(model: "Any", sd: dict) -> dict:
    """Auto-detect architecture-specific state-dict key prefixes.

    Returns a dict with:
      - ``inner_prefix``  : prefix for the base transformer (e.g. "model." or "transformer.")
      - ``embed_prefix``  : prefix for the token embedding (e.g. "model.embed_tokens.")
      - ``layer_prefix``  : prefix template for decoder layers (e.g. "model.layers.{i}.")
      - ``norm_prefix``   : prefix for the final norm (e.g. "model.norm.")
      - ``lm_head_prefix``: prefix for the LM head (e.g. "lm_head.")
      - ``num_layers``    : total number of decoder layers in the model
    """
    # Find the inner model attribute and its state-dict prefix.
    inner = None
    inner_attr = None
    for attr in ("model", "transformer"):
        cand = getattr(model, attr, None)
        if cand is not None:
            inner = cand
            inner_attr = attr
            break
    if inner is None:
        raise NotImplementedError(
            f"{type(model).__name__} has neither .model nor .transformer. "
            "Please open an issue to add support for this architecture."
        )
    inner_prefix = f"{inner_attr}."

    # Embedding
    embed_prefix = None
    for attr in ("embed_tokens", "word_embeddings", "wte", "embed_in", "embeddings"):
        if hasattr(inner, attr):
            cand = f"{inner_prefix}{attr}."
            if _find_prefix(sd, [cand]):
                embed_prefix = cand
                break
    if embed_prefix is None:
        raise NotImplementedError(
            f"Cannot detect embedding prefix in {type(inner).__name__}. "
            f"Tried: embed_tokens, word_embeddings, wte, embed_in, embeddings"
        )

    # Decoder layers
    layers_attr = None
    for attr in ("layers", "h", "blocks", "layer"):
        if hasattr(inner, attr):
            cand = f"{inner_prefix}{attr}.0."
            if _find_prefix(sd, [cand]):
                layers_attr = attr
                break
    if layers_attr is None:
        raise NotImplementedError(
            f"Cannot detect layers attribute in {type(inner).__name__}. "
            f"Tried: layers, h, blocks, layer"
        )
    layer_prefix_tmpl = f"{inner_prefix}{layers_attr}.{{i}}."
    num_layers = len(getattr(inner, layers_attr))

    # Final norm
    norm_prefix = None
    for attr in ("norm", "final_layernorm", "ln_f", "layer_norm", "out_norm"):
        if hasattr(inner, attr):
            cand = f"{inner_prefix}{attr}."
            if _find_prefix(sd, [cand]):
                norm_prefix = cand
                break
    if norm_prefix is None:
        raise NotImplementedError(
            f"Cannot detect norm prefix in {type(inner).__name__}. "
            f"Tried: norm, final_layernorm, ln_f, layer_norm, out_norm"
        )

    # LM head
    lm_head_prefix = None
    import torch
    for attr in ("lm_head", "embed_out", "output"):
        head = getattr(model, attr, None)
        if head is not None and isinstance(head, torch.nn.Linear):
            lm_head_prefix = f"{attr}."
            break
    if lm_head_prefix is None:
        raise NotImplementedError(
            f"Cannot detect lm_head in {type(model).__name__}. "
            f"Tried: lm_head, embed_out, output"
        )

    return {
        "inner_prefix": inner_prefix,
        "embed_prefix": embed_prefix,
        "layer_prefix_tmpl": layer_prefix_tmpl,
        "norm_prefix": norm_prefix,
        "lm_head_prefix": lm_head_prefix,
        "num_layers": num_layers,
    }


# ── Main split logic ───────────────────────────────────────────────────────────

def split_model(model_name: str, splits: list[tuple[int, int]], output_dir: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    print(f"Loading {model_name} …")
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
    )
    model.eval()
    sd = model.state_dict()

    print("Detecting model architecture …")
    struct = _detect_structure(model, sd)
    print(
        f"  inner_prefix    : {struct['inner_prefix']}\n"
        f"  embed_prefix    : {struct['embed_prefix']}\n"
        f"  layer_prefix    : {struct['layer_prefix_tmpl'].format(i='N')}\n"
        f"  norm_prefix     : {struct['norm_prefix']}\n"
        f"  lm_head_prefix  : {struct['lm_head_prefix']}\n"
        f"  total layers    : {struct['num_layers']}"
    )

    os.makedirs(output_dir, exist_ok=True)

    # Save config alongside the splits so workers can reconstruct components.
    config_path = os.path.join(output_dir, "config.json")
    config.save_pretrained(output_dir)
    print(f"Saved config to {config_path}")

    tied = getattr(config, "tie_word_embeddings", False)

    for i, (start, end) in enumerate(splits):
        split_sd: dict[str, torch.Tensor] = {}

        # First split: include the token embedding.
        if i == 0:
            ep = struct["embed_prefix"]
            for k, v in sd.items():
                if k.startswith(ep):
                    split_sd[k] = v

        # Decoder layers in [start, end).
        for layer_idx in range(start, end):
            prefix = struct["layer_prefix_tmpl"].format(i=layer_idx)
            for k, v in sd.items():
                if k.startswith(prefix):
                    split_sd[k] = v

        # Last split: include final norm and lm_head.
        if i == len(splits) - 1:
            np_ = struct["norm_prefix"]
            for k, v in sd.items():
                if k.startswith(np_):
                    split_sd[k] = v
            if tied:
                # With tied embeddings, lm_head shares the embed weight.
                # Copy it explicitly so the last split is self-contained.
                embed_weight_key = struct["embed_prefix"] + "weight"
                if embed_weight_key in sd:
                    lm_head_key = struct["lm_head_prefix"] + "weight"
                    split_sd[lm_head_key] = sd[embed_weight_key].clone()
                    print(
                        f"  tied embeddings: copied {embed_weight_key} "
                        f"→ {lm_head_key}"
                    )
            else:
                lp = struct["lm_head_prefix"]
                for k, v in sd.items():
                    if k.startswith(lp):
                        split_sd[k] = v

        meta = {
            "split_index": i,
            "layer_start": start,
            "layer_end": end,
            "is_first": i == 0,
            "is_last": i == len(splits) - 1,
            "model_name": model_name,
            "num_layers": struct["num_layers"],
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
            "tied_embeddings": tied,
            # Store the key prefixes so workers can validate the split file.
            "key_structure": struct,
        }

        out_path = os.path.join(output_dir, f"split_{i}.pt")
        torch.save({"state_dict": split_sd, "meta": meta}, out_path)
        print(
            f"  split {i}: layers {start}-{end - 1} "
            f"({len(split_sd)} tensors) → {out_path}"
        )

    # Write a manifest describing the full pipeline.
    manifest = {
        "model_name": model_name,
        "num_layers": struct["num_layers"],
        "splits": [
            {"index": i, "layer_start": s, "layer_end": e}
            for i, (s, e) in enumerate(splits)
        ],
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest → {manifest_path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Split a model into per-worker shards")
    p.add_argument("--model", required=True, help="HF model name or local path")
    p.add_argument(
        "--splits",
        required=True,
        help="Layer ranges, e.g. '0:11,11:22'",
    )
    p.add_argument("--output", default="./model_splits", help="Output directory")
    args = p.parse_args(argv)

    splits = _parse_splits(args.splits)
    split_model(args.model, splits, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Offline tool that splits a HuggingFace model's state dict by layer ranges.

Usage::

    python -m tools.split_model \\
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --splits 0:11,11:22 \\
        --output ./model_splits

This produces ``split_0.pt`` (embed_tokens + layers 0-10) and ``split_1.pt``
(layers 11-21 + final norm + lm_head) inside the output directory.
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


def split_model(model_name: str, splits: list[tuple[int, int]], output_dir: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    print(f"Loading {model_name} …")
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
    )
    sd = model.state_dict()

    os.makedirs(output_dir, exist_ok=True)

    # Save config alongside the splits so workers can reconstruct components.
    config_path = os.path.join(output_dir, "config.json")
    config.save_pretrained(output_dir)
    print(f"Saved config to {config_path}")

    tied = getattr(config, "tie_word_embeddings", False)

    for i, (start, end) in enumerate(splits):
        split_sd: dict[str, torch.Tensor] = {}

        # First split gets the embedding layer.
        if i == 0:
            for k, v in sd.items():
                if k.startswith("model.embed_tokens"):
                    split_sd[k] = v
            if tied:
                # lm_head shares weights with embed_tokens — copy for the last
                # split as well so it can load independently.
                pass

        # Layers in range.
        for layer_idx in range(start, end):
            prefix = f"model.layers.{layer_idx}."
            for k, v in sd.items():
                if k.startswith(prefix):
                    split_sd[k] = v

        # Last split gets final norm and lm_head.
        if i == len(splits) - 1:
            for k, v in sd.items():
                if k.startswith("model.norm"):
                    split_sd[k] = v
            if tied:
                # With tied embeddings, lm_head.weight == model.embed_tokens.weight.
                # Copy the shared weight so the last split is self-contained.
                emb_key = "model.embed_tokens.weight"
                if emb_key in sd:
                    split_sd["lm_head.weight"] = sd[emb_key].clone()
            else:
                for k, v in sd.items():
                    if k.startswith("lm_head"):
                        split_sd[k] = v

        meta = {
            "split_index": i,
            "layer_start": start,
            "layer_end": end,
            "is_first": i == 0,
            "is_last": i == len(splits) - 1,
            "model_name": model_name,
            "num_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
            "tied_embeddings": tied,
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
        "num_layers": config.num_hidden_layers,
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

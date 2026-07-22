from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger("relay.worker.inference")


class InferenceRunner:
    """Holds a contiguous range of model layers and runs forward passes.

    When ``is_first`` is True the runner owns the token embedding layer and
    accepts raw token IDs.  When ``is_last`` is True it owns the final norm
    and LM head and returns logits instead of hidden states.

    The simplest loading path downloads the full model and keeps only the
    relevant sub-modules — sufficient for a 1-3 B parameter prototype.
    Optionally a ``split_path`` produced by ``tools.split_model`` can be
    supplied to avoid loading the full weights.
    """

    def __init__(
        self,
        model_name: str,
        start_layer: int,
        end_layer: int,
        is_first: bool,
        is_last: bool,
        split_path: Optional[str] = None,
        dtype: str = "float32",
    ) -> None:
        import torch

        self.model_name = model_name
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.is_first = is_first
        self.is_last = is_last
        self.torch = torch

        torch_dtype = torch.float32 if dtype == "float32" else torch.float16

        if split_path is not None:
            self._load_from_split(split_path, torch_dtype)
        else:
            self._load_from_full_model(model_name, torch_dtype)

        n = len(self.layers)
        log.info(
            "InferenceRunner ready: layers %d-%d (%d), first=%s last=%s",
            start_layer, end_layer - 1, n, is_first, is_last,
        )

    # ── loading ──────────────────────────────────────────────────────

    def _load_from_full_model(self, model_name: str, dtype: Any) -> None:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype,
        )
        model.eval()

        self.config = model.config

        if self.is_first:
            self.embed_tokens = model.model.embed_tokens
        else:
            self.embed_tokens = None

        self.layers = model.model.layers[self.start_layer:self.end_layer]

        if self.is_last:
            self.norm = model.model.norm
            self.lm_head = model.lm_head
        else:
            self.norm = None
            self.lm_head = None

    def _load_from_split(self, split_path: str, dtype: Any) -> None:
        import torch
        from transformers import AutoConfig

        checkpoint = torch.load(split_path, map_location="cpu", weights_only=False)
        sd = checkpoint["state_dict"]
        meta = checkpoint["meta"]

        self.config = AutoConfig.from_pretrained(
            meta["model_name"], torch_dtype=dtype,
        )

        # Build components from scratch and load only the needed weights.
        if self.is_first:
            self.embed_tokens = torch.nn.Embedding(
                meta["vocab_size"], meta["hidden_size"],
            )
            emb_sd = {
                k.replace("model.embed_tokens.", ""): v
                for k, v in sd.items()
                if k.startswith("model.embed_tokens")
            }
            self.embed_tokens.load_state_dict(emb_sd)
        else:
            self.embed_tokens = None

        from transformers.models.llama.modeling_llama import (
            LlamaDecoderLayer,
            LlamaRMSNorm,
        )

        self.layers = torch.nn.ModuleList()
        for idx in range(self.start_layer, self.end_layer):
            layer = LlamaDecoderLayer(self.config, idx)
            prefix = f"model.layers.{idx}."
            layer_sd = {
                k.replace(prefix, ""): v
                for k, v in sd.items()
                if k.startswith(prefix)
            }
            layer.load_state_dict(layer_sd)
            self.layers.append(layer)

        if self.is_last:
            self.norm = LlamaRMSNorm(meta["hidden_size"], eps=self.config.rms_norm_eps)
            norm_sd = {
                k.replace("model.norm.", ""): v
                for k, v in sd.items()
                if k.startswith("model.norm")
            }
            self.norm.load_state_dict(norm_sd)

            self.lm_head = torch.nn.Linear(
                meta["hidden_size"], meta["vocab_size"], bias=False,
            )
            if "lm_head.weight" in sd:
                self.lm_head.load_state_dict(
                    {"weight": sd["lm_head.weight"]}
                )
            else:
                # Tied embeddings — share the embedding weight.
                emb_w = sd.get("model.embed_tokens.weight")
                if emb_w is not None:
                    self.lm_head.weight = torch.nn.Parameter(emb_w.clone())
                else:
                    raise RuntimeError(
                        "lm_head weight missing and no embedding to tie"
                    )
        else:
            self.norm = None
            self.lm_head = None

    # ── forward ──────────────────────────────────────────────────────

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def _position_ids(self, seq_len: int) -> Any:
        return self.torch.arange(seq_len).unsqueeze(0)

    def forward_from_tokens(self, token_ids: list[int]) -> Any:
        """First-worker entry point: embed tokens then run assigned layers."""
        torch = self.torch
        with torch.no_grad():
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            hidden = self.embed_tokens(input_ids)
            seq_len = input_ids.shape[1]
            pos = self._position_ids(seq_len)
            for layer in self.layers:
                hidden = layer(
                    hidden, position_ids=pos, use_cache=False,
                )[0]
        return hidden

    def forward_from_tokens_to_logits(self, token_ids: list[int]) -> Any:
        """Single-worker entry point: embed → layers → norm → lm_head → logits."""
        torch = self.torch
        with torch.no_grad():
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            hidden = self.embed_tokens(input_ids)
            seq_len = input_ids.shape[1]
            pos = self._position_ids(seq_len)
            for layer in self.layers:
                hidden = layer(
                    hidden, position_ids=pos, use_cache=False,
                )[0]
            hidden = self.norm(hidden)
            logits = self.lm_head(hidden)
        return logits

    def forward_from_activations(
        self, hidden_states: Any, is_last: bool,
    ) -> Any:
        """Subsequent-worker entry point: run layers on received activations."""
        torch = self.torch
        with torch.no_grad():
            seq_len = hidden_states.shape[1]
            pos = self._position_ids(seq_len)
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states, position_ids=pos, use_cache=False,
                )[0]
            if is_last:
                hidden_states = self.norm(hidden_states)
                logits = self.lm_head(hidden_states)
                return logits
        return hidden_states

from __future__ import annotations

import logging
import traceback
from typing import Any, Optional

log = logging.getLogger("relay.worker.inference")


# ── Architecture-agnostic introspection helpers ───────────────────────────────

def _detect_attn_implementation() -> str:
    """Return the attention backend transformers auto-selects.

    Priority order (mirrors ``transformers.modeling_utils``):
      1. ``flash_attention_2`` — if the ``flash_attn`` package is installed.
      2. ``sdpa``              — if PyTorch ≥ 2.0 (F.scaled_dot_product_attention).
      3. ``eager``             — always available pure-Python fallback.

    Using a different backend than ``AutoModelForCausalLM.from_pretrained``
    chooses produces numerically different attention outputs that compound
    across many layers and corrupt output tokens.
    """
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        pass
    import torch
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return "sdpa"
    return "eager"


def _get_inner_model(shell: Any) -> Any:
    """Return the base transformer model from a CausalLM wrapper.

    Most HuggingFace CausalLM wrappers expose the inner model as ``.model``.
    Some older / non-standard ones use ``.transformer``.
    """
    for attr in ("model", "transformer"):
        inner = getattr(shell, attr, None)
        if inner is not None:
            return inner
    raise NotImplementedError(
        f"{type(shell).__name__} exposes neither .model nor .transformer. "
        "Open an issue or use the full-model loading path (omit --split-path)."
    )


def _get_decoder_layers(inner: Any) -> Any:
    """Return the sequence of decoder layers for any architecture."""
    for attr in ("layers", "h", "blocks", "layer"):
        layers = getattr(inner, attr, None)
        if layers is not None:
            return layers
    raise NotImplementedError(
        f"{type(inner).__name__} has no .layers / .h / .blocks attribute. "
        "Open an issue or use the full-model loading path (omit --split-path)."
    )


def _get_embed(inner: Any) -> Any:
    """Return the token embedding module for any architecture."""
    for attr in ("embed_tokens", "word_embeddings", "wte", "embed_in", "embeddings"):
        emb = getattr(inner, attr, None)
        if emb is not None:
            return emb
    raise NotImplementedError(
        f"{type(inner).__name__} has no embed_tokens / word_embeddings / wte attribute."
    )


def _get_norm(inner: Any) -> Any:
    """Return the final layer-norm for any architecture."""
    for attr in ("norm", "final_layernorm", "ln_f", "layer_norm", "out_norm"):
        norm = getattr(inner, attr, None)
        if norm is not None:
            return norm
    raise NotImplementedError(
        f"{type(inner).__name__} has no norm / final_layernorm / ln_f attribute."
    )


def _get_lm_head(shell: Any) -> Any:
    """Return the LM-head Linear layer from a CausalLM wrapper."""
    import torch
    for attr in ("lm_head", "embed_out", "output"):
        head = getattr(shell, attr, None)
        if head is not None and isinstance(head, torch.nn.Linear):
            return head
    raise NotImplementedError(
        f"{type(shell).__name__} has no lm_head / embed_out / output Linear layer."
    )


# ── InferenceRunner ────────────────────────────────────────────────────────────

class InferenceRunner:
    """Holds a contiguous range of model layers and runs forward passes.

    When ``is_first`` is True the runner owns the token embedding layer and
    accepts raw token IDs.  When ``is_last`` is True it owns the final norm
    and LM head and returns logits instead of hidden states.

    Two loading paths:
    - ``_load_from_full_model``: downloads the full model and keeps only the
      relevant sub-modules.  Works for **any** HuggingFace CausalLM.
    - ``_load_from_split``: loads from a shard produced by ``tools.split_model``.
      Also works for **any** architecture — module types are derived via
      ``AutoModelForCausalLM.from_config`` (no architecture-specific imports).

    Device placement is handled by an ``InferenceBackend`` (Phase 4).
    The coordinator doesn't know or care which backend a worker uses — it
    just sends/receives activations.  The backend moves modules to the target
    device at load time and moves tensors to/from the device at forward time.

    Compatible with transformers ≥ 4.40 and transformers 5.x.
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
        device: str = "auto",
    ) -> None:
        import torch

        self.model_name = model_name
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.is_first = is_first
        self.is_last = is_last
        self.torch = torch
        self.device = device

        from worker.backends import create_backend
        self.backend = create_backend(device)

        torch_dtype = torch.float32 if dtype == "float32" else torch.float16

        if split_path is not None:
            self._load_from_split(split_path, torch_dtype)
        else:
            self._load_from_full_model(model_name, torch_dtype)

        self._move_to_device()

        n = len(self.layers)
        log.info(
            "InferenceRunner ready: layers %d-%d (%d), first=%s last=%s, device=%s",
            start_layer, end_layer - 1, n, is_first, is_last, self.backend.device,
        )

    # ── loading ──────────────────────────────────────────────────────────────

    def _move_to_device(self) -> None:
        """Move all loaded modules to the backend's device."""
        if self.embed_tokens is not None:
            self.backend.move_module(self.embed_tokens)
        for layer in self.layers:
            self.backend.move_module(layer)
        if self.norm is not None:
            self.backend.move_module(self.norm)
        if self.lm_head is not None:
            self.backend.move_module(self.lm_head)
        if self.rotary_emb is not None:
            self.backend.move_module(self.rotary_emb)

    def _load_from_full_model(self, model_name: str, dtype: Any) -> None:
        """Download the full model, keep only the layers assigned to this worker."""
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        model.eval()
        self.config = model.config

        inner = _get_inner_model(model)
        self.embed_tokens = _get_embed(inner) if self.is_first else None
        self.layers = torch.nn.ModuleList(
            list(_get_decoder_layers(inner)[self.start_layer:self.end_layer])
        )
        self.rotary_emb = getattr(inner, "rotary_emb", None)

        if self.is_last:
            self.norm = _get_norm(inner)
            self.lm_head = _get_lm_head(model)
        else:
            self.norm = None
            self.lm_head = None

        del model  # free rest of weights from memory

    def _load_from_split(self, split_path: str, dtype: Any) -> None:
        """Load from a shard file produced by tools.split_model.

        Strategy (fully architecture-agnostic):
          1. Detect the dtype actually stored in the file.
          2. Build a typed, correctly configured model shell with
             ``AutoModelForCausalLM.from_config`` — no download, random weights.
             This gives us the right module classes (LlamaDecoderLayer,
             GemmaDecoderLayer, Phi3DecoderLayer …) and auto-selects the correct
             ``_attn_implementation`` for the installed PyTorch / flash-attn.
          3. Load the split state dict into the shell with ``strict=False``
             (only the layers in our shard are present; the rest stay random).
          4. Extract and keep only the sub-modules for our layer range.
        """
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        checkpoint = torch.load(split_path, map_location="cpu", weights_only=False)
        sd: dict = checkpoint["state_dict"]
        meta: dict = checkpoint["meta"]

        # 1. Detect dtype from the actual saved tensors — never trust config.json
        #    (e.g. TinyLlama has "dtype": "bfloat16" but was split as float32).
        sd_dtype = next(iter(sd.values())).dtype
        log.info(
            "Split file dtype: %s  (layers %d-%d of %s)",
            sd_dtype, self.start_layer, self.end_layer - 1, meta["model_name"],
        )

        # 2. Build the model config and override dtype + attn implementation.
        config = AutoConfig.from_pretrained(meta["model_name"])
        config.torch_dtype = sd_dtype  # override e.g. bfloat16 from config.json

        # 3. Build the shell — random weights, correct types, correct attn backend.
        #    set_default_dtype ensures every sub-module is born in sd_dtype.
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(sd_dtype)
        shell = None
        try:
            shell = AutoModelForCausalLM.from_config(config)
            shell.eval()
            # from_config sets _attn_implementation correctly (sdpa / eager / flash).
            self.config = shell.config

            # 4. Load split weights (strict=False — other layers stay random).
            missing, unexpected = shell.load_state_dict(sd, strict=False)
            unexpected_real = [k for k in unexpected if not k.startswith("_")]
            if unexpected_real:
                log.warning(
                    "Unexpected keys in split file (first 5): %s",
                    unexpected_real[:5],
                )
            log.debug(
                "load_state_dict: %d missing keys (layers outside our shard), "
                "%d unexpected",
                len(missing), len(unexpected_real),
            )

            # 5. Locate the inner transformer model using introspection.
            inner = _get_inner_model(shell)

            # 6. Extract only our sub-modules (other modules stay in shell and
            #    will be garbage-collected when `shell` is deleted below).
            self.embed_tokens = _get_embed(inner) if self.is_first else None
            layers_all = _get_decoder_layers(inner)
            self.layers = torch.nn.ModuleList(
                list(layers_all[self.start_layer:self.end_layer])
            )
            self.rotary_emb = getattr(inner, "rotary_emb", None)

            if self.is_last:
                self.norm = _get_norm(inner)
                self.lm_head = _get_lm_head(shell)
                # Sanity check: lm_head should have non-random weights
                # (either loaded from sd["lm_head.weight"] or tied to embed).
                if "lm_head.weight" not in sd:
                    # Check tied embeddings — most small models tie weights.
                    emb_w = sd.get("model.embed_tokens.weight")
                    if emb_w is None:
                        log.warning(
                            "lm_head.weight not found in split and no embed weight "
                            "to tie — lm_head may have random weights."
                        )
            else:
                self.norm = None
                self.lm_head = None

        finally:
            torch.set_default_dtype(prev_dtype)
            if shell is not None:
                del shell   # free random-weight modules outside our shard

    # ── helpers ──────────────────────────────────────────────────────────────

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def _position_ids(self, seq_len: int) -> Any:
        return self.torch.arange(seq_len).unsqueeze(0)

    def _position_embeddings(self, hidden: Any, pos: Any) -> tuple | None:
        """Return pre-computed (cos, sin) RoPE tensors, or None if not needed.

        transformers 5.x requires ``position_embeddings=(cos, sin)`` to be
        passed to each decoder layer.  4.x computed this internally from
        ``position_ids``.  Models without ``rotary_emb`` (e.g. some older GPT
        variants) return None and rely on the ``position_ids`` kwarg alone.
        """
        if self.rotary_emb is None:
            return None
        return self.rotary_emb(hidden, pos)

    def _layer_forward(self, layer: Any, hidden: Any, pos: Any) -> Any:
        """Forward through one decoder layer — handles both transformers 4.x and 5.x.

        - transformers 4.x: layer returns ``(hidden_states, ...)`` tuple; we
          take index ``[0]``.  It does NOT accept ``position_embeddings``.
        - transformers 5.x: layer returns a plain ``torch.Tensor`` and
          REQUIRES ``position_embeddings=(cos, sin)`` for correct RoPE.

        Both versions accept ``position_ids`` and ``use_cache``.
        """
        pe = self._position_embeddings(hidden, pos)
        kwargs: dict = {"position_ids": pos, "use_cache": False}
        if pe is not None:
            kwargs["position_embeddings"] = pe
        out = layer(hidden, **kwargs)
        # transformers 4.x returns (hidden_states, present_key_value, ...)
        if isinstance(out, tuple):
            return out[0]
        return out

    # ── forward passes ───────────────────────────────────────────────────────

    def forward_from_tokens(self, token_ids: list[int]) -> Any:
        """First-worker entry point: embed tokens then run assigned layers."""
        torch = self.torch
        backend = getattr(self, "backend", None)
        if backend is None:
            from worker.backends import CpuBackend
            backend = CpuBackend()
            self.backend = backend
        with torch.no_grad():
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            input_ids = self.backend.move_tensor(input_ids)
            hidden = self.embed_tokens(input_ids)
            seq_len = input_ids.shape[1]
            pos = self._position_ids(seq_len)
            pos = self.backend.move_tensor(pos)
            for layer in self.layers:
                hidden = self._layer_forward(layer, hidden, pos)
        return self.backend.to_cpu(hidden)

    def forward_from_tokens_to_logits(self, token_ids: list[int]) -> Any:
        """Single-worker pipeline: embed → layers → norm → lm_head → logits."""
        torch = self.torch
        backend = getattr(self, "backend", None)
        if backend is None:
            from worker.backends import CpuBackend
            backend = CpuBackend()
            self.backend = backend
        with torch.no_grad():
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            input_ids = self.backend.move_tensor(input_ids)
            hidden = self.embed_tokens(input_ids)
            seq_len = input_ids.shape[1]
            pos = self._position_ids(seq_len)
            pos = self.backend.move_tensor(pos)
            for layer in self.layers:
                hidden = self._layer_forward(layer, hidden, pos)
            hidden = self.norm(hidden)
            logits = self.lm_head(hidden)
        return self.backend.to_cpu(logits)

    def forward_from_activations(self, hidden_states: Any, is_last: bool) -> Any:
        """Subsequent-worker entry point: run layers on received activations."""
        torch = self.torch
        backend = getattr(self, "backend", None)
        if backend is None:
            from worker.backends import CpuBackend
            backend = CpuBackend()
            self.backend = backend
        with torch.no_grad():
            hidden_states = self.backend.move_tensor(hidden_states)
            seq_len = hidden_states.shape[1]
            pos = self._position_ids(seq_len)
            pos = self.backend.move_tensor(pos)
            for layer in self.layers:
                hidden_states = self._layer_forward(layer, hidden_states, pos)
            if is_last:
                hidden_states = self.norm(hidden_states)
                logits = self.lm_head(hidden_states)
                return self.backend.to_cpu(logits)
        return self.backend.to_cpu(hidden_states)

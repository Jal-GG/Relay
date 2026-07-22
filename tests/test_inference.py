"""Tests for Phase 3 inference components.

Tests that require torch/transformers are skipped when those packages
are not installed — the rest of the Phase 1–2 test suite should still run.
"""
from __future__ import annotations

import pytest

from common.models import ForwardRequest, ForwardResponse


# ── Tensor serialization round-trip ───────────────────────────────────

def test_serialize_deserialize_tensor():
    """torch tensor round-trips through base64 serialization."""
    torch = pytest.importorskip("torch")
    from common.serialization import serialize_tensor, deserialize_tensor

    original = torch.randn(2, 3, 4)
    encoded = serialize_tensor(original)
    assert isinstance(encoded, str)

    decoded = deserialize_tensor(encoded)
    assert torch.equal(original, decoded)


def test_serialize_deserialize_nested():
    """Non-tensor objects (dicts, lists) also round-trip via torch.save."""
    torch = pytest.importorskip("torch")
    from common.serialization import serialize_tensor, deserialize_tensor

    obj = {"a": [1, 2, 3], "b": torch.tensor([1.0, 2.0])}
    decoded = deserialize_tensor(serialize_tensor(obj))
    assert decoded["a"] == [1, 2, 3]
    assert torch.equal(decoded["b"], torch.tensor([1.0, 2.0]))


# ── Forward protocol models ───────────────────────────────────────────

def test_forward_request_with_token_ids():
    req = ForwardRequest(token_ids=[1, 2, 3])
    assert req.token_ids == [1, 2, 3]
    assert req.activations_b64 is None
    assert req.is_last is False


def test_forward_request_with_activations():
    req = ForwardRequest(activations_b64="abc123", is_last=True)
    assert req.token_ids is None
    assert req.activations_b64 == "abc123"
    assert req.is_last is True


def test_forward_response_activations():
    resp = ForwardResponse(activations_b64="abc")
    assert resp.activations_b64 == "abc"
    assert resp.logits_b64 is None


def test_forward_response_logits():
    resp = ForwardResponse(logits_b64="xyz")
    assert resp.logits_b64 == "xyz"
    assert resp.activations_b64 is None


# ── InferenceRunner forward pass ──────────────────────────────────────

def test_inference_runner_split_forward():
    """Split-model forward pass should produce the same hidden states as
    the full model's intermediate output.

    Uses a tiny model so it runs in CI without downloading GBs of weights.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import LlamaConfig, AutoModelForCausalLM
    from worker.inference import InferenceRunner

    # Build a tiny Llama config from scratch (no download needed).
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        torch_dtype=torch.float32,
    )
    model = AutoModelForCausalLM.from_config(config).eval()

    # Create runners: worker A = layers 0-2, worker B = layers 2-4.
    runner_a = InferenceRunner.__new__(InferenceRunner)
    runner_a.torch = torch
    runner_a.model_name = "test"
    runner_a.start_layer = 0
    runner_a.end_layer = 2
    runner_a.is_first = True
    runner_a.is_last = False
    runner_a.config = config
    runner_a.embed_tokens = model.model.embed_tokens
    runner_a.layers = model.model.layers[0:2]
    runner_a.norm = None
    runner_a.lm_head = None

    runner_b = InferenceRunner.__new__(InferenceRunner)
    runner_b.torch = torch
    runner_b.model_name = "test"
    runner_b.start_layer = 2
    runner_b.end_layer = 4
    runner_b.is_first = False
    runner_b.is_last = True
    runner_b.config = config
    runner_b.embed_tokens = None
    runner_b.layers = model.model.layers[2:4]
    runner_b.norm = model.model.norm
    runner_b.lm_head = model.lm_head

    token_ids = [5, 10, 15, 20]

    # Split forward pass.
    hidden = runner_a.forward_from_tokens(token_ids)
    logits_split = runner_b.forward_from_activations(hidden, is_last=True)

    # Full-model reference.
    with torch.no_grad():
        logits_full = model(torch.tensor([token_ids])).logits

    assert logits_split.shape == logits_full.shape
    assert torch.allclose(logits_split, logits_full, atol=1e-5)


def test_inference_runner_split_matches_full_argmax():
    """Greedy decode from a split pipeline should match the full model."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import LlamaConfig, AutoModelForCausalLM
    from worker.inference import InferenceRunner

    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        torch_dtype=torch.float32,
    )
    model = AutoModelForCausalLM.from_config(config).eval()

    runner_a = InferenceRunner.__new__(InferenceRunner)
    runner_a.torch = torch
    runner_a.model_name = "test"
    runner_a.start_layer = 0
    runner_a.end_layer = 2
    runner_a.is_first = True
    runner_a.is_last = False
    runner_a.config = config
    runner_a.embed_tokens = model.model.embed_tokens
    runner_a.layers = model.model.layers[0:2]
    runner_a.norm = None
    runner_a.lm_head = None

    runner_b = InferenceRunner.__new__(InferenceRunner)
    runner_b.torch = torch
    runner_b.model_name = "test"
    runner_b.start_layer = 2
    runner_b.end_layer = 4
    runner_b.is_first = False
    runner_b.is_last = True
    runner_b.config = config
    runner_b.embed_tokens = None
    runner_b.layers = model.model.layers[2:4]
    runner_b.norm = model.model.norm
    runner_b.lm_head = model.lm_head

    token_ids = [5, 10, 15, 20]

    hidden = runner_a.forward_from_tokens(token_ids)
    logits = runner_b.forward_from_activations(hidden, is_last=True)

    with torch.no_grad():
        full_logits = model(torch.tensor([token_ids])).logits

    split_token = int(torch.argmax(logits[0, -1, :]).item())
    full_token = int(torch.argmax(full_logits[0, -1, :]).item())
    assert split_token == full_token


def test_inference_runner_single_worker_matches_full():
    """A single-worker pipeline (all layers on one runner) should match the full model."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import LlamaConfig, AutoModelForCausalLM
    from worker.inference import InferenceRunner

    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        torch_dtype=torch.float32,
    )
    model = AutoModelForCausalLM.from_config(config).eval()

    # Single runner: all layers, both first and last.
    runner = InferenceRunner.__new__(InferenceRunner)
    runner.torch = torch
    runner.model_name = "test"
    runner.start_layer = 0
    runner.end_layer = 4
    runner.is_first = True
    runner.is_last = True
    runner.config = config
    runner.embed_tokens = model.model.embed_tokens
    runner.layers = model.model.layers[0:4]
    runner.norm = model.model.norm
    runner.lm_head = model.lm_head

    token_ids = [5, 10, 15, 20]
    logits = runner.forward_from_tokens_to_logits(token_ids)

    with torch.no_grad():
        full_logits = model(torch.tensor([token_ids])).logits

    assert torch.allclose(logits, full_logits, atol=1e-5)
    single_token = int(torch.argmax(logits[0, -1, :]).item())
    full_token = int(torch.argmax(full_logits[0, -1, :]).item())
    assert single_token == full_token


# ── Pipeline endpoint ─────────────────────────────────────────────────

def test_pipeline_endpoint_empty():
    from fastapi.testclient import TestClient
    from coordinator.main import create_app

    with TestClient(create_app()) as c:
        resp = c.get("/pipeline")
        assert resp.status_code == 200
        assert resp.json() == []


def test_generate_no_workers():
    from fastapi.testclient import TestClient
    from coordinator.main import create_app

    with TestClient(create_app()) as c:
        resp = c.post("/generate", json={"prompt": "hello"})
        assert resp.status_code == 400
        assert "no inference workers" in resp.json()["detail"].lower()

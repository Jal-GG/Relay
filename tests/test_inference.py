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
    runner_a.rotary_emb = model.model.rotary_emb
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
    runner_b.rotary_emb = model.model.rotary_emb
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
    runner_a.rotary_emb = model.model.rotary_emb
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
    runner_b.rotary_emb = model.model.rotary_emb
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
    runner.rotary_emb = model.model.rotary_emb
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


# ── split_model validation ─────────────────────────────────────────────

def test_parse_splits_valid():
    from tools.split_model import _parse_splits

    assert _parse_splits("0:11,11:22") == [(0, 11), (11, 22)]
    assert _parse_splits("0:4,4:8,8:12") == [(0, 4), (4, 8), (8, 12)]


def test_parse_splits_overlap():
    import pytest
    from tools.split_model import _parse_splits

    with pytest.raises(ValueError, match="overlapping"):
        _parse_splits("0:11,10:22")


def test_parse_splits_gap():
    import pytest
    from tools.split_model import _parse_splits

    with pytest.raises(ValueError, match="gap"):
        _parse_splits("0:5,10:22")


# ── serialization: weights_only explicitly False ───────────────────────

def test_deserialize_tensor_weights_only():
    """torch.load must not raise FutureWarning about weights_only."""
    import warnings
    torch = pytest.importorskip("torch")
    from common.serialization import serialize_tensor, deserialize_tensor

    t = torch.randn(3, 4)
    encoded = serialize_tensor(t)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        decoded = deserialize_tensor(encoded)  # would raise if weights_only missing
    assert torch.equal(t, decoded)


# ── Phase 4 — Backend abstraction ────────────────────────────────────

def test_cpu_backend_device():
    from worker.backends import CpuBackend
    backend = CpuBackend()
    assert backend.device == "cpu"


def test_cuda_backend_fallback_without_gpu():
    """CudaBackend should fall back to CPU gracefully when no GPU is available."""
    torch = pytest.importorskip("torch")
    from worker.backends import CudaBackend

    if torch.cuda.is_available():
        pytest.skip("CUDA is available — fallback path not testable on this machine")

    backend = CudaBackend()
    assert backend.device == "cpu"
    assert backend.is_cuda is False


def test_create_backend_auto():
    """create_backend('auto') should pick CUDA if available, else CPU."""
    torch = pytest.importorskip("torch")
    from worker.backends import create_backend, CpuBackend, CudaBackend

    backend = create_backend("auto")
    if torch.cuda.is_available():
        assert isinstance(backend, CudaBackend)
    else:
        assert isinstance(backend, CpuBackend)


def test_create_backend_cpu_forced():
    from worker.backends import create_backend, CpuBackend
    backend = create_backend("cpu")
    assert isinstance(backend, CpuBackend)
    assert backend.device == "cpu"


def test_backend_move_tensor():
    """CpuBackend.move_tensor should keep tensor on CPU."""
    torch = pytest.importorskip("torch")
    from worker.backends import CpuBackend

    backend = CpuBackend()
    t = torch.randn(2, 3)
    moved = backend.move_tensor(t)
    assert moved.device.type == "cpu"


def test_backend_to_cpu():
    """to_cpu should return a CPU tensor regardless of backend."""
    torch = pytest.importorskip("torch")
    from worker.backends import CpuBackend

    backend = CpuBackend()
    t = torch.randn(2, 3)
    result = backend.to_cpu(t)
    assert result.device.type == "cpu"
    assert torch.equal(result, t)


def test_inference_runner_accepts_device_param():
    """InferenceRunner should accept and store the device parameter."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import LlamaConfig, AutoModelForCausalLM
    from worker.inference import InferenceRunner

    config = LlamaConfig(
        vocab_size=100, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, torch_dtype=torch.float32,
    )
    model = AutoModelForCausalLM.from_config(config).eval()

    runner = InferenceRunner.__new__(InferenceRunner)
    runner.torch = torch
    runner.model_name = "test"
    runner.start_layer = 0
    runner.end_layer = 4
    runner.is_first = True
    runner.is_last = True
    runner.device = "cpu"
    from worker.backends import create_backend
    runner.backend = create_backend("cpu")
    runner.config = config
    runner.embed_tokens = model.model.embed_tokens
    runner.layers = model.model.layers[0:4]
    runner.rotary_emb = getattr(model.model, "rotary_emb", None)
    runner.norm = model.model.norm
    runner.lm_head = model.lm_head
    runner._move_to_device()

    token_ids = [5, 10, 15, 20]
    logits = runner.forward_from_tokens_to_logits(token_ids)
    assert logits.device.type == "cpu"

    with torch.no_grad():
        full_logits = model(torch.tensor([token_ids])).logits
    assert torch.allclose(logits, full_logits, atol=1e-5)


def test_pipeline_stage_includes_device():
    """PipelineStage model should accept the device field."""
    from common.models import PipelineStage

    stage = PipelineStage(
        worker_id="w1", ip="10.0.0.1", inference_port=9001,
        layer_start=0, layer_end=11, model_name="test", status="online",
        device="cuda",
    )
    assert stage.device == "cuda"


def test_register_with_device():
    """RegisterRequest should accept and serialize the device field."""
    from common.models import RegisterRequest, HardwareInfo

    req = RegisterRequest(
        worker_id="w1", hostname="w1", ip="10.0.0.1",
        hardware=HardwareInfo(
            os="test", cpu_cores=4, ram_total_mb=8192, ram_free_mb=4096,
        ),
        inference_port=9001, layer_start=0, layer_end=11,
        model_name="test", device="cuda",
    )
    d = req.model_dump()
    assert d["device"] == "cuda"

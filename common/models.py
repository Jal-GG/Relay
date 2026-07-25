from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GpuInfo(BaseModel):
    name: str
    vram_total_mb: int
    vram_free_mb: int


class HardwareInfo(BaseModel):
    os: str
    cpu_cores: int
    ram_total_mb: int
    ram_free_mb: int
    cpu_percent: float = 0.0
    gpu: Optional[GpuInfo] = None  # Optional kept: pynvml may be absent
    network_latency_ms: float = 0.0
    network_bandwidth_mbps: float = 0.0


class RegisterRequest(BaseModel):
    worker_id: str = Field(description="Stable client-generated id (e.g. hostname)")
    hostname: str
    ip: str
    hardware: HardwareInfo
    # Phase 3 — inference capabilities
    inference_port: Optional[int] = None
    layer_start: Optional[int] = None
    layer_end: Optional[int] = None
    model_name: Optional[str] = None
    # Phase 4 — heterogeneous execution
    device: Optional[str] = None  # "cpu", "cuda", or None if not an inference worker


class HeartbeatRequest(BaseModel):
    worker_id: str
    hardware: HardwareInfo


class WorkerStatus(BaseModel):
    worker_id: str
    hostname: str
    ip: str
    status: Literal["online", "offline"] = "online"
    last_seen: datetime
    hardware: HardwareInfo
    # Phase 3 — inference capabilities
    inference_port: Optional[int] = None
    layer_start: Optional[int] = None
    layer_end: Optional[int] = None
    model_name: Optional[str] = None
    # Phase 4 — heterogeneous execution
    device: Optional[str] = None

    def is_stale(self, heartbeat_timeout_s: float) -> bool:
        age = (utcnow() - self.last_seen).total_seconds()
        return age > heartbeat_timeout_s

    @property
    def has_inference(self) -> bool:
        """True if this worker registered with inference capabilities."""
        return self.inference_port is not None and self.model_name is not None

    @property
    def inference_url(self) -> str:
        """Base URL for the worker's inference HTTP server."""
        return f"http://{self.ip}:{self.inference_port}"

    @property
    def network_bandwidth_mbps(self) -> float:
        return self.hardware.network_bandwidth_mbps


# ── Phase 2 — Remote task execution ──────────────────────────────────


class TaskType(str, Enum):
    matrix_multiply = "matrix_multiply"
    hash_file = "hash_file"
    sleep = "sleep"


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    timed_out = "timed_out"


class TaskRequest(BaseModel):
    type: TaskType
    params: dict[str, Any] = Field(default_factory=dict)
    deadline_s: float = 60.0


class Task(BaseModel):
    task_id: str
    type: TaskType
    params: dict[str, Any] = Field(default_factory=dict)
    status: TaskStatus = TaskStatus.pending
    worker_id: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline_s: float = 60.0
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class TaskResultRequest(BaseModel):
    task_id: str
    worker_id: str
    success: bool
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


# ── Phase 3 — Distributed model loading ──────────────────────────────


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.0  # 0 = greedy/deterministic
    model_name: Optional[str] = None


class GenerateResponse(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    elapsed_ms: float


class ForwardRequest(BaseModel):
    """Sent by the coordinator to a worker's inference server."""
    token_ids: Optional[list[int]] = None  # for the first worker
    activations_b64: Optional[str] = None  # for subsequent workers
    is_last: bool = False  # whether this worker should produce final logits


class ForwardResponse(BaseModel):
    """Returned by a worker's inference server after a forward pass."""
    activations_b64: Optional[str] = None  # intermediate output
    logits_b64: Optional[str] = None  # final output (last worker only)


class ConfigureRequest(BaseModel):
    model_name: str
    layer_start: int
    layer_end: int
    is_last: bool = False
    split_path: Optional[str] = None
    device: str = "auto"


class PipelineStage(BaseModel):
    """Read-only view of one stage in the inference pipeline."""
    worker_id: str
    ip: str
    inference_port: int
    layer_start: int
    layer_end: int
    model_name: str
    status: str
    device: Optional[str] = None  # Phase 4 — "cpu" or "cuda"


class LayerRange(BaseModel):
    start: int
    end: int


class ModelMetadata(BaseModel):
    model_name: str
    num_layers: int
    size_mb: int

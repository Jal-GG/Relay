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


class RegisterRequest(BaseModel):
    worker_id: str = Field(description="Stable client-generated id (e.g. hostname)")
    hostname: str
    ip: str
    hardware: HardwareInfo


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

    def is_stale(self, heartbeat_timeout_s: float) -> bool:
        age = (utcnow() - self.last_seen).total_seconds()
        return age > heartbeat_timeout_s


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

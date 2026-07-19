from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

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
    gpu: Optional[GpuInfo] = None


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
    status: str = "online"
    last_seen: datetime
    hardware: HardwareInfo

    def is_stale(self, heartbeat_timeout_s: float) -> bool:
        age = (utcnow() - self.last_seen).total_seconds()
        return age > heartbeat_timeout_s

from __future__ import annotations

import threading

from common.models import HeartbeatRequest, RegisterRequest, WorkerStatus, utcnow
from common.protocol import HEARTBEAT_TIMEOUT_S


class ClusterState:
    def __init__(self, heartbeat_timeout_s: float = HEARTBEAT_TIMEOUT_S) -> None:
        self._workers: dict[str, WorkerStatus] = {}
        self._lock = threading.Lock()
        self._heartbeat_timeout_s = heartbeat_timeout_s

    def register(self, req: RegisterRequest) -> WorkerStatus:
        with self._lock:
            status = WorkerStatus(
                worker_id=req.worker_id,
                hostname=req.hostname,
                ip=req.ip,
                status="online",
                last_seen=utcnow(),
                hardware=req.hardware,
            )
            self._workers[req.worker_id] = status
            return status

    def heartbeat(self, req: HeartbeatRequest) -> WorkerStatus:
        with self._lock:
            existing = self._workers.get(req.worker_id)
            if existing is None:
                raise KeyError(f"unknown worker_id {req.worker_id!r}; must register first")
            existing.hardware = req.hardware
            existing.last_seen = utcnow()
            existing.status = "online"
            return existing

    def sweep_stale(self) -> None:
        """Mark workers offline if they haven't sent a heartbeat within the timeout window.
        Called periodically by the background sweeper in the coordinator process.
        """
        with self._lock:
            for status in self._workers.values():
                if status.status == "online" and status.is_stale(self._heartbeat_timeout_s):
                    status.status = "offline"

    def _mark_stale_if_needed(self, status: WorkerStatus) -> None:
        """Inline check used by read paths that must reflect staleness immediately."""
        if status.is_stale(self._heartbeat_timeout_s) and status.status == "online":
            status.status = "offline"

    def list_workers(self) -> list[WorkerStatus]:
        with self._lock:
            # Ensure reads see up-to-date status even between sweep intervals.
            for status in self._workers.values():
                self._mark_stale_if_needed(status)
            return list(self._workers.values())

    def get_worker(self, worker_id: str) -> WorkerStatus | None:
        with self._lock:
            status = self._workers.get(worker_id)
            if status is not None:
                self._mark_stale_if_needed(status)
            return status

    def remove_worker(self, worker_id: str) -> None:
        with self._lock:
            self._workers.pop(worker_id, None)

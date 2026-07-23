from __future__ import annotations

import threading
import uuid

from common.models import (
    HeartbeatRequest,
    RegisterRequest,
    Task,
    TaskRequest,
    TaskResultRequest,
    TaskStatus,
    WorkerStatus,
    utcnow,
)
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
                inference_port=req.inference_port,
                layer_start=req.layer_start,
                layer_end=req.layer_end,
                model_name=req.model_name,
                device=req.device,
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


class TaskState:
    """In-memory task store. Coordinates with ClusterState to detect dead workers."""

    def __init__(
        self,
        cluster_state: ClusterState,
    ) -> None:
        self._tasks: dict[str, Task] = {}
        self._cluster = cluster_state
        self._lock = threading.Lock()

    def submit(self, req: TaskRequest) -> Task:
        with self._lock:
            task = Task(
                task_id=str(uuid.uuid4()),
                type=req.type,
                params=req.params,
                status=TaskStatus.pending,
                created_at=utcnow(),
                deadline_s=req.deadline_s,
            )
            self._tasks[task.task_id] = task
            return task

    def poll(self, worker_id: str) -> Task | None:
        """Assign the next pending task to *worker_id* and return it."""
        with self._lock:
            for task in self._tasks.values():
                if task.status == TaskStatus.pending:
                    task.status = TaskStatus.running
                    task.worker_id = worker_id
                    task.started_at = utcnow()
                    return task
            return None

    def submit_result(self, req: TaskResultRequest) -> Task:
        with self._lock:
            task = self._tasks.get(req.task_id)
            if task is None:
                raise KeyError(f"unknown task_id {req.task_id!r}")
            # Ignore late results for tasks already in a terminal state.
            if task.status in (TaskStatus.completed, TaskStatus.failed, TaskStatus.timed_out):
                return task
            task.status = TaskStatus.completed if req.success else TaskStatus.failed
            task.result = req.result
            task.error = req.error
            task.completed_at = utcnow()
            return task

    def sweep(self) -> None:
        """Mark running tasks as failed (worker offline) or timed_out (deadline exceeded).

        Snapshot worker statuses first (under cluster's lock), then process tasks
        under our own lock. This avoids holding two locks simultaneously.
        """
        # Step 1: snapshot — only cluster._lock held here.
        offline_workers: set[str] = {
            w.worker_id
            for w in self._cluster.list_workers()
            if w.status == "offline"
        }
        # Step 2: update tasks — only self._lock held here.
        with self._lock:
            now = utcnow()
            for task in self._tasks.values():
                if task.status != TaskStatus.running:
                    continue
                if task.worker_id and task.worker_id in offline_workers:
                    task.status = TaskStatus.failed
                    task.error = "worker went offline"
                    task.completed_at = now
                elif task.started_at is not None:
                    elapsed = (now - task.started_at).total_seconds()
                    if elapsed > task.deadline_s:
                        task.status = TaskStatus.timed_out
                        task.error = f"timed out after {task.deadline_s}s"
                        task.completed_at = now

    def list_tasks(self) -> list[Task]:
        with self._lock:
            return list(self._tasks.values())

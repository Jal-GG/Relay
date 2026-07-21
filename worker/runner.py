from __future__ import annotations

import base64
import hashlib
import logging
import threading
import time

import numpy as np

from common.models import Task, TaskResultRequest, TaskType

log = logging.getLogger("relay.worker.runner")


def execute_task(
    task: Task,
    stop_event: threading.Event | None = None,
) -> TaskResultRequest:
    """Execute a task and produce a result request to send back to the coordinator."""
    if task.type == TaskType.matrix_multiply:
        return _matrix_multiply(task)
    if task.type == TaskType.hash_file:
        return _hash_file(task)
    if task.type == TaskType.sleep:
        return _sleep(task, stop_event)
    return TaskResultRequest(
        task_id=task.task_id,
        worker_id=task.worker_id or "",
        success=False,
        error=f"unknown task type: {task.type}",
    )


def _matrix_multiply(task: Task) -> TaskResultRequest:
    n = int(task.params.get("n", 100))
    start = time.perf_counter()
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    c = a @ b
    elapsed_ms = (time.perf_counter() - start) * 1000
    log.info("matrix_multiply n=%d done in %.1fms", n, elapsed_ms)
    return TaskResultRequest(
        task_id=task.task_id,
        worker_id=task.worker_id or "",
        success=True,
        result={"shape": [n, n], "time_ms": round(elapsed_ms, 2)},
    )


def _hash_file(task: Task) -> TaskResultRequest:
    payload_b64 = task.params.get("payload", "")
    payload = base64.b64decode(payload_b64) if payload_b64 else b""
    h = hashlib.sha256(payload).hexdigest()
    log.info("hash_file payload=%d bytes sha256=%s…", len(payload), h[:16])
    return TaskResultRequest(
        task_id=task.task_id,
        worker_id=task.worker_id or "",
        success=True,
        result={"hash": h, "size": len(payload)},
    )


def _sleep(
    task: Task,
    stop_event: threading.Event | None = None,
) -> TaskResultRequest:
    seconds = float(task.params.get("seconds", 1.0))
    # Use stop_event.wait so shutdown signals interrupt the sleep immediately.
    if stop_event is not None:
        stop_event.wait(timeout=seconds)
    else:
        time.sleep(seconds)
    log.info("sleep %.1fs done", seconds)
    return TaskResultRequest(
        task_id=task.task_id,
        worker_id=task.worker_id or "",
        success=True,
        result={"slept": seconds},
    )

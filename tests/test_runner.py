from __future__ import annotations

import base64
import hashlib
import time

from common.models import Task, TaskType
from common.models import utcnow
from worker.runner import execute_task


def _make_task(task_type: TaskType, params: dict | None = None) -> Task:
    return Task(
        task_id="test-1",
        type=task_type,
        params=params or {},
        status="pending",
        created_at=utcnow(),
        worker_id="w1",
    )


def test_matrix_multiply():
    task = _make_task(TaskType.matrix_multiply, {"n": 50})
    result = execute_task(task)
    assert result.success
    assert result.result["shape"] == [50, 50]
    assert result.result["time_ms"] >= 0


def test_hash_file():
    payload = b"hello relay"
    task = _make_task(TaskType.hash_file, {"payload": base64.b64encode(payload).decode()})
    result = execute_task(task)
    assert result.success
    assert result.result["hash"] == hashlib.sha256(payload).hexdigest()
    assert result.result["size"] == len(payload)


def test_hash_file_empty_payload():
    task = _make_task(TaskType.hash_file, {})
    result = execute_task(task)
    assert result.success
    assert result.result["hash"] == hashlib.sha256(b"").hexdigest()
    assert result.result["size"] == 0


def test_sleep():
    task = _make_task(TaskType.sleep, {"seconds": 0.05})
    start = time.perf_counter()
    result = execute_task(task)
    elapsed = time.perf_counter() - start
    assert result.success
    assert result.result["slept"] == 0.05
    assert elapsed >= 0.04


def test_unknown_task_type():
    task = _make_task(TaskType.matrix_multiply)
    # Bypass the enum to simulate an unsupported type arriving over the wire.
    bad_task = task.model_copy(update={"type": "bogus"})
    result = execute_task(bad_task)
    assert not result.success
    assert "unknown task type" in (result.error or "")


def test_sleep_interruptible():
    """sleep task should return immediately when stop_event is already set."""
    import threading

    stop = threading.Event()
    stop.set()
    task = _make_task(TaskType.sleep, {"seconds": 60.0})
    start = time.perf_counter()
    result = execute_task(task, stop_event=stop)
    elapsed = time.perf_counter() - start
    assert result.success
    assert elapsed < 1.0, f"sleep was not interrupted: took {elapsed:.2f}s"

from __future__ import annotations

import time
from datetime import timedelta

from common.models import (
    HardwareInfo,
    RegisterRequest,
    TaskRequest,
    TaskResultRequest,
    TaskStatus,
    TaskType,
    utcnow,
)
from common.protocol import HEARTBEAT_TIMEOUT_S
from coordinator.state import ClusterState, TaskState


def _register_worker(cluster: ClusterState, worker_id: str = "w1") -> None:
    cluster.register(RegisterRequest(
        worker_id=worker_id,
        hostname=worker_id,
        ip="127.0.0.1",
        hardware=HardwareInfo(
            os="test",
            cpu_cores=4,
            ram_total_mb=8192,
            ram_free_mb=4096,
        ),
    ))


def test_submit_creates_pending_task():
    cluster = ClusterState()
    tasks = TaskState(cluster)
    task = tasks.submit(TaskRequest(type=TaskType.sleep, params={"seconds": 0.1}))
    assert task.status == TaskStatus.pending
    assert task.type == TaskType.sleep
    assert task.task_id


def test_poll_assigns_task():
    cluster = ClusterState()
    tasks = TaskState(cluster)
    tasks.submit(TaskRequest(type=TaskType.sleep, params={"seconds": 0.1}))
    task = tasks.poll("w1")
    assert task is not None
    assert task.status == TaskStatus.running
    assert task.worker_id == "w1"
    assert task.started_at is not None


def test_poll_returns_none_when_empty():
    cluster = ClusterState()
    tasks = TaskState(cluster)
    assert tasks.poll("w1") is None


def test_poll_skips_non_pending_tasks():
    cluster = ClusterState()
    tasks = TaskState(cluster)
    tasks.submit(TaskRequest(type=TaskType.sleep))
    task = tasks.poll("w1")
    assert task is not None
    # Second poll should find nothing — the only task is now running.
    assert tasks.poll("w1") is None


def test_submit_result_completed():
    cluster = ClusterState()
    tasks = TaskState(cluster)
    task = tasks.submit(TaskRequest(type=TaskType.sleep))
    tasks.poll("w1")
    result = tasks.submit_result(TaskResultRequest(
        task_id=task.task_id,
        worker_id="w1",
        success=True,
        result={"slept": 1.0},
    ))
    assert result.status == TaskStatus.completed
    assert result.result == {"slept": 1.0}
    assert result.completed_at is not None


def test_submit_result_failed():
    cluster = ClusterState()
    tasks = TaskState(cluster)
    task = tasks.submit(TaskRequest(type=TaskType.sleep))
    tasks.poll("w1")
    result = tasks.submit_result(TaskResultRequest(
        task_id=task.task_id,
        worker_id="w1",
        success=False,
        error="boom",
    ))
    assert result.status == TaskStatus.failed
    assert result.error == "boom"


def test_submit_result_ignored_for_terminal_task():
    cluster = ClusterState()
    tasks = TaskState(cluster)
    task = tasks.submit(TaskRequest(type=TaskType.sleep))
    tasks.poll("w1")
    tasks.submit_result(TaskResultRequest(
        task_id=task.task_id, worker_id="w1", success=True, result={"slept": 1},
    ))
    # Late result — should not change anything.
    result = tasks.submit_result(TaskResultRequest(
        task_id=task.task_id, worker_id="w1", success=False, error="late",
    ))
    assert result.status == TaskStatus.completed
    assert result.result == {"slept": 1}


def test_sweep_marks_failed_on_offline_worker():
    cluster = ClusterState(heartbeat_timeout_s=HEARTBEAT_TIMEOUT_S)
    tasks = TaskState(cluster)
    _register_worker(cluster, "w1")
    task = tasks.submit(TaskRequest(type=TaskType.sleep, params={"seconds": 100}))
    tasks.poll("w1")
    # Simulate worker going offline by backdating last_seen past the heartbeat timeout.
    with cluster._lock:
        cluster._workers["w1"].last_seen = utcnow() - timedelta(seconds=HEARTBEAT_TIMEOUT_S + 1)
    cluster.sweep_stale()
    tasks.sweep()
    updated = tasks.list_tasks()[0]
    assert updated.status == TaskStatus.failed
    assert "offline" in (updated.error or "")


def test_sweep_marks_timed_out():
    cluster = ClusterState()
    tasks = TaskState(cluster)
    _register_worker(cluster, "w1")
    task = tasks.submit(TaskRequest(type=TaskType.sleep, deadline_s=0.01))
    tasks.poll("w1")
    time.sleep(0.05)
    tasks.sweep()
    updated = tasks.list_tasks()[0]
    assert updated.status == TaskStatus.timed_out
    assert "timed out" in (updated.error or "")


def test_list_tasks_returns_all():
    cluster = ClusterState()
    tasks = TaskState(cluster)
    tasks.submit(TaskRequest(type=TaskType.sleep))
    tasks.submit(TaskRequest(type=TaskType.hash_file))
    assert len(tasks.list_tasks()) == 2


def test_sweep_does_not_deadlock_with_concurrent_list_workers():
    """sweep() snapshots worker status without holding both locks simultaneously.
    Run sweep() and list_workers() concurrently to confirm no deadlock occurs.
    """
    import threading as _threading

    cluster = ClusterState()
    tasks = TaskState(cluster)
    _register_worker(cluster, "w1")
    tasks.submit(TaskRequest(type=TaskType.sleep, params={"seconds": 100}))
    tasks.poll("w1")

    errors: list[Exception] = []

    def _sweep_loop():
        for _ in range(50):
            try:
                tasks.sweep()
            except Exception as exc:
                errors.append(exc)

    def _list_loop():
        for _ in range(50):
            try:
                cluster.list_workers()
            except Exception as exc:
                errors.append(exc)

    t1 = _threading.Thread(target=_sweep_loop)
    t2 = _threading.Thread(target=_list_loop)
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)
    assert not t1.is_alive(), "sweep thread did not finish — possible deadlock"
    assert not t2.is_alive(), "list_workers thread did not finish — possible deadlock"
    assert not errors

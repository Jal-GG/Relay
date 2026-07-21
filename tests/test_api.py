from __future__ import annotations

from fastapi.testclient import TestClient

from coordinator.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_submit_task():
    with _client() as c:
        resp = c.post("/tasks", json={"type": "sleep", "params": {"seconds": 0.01}})
        assert resp.status_code == 200
        body = resp.json()
        assert body["task_id"]
        assert body["status"] == "pending"


def test_poll_no_tasks():
    with _client() as c:
        resp = c.get("/tasks/poll", params={"worker_id": "w1"})
        assert resp.status_code == 200
        assert resp.json()["task_id"] is None


def test_poll_returns_task():
    with _client() as c:
        c.post("/tasks", json={"type": "sleep", "params": {"seconds": 0.01}})
        resp = c.get("/tasks/poll", params={"worker_id": "w1"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["task_id"] is not None
        assert body["status"] == "running"
        assert body["worker_id"] == "w1"


def test_submit_result():
    with _client() as c:
        submit = c.post("/tasks", json={"type": "sleep", "params": {"seconds": 0.01}})
        task_id = submit.json()["task_id"]
        c.get("/tasks/poll", params={"worker_id": "w1"})
        resp = c.post("/tasks/result", json={
            "task_id": task_id,
            "worker_id": "w1",
            "success": True,
            "result": {"slept": 0.01},
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"


def test_list_tasks():
    with _client() as c:
        c.post("/tasks", json={"type": "sleep", "params": {"seconds": 0.01}})
        c.post("/tasks", json={"type": "hash_file", "params": {}})
        resp = c.get("/tasks")
        assert resp.status_code == 200
        tasks = resp.json()
        assert len(tasks) == 2


def test_submit_result_unknown_task():
    with _client() as c:
        resp = c.post("/tasks/result", json={
            "task_id": "nonexistent",
            "worker_id": "w1",
            "success": True,
        })
        assert resp.status_code == 404


def test_workers_and_dashboard_still_work():
    with _client() as c:
        assert c.get("/workers").status_code == 200
        assert c.get("/").status_code == 200

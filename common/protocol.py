from __future__ import annotations

HEARTBEAT_INTERVAL_S = 5.0
HEARTBEAT_TIMEOUT_S = 15.0
DASHBOARD_REFRESH_S = 3

TASK_POLL_INTERVAL_S = 1.0
DEFAULT_TASK_DEADLINE_S = 60.0
TASK_SWEEP_INTERVAL_S = 1.0

# Phase 3 — inference defaults
DEFAULT_INFER_TIMEOUT_S = 120.0
DEFAULT_MAX_NEW_TOKENS = 50

ROUTES = {
    "register": "/register",
    "heartbeat": "/heartbeat",
    "workers": "/workers",
    "dashboard": "/",
    "tasks": "/tasks",
    "task_poll": "/tasks/poll",
    "task_result": "/tasks/result",
    # Phase 3
    "generate": "/generate",
    "pipeline": "/pipeline",
    "forward": "/forward",
}

DEFAULT_PORT = 8000
DEFAULT_INFERENCE_PORT = 9001

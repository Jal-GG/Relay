from __future__ import annotations

HEARTBEAT_INTERVAL_S = 5.0
HEARTBEAT_TIMEOUT_S = 15.0
DASHBOARD_REFRESH_S = 3

ROUTES = {
    "register": "/register",
    "heartbeat": "/heartbeat",
    "workers": "/workers",
    "dashboard": "/",
}

DEFAULT_PORT = 8000

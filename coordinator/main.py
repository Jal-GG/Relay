from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from common.models import HeartbeatRequest, RegisterRequest
from common.protocol import ROUTES
from coordinator.state import ClusterState


DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"
_SWEEP_INTERVAL_S = 1.0  # how often the background thread checks for stale workers


def _start_staleness_sweeper(state: ClusterState, stop_event: threading.Event) -> None:
    """Flip stale workers to offline in the background so status is current at all times."""
    def _sweep():
        while not stop_event.wait(timeout=_SWEEP_INTERVAL_S):
            state.sweep_stale()
    t = threading.Thread(target=_sweep, daemon=True, name="staleness-sweeper")
    t.start()


def create_app() -> FastAPI:
    state = ClusterState()
    stop_event = threading.Event()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _start_staleness_sweeper(state, stop_event)
        yield
        stop_event.set()

    app = FastAPI(title="Relay Coordinator", version="0.1.0", lifespan=lifespan)

    @app.post(ROUTES["register"])
    def register(req: RegisterRequest) -> dict:
        status = state.register(req)
        return {"status": "registered", "worker_id": status.worker_id}

    @app.post(ROUTES["heartbeat"])
    def heartbeat(req: HeartbeatRequest) -> dict:
        try:
            status = state.heartbeat(req)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"status": "ok", "worker_id": status.worker_id}

    @app.get(ROUTES["workers"])
    def list_workers() -> JSONResponse:
        workers = state.list_workers()
        return JSONResponse([
            w.model_dump(mode="json") for w in workers
        ])

    @app.get(ROUTES["dashboard"], response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        index_path = DASHBOARD_DIR / "index.html"
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    return app


app = create_app()


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Relay coordinator")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()
    uvicorn.run("coordinator.main:app", host=args.host, port=args.port, reload=False)

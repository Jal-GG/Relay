from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from common.models import HeartbeatRequest, RegisterRequest
from common.protocol import ROUTES
from coordinator.state import ClusterState, workers_to_dict


DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"


def create_app() -> FastAPI:
    app = FastAPI(title="Relay Coordinator", version="0.1.0")
    state = ClusterState()

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
        return JSONResponse(workers_to_dict(state.list_workers()))

    @app.get(ROUTES["dashboard"], response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        index_path = DASHBOARD_DIR / "index.html"
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("coordinator.main:app", host="0.0.0.0", port=8000, reload=False)

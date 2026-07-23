from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from common.models import (
    GenerateRequest,
    HeartbeatRequest,
    RegisterRequest,
    TaskRequest,
    TaskResultRequest,
)
from common.protocol import ROUTES, TASK_SWEEP_INTERVAL_S
from coordinator.pipeline import Pipeline
from coordinator.state import ClusterState, TaskState


DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"


def _start_staleness_sweeper(state: ClusterState, stop_event: threading.Event) -> None:
    """Flip stale workers to offline in the background so status is current at all times."""
    def _sweep():
        while not stop_event.wait(timeout=TASK_SWEEP_INTERVAL_S):
            state.sweep_stale()
    t = threading.Thread(target=_sweep, daemon=True, name="staleness-sweeper")
    t.start()


def _start_task_sweeper(tasks: TaskState, stop_event: threading.Event) -> None:
    """Mark running tasks failed/timed_out when their worker dies or deadline expires."""
    def _sweep():
        while not stop_event.wait(timeout=TASK_SWEEP_INTERVAL_S):
            tasks.sweep()
    t = threading.Thread(target=_sweep, daemon=True, name="task-sweeper")
    t.start()


def create_app() -> FastAPI:
    state = ClusterState()
    tasks = TaskState(state)
    pipeline = Pipeline(state)
    stop_event = threading.Event()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _start_staleness_sweeper(state, stop_event)
        _start_task_sweeper(tasks, stop_event)
        yield
        stop_event.set()

    app = FastAPI(title="Relay Coordinator", version="0.3.0", lifespan=lifespan)

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

    @app.post(ROUTES["tasks"])
    def submit_task(req: TaskRequest) -> JSONResponse:
        task = tasks.submit(req)
        return JSONResponse({"task_id": task.task_id, "status": task.status.value})

    @app.get(ROUTES["task_poll"])
    def poll_task(worker_id: str = Query(...)) -> JSONResponse:
        task = tasks.poll(worker_id)
        if task is None:
            return JSONResponse({"task_id": None})
        return JSONResponse(task.model_dump(mode="json"))

    @app.post(ROUTES["task_result"])
    def submit_result(req: TaskResultRequest) -> JSONResponse:
        try:
            task = tasks.submit_result(req)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return JSONResponse({"task_id": task.task_id, "status": task.status.value})

    @app.get(ROUTES["tasks"])
    def list_tasks() -> JSONResponse:
        return JSONResponse([t.model_dump(mode="json") for t in tasks.list_tasks()])

    # ── Phase 3 — distributed inference ──────────────────────────────

    @app.post(ROUTES["generate"])
    async def generate(req: GenerateRequest) -> JSONResponse:
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, pipeline.generate, req)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return JSONResponse(result.model_dump())

    @app.get(ROUTES["pipeline"])
    def get_pipeline() -> JSONResponse:
        return JSONResponse([s.model_dump() for s in pipeline.get_pipeline_view()])

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

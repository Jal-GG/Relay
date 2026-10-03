from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from urllib.parse import urljoin

import httpx

from common.models import HeartbeatRequest, RegisterRequest, Task, TaskResultRequest
from common.protocol import (
    DEFAULT_INFERENCE_PORT,
    HEARTBEAT_INTERVAL_S,
    ROUTES,
    TASK_POLL_INTERVAL_S,
)
from worker.hardware import collect_hardware, local_hostname, local_ip
from worker.runner import execute_task

log = logging.getLogger("relay.worker")


def build_register_request(
    worker_id: str | None,
    inference_port: int | None = None,
    layer_start: int | None = None,
    layer_end: int | None = None,
    model_name: str | None = None,
    device: str | None = None,
) -> RegisterRequest:
    hw = collect_hardware()
    return RegisterRequest(
        worker_id=worker_id or local_hostname(),
        hostname=local_hostname(),
        ip=local_ip(),
        hardware=hw,
        inference_port=inference_port,
        layer_start=layer_start,
        layer_end=layer_end,
        model_name=model_name,
        device=device,
    )


def make_heartbeat(worker_id: str) -> HeartbeatRequest:
    hw = collect_hardware()
    return HeartbeatRequest(worker_id=worker_id, hardware=hw)


def url(base: str, route: str) -> str:
    return urljoin(base if base.endswith("/") else base + "/", route.lstrip("/"))


def measure_network(coordinator_url: str, client: httpx.Client) -> tuple[float, float]:
    payload = "x" * (256 * 1024)
    started = time.perf_counter()
    response = client.post(
        url(coordinator_url, ROUTES["network_probe"]),
        json={"payload": payload},
    )
    response.raise_for_status()
    elapsed = time.perf_counter() - started
    bandwidth = (len(payload) * 8 / elapsed) / 1_000_000 if elapsed else 0.0
    return elapsed * 1000, bandwidth


def task_loop(
    coordinator_url: str,
    worker_id: str,
    client: httpx.Client,
    stop_event: threading.Event,
) -> None:
    """Poll the coordinator for tasks, execute them, and report results back."""
    while not stop_event.is_set():
        try:
            resp = client.get(
                url(coordinator_url, ROUTES["task_poll"]),
                params={"worker_id": worker_id},
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("task_id") is None:
                stop_event.wait(timeout=TASK_POLL_INTERVAL_S)
                continue

            task = Task.model_validate(data)
            log.info("received task %s (%s)", task.task_id, task.type.value)
            result = execute_task(task, stop_event)

            resp = client.post(
                url(coordinator_url, ROUTES["task_result"]),
                json=result.model_dump(),
            )
            resp.raise_for_status()
            log.info(
                "task %s %s",
                task.task_id,
                "completed" if result.success else f"failed: {result.error}",
            )
        except httpx.HTTPError as exc:
            log.warning("task poll/execute failed: %s", exc)
            stop_event.wait(timeout=TASK_POLL_INTERVAL_S)
        except Exception as exc:
            log.exception("unexpected task error: %s", exc)
            stop_event.wait(timeout=TASK_POLL_INTERVAL_S)


def _parse_layer_spec(spec: str) -> tuple[int, int]:
    """Parse '0:11' → (0, 11)."""
    start_s, end_s = spec.split(":")
    return int(start_s), int(end_s)


def _start_inference_server(
    model_name: str,
    layer_start: int,
    layer_end: int,
    inference_port: int,
    split_path: str | None,
    is_last: bool = True,
    device: str = "auto",
) -> tuple[object, int]:
    """Load model layers and start the inference HTTP server in a daemon thread.

    ``is_last`` controls whether the worker loads the final norm and LM-head
    weights.  Set it to True only for the last stage in the pipeline (or when
    running a single-worker pipeline).  Keeping it False on intermediate workers
    avoids loading ~2 × hidden_size² parameters unnecessarily.

    ``device`` selects the inference backend: 'auto' (GPU if available),
    'cpu', or 'cuda'.  See ``worker.backends.create_backend``.

    Returns ``(runner, port)``.
    """
    import uvicorn

    from worker.inference import InferenceRunner
    from worker.server import create_inference_app

    is_first = layer_start == 0

    runner = InferenceRunner(
        model_name=model_name,
        start_layer=layer_start,
        end_layer=layer_end,
        is_first=is_first,
        is_last=is_last,
        split_path=split_path,
        device=device,
    )
    app = create_inference_app(runner)

    config = uvicorn.Config(
        app, host="0.0.0.0", port=inference_port, log_level="warning",
    )
    server = uvicorn.Server(config)

    t = threading.Thread(target=server.run, daemon=True, name="inference-server")
    t.start()
    log.info("Inference server listening on port %d (device=%s)", inference_port, runner.backend.device)
    return runner, inference_port


def run(
    coordinator_url: str,
    worker_id: str | None,
    model_name: str | None = None,
    layer_spec: str | None = None,
    inference_port: int = DEFAULT_INFERENCE_PORT,
    split_path: str | None = None,
    is_last: bool = True,
    device: str = "auto",
) -> int:
    layer_start: int | None = None
    layer_end: int | None = None
    actual_device: str | None = None

    if model_name:
        layer_start, layer_end = _parse_layer_spec(layer_spec) if layer_spec else (0, 0)
        if layer_spec:
            runner, _ = _start_inference_server(
                model_name, layer_start, layer_end, inference_port, split_path,
                is_last=is_last, device=device,
            )
            actual_device = runner.backend.device
        else:
            from worker.server import create_inference_app
            import uvicorn

            app = create_inference_app()
            config = uvicorn.Config(app, host="0.0.0.0", port=inference_port, log_level="warning")
            server = uvicorn.Server(config)
            threading.Thread(target=server.run, daemon=True, name="inference-server").start()

    reg = build_register_request(
        worker_id,
        inference_port=inference_port if model_name else None,
        layer_start=layer_start,
        layer_end=layer_end,
        model_name=model_name,
        device=actual_device,
    )
    log.info("registering %s (%s) with %s", reg.worker_id, reg.ip, coordinator_url)

    with httpx.Client(timeout=10.0) as client:
        try:
            latency, bandwidth = measure_network(coordinator_url, client)
            reg.hardware.network_latency_ms = latency
            reg.hardware.network_bandwidth_mbps = bandwidth
        except httpx.HTTPError as exc:
            log.warning("network measurement failed: %s", exc)
        try:
            resp = client.post(url(coordinator_url, ROUTES["register"]), json=reg.model_dump())
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            log.error("registration failed: %s", exc)
            return 1
        log.info("registered as %s", reg.worker_id)

        stop_event = threading.Event()

        def handle_signal(signum, frame):
            log.info("received signal %s, stopping", signum)
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handle_signal)
            except (ValueError, OSError):
                # SIGTERM cannot always be installed on Windows — skip quietly.
                pass

        task_thread = threading.Thread(
            target=task_loop,
            args=(coordinator_url, reg.worker_id, client, stop_event),
            daemon=True,
            name="task-loop",
        )
        task_thread.start()

        while not stop_event.is_set():
            try:
                hb = make_heartbeat(reg.worker_id)
                resp = client.post(url(coordinator_url, ROUTES["heartbeat"]), json=hb.model_dump())
                resp.raise_for_status()
                log.debug("heartbeat ok: cpu=%.1f%% ram_free=%dMB",
                          hb.hardware.cpu_percent, hb.hardware.ram_free_mb)
            except httpx.HTTPError as exc:
                log.warning("heartbeat failed: %s", exc)
            except Exception as exc:
                log.exception("unexpected heartbeat error: %s", exc)

            stop_event.wait(timeout=HEARTBEAT_INTERVAL_S)

        # Wait for the task thread to finish any in-flight request before closing the client.
        task_thread.join(timeout=5.0)

    log.info("worker %s shut down cleanly", reg.worker_id)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Relay worker process")
    p.add_argument("coordinator", help="Coordinator base URL, e.g. http://192.168.1.10:8000")
    p.add_argument("--worker-id", default=None, help="Override worker id (defaults to hostname)")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    # Phase 3 — inference
    p.add_argument("--model", default=None, help="HF model name for inference (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0)")
    p.add_argument("--layers", default=None, help="Layer range, e.g. '0:11' or '11:22'")
    p.add_argument("--inference-port", type=int, default=DEFAULT_INFERENCE_PORT, help=f"Inference server port (default: {DEFAULT_INFERENCE_PORT})")
    p.add_argument("--split-path", default=None, help="Path to a split model file from tools.split_model")
    p.add_argument(
        "--is-last",
        action="store_true",
        default=True,
        help=(
            "Mark this worker as the last stage in the pipeline (loads final norm "
            "and LM-head). Set --no-is-last for intermediate workers to save RAM."
        ),
    )
    p.add_argument("--no-is-last", dest="is_last", action="store_false")
    # Phase 4 — heterogeneous execution
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device: auto (GPU if available), cpu, or cuda (default: auto)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    return run(
        coordinator_url=args.coordinator,
        worker_id=args.worker_id,
        model_name=args.model,
        layer_spec=args.layers,
        inference_port=args.inference_port,
        split_path=args.split_path,
        is_last=args.is_last,
        device=args.device,
    )


if __name__ == "__main__":
    sys.exit(main())

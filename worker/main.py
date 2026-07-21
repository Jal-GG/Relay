from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from urllib.parse import urljoin

import httpx

from common.models import HeartbeatRequest, RegisterRequest, Task, TaskResultRequest
from common.protocol import HEARTBEAT_INTERVAL_S, ROUTES, TASK_POLL_INTERVAL_S
from worker.hardware import collect_hardware, local_hostname, local_ip
from worker.runner import execute_task

log = logging.getLogger("relay.worker")


def build_register_request(worker_id: str | None) -> RegisterRequest:
    hw = collect_hardware()
    return RegisterRequest(
        worker_id=worker_id or local_hostname(),
        hostname=local_hostname(),
        ip=local_ip(),
        hardware=hw,
    )


def make_heartbeat(worker_id: str) -> HeartbeatRequest:
    hw = collect_hardware()
    return HeartbeatRequest(worker_id=worker_id, hardware=hw)


def url(base: str, route: str) -> str:
    return urljoin(base if base.endswith("/") else base + "/", route.lstrip("/"))


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


def run(coordinator_url: str, worker_id: str | None) -> int:
    reg = build_register_request(worker_id)
    log.info("registering %s (%s) with %s", reg.worker_id, reg.ip, coordinator_url)

    with httpx.Client(timeout=10.0) as client:
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
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    return run(args.coordinator, args.worker_id)


if __name__ == "__main__":
    sys.exit(main())

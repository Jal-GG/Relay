from __future__ import annotations

import logging
import platform
import socket
from typing import Optional

import psutil

from common.models import GpuInfo, HardwareInfo

log = logging.getLogger("relay.worker.hardware")

_pynvml_ready = False
_pynvml_init_attempted = False


def _try_init_pynvml() -> bool:
    global _pynvml_ready, _pynvml_init_attempted
    if _pynvml_init_attempted:
        return _pynvml_ready
    _pynvml_init_attempted = True
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        _pynvml_ready = True
        log.info("pynvml initialized — NVIDIA GPU stats available")
    except Exception as exc:
        log.info("pynvml unavailable (%s) — running without GPU stats", exc)
        _pynvml_ready = False
    return _pynvml_ready


def detect_gpu() -> Optional[GpuInfo]:
    if not _try_init_pynvml():
        return None
    try:
        import pynvml  # type: ignore

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        total = pynvml.nvmlDeviceGetMemoryInfo(handle).total // (1024 * 1024)
        free = pynvml.nvmlDeviceGetMemoryInfo(handle).free // (1024 * 1024)
        return GpuInfo(name=name, vram_total_mb=int(total), vram_free_mb=int(free))
    except Exception as exc:
        log.warning("GPU detection failed: %s", exc)
        return None


def collect_hardware() -> HardwareInfo:
    vm = psutil.virtual_memory()
    return HardwareInfo(
        os=platform.platform(),
        cpu_cores=psutil.cpu_count(logical=True) or 0,
        ram_total_mb=int(vm.total // (1024 * 1024)),
        ram_free_mb=int(vm.available // (1024 * 1024)),
        cpu_percent=psutil.cpu_percent(interval=None),
        gpu=detect_gpu(),
    )


def local_hostname() -> str:
    return socket.gethostname()


def local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()

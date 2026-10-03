from __future__ import annotations

import logging
from typing import Any

from worker.backends.base import InferenceBackend

log = logging.getLogger("relay.worker.backends.cuda")


class CudaBackend(InferenceBackend):
    """CUDA inference backend — requires torch.cuda.is_available().

    Falls back to CPU with a warning if CUDA is not available, so the worker
    never crashes on a machine without a GPU.
    """

    def __init__(self) -> None:
        import torch

        if not torch.cuda.is_available():
            log.warning(
                "CUDA requested but not available — falling back to CPU. "
                "Install torch with CUDA support or use --device cpu."
            )
            self._device = "cpu"
            self._cuda = False
        else:
            self._device = "cuda"
            self._cuda = True
            gpu_name = torch.cuda.get_device_name(0)
            log.info("CudaBackend using %s", gpu_name)

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_cuda(self) -> bool:
        return self._cuda

    def move_module(self, module: Any) -> Any:
        return module.to(self._device)

    def move_tensor(self, tensor: Any) -> Any:
        return tensor.to(self._device)

    def to_cpu(self, tensor: Any) -> Any:
        return tensor.cpu()

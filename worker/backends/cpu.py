from __future__ import annotations

from typing import Any

from worker.backends.base import InferenceBackend


class CpuBackend(InferenceBackend):
    """CPU inference backend — always available, no special dependencies."""

    @property
    def device(self) -> str:
        return "cpu"

    def move_module(self, module: Any) -> Any:
        return module.to("cpu")

    def move_tensor(self, tensor: Any) -> Any:
        return tensor.to("cpu")

    def to_cpu(self, tensor: Any) -> Any:
        return tensor

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class InferenceBackend(ABC):
    """Device abstraction for inference forward passes.

    The coordinator doesn't know or care which backend a worker uses — it
    just sends/receives activations.  The backend handles:
    - Moving loaded modules to the target device
    - Moving input tensors to the target device before forward
    - Moving output tensors back to CPU before serialization
    """

    @property
    @abstractmethod
    def device(self) -> str:
        """Torch device string, e.g. 'cpu' or 'cuda'."""

    @abstractmethod
    def move_module(self, module: Any) -> Any:
        """Move a torch.nn.Module to this backend's device."""

    @abstractmethod
    def move_tensor(self, tensor: Any) -> Any:
        """Move a tensor to this backend's device."""

    @abstractmethod
    def to_cpu(self, tensor: Any) -> Any:
        """Move a tensor back to CPU for serialization."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(device={self.device!r})"

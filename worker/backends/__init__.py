from worker.backends.base import InferenceBackend
from worker.backends.cpu import CpuBackend
from worker.backends.cuda import CudaBackend


def create_backend(device: str = "auto") -> InferenceBackend:
    """Factory: pick the right backend by device string.

    'auto'  — CUDA if available, else CPU.
    'cpu'   — force CPU.
    'cuda'  — force CUDA (falls back to CPU with warning if unavailable).
    """
    if device == "cpu":
        return CpuBackend()
    if device == "cuda":
        return CudaBackend()
    # auto
    try:
        import torch
        if torch.cuda.is_available():
            return CudaBackend()
    except ImportError:
        pass
    return CpuBackend()


__all__ = ["InferenceBackend", "CpuBackend", "CudaBackend", "create_backend"]

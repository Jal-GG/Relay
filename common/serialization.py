from __future__ import annotations

import base64
import io
from typing import Any


def _torch():
    import torch
    return torch


def serialize_tensor(tensor: Any) -> str:
    """Serialize a tensor (or any picklable object) to a base64 string."""
    torch = _torch()
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def deserialize_tensor(b64: str) -> Any:
    """Deserialize a base64 string back into a tensor (or any picklable object).

    ``weights_only=False`` is explicit: activation tensors are coordinator-
    generated data we trust.  PyTorch ≥ 2.4 warns when this kwarg is omitted
    because the default is changing to True in a future release.
    """
    torch = _torch()
    raw = base64.b64decode(b64)
    return torch.load(io.BytesIO(raw), weights_only=False)

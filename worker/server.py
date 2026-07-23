from __future__ import annotations

import logging
import traceback

from fastapi import FastAPI, HTTPException

from common.models import ForwardRequest, ForwardResponse
from common.serialization import deserialize_tensor, serialize_tensor

log = logging.getLogger("relay.worker.server")


def create_inference_app(runner) -> FastAPI:
    """Build a tiny FastAPI app that exposes a /forward endpoint.

    *runner* is an ``InferenceRunner`` instance (duck-typed — only needs
    ``forward_from_tokens`` / ``forward_from_activations`` and the
    ``is_first`` / ``is_last`` flags).
    """
    app = FastAPI(title="Relay Worker Inference", version="0.4.0")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "is_first": runner.is_first,
            "is_last": runner.is_last,
            "layer_start": runner.start_layer,
            "layer_end": runner.end_layer,
            "num_layers": runner.num_layers,
            "device": runner.backend.device,
        }

    @app.post("/forward")
    def forward(req: ForwardRequest) -> ForwardResponse:
        try:
            return _forward(runner, req)
        except HTTPException:
            raise
        except Exception as exc:
            # Log the full traceback so operator can diagnose the issue
            # without having to check both the coordinator and worker logs.
            log.error(
                "forward pass failed:\n%s", traceback.format_exc()
            )
            raise HTTPException(
                status_code=500,
                detail=f"{type(exc).__name__}: {exc}",
            )

    return app


def _forward(runner, req: ForwardRequest) -> ForwardResponse:
    """Inner forward logic, separated so exceptions propagate cleanly."""
    if req.token_ids is not None:
        if not runner.is_first:
            raise HTTPException(
                status_code=400,
                detail="token_ids sent to a non-first worker",
            )
        if req.is_last:
            # Single-worker pipeline: embed → layers → norm → lm_head
            output = runner.forward_from_tokens_to_logits(req.token_ids)
            return ForwardResponse(logits_b64=serialize_tensor(output))
        output = runner.forward_from_tokens(req.token_ids)
        return ForwardResponse(activations_b64=serialize_tensor(output))

    if req.activations_b64 is not None:
        activations = deserialize_tensor(req.activations_b64)
        output = runner.forward_from_activations(activations, is_last=req.is_last)
        if req.is_last:
            return ForwardResponse(logits_b64=serialize_tensor(output))
        return ForwardResponse(activations_b64=serialize_tensor(output))

    raise HTTPException(
        status_code=400,
        detail="forward request must include token_ids or activations_b64",
    )

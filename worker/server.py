from __future__ import annotations

import logging

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
    app = FastAPI(title="Relay Worker Inference", version="0.3.0")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "is_first": runner.is_first,
            "is_last": runner.is_last,
            "layer_start": runner.start_layer,
            "layer_end": runner.end_layer,
            "num_layers": runner.num_layers,
        }

    @app.post("/forward")
    def forward(req: ForwardRequest) -> ForwardResponse:
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

    return app

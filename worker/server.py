from __future__ import annotations

import logging
import traceback

from fastapi import FastAPI, HTTPException

from common.models import ConfigureRequest, ForwardRequest, ForwardResponse
from common.serialization import deserialize_tensor, serialize_tensor

log = logging.getLogger("relay.worker.server")


def create_inference_app(runner=None) -> FastAPI:
    """Build a tiny FastAPI app that exposes inference and configuration endpoints."""
    app = FastAPI(title="Relay Worker Inference", version="0.6.0")
    current = {"runner": runner}

    @app.get("/health")
    def health() -> dict:
        active = current["runner"]
        if active is None:
            return {"status": "unconfigured", "ready": False}
        return {
            "status": "ok",
            "ready": True,
            "is_first": active.is_first,
            "is_last": active.is_last,
            "layer_start": active.start_layer,
            "layer_end": active.end_layer,
            "num_layers": active.num_layers,
            "device": active.backend.device,
        }

    @app.post("/configure")
    def configure(req: ConfigureRequest) -> dict:
        try:
            from worker.inference import InferenceRunner

            active = InferenceRunner(
                model_name=req.model_name,
                start_layer=req.layer_start,
                end_layer=req.layer_end,
                is_first=req.layer_start == 0,
                is_last=req.is_last,
                split_path=req.split_path,
                device=req.device,
            )
            current["runner"] = active
            return {
                "status": "ready",
                "layer_start": active.start_layer,
                "layer_end": active.end_layer,
                "device": active.backend.device,
            }
        except Exception as exc:
            log.error("worker configuration failed:\n%s", traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

    @app.post("/forward")
    def forward(req: ForwardRequest) -> ForwardResponse:
        active = current["runner"]
        if active is None:
            raise HTTPException(status_code=503, detail="worker is not configured")
        try:
            return _forward(active, req)
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

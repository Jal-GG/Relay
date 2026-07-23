from __future__ import annotations

import logging

import httpx

from common.models import (
    ForwardRequest,
    ForwardResponse,
    GenerateRequest,
    GenerateResponse,
    PipelineStage,
    WorkerStatus,
)
from common.protocol import DEFAULT_MAX_NEW_TOKENS, DEFAULT_INFER_TIMEOUT_S, ROUTES
from common.serialization import deserialize_tensor
from coordinator.state import ClusterState

log = logging.getLogger("relay.coordinator.pipeline")


class Pipeline:
    """Builds and executes a token-generation pipeline from registered workers.

    Workers that registered with ``layer_start`` / ``layer_end`` / ``model_name``
    are sorted by ``layer_start`` to form an ordered chain.  The first worker
    receives token IDs; each subsequent worker receives the hidden states from
    the previous one; the last worker returns logits.
    """

    def __init__(self, cluster_state: ClusterState) -> None:
        self._cluster = cluster_state

    def get_stages(self) -> list[WorkerStatus]:
        """Return inference-capable workers sorted by layer_start."""
        workers = self._cluster.list_workers()
        stages = [
            w for w in workers
            if w.has_inference and w.status == "online"
        ]
        stages.sort(key=lambda w: w.layer_start or 0)
        return stages

    def get_pipeline_view(self) -> list[PipelineStage]:
        stages = self.get_stages()
        return [
            PipelineStage(
                worker_id=s.worker_id,
                ip=s.ip,
                inference_port=s.inference_port or 0,
                layer_start=s.layer_start or 0,
                layer_end=s.layer_end or 0,
                model_name=s.model_name or "",
                status=s.status,
                device=s.device,
            )
            for s in stages
        ]

    def _validate_pipeline(self, stages: list[WorkerStatus]) -> None:
        if not stages:
            raise ValueError("no inference workers registered")
        if stages[0].layer_start != 0:
            raise ValueError(
                f"first worker starts at layer {stages[0].layer_start}, expected 0"
            )
        for i in range(1, len(stages)):
            prev_end = stages[i - 1].layer_end
            curr_start = stages[i].layer_start
            if prev_end != curr_start:
                raise ValueError(
                    f"gap or overlap: worker {stages[i-1].worker_id} ends at "
                    f"layer {prev_end}, worker {stages[i].worker_id} starts at "
                    f"layer {curr_start}"
                )

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        import time

        stages = self.get_stages()
        self._validate_pipeline(stages)  # fail fast before any heavy imports

        from transformers import AutoTokenizer
        model_name = stages[0].model_name or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        input_ids = tokenizer(req.prompt, return_tensors="pt")["input_ids"][0].tolist()
        input_len = len(input_ids)

        generated_ids: list[int] = []
        max_tokens = req.max_new_tokens or DEFAULT_MAX_NEW_TOKENS
        start = time.perf_counter()

        # Use a single persistent client for the entire generation loop so we
        # avoid creating a new TCP connection for every token × every worker.
        with httpx.Client(timeout=DEFAULT_INFER_TIMEOUT_S) as client:
            for _ in range(max_tokens):
                next_token = self._forward_one_token(stages, input_ids + generated_ids, client)
                if next_token == tokenizer.eos_token_id:
                    break
                generated_ids.append(next_token)

        elapsed_ms = (time.perf_counter() - start) * 1000
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return GenerateResponse(
            text=text,
            input_tokens=input_len,
            output_tokens=len(generated_ids),
            elapsed_ms=round(elapsed_ms, 1),
        )

    def _forward_one_token(
        self,
        stages: list[WorkerStatus],
        token_ids: list[int],
        client: httpx.Client,
    ) -> int:
        """Run a single forward pass through the pipeline and return the next token id."""
        torch = _torch()

        # First worker: token_ids → hidden states
        first = stages[0]
        req = ForwardRequest(token_ids=token_ids, is_last=len(stages) == 1)
        resp = self._post_forward(first, req, client)

        if len(stages) == 1:
            logits = deserialize_tensor(resp.logits_b64)
            return int(torch.argmax(logits[0, -1, :]).item())

        # Intermediate + last workers
        for i, stage in enumerate(stages[1:], start=1):
            is_last = i == len(stages) - 1
            req = ForwardRequest(
                activations_b64=resp.activations_b64,
                is_last=is_last,
            )
            resp = self._post_forward(stage, req, client)

        logits = deserialize_tensor(resp.logits_b64)
        return int(torch.argmax(logits[0, -1, :]).item())

    def _post_forward(
        self, worker: WorkerStatus, req: ForwardRequest, client: httpx.Client,
    ) -> ForwardResponse:
        url = worker.inference_url + ROUTES["forward"]
        r = client.post(url, json=req.model_dump())
        r.raise_for_status()
        return ForwardResponse(**r.json())


def _torch():
    import torch
    return torch

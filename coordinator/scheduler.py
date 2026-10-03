from __future__ import annotations

from dataclasses import dataclass

from common.models import LayerRange, ModelMetadata, WorkerStatus


class InsufficientResourcesError(ValueError):
    pass


def worker_memory_mb(worker: WorkerStatus) -> int:
    gpu = worker.hardware.gpu
    if gpu is not None and gpu.vram_free_mb > 0:
        return gpu.vram_free_mb
    return worker.hardware.ram_free_mb


def plan_layers(
    workers: list[WorkerStatus],
    model: ModelMetadata,
) -> dict[str, LayerRange]:
    eligible = [
        worker for worker in workers
        if worker.status == "online" and worker.inference_port is not None
    ]
    if not eligible:
        raise InsufficientResourcesError("no inference workers registered")

    capacities = {worker.worker_id: worker_memory_mb(worker) for worker in eligible}
    total_capacity = sum(capacities.values())
    if total_capacity < model.size_mb:
        raise InsufficientResourcesError(
            f"Insufficient cluster resources for this model: "
            f"requires {model.size_mb}MB, only {total_capacity}MB available"
        )

    usable = [worker for worker in eligible if capacities[worker.worker_id] > 0]
    if not usable:
        raise InsufficientResourcesError("Insufficient cluster resources for this model")

    usable.sort(
        key=lambda worker: (
            worker.hardware.gpu is None,
            -capacities[worker.worker_id],
            -worker.hardware.cpu_cores,
            -worker.network_bandwidth_mbps,
            worker.worker_id,
        )
    )
    if len(usable) > model.num_layers:
        usable = usable[:model.num_layers]

    total_weight = sum(capacities[worker.worker_id] for worker in usable)
    raw_counts = [
        model.num_layers * capacities[worker.worker_id] / total_weight
        for worker in usable
    ]
    counts = [max(1, int(count)) for count in raw_counts]

    while sum(counts) > model.num_layers:
        index = max(
            (i for i, count in enumerate(counts) if count > 1),
            key=lambda i: counts[i] - raw_counts[i],
        )
        counts[index] -= 1
    while sum(counts) < model.num_layers:
        index = max(
            range(len(counts)),
            key=lambda i: raw_counts[i] - counts[i],
        )
        counts[index] += 1

    plan: dict[str, LayerRange] = {}
    start = 0
    for worker, count in zip(usable, counts):
        plan[worker.worker_id] = LayerRange(start=start, end=start + count)
        start += count
    return plan


def estimate_model_metadata(model_name: str) -> ModelMetadata:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    num_layers = int(
        getattr(config, "num_hidden_layers", 0)
        or getattr(config, "n_layer", 0)
        or getattr(config, "num_layers", 0)
    )
    hidden_size = int(getattr(config, "hidden_size", 0) or getattr(config, "n_embd", 0))
    intermediate_size = int(
        getattr(config, "intermediate_size", 0) or hidden_size * 4
    )
    vocab_size = int(getattr(config, "vocab_size", 0))
    if num_layers <= 0 or hidden_size <= 0:
        raise ValueError(f"cannot determine layer metadata for model {model_name!r}")

    attention_params = 4 * hidden_size * hidden_size
    mlp_params = 3 * hidden_size * intermediate_size
    layer_norm_params = 4 * hidden_size
    embedding_params = vocab_size * hidden_size
    parameter_count = (
        num_layers * (attention_params + mlp_params + layer_norm_params)
        + embedding_params * 2
    )
    size_mb = max(1, int(parameter_count * 4 / (1024 * 1024)))
    return ModelMetadata(model_name=model_name, num_layers=num_layers, size_mb=size_mb)

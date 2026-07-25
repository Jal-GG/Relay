from common.models import GpuInfo, HardwareInfo, ModelMetadata, RegisterRequest
from coordinator.scheduler import plan_layers
from coordinator.state import ClusterState


def _worker(worker_id: str, ram: int, vram: int = 0):
    return RegisterRequest(
        worker_id=worker_id,
        hostname=worker_id,
        ip="127.0.0.1",
        hardware=HardwareInfo(
            os="test", cpu_cores=4, ram_total_mb=ram * 2, ram_free_mb=ram,
            gpu=GpuInfo(name="GPU", vram_total_mb=vram, vram_free_mb=vram) if vram else None,
        ),
        inference_port=9001,
        model_name="test-model",
    )


def test_plan_layers_is_contiguous_and_memory_weighted():
    state = ClusterState()
    state.register(_worker("gpu", 4_000, 12_000))
    state.register(_worker("laptop", 8_000))
    state.register(_worker("mini", 2_000))

    plan = plan_layers(
        state.list_workers(),
        ModelMetadata(model_name="test-model", num_layers=22, size_mb=10_000),
    )

    assert set(plan) == {"gpu", "laptop", "mini"}
    ranges = [plan[key] for key in ("gpu", "laptop", "mini")]
    assert sorted((item.start, item.end) for item in ranges) == [(0, 12), (12, 20), (20, 22)]
    assert sum(item.end - item.start for item in ranges) == 22


def test_plan_layers_rejects_insufficient_resources():
    import pytest

    state = ClusterState()
    state.register(_worker("small", 100))
    with pytest.raises(ValueError, match="Insufficient cluster resources"):
        plan_layers(
            state.list_workers(),
            ModelMetadata(model_name="test-model", num_layers=4, size_mb=1_000),
        )

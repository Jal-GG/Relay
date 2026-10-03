# Relay — Development Log

A chronological record of every phase: what was built, what broke, and how it was fixed.
Sourced from git history (`git log`, `git diff`) and code analysis.

---

## Phase 1 — Cluster Basics (Jul 19, 2026)

**Commit:** `4609894` — `(feat): Cluster coordinator and worker`
**Lines changed:** +643 across 13 files

### What was built

| File | Purpose |
|---|---|
| `common/models.py` | Pydantic models: `GpuInfo`, `HardwareInfo`, `RegisterRequest`, `HeartbeatRequest`, `WorkerStatus` |
| `common/protocol.py` | Route constants, heartbeat intervals, timeout values |
| `coordinator/main.py` | FastAPI app with `/register`, `/heartbeat`, `/workers`, `/` (dashboard) endpoints |
| `coordinator/state.py` | `ClusterState` — thread-safe in-memory worker registry with staleness detection |
| `coordinator/dashboard/index.html` | Single-page HTML+JS dashboard, polls every 3s, shows worker table |
| `worker/main.py` | Worker process: register → heartbeat loop → graceful shutdown |
| `worker/hardware.py` | Hardware detection: `psutil` for CPU/RAM, `pynvml` for NVIDIA GPU, IP/hostname helpers |
| `requirements.txt` | `fastapi`, `uvicorn`, `psutil`, `pynvml`, `pydantic`, `httpx`, `numpy` |

### Architecture decisions

- **In-memory state, no database** — MVP constraint. Restart = clean slate.
- **Pull-based polling for dashboard** — simpler than SSE/WebSockets, 3s refresh is fine.
- **Heartbeat every 5s, offline after 15s (3 missed beats)** — balances responsiveness with noise tolerance.
- **`pynvml` optional** — worker reports `gpu: null` if no NVIDIA driver, runs fine without.
- **Manual IP discovery** — workers take coordinator IP as CLI arg. No mDNS, no auto-discovery.

### Problems and solutions

#### Problem 1: Stale workers not marked offline in real-time

**Symptom:** Dashboard showed workers as "online" long after they disconnected, because staleness was only checked at read time.

**Root cause:** `ClusterState.list_workers()` checked staleness inline, but the check was only triggered when the dashboard polled. Between polls, the status was stale.

**Fix (commit `48a8918`):** Added a background sweeper thread (`_start_staleness_sweeper`) that runs every 1s and marks workers offline proactively. Also added `_mark_stale_if_needed` for inline checks on read paths, ensuring consistency even between sweep intervals.

```python
def sweep_stale(self) -> None:
    with self._lock:
        for status in self._workers.values():
            if status.status == "online" and status.is_stale(self._heartbeat_timeout_s):
                status.status = "offline"
```

#### Problem 2: Manual JSON serialization was verbose and error-prone

**Symptom:** `workers_to_dict()` was a 15-line function that manually serialized each field.

**Fix (commit `48a8918`):** Replaced with Pydantic's built-in `model_dump(mode="json")` — one line, handles all nested models, ISO dates, Optional fields automatically.

```python
# Before:
return JSONResponse(workers_to_dict(state.list_workers()))

# After:
return JSONResponse([w.model_dump(mode="json") for w in workers])
```

#### Problem 3: No CLI args for coordinator host/port

**Symptom:** Coordinator always bound to `0.0.0.0:8000` with no way to override.

**Fix (commit `48a8918`):** Added `argparse` with `--host` and `--port` flags.

---

## Phase 2 — Remote Task Execution (Jul 21, 2026)

**Commits:**
- `48a8918` — `worker fixes` (Jul 21, 21:36)
- `006ea7b` — `(feat): Task execution and worker protocol` (Jul 21, 23:04)

**Lines changed:** +845 across 14 files (net)

### What was built

| File | Purpose |
|---|---|
| `common/models.py` | Added `TaskType` (matrix_multiply, hash_file, sleep), `TaskStatus`, `TaskRequest`, `Task`, `TaskResultRequest` |
| `common/protocol.py` | Added task poll interval, default deadline, sweep interval, task routes |
| `coordinator/state.py` | Added `TaskState` — in-memory task store, poll/assign/submit_result/sweep |
| `coordinator/main.py` | Added `/tasks`, `/tasks/poll`, `/tasks/result` endpoints + task sweeper thread |
| `coordinator/dashboard/index.html` | Added tasks table with status badges, summary pills, polling |
| `worker/main.py` | Added task polling loop (daemon thread), task execution + result reporting |
| `worker/runner.py` | Task executors: `_matrix_multiply`, `_hash_file`, `_sleep` |
| `tests/test_api.py` | API integration tests via FastAPI TestClient |
| `tests/test_runner.py` | Unit tests for task execution (all 3 types + unknown type + interruptible sleep) |
| `tests/test_task_state.py` | Unit tests for task state (submit, poll, result, sweep, deadlock prevention) |

### Architecture decisions

- **Pull-based polling (not push)** — workers poll `/tasks/poll` every 1s. Simpler than WebSockets/gRPC, sufficient for MVP.
- **Round-robin scheduling** — first pending task assigned to whichever worker polls. No smart scheduling yet.
- **No retry logic** — failed tasks stay failed. Workers coming back online pick up new tasks only.
- **Deadline-based timeout** — each task has a `deadline_s` (default 60s). Background sweeper marks timed-out tasks.
- **Two-lock deadlock prevention** — `TaskState.sweep()` snapshots worker status under cluster lock, then updates tasks under task lock. Never holds both simultaneously.

### Problems and solutions

#### Problem 1: Deadlock between ClusterState and TaskState locks

**Symptom:** `sweep()` needed to check worker status (cluster's lock) and update task status (task's lock) simultaneously. Naive implementation could deadlock if another thread acquired the locks in reverse order.

**Fix:** Two-phase snapshot pattern:
1. Phase 1: Acquire cluster lock, snapshot offline worker IDs, release lock.
2. Phase 2: Acquire task lock, update tasks based on snapshot, release lock.

```python
def sweep(self) -> None:
    # Step 1: snapshot — only cluster._lock held here.
    offline_workers: set[str] = {
        w.worker_id for w in self._cluster.list_workers() if w.status == "offline"
    }
    # Step 2: update tasks — only self._lock held here.
    with self._lock:
        for task in self._tasks.values():
            if task.worker_id and task.worker_id in offline_workers:
                task.status = TaskStatus.failed
```

A dedicated test (`test_sweep_does_not_deadlock_with_concurrent_list_workers`) runs 50 concurrent sweep + list_workers iterations to verify no deadlock.

#### Problem 2: Late task results corrupting final state

**Symptom:** If a worker sent a result after the task was already marked failed (e.g., worker was offline, then came back), the late result would overwrite the failed status.

**Fix:** `submit_result` ignores results for tasks already in a terminal state (completed, failed, timed_out):

```python
if task.status in (TaskStatus.completed, TaskStatus.failed, TaskStatus.timed_out):
    return task  # Ignore late result
```

#### Problem 3: Sleep task not interruptible on shutdown

**Symptom:** When the worker received SIGINT, the task thread was blocked in `time.sleep()` and couldn't exit until the sleep completed.

**Fix:** Use `stop_event.wait(timeout=seconds)` instead of `time.sleep(seconds)`. When the stop event is set, the wait returns immediately, interrupting the sleep.

---

## Phase 3 — Distributed Model Loading (Jul 23, 2026)

**Commits:**
- `3ede18e` — `(feat): Distributed model inference workers` (Jul 23, 00:18)
- `d8ab739` — `(fix): Phase 3 enhancements` (Jul 23, 18:21)
- `f6a9c99` — `(enhancement): Inference support for distributed layer` (Jul 23, 19:36)

**Lines changed:** +1420 initial, then +115 fix, then +438 enhancement = ~1973 total

### What was built

| File | Purpose |
|---|---|
| `common/serialization.py` | `serialize_tensor` / `deserialize_tensor` — base64-encoded `torch.save`/`torch.load` |
| `common/models.py` | Added `GenerateRequest`, `GenerateResponse`, `ForwardRequest`, `ForwardResponse`, `PipelineStage`; inference fields on `RegisterRequest`/`WorkerStatus` |
| `common/protocol.py` | Added `generate`, `pipeline`, `forward` routes + inference constants |
| `tools/split_model.py` | Offline model splitter — architecture-agnostic key detection, saves per-worker shards + manifest |
| `tools/__init__.py` | Package init |
| `worker/inference.py` | `InferenceRunner` — loads layer range, runs forward passes (from tokens, from activations, full single-worker) |
| `worker/server.py` | FastAPI app with `/forward` and `/health` endpoints on the worker |
| `coordinator/pipeline.py` | `Pipeline` — builds worker chain, orchestrates token-by-token generation, greedy sampling |
| `coordinator/main.py` | Added `/generate` and `/pipeline` endpoints |
| `coordinator/dashboard/index.html` | Pipeline visualization + inference UI (prompt box, generate button, output) |
| `worker/main.py` | Added `--model`, `--layers`, `--inference-port`, `--split-path`, `--is-last` args |
| `tests/test_inference.py` | Unit tests: tensor round-trip, forward protocol, split vs full correctness, pipeline API |
| `tests/correctness.py` | Standalone script: full model vs split pipeline token-for-token comparison |

### Architecture decisions

- **PyTorch/transformers (not llama.cpp)** — direct access to model graph for layer splitting. Hackable for prototype.
- **CPU-only, greedy sampling, no KV cache** — stateless per-request. Each token is a full forward pass. Simple and correct.
- **Base64 torch.save for activation transfer** — simplest serialization. Over HTTP/JSON. Optimizable later.
- **Architecture-agnostic introspection** — `_get_inner_model`, `_get_decoder_layers`, `_get_embed`, `_get_norm`, `_get_lm_head` detect module structure at runtime for any HuggingFace CausalLM (Llama, Gemma, Mistral, Phi, GPT, etc.).
- **Single persistent HTTP client per generation** — avoids creating a new TCP connection per token × per worker.

### Problems and solutions

#### Problem 1: torch.load FutureWarning on weights_only

**Symptom:** PyTorch ≥ 2.4 emits a `FutureWarning` when `torch.load()` is called without `weights_only` parameter, because the default is changing to `True` in a future release. This warning appeared on every activation deserialization.

**Fix (commit `d8ab739`):** Explicitly pass `weights_only=False` in `deserialize_tensor`. Activation tensors are coordinator-generated data we trust. A test (`test_deserialize_tensor_weights_only`) verifies no FutureWarning is raised.

```python
# Before:
return torch.load(io.BytesIO(raw))

# After:
return torch.load(io.BytesIO(raw), weights_only=False)
```

#### Problem 2: Blocking the event loop during /generate

**Symptom:** The `/generate` endpoint was a synchronous function that ran the entire generation loop (which can take 30+ seconds for 50 tokens). This blocked FastAPI's event loop, preventing all other requests (heartbeats, dashboard polls, task polling) from being processed during generation.

**Fix (commit `d8ab739`):** Made `/generate` async and ran the blocking `pipeline.generate()` in a thread executor:

```python
@app.post(ROUTES["generate"])
async def generate(req: GenerateRequest) -> JSONResponse:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, pipeline.generate, req)
```

#### Problem 3: New TCP connection per forward pass

**Symptom:** `_post_forward` created a new `httpx.Client` for every single token's forward pass. For 50 tokens × 2 workers = 100 TCP connections. Connection overhead dominated latency.

**Fix (commit `d8ab739`):** One persistent `httpx.Client` for the entire generation loop:

```python
# Before:
def _post_forward(self, worker, req):
    with httpx.Client(timeout=120.0) as client:
        r = client.post(url, json=req.model_dump())

# After:
def generate(self, req):
    with httpx.Client(timeout=DEFAULT_INFER_TIMEOUT_S) as client:
        for _ in range(max_tokens):
            next_token = self._forward_one_token(stages, ..., client)
```

#### Problem 4: Split-path loading only worked for Llama architectures

**Symptom:** `_load_from_split` imported `LlamaDecoderLayer` and `LlamaRMSNorm` directly. Non-Llama models (Gemma, Phi, GPT) would crash on import.

**Fix (commit `d8ab739`, then fully rewritten in `f6a9c99`):**
- Initial fix (d8ab739): Added a guard to reject non-Llama models with a clear error message.
- Full fix (f6a9c99): Complete rewrite using `AutoModelForCausalLM.from_config()` to build a typed model shell with the correct module classes for any architecture, then load the split state dict with `strict=False`. Architecture-agnostic introspection functions (`_get_inner_model`, `_get_decoder_layers`, etc.) locate sub-modules at runtime.

```python
# f6a9c99 approach: build a shell with correct types, load weights with strict=False
shell = AutoModelForCausalLM.from_config(config)
shell.load_state_dict(sd, strict=False)
inner = _get_inner_model(shell)
self.layers = torch.nn.ModuleList(list(_get_decoder_layers(inner)[start:end]))
```

#### Problem 5: transformers 5.x API change — layer forward signature

**Symptom:** transformers 5.x changed decoder layer forward to return a plain tensor (not a tuple) and to require `position_embeddings=(cos, sin)` instead of computing RoPE internally from `position_ids`.

**Fix (commit `f6a9c99`):** `_layer_forward` helper handles both APIs:

```python
def _layer_forward(self, layer, hidden, pos):
    pe = self._position_embeddings(hidden, pos)  # None if no rotary_emb
    kwargs = {"position_ids": pos, "use_cache": False}
    if pe is not None:
        kwargs["position_embeddings"] = pe  # transformers 5.x
    out = layer(hidden, **kwargs)
    if isinstance(out, tuple):
        return out[0]  # transformers 4.x
    return out  # transformers 5.x
```

#### Problem 6: Attention backend mismatch corrupting output tokens

**Symptom:** When `_load_from_split` built a model with `from_config`, transformers auto-selected the best attention implementation (sdpa, flash_attention_2, or eager). If the split was created with a different backend than the worker used for loading, attention outputs differed numerically, compounding across layers and producing different tokens.

**Fix (commit `f6a9c99`):** `_detect_attn_implementation()` mirrors transformers' auto-selection logic so the split tool and the worker choose the same backend. The split model saves config which preserves the `_attn_implementation` attribute.

#### Problem 7: dtype mismatch between config.json and actual tensors

**Symptom:** TinyLlama's `config.json` declares `"torch_dtype": "bfloat16"` but the model was split as `float32`. When the worker loaded the split, it would create modules in `bfloat16` and then load `float32` weights, causing dtype mismatch errors.

**Fix (commit `f6a9c99`):** Detect dtype from the actual saved tensors, not from config.json:

```python
sd_dtype = next(iter(sd.values())).dtype  # actual dtype from file
config.torch_dtype = sd_dtype  # override config
torch.set_default_dtype(sd_dtype)  # ensure modules are born in this dtype
```

#### Problem 8: Forward pass errors returning 500 without useful diagnostics

**Symptom:** When a worker's forward pass crashed (e.g., wrong dtype, missing module), the 500 error returned a bare "Internal Server Error" with no traceback. Operators had to check worker logs separately from coordinator logs.

**Fix (commit `f6a9c99`):** Wrap forward in try/except, log the full traceback on the worker, and return the exception type + message in the HTTP response:

```python
except Exception as exc:
    log.error("forward pass failed:\n%s", traceback.format_exc())
    raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
```

#### Problem 9: Intermediate workers loading unnecessary norm + lm_head

**Symptom:** The initial implementation had `is_last = True` hardcoded for all workers. Every worker loaded the final norm and LM head weights (~2 × hidden_size² parameters), wasting RAM on intermediate workers that never use them.

**Fix (commit `d8ab739`):** Added `--is-last` / `--no-is-last` CLI flags. Intermediate workers use `--no-is-last` to skip loading norm + lm_head. The coordinator still sends `is_last=True` per-request to control when logits are produced.

---

## Phase 4 — Heterogeneous Execution (Jul 23, 2026)

**Status:** Implemented (uncommitted at time of writing)

**Lines changed:** ~500 across 8 new/modified files

### What was built

| File | Purpose |
|---|---|
| `worker/backends/__init__.py` | `create_backend(device)` factory: auto/cpu/cuda |
| `worker/backends/base.py` | `InferenceBackend` ABC — `device`, `move_module`, `move_tensor`, `to_cpu` |
| `worker/backends/cpu.py` | `CpuBackend` — no-op device placement (already on CPU) |
| `worker/backends/cuda.py` | `CudaBackend` — `.to("cuda")` for modules/tensors, `.cpu()` for output; graceful fallback to CPU if CUDA unavailable |
| `worker/inference.py` | Updated `InferenceRunner` — accepts `device` param, uses backend for all device placement, `_move_to_device()` after loading |
| `worker/main.py` | Added `--device` arg (auto/cpu/cuda), passes actual device to coordinator at registration |
| `worker/server.py` | `/health` endpoint now reports `device` field |
| `common/models.py` | Added `device` field to `RegisterRequest`, `WorkerStatus`, `PipelineStage` |
| `coordinator/state.py` | Passes `device` through registration |
| `coordinator/pipeline.py` | `PipelineStage` includes `device` |
| `coordinator/main.py` | Version bump to 0.4.0 |
| `coordinator/dashboard/index.html` | Pipeline view shows GPU/CPU badges per stage |
| `tools/benchmark.py` | Benchmark: single-CPU vs single-GPU vs split (GPU+CPU), prints table + saves JSON |
| `tests/test_inference.py` | Added 10 Phase 4 tests: backend selection, device placement, device param, model fields |

### Architecture decisions

- **Backend abstraction, not device strings scattered everywhere** — `InferenceBackend` interface with `CpuBackend`/`CudaBackend`. The `InferenceRunner` delegates all device operations to the backend. No `if device == "cuda"` scattered through forward code.
- **Auto-detection by default** — `--device auto` (default) checks `torch.cuda.is_available()`. Workers without CUDA silently use CPU. No crash, no manual config needed.
- **Graceful CUDA fallback** — `CudaBackend()` on a machine without CUDA logs a warning and falls back to CPU. The worker never crashes.
- **Coordinator is device-agnostic** — the pipeline doesn't know or care which backend a worker uses. It sends/receives activations. Device placement is internal to the worker.
- **Device reported at registration** — workers send their actual device string to the coordinator, which displays it in the dashboard pipeline view. The coordinator never uses the device for scheduling (that's Phase 5).

### Design: the InferenceBackend interface

```python
class InferenceBackend(ABC):
    @property
    def device(self) -> str: ...       # "cpu" or "cuda"
    def move_module(self, m) -> Any: ...  # .to(device) on a Module
    def move_tensor(self, t) -> Any: ...  # .to(device) on a Tensor
    def to_cpu(self, t) -> Any: ...       # .cpu() on a Tensor (for serialization)
```

The `InferenceRunner` uses the backend at three points:
1. **Load time** — `_move_to_device()` moves all loaded modules (embed, layers, norm, lm_head, rotary_emb) to the backend's device.
2. **Forward input** — `move_tensor()` moves incoming token IDs or activations to the device before the forward pass.
3. **Forward output** — `to_cpu()` moves output tensors back to CPU before base64 serialization, so the coordinator always receives CPU tensors.

### Problems and solutions

#### Problem 1: CUDA tensors can't be serialized via torch.save directly

**Symptom:** If a worker on GPU returned a CUDA tensor, `torch.save` would serialize it as a CUDA tensor. When the coordinator (which may not have CUDA) tried to `torch.load` it, it would fail or implicitly move to CPU with a warning.

**Fix:** `to_cpu()` is always called on output tensors before serialization. The coordinator always receives CPU tensors. The backend handles the CPU transfer transparently.

#### Problem 2: Position IDs tensor not on the right device

**Symptom:** `_position_ids()` creates a tensor on CPU. If the model is on CUDA, the layer forward would fail with a device mismatch error.

**Fix:** All forward methods now call `self.backend.move_tensor(pos)` after creating position IDs:

```python
pos = self._position_ids(seq_len)
pos = self.backend.move_tensor(pos)  # moves to CUDA if needed
```

#### Problem 3: Rotary embedding module not moved to device

**Symptom:** The `rotary_emb` module (used for RoPE in Llama models) was not moved to the device. When a CUDA worker tried to compute position embeddings, it would fail because the rotary embedding was on CPU while the hidden states were on CUDA.

**Fix:** `_move_to_device()` now includes rotary_emb:

```python
if self.rotary_emb is not None:
    self.backend.move_module(self.rotary_emb)
```

#### Problem 4: Benchmark needing both GPU and CPU runners simultaneously

**Symptom:** The benchmark tool needs to run the same model on both GPU and CPU in the same process. Loading the model twice would double memory usage.

**Fix:** The benchmark builds `InferenceRunner` instances that share the loaded model's sub-modules (no copy). Each runner's `_move_to_device()` moves only its own sub-modules to the target device. The GPU runner's modules are on CUDA, the CPU runner's are on CPU — they reference different objects.

---

## Summary: Lines of code by phase

| Phase | Commit | Files | Lines added | Key metric |
|---|---|---|---|---|
| 1 | `4609894` | 13 | +643 | 3 machines visible in dashboard |
| 1 fixes | `48a8918` | 5 | +56/-41 | Staleness sweeper, CLI args |
| 2 | `006ea7b` | 14 | +845 | 10 tasks distributed across workers |
| 3 | `3ede18e` | 16 | +1420 | Split model produces correct tokens |
| 3 fixes | `d8ab739` | 7 | +115 | async generate, persistent client, weights_only |
| 3 enhancement | `f6a9c99` | 5 | +438 | Architecture-agnostic, transformers 5.x compat |
| 4 | (this commit) | 14 | ~500 | GPU+CPU in same pipeline |
| **Total** | | | **~4000** | |

## Key lessons

1. **Test the split path early** — the split model loader was initially Llama-specific and broke for other architectures. Architecture-agnostic introspection should have been the first approach.
2. **Device placement must be centralized** — scattering `.to(device)` calls through the codebase is error-prone. A backend abstraction that owns all device operations is cleaner and makes the coordinator device-agnostic.
3. **Always return CPU tensors over the wire** — serialization of CUDA tensors causes subtle issues. Explicitly call `.cpu()` before serialization.
4. **Attention backend matters for correctness** — different attention implementations (eager vs sdpa vs flash) produce numerically different results. The split tool and worker must use the same backend.
5. **Async endpoints for long-running operations** — a 30-second generation loop blocks the event loop. Use `run_in_executor` for CPU-bound work in async frameworks.
6. **Two-phase locking prevents deadlocks** — when two state objects (cluster + tasks) need to be checked together, snapshot under one lock, then update under the other. Never hold both simultaneously.
7. **Graceful degradation over crashes** — CUDA backend falling back to CPU with a warning is better than crashing. Workers should never fail to start because of optional hardware.

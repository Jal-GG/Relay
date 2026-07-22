# Relay

Distributed inference MVP — heterogeneous consumer hardware collaboratively running a model too large for any single machine.

## Status — Distributed model loading

Coordinator accepts worker registration and heartbeats, dispatches tasks to workers, and tracks results. Workers poll for tasks, execute them, and report results back. Three task types: `matrix_multiply`, `hash_file`, `sleep`. Round-robin scheduling via pull-based polling. Error handling: worker crash → task marked `failed`; task deadline exceeded → task marked `timed_out`. No auth, no TLS, no persistent storage, no retry logic.

**Phase 3:** A model is split across two or more workers by layer range. The coordinator tokenizes a prompt, sends token IDs to the first worker, forwards hidden-state activations between workers, and samples greedily from the final worker's logits. Correctness is verified by comparing split-pipeline output to single-model output token-for-token.

### Layout

```
coordinator/   FastAPI app + in-memory cluster state + task state + dashboard + pipeline
worker/        Hardware detection + registration/heartbeat loop + task runner + inference
common/        Shared Pydantic models + protocol constants + tensor serialization
tools/         Offline model splitter (Phase 3)
tests/         pytest tests (runner, task state, API, inference, correctness)
```

### Setup

```bash
pip install -r requirements.txt
```

### Run — cluster basics + task execution

1. Start the coordinator (on the machine that other workers can reach):

```bash
python -m coordinator.main
# or: uvicorn coordinator.main:app --host 0.0.0.0 --port 8000
```

2. Open the dashboard: <http://localhost:8000/>

3. On each worker machine, start a worker pointing at the coordinator:

```bash
python -m worker.main http://<coordinator-ip>:8000
# override worker id:
python -m worker.main http://<coordinator-ip>:8000 --worker-id gaming-pc
# debug logging:
python -m worker.main http://<coordinator-ip>:8000 -v
```

4. Submit tasks (from any machine with curl):

```bash
# matrix multiply (100x100)
curl -X POST http://localhost:8000/tasks -H "Content-Type: application/json" \
  -d '{"type":"matrix_multiply","params":{"n":100}}'

# hash a byte payload (base64-encoded)
curl -X POST http://localhost:8000/tasks -H "Content-Type: application/json" \
  -d '{"type":"hash_file","params":{"payload":"aGVsbG8="}}'

# sleep for 2 seconds
curl -X POST http://localhost:8000/tasks -H "Content-Type: application/json" \
  -d '{"type":"sleep","params":{"seconds":2}}'

# with a custom 10s deadline
curl -X POST http://localhost:8000/tasks -H "Content-Type: application/json" \
  -d '{"type":"sleep","params":{"seconds":5},"deadline_s":10}'
```

### Run — distributed inference (Phase 3)

1. Start the coordinator:

```bash
python -m coordinator.main
```

2. On **Machine A** (layers 0–10):

```bash
python -m worker.main http://<coordinator-ip>:8000 \
  --worker-id worker-a \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --layers 0:11 \
  --inference-port 9001
```

3. On **Machine B** (layers 11–21):

```bash
python -m worker.main http://<coordinator-ip>:8000 \
  --worker-id worker-b \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --layers 11:22 \
  --inference-port 9001
```

4. Generate text via the coordinator:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_new_tokens":50}'
```

5. Or use the dashboard at <http://localhost:8000/> — the Pipeline section shows layer assignments, and the Inference section has a prompt box.

### Model splitter (optional)

Split a model into per-worker shard files so each worker loads only its layers:

```bash
python -m tools.split_model \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --splits 0:11,11:22 \
  --output ./model_splits
```

Workers can then load from a split file:

```bash
python -m worker.main http://<coordinator-ip>:8000 \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --layers 0:11 \
  --split-path ./model_splits/split_0.pt
```

### Correctness check

Verify that the split pipeline produces the same tokens as the full model:

```bash
python -m tests.correctness --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### Behavior

- Worker registers on startup, then POSTs a heartbeat every 5 seconds with refreshed CPU/RAM/GPU stats.
- Coordinator marks a worker `offline` after 15 seconds with no heartbeat (3 missed beats).
- Worker polls `/tasks/poll` every 1 second for new tasks. When a pending task is found, the coordinator assigns it (status → `running`) and the worker executes it.
- Worker reports the result back via `/tasks/result` (status → `completed` or `failed`).
- If the assigned worker goes offline mid-task, the coordinator's task sweeper marks the task `failed`.
- If a running task exceeds its `deadline_s` (default 60s), the sweeper marks it `timed_out`.
- Workers coming back online pick up new tasks but do NOT retry old failed ones.
- Dashboard polls `/workers`, `/tasks`, and `/pipeline` every 3 seconds.
- **Phase 3:** When inference workers register with `--model` and `--layers`, the coordinator builds a pipeline sorted by layer range. `POST /generate` tokenizes the prompt, sends token IDs to the first worker, forwards activations between workers, and samples greedily from the last worker's logits. No KV cache — each token is a full forward pass.

### API

| Method | Path             | Body                | Returns |
|--------|------------------|---------------------|---------|
| POST   | `/register`      | `RegisterRequest`   | `{"status":"registered","worker_id":...}` |
| POST   | `/heartbeat`     | `HeartbeatRequest`  | `{"status":"ok","worker_id":...}` |
| GET    | `/workers`       | —                   | list of worker status objects |
| POST   | `/tasks`         | `TaskRequest`       | `{"task_id":...,"status":"pending"}` |
| GET    | `/tasks/poll`    | `?worker_id=X`      | task object or `{"task_id":null}` |
| POST   | `/tasks/result`  | `TaskResultRequest` | `{"task_id":...,"status":"completed"}` |
| GET    | `/tasks`         | —                   | list of task objects |
| POST   | `/generate`      | `GenerateRequest`   | `GenerateResponse` |
| GET    | `/pipeline`      | —                   | list of `PipelineStage` objects |
| GET    | `/`              | —                   | dashboard HTML |

### Verifying

```bash
# Terminal 1 — coordinator
python -m coordinator.main

# Terminal 2 — worker A (layers 0-10)
python -m worker.main http://127.0.0.1:8000 --worker-id worker-a \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --layers 0:11 --inference-port 9001

# Terminal 3 — worker B (layers 11-21)
python -m worker.main http://127.0.0.1:8000 --worker-id worker-b \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --layers 11:22 --inference-port 9002

# Terminal 4 — generate
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_new_tokens":20}'

# Correctness check (single machine, split pipeline)
python -m tests.correctness
```

### Running tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

### Notes

- `pynvml` is required for NVIDIA GPU stats; if it can't initialize (no NVIDIA driver / no GPU), the worker reports `gpu: null` and runs fine.
- `numpy` is required for `matrix_multiply` tasks.
- `torch` and `transformers` are required for Phase 3 inference. Tests that need them are automatically skipped when not installed.
- All state lives in coordinator memory — restart the coordinator and the cluster is empty until workers re-register (which they won't, since they only register on startup. Stop and restart workers to repopulate).
- Inference is CPU-only, stateless per-request (no KV cache), and uses greedy/temperature=0 sampling.

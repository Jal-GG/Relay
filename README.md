# Relay

Distributed inference MVP — heterogeneous consumer hardware collaboratively running a model too large for any single machine.

## Status — Remote task execution

Coordinator accepts worker registration and heartbeats, dispatches tasks to workers, and tracks results. Workers poll for tasks, execute them, and report results back. Three task types: `matrix_multiply`, `hash_file`, `sleep`. Round-robin scheduling via pull-based polling. Error handling: worker crash → task marked `failed`; task deadline exceeded → task marked `timed_out`. No auth, no TLS, no persistent storage, no retry logic.

### Layout

```
coordinator/   FastAPI app + in-memory cluster state + task state + dashboard
worker/        Hardware detection + registration/heartbeat loop + task runner
common/        Shared Pydantic models + protocol constants
tests/         pytest tests (runner, task state, API)
```

### Setup

```bash
pip install -r requirements.txt
```

### Run

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

### Behavior

- Worker registers on startup, then POSTs a heartbeat every 5 seconds with refreshed CPU/RAM/GPU stats.
- Coordinator marks a worker `offline` after 15 seconds with no heartbeat (3 missed beats).
- Worker polls `/tasks/poll` every 1 second for new tasks. When a pending task is found, the coordinator assigns it (status → `running`) and the worker executes it.
- Worker reports the result back via `/tasks/result` (status → `completed` or `failed`).
- If the assigned worker goes offline mid-task, the coordinator's task sweeper marks the task `failed`.
- If a running task exceeds its `deadline_s` (default 60s), the sweeper marks it `timed_out`.
- Workers coming back online pick up new tasks but do NOT retry old failed ones.
- Dashboard polls `/workers` and `/tasks` every 3 seconds, rendering both tables.

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
| GET    | `/`              | —                   | dashboard HTML |

### Verifying

```bash
# Terminal 1 — coordinator
python -m coordinator.main

# Terminal 2 — worker
python -m worker.main http://127.0.0.1:8000 --worker-id laptop

# Terminal 3 — submit 10 matrix_multiply tasks
for i in $(seq 1 10); do
  curl -s -X POST http://localhost:8000/tasks \
    -H "Content-Type: application/json" \
    -d '{"type":"matrix_multiply","params":{"n":200}}'
  echo
done

# Open http://localhost:8000/ — tasks show as running then completed.
# Ctrl-C the worker mid-task — that task flips to failed within 15s.
```

### Running tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

### Notes

- `pynvml` is required for NVIDIA GPU stats; if it can't initialize (no NVIDIA driver / no GPU), the worker reports `gpu: null` and runs fine.
- `numpy` is required for `matrix_multiply` tasks.
- All state lives in coordinator memory — restart the coordinator and the cluster is empty until workers re-register (which they won't, since they only register on startup. Stop and restart workers to repopulate).

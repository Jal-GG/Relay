# Relay

Distributed inference MVP — heterogeneous consumer hardware collaboratively running a model too large for any single machine.

## Phase 1 status — Cluster basics

Coordinator accepts worker registration and heartbeats, serves a live dashboard. Workers collect local hardware info and report it every 5 seconds. No auth, no TLS, no persistent storage.

### Layout

```
coordinator/   FastAPI app + in-memory cluster state + dashboard
worker/        Hardware detection + registration/heartbeat loop
common/        Shared Pydantic models + protocol constants
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

### Behavior

- Worker registers on startup, then POSTs a heartbeat every 5 seconds with refreshed CPU/RAM/GPU stats.
- Coordinator marks a worker `offline` after 15 seconds with no heartbeat (3 missed beats).
- Dashboard polls `/workers` every 3 seconds and renders the table.

### API

| Method | Path           | Body                | Returns |
|--------|----------------|---------------------|---------|
| POST   | `/register`    | `RegisterRequest`   | `{"status":"registered","worker_id":...}` |
| POST   | `/heartbeat`   | `HeartbeatRequest`  | `{"status":"ok","worker_id":...}` |
| GET    | `/workers`     | —                   | list of worker status objects |
| GET    | `/`            | —                   | dashboard HTML |

### Verifying

```bash
# Terminal 1
python -m coordinator.main
# Terminal 2 (simulate a worker on the same machine)
python -m worker.main http://127.0.0.1:8000 --worker-id laptop
# Terminal 3
python -m worker.main http://127.0.0.1:8000 --worker-id mini-pc
```

Open <http://localhost:8000/> — both workers show as `online`. Ctrl-C one; within 15 seconds it flips to `offline`. Restart it and it re-registers as `online`.

### Notes

- `pynvml` is required for NVIDIA GPU stats; if it can't initialize (no NVIDIA driver / no GPU), the worker reports `gpu: null` and runs fine.
- All state lives in coordinator memory — restart the coordinator and the cluster is empty until workers re-register (which they won't, since they only register on startup. Stop and restart workers to repopulate).

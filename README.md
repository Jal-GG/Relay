<div align="center">
  <h1>🔗 Relay</h1>
  <p><b>Distributed LLM Inference Engine</b></p>
  <p>Run Large Language Models collaboratively across heterogeneous consumer hardware (CPU + GPU). Don't rent a $30,000 GPU—pool your existing hardware together.</p>
</div>

---

## 📖 Overview

Relay is a proof-of-concept distributed systems platform designed to run Large Language Models (LLMs) that are too large to fit into the memory (VRAM/RAM) of any single machine. By utilizing **Pipeline Parallelism**, Relay splits the layers of a model across multiple network-connected nodes. 

Instead of requiring expensive, uniform enterprise hardware, Relay dynamically schedules workloads across **heterogeneous nodes** (e.g., a gaming PC with a GPU and a standard laptop with a CPU) in the same pipeline.


## ✨ Key Features

- **Heterogeneous Hardware:** Mix and match CPUs and GPUs seamlessly. The system detects CUDA availability and routes layers appropriately.
- **Dynamic Layer Splitting:** Model state dicts are split by layer ranges across machines, ensuring no single node runs out of memory.
- **Fault-Tolerant Heartbeats:** Workers report live CPU/RAM/VRAM metrics every 5 seconds. If a worker goes offline, tasks fail gracefully and are marked for reallocation.
- **Backend Agnostic:** The coordinator orchestrates tensor serialization (activations) without needing to know if the underlying execution environment is PyTorch CUDA or CPU.

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
git clone https://github.com/your-username/relay.git
cd relay
pip install -r requirements.txt
```
*(Requires `torch`, `transformers`, `fastapi`, and `uvicorn`)*

### 2. Start the Coordinator
The Coordinator tracks cluster state and serves the API and Dashboard.
```bash
python -m coordinator.main
```
👉 Open the live cluster dashboard at: **[http://localhost:8000/](http://localhost:8000/)**

### 3. Attach Worker Nodes
Open new terminals (or run these on separate machines on the same network). Provide the Coordinator's IP address.

**Start Worker A (GPU) - Assigned to load the first half of the model:**
```bash
python -m worker.main http://localhost:8000 \
  --worker-id gpu-node-1 \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --layers 0:11 \
  --inference-port 9001
```

**Start Worker B (CPU) - Assigned to load the second half of the model:**
```bash
python -m worker.main http://localhost:8000 \
  --worker-id cpu-node-1 \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --layers 11:22 \
  --inference-port 9002
```

### 4. Run Distributed Inference!
Submit a prompt to the coordinator. It will route the request through the pipeline (Worker A -> Worker B) and return the response.

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_new_tokens":20}'
```

---

## 🛠️ Testing & Benchmarks

Verify that the distributed split-pipeline produces the exact same tokens as running the full model on a single machine:
```bash
python -m tests.correctness --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Measure tokens/second across single-CPU, single-GPU, and Split (GPU+CPU) configurations:
```bash
python -m tools.benchmark \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "The capital of France is" \
  --max-tokens 20 \
  --split-at 11
```

---

## 📂 Project Structure

- `coordinator/`: FastAPI server, in-memory cluster state, task sweeping, and the web dashboard.
- `worker/`: Hardware capability detection, registration logic, task polling, and model inference runners (`backends/`).
- `common/`: Shared protocols, Pydantic models, and wire-format tensor serialization.
- `tools/`: Offline model-splitting scripts and benchmark utilities.
# EfficientGraphBench — Implementation Guide

## 1. Implementation Strategy

The project should be built in layers.

Do **not** begin with:

- the web dashboard
- large datasets
- every model
- hardware recommendation logic

Start with one reproducible benchmark pipeline.

The first goal is:

> **Run GCN, GraphSAGE, GAT, and SGFormer/GraphGPS on one small dataset and save comparable metrics.**

---

## 2. Recommended Project Structure

```text
efficientgraphbench/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── datasets/
│   ├── models/
│   └── experiments/
│
├── src/
│   └── efficientgraphbench/
│       ├── __init__.py
│       │
│       ├── datasets/
│       │   ├── registry.py
│       │   ├── cora.py
│       │   ├── pubmed.py
│       │   └── ogbn_arxiv.py
│       │
│       ├── models/
│       │   ├── registry.py
│       │   ├── gcn.py
│       │   ├── graphsage.py
│       │   ├── gat.py
│       │   ├── sgformer.py
│       │   └── graphgps.py
│       │
│       ├── training/
│       │   ├── trainer.py
│       │   ├── evaluator.py
│       │   └── seed.py
│       │
│       ├── profiling/
│       │   ├── runtime.py
│       │   ├── gpu.py
│       │   ├── cpu.py
│       │   └── latency.py
│       │
│       ├── benchmark/
│       │   ├── runner.py
│       │   ├── experiment.py
│       │   └── scaling.py
│       │
│       ├── results/
│       │   ├── schema.py
│       │   ├── writer.py
│       │   └── reader.py
│       │
│       ├── recommend/
│       │   ├── constraints.py
│       │   └── selector.py
│       │
│       └── cli/
│           └── main.py
│
├── dashboard/
│   └── app.py
│
├── experiments/
│   ├── raw/
│   ├── processed/
│   └── reports/
│
└── tests/
```

---

## 3. Phase 0 — Environment Setup

### Create Environment

Use either:

```bash
python -m venv .venv
```

or:

```bash
conda create -n efficientgraphbench python=3.11
```

### Install Core Packages

Initial packages:

```bash
pip install torch
pip install torch-geometric
pip install ogb
pip install pandas numpy scikit-learn
pip install psutil pynvml
pip install pyyaml typer rich
pip install matplotlib
```

Optional later:

```bash
pip install streamlit
pip install mlflow
```

---

## 4. Phase 1 — Dataset Layer

Create a common dataset interface.

Each dataset loader should return:

```python
{
    "data": ...,
    "task_type": ...,
    "num_nodes": ...,
    "num_edges": ...,
    "num_features": ...,
    "num_classes": ...,
    "train_mask": ...,
    "val_mask": ...,
    "test_mask": ...,
}
```

### First Dataset

Start with:

- Cora

Then:

- PubMed

Then:

- ogbn-arxiv

### Why this order?

Cora:
- very small
- easy to debug
- fast training

PubMed:
- larger
- still manageable

ogbn-arxiv:
- more realistic scalability test

---

## 5. Phase 2 — Baseline Models

Implement or wrap models in this order:

1. GCN
2. GraphSAGE
3. GAT
4. SGFormer or GraphGPS

All models should expose a consistent interface.

Example:

```python
class GraphModel:
    def forward(self, data): ...
```

Store metadata for each model:

```python
MODEL_METADATA = {
    "name": "GraphSAGE",
    "family": "GNN",
    "attention": "none",
    "scalable": True,
}
```

---

## 6. Phase 3 — Common Trainer

Create one reusable trainer.

Responsibilities:

- move model/data to device
- train for fixed number of epochs
- compute validation score
- save best checkpoint
- evaluate test score
- record runtime
- record memory
- log configuration

Pseudo-flow:

```text
load config
↓
set random seed
↓
load dataset
↓
initialize model
↓
start profiler
↓
train
↓
stop profiler
↓
evaluate
↓
save result
```

---

## 7. Phase 4 — Reproducibility

Create a seed utility.

For each run set:

- Python seed
- NumPy seed
- PyTorch seed
- CUDA seed

Store:

- seed
- PyTorch version
- PyG version
- CUDA version
- GPU name

Each benchmark should ideally run on multiple seeds.

Recommended MVP:

- 3 seeds

Research-grade later:

- 5 or 10 seeds

---

## 8. Phase 5 — Profiling

### Training Time

Use:

```python
time.perf_counter()
```

Measure:

- total training time
- time per epoch

### GPU Memory

Use PyTorch:

```python
torch.cuda.reset_peak_memory_stats()
torch.cuda.max_memory_allocated()
```

Also optionally query:

- `nvidia-smi`
- `pynvml`

### CPU Memory

Use:

```python
psutil.Process().memory_info().rss
```

### Inference Latency

Use:

- warm-up runs
- multiple measured runs
- average or median latency

For GPU latency, synchronize CUDA before and after timing.

Example concept:

```python
torch.cuda.synchronize()
start = time.perf_counter()

# inference

torch.cuda.synchronize()
end = time.perf_counter()
```

---

## 9. Phase 6 — Result Schema

Every experiment should output one structured record.

Example:

```json
{
  "run_id": "2026-09-16-cora-gcn-seed42",
  "dataset": "cora",
  "model": "gcn",
  "seed": 42,
  "num_nodes": 2708,
  "num_edges": 5429,
  "accuracy": 0.82,
  "train_time_sec": 8.4,
  "inference_ms": 2.1,
  "peak_gpu_memory_mb": 1024,
  "peak_cpu_memory_mb": 650,
  "num_parameters": 120000,
  "device": "RTX GPU"
}
```

Write results to:

```text
experiments/raw/results.csv
experiments/raw/results.jsonl
```

---

## 10. Phase 7 — CLI

Use Typer.

Example commands:

```bash
egbench run --dataset cora --model gcn
```

```bash
egbench run --dataset pubmed --model graphsage --seed 42
```

```bash
egbench compare --dataset cora
```

```bash
egbench recommend --nodes 100000 --edges 500000 --gpu-memory 8
```

---

## 11. Phase 8 — Benchmark Configs

Use YAML configuration files.

Example:

```yaml
dataset: cora

model:
  name: graphsage
  hidden_dim: 128
  num_layers: 2

training:
  epochs: 200
  lr: 0.001
  weight_decay: 0.0005

hardware:
  device: cuda

reproducibility:
  seed: 42
```

This makes experiments repeatable.

---

## 12. Phase 9 — Initial Benchmark Matrix

Run:

| Dataset | GCN | GraphSAGE | GAT | SGFormer |
|---|---|---|---|---|
| Cora | Yes | Yes | Yes | Yes |
| PubMed | Yes | Yes | Yes | Yes |
| ogbn-arxiv | Later | Later | Later | Later |

For each run collect:

- task score
- train time
- inference latency
- peak memory
- parameter count

---

## 13. Phase 10 — Scaling Experiments

Use controlled graph sizes.

Possible approach:

- sample subgraphs
- use fixed node counts

Example:

```text
10K nodes
25K nodes
50K nodes
100K nodes
```

Record:

```text
node_count
edge_count
runtime
memory
accuracy
```

Then produce:

- graph size vs runtime
- graph size vs GPU memory
- graph size vs latency

---

## 14. Phase 11 — Sparsity Experiments

Take a graph and vary edge density.

Example:

```text
100% edges
75% edges
50% edges
25% edges
```

Then compare model behavior.

Important:

- edge removal strategy must be documented
- preserve train/validation/test protocol
- avoid accidentally disconnecting the graph in a misleading way

---

## 15. Phase 12 — OOM Tracking

Do not treat OOM as an error to hide.

Treat it as a benchmark result.

Example result:

```json
{
  "model": "graph_transformer",
  "num_nodes": 100000,
  "status": "oom",
  "gpu_memory_limit_gb": 8
}
```

This is valuable deployment information.

---

## 16. Phase 13 — Recommendation Engine

Start simple.

### Inputs

- node count
- edge count
- GPU memory
- CPU-only flag
- max latency
- min accuracy

### Step 1

Filter benchmark history for similar graph sizes.

### Step 2

Remove models violating constraints.

### Step 3

Rank feasible models.

Simple ranking:

```text
1. highest accuracy
2. lower memory
3. lower latency
```

### Example

Input:

```text
GPU memory <= 8 GB
latency <= 40 ms
accuracy >= 0.84
```

Output:

```text
Recommended: SGFormer

Alternatives:
1. GraphSAGE
2. GAT

Rejected:
Graph Transformer — exceeds memory budget
```

---

## 17. Phase 14 — Dashboard

Only build the dashboard after the package works.

Recommended MVP:

- Streamlit

Pages:

### Overview

- total runs
- supported models
- supported datasets

### Benchmark Explorer

Filters:

- dataset
- model
- graph size
- hardware

### Comparison

Charts:

- accuracy vs memory
- accuracy vs training time
- latency vs graph size
- memory vs graph size

### Recommendation

Inputs:

- graph size
- VRAM
- latency
- minimum accuracy

Output:

- recommended model
- alternatives
- reason

---

## 18. Phase 15 — Add GraphGPS

GraphGPS is useful because it allows combinations of:

- local message passing
- global attention
- positional/structural encodings

Possible experiments:

```text
GraphGPS + full Transformer
GraphGPS + Performer
```

Compare:

- accuracy
- memory
- runtime

---

## 19. Phase 16 — Add SGFormer

SGFormer should be treated as an efficient global-attention model.

Measure:

- scalability
- memory
- latency
- accuracy

Compare directly against:

- GCN
- GraphSAGE
- GAT
- GraphGPS

---

## 20. Phase 17 — Advanced Profiling

Later add:

- preprocessing time
- positional encoding computation
- energy estimate
- GPU utilization
- throughput

Preprocessing should be measured separately because some Graph Transformer methods have expensive positional encodings.

---

## 21. Testing Strategy

### Unit Tests

Test:

- dataset loaders
- metric functions
- profiler functions
- result serialization
- constraint filtering

### Integration Tests

Verify:

```text
dataset → model → trainer → profiler → result
```

### Reproducibility Tests

Same:

- model
- dataset
- seed
- config

should produce similar results.

---

## 22. Logging

Use Python logging.

Store:

- run start
- dataset metadata
- model config
- device
- epoch progress
- best validation score
- test score
- memory
- runtime
- errors

---

## 23. Error Handling

Handle:

- CUDA OOM
- missing dataset
- unsupported model
- invalid configuration
- CPU fallback
- corrupted experiment output

OOM should be saved as:

```text
status = OOM
```

rather than silently skipped.

---

## 24. Suggested Development Order

### Week 1

- Learn PyTorch Geometric basics
- Load Cora
- Implement GCN

### Week 2

- GraphSAGE
- GAT
- common trainer

### Week 3

- profiling
- result storage
- reproducibility

### Week 4

- SGFormer or GraphGPS integration
- first comparison table

### Week 5

- PubMed
- graph-size experiments

### Week 6

- hardware constraint filtering
- OOM tracking

### Week 7

- recommendation logic

### Week 8

- Streamlit dashboard
- plots
- documentation

---

## 25. MVP Completion Checklist

- [ ] Cora works
- [ ] PubMed works
- [ ] GCN works
- [ ] GraphSAGE works
- [ ] GAT works
- [ ] SGFormer or GraphGPS works
- [ ] Accuracy recorded
- [ ] Training time recorded
- [ ] Inference latency recorded
- [ ] Peak GPU memory recorded
- [ ] CPU memory recorded
- [ ] Results saved to CSV
- [ ] Results saved to JSON
- [ ] Reproducible config saved
- [ ] 3-seed experiment supported
- [ ] OOM captured as result
- [ ] Constraint-based recommendation works
- [ ] Basic dashboard works

---

## 26. First Concrete Milestone

Do not proceed to a dashboard until this table can be generated automatically:

| Model | Dataset | Accuracy | Train Time | Latency | GPU Memory |
|---|---|---:|---:|---:|---:|
| GCN | Cora | ... | ... | ... | ... |
| GraphSAGE | Cora | ... | ... | ... | ... |
| GAT | Cora | ... | ... | ... | ... |
| SGFormer | Cora | ... | ... | ... | ... |

Once this works reliably, the core of EfficientGraphBench exists.

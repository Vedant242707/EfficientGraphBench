# EfficientGraphBench — Tech Stack

## 1. Core Language

### Python 3.11

Use Python for:

- graph model implementation
- training
- profiling
- experiment management
- result analysis
- CLI
- dashboard backend

Reason:
- strongest ecosystem for PyTorch, PyTorch Geometric, OGB, and graph-learning research

---

## 2. Deep Learning Framework

### PyTorch

Use for:

- neural-network training
- GPU execution
- automatic differentiation
- CUDA memory profiling
- checkpointing

Recommended:

```text
PyTorch 2.x
```

---

## 3. Graph Learning Framework

### PyTorch Geometric (PyG)

Primary graph-learning library.

Use for:

- GCN
- GraphSAGE
- GAT
- graph datasets
- neighborhood sampling
- graph mini-batching
- utilities

Why PyG:

- widely used in graph ML research
- integrates directly with PyTorch
- strong dataset support
- easy to extend with custom architectures

---

## 4. Dataset Libraries

### PyTorch Geometric Datasets

For:

- Cora
- CiteSeer
- PubMed
- other standard graph datasets

### Open Graph Benchmark (OGB)

Use for larger-scale datasets such as:

- ogbn-arxiv
- ogbn-products
- other OGB tasks

Package:

```bash
pip install ogb
```

---

## 5. Models

### Baseline GNNs

Implement using PyG:

- GCN
- GraphSAGE
- GAT

### Efficient Graph Transformer

Initial:

- SGFormer

Alternative / additional:

- GraphGPS

Later:

- LightGCN
- Performer-based GraphGPS
- masked Graph Transformer
- kernelized Transformer
- RELGT

---

## 6. Experiment Configuration

### YAML

Use YAML files for:

- model settings
- dataset settings
- training parameters
- hardware settings

Library:

```bash
pip install pyyaml
```

Example:

```yaml
model:
  name: graphsage
  hidden_dim: 128

training:
  epochs: 200
  lr: 0.001
```

---

## 7. Profiling

### Python `time`

Use:

```python
time.perf_counter()
```

for:

- training runtime
- preprocessing runtime
- inference timing

### PyTorch CUDA Memory Tools

Use:

```python
torch.cuda.max_memory_allocated()
```

for:

- peak GPU memory

### NVIDIA Management Library

Package:

```bash
pip install pynvml
```

Use for:

- GPU name
- GPU memory
- GPU utilization
- optional power readings later

### psutil

Package:

```bash
pip install psutil
```

Use for:

- CPU memory
- process memory
- system information

---

## 8. Data Handling

### Pandas

Use for:

- experiment result tables
- CSV export
- comparison analysis

### NumPy

Use for:

- numerical operations
- aggregation
- experiment statistics

---

## 9. Evaluation

### scikit-learn

Use for:

- ROC-AUC
- F1
- precision
- recall
- confusion matrix
- other metrics where needed

---

## 10. Experiment Tracking

### MVP

Use:

- CSV
- JSONL
- YAML configs

This is sufficient initially.

### Later

Use one of:

- MLflow
- Weights & Biases

Recommendation:

- **MLflow** if you want local/open-source experiment tracking
- **Weights & Biases** if you want polished cloud experiment tracking

---

## 11. Storage

### MVP

Use:

```text
CSV
JSON
JSONL
```

### Later

Use:

```text
SQLite
```

if you need structured querying.

### Advanced

Use:

```text
PostgreSQL
```

only if the dashboard becomes multi-user or remote.

---

## 12. CLI

### Typer

Package:

```bash
pip install typer
```

Use for commands like:

```bash
egbench run --dataset cora --model gcn
```

### Rich

Package:

```bash
pip install rich
```

Use for:

- formatted terminal output
- tables
- progress
- status messages

---

## 13. Dashboard

### Recommended MVP: Streamlit

Use for:

- benchmark comparison
- charts
- filters
- model recommendation

Why Streamlit:

- very fast to build
- Python-only
- ideal for research dashboards

### Later Upgrade

If needed:

Frontend:
- React / Next.js

Backend:
- FastAPI

Only move to this if the project becomes a full software product.

---

## 14. Visualization

### Matplotlib

Use for:

- static research plots
- exported figures
- paper/report plots

### Streamlit Charts / Plotly

Use inside dashboard for:

- interactive comparison plots

Suggested charts:

- accuracy vs memory
- accuracy vs training time
- latency vs graph size
- memory vs graph size

---

## 15. Testing

### pytest

Use for:

- unit tests
- integration tests
- reproducibility tests

Package:

```bash
pip install pytest
```

---

## 16. Code Quality

Recommended tools:

### Ruff

Use for:
- linting
- formatting

### mypy

Optional:
- type checking

### pre-commit

Optional:
- automatic checks before commits

---

## 17. Packaging

### `pyproject.toml`

Package the project as:

```text
efficientgraphbench
```

Recommended build system:

```text
setuptools
```

or:

```text
hatchling
```

Expose CLI command:

```text
egbench
```

---

## 18. Version Control

### Git

Use Git for:

- code
- configs
- experiment scripts
- documentation

### GitHub

Repository should include:

```text
README.md
PRD.md
IMPLEMENTATION_GUIDE.md
TECHSTACK.md
LICENSE
requirements.txt
pyproject.toml
configs/
src/
tests/
```

---

## 19. Recommended Hardware

### Minimum Development

- 16 GB RAM preferred
- CPU sufficient for Cora and small tests

### Recommended for Experiments

- NVIDIA GPU
- 8–12 GB VRAM minimum for meaningful graph experiments

### Better Research Setup

- 16–24 GB VRAM

Large-scale datasets may require:

- cloud GPU
- lab GPU
- Google Colab
- Kaggle
- university compute server

---

## 20. Cloud Options

If local hardware is limited:

- Google Colab
- Kaggle notebooks
- university lab GPU
- cloud GPU provider

For repeatable benchmarking, prefer a fixed machine once the final experiments begin.

---

## 21. Proposed Stack Summary

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Deep Learning | PyTorch |
| Graph ML | PyTorch Geometric |
| Large Graph Datasets | OGB |
| Data Analysis | Pandas, NumPy |
| Metrics | scikit-learn |
| GPU Profiling | PyTorch CUDA APIs, pynvml |
| CPU Profiling | psutil |
| Configs | YAML |
| CLI | Typer + Rich |
| Storage | CSV + JSONL |
| Experiment Tracking | MLflow later |
| Dashboard | Streamlit |
| Visualization | Matplotlib / Plotly |
| Testing | pytest |
| Packaging | pyproject.toml |
| Version Control | Git + GitHub |

---

## 22. Install Set for MVP

```bash
pip install torch
pip install torch-geometric
pip install ogb
pip install pandas numpy scikit-learn
pip install psutil pynvml
pip install pyyaml
pip install typer rich
pip install matplotlib
pip install pytest
pip install streamlit
```

---

## 23. Do Not Add Initially

Avoid these in the MVP unless required:

- Docker
- Redis
- Celery
- MongoDB
- Kubernetes
- React
- Next.js
- FastAPI
- PostgreSQL
- distributed training

They add engineering complexity without improving the initial research result.

---

## 24. Best Initial Stack

For the first working version, keep it extremely simple:

```text
Python
+
PyTorch
+
PyTorch Geometric
+
Pandas
+
psutil / CUDA profiler
+
CSV
+
Typer
```

Then add:

```text
Streamlit
```

only after the benchmark pipeline works.

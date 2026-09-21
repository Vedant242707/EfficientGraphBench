# EfficientGraphBench — Product Requirements Document (PRD)

## 1. Product Overview

**EfficientGraphBench** is a hardware-aware benchmarking and model-selection framework for graph learning.

The system compares Graph Neural Networks (GNNs) and Graph Transformers under practical constraints such as:

- Graph size
- Number of edges
- GPU memory
- CPU memory
- Training time
- Inference latency
- Accuracy
- Preprocessing cost

The goal is not only to identify the most accurate model, but to determine:

> **Which graph-learning model gives the best deployable accuracy–efficiency trade-off for a given graph and hardware budget?**

---

## 2. Problem Statement

Graph Transformers can capture long-range dependencies and global graph structure better than traditional local message-passing GNNs in some settings.

However, they can also be significantly more expensive in:

- GPU memory
- Runtime
- Preprocessing
- Attention computation
- Hardware requirements

A model with the best accuracy may not be the most practical model.

For example:

- Model A: 87% accuracy, 11 GB GPU memory
- Model B: 85.5% accuracy, 4 GB GPU memory

If the available GPU has only 8 GB VRAM, Model A cannot be deployed.

EfficientGraphBench aims to make these trade-offs measurable and actionable.

---

## 3. Target Users

### Primary Users

- ML engineers working with graph data
- Graph ML researchers
- Students experimenting with GNNs and Graph Transformers
- Recommendation-system engineers
- Fraud-detection teams
- Knowledge-graph teams
- Data scientists evaluating graph-learning architectures

### Secondary Users

- Research labs
- University courses and student projects
- Small teams with limited GPU resources
- Engineers deploying graph models on CPU-only systems

---

## 4. Core User Need

Users need a way to answer:

- Which graph model should I use for my dataset?
- Is a Graph Transformer worth its additional cost?
- Can this model fit on my GPU?
- How does performance change as graph size grows?
- Which model gives the best accuracy under a memory or latency limit?
- Which models fail because of out-of-memory constraints?
- How expensive is preprocessing such as positional encoding?

---

## 5. Research Objective

The primary research question is:

> **How does the accuracy–efficiency trade-off of graph-learning models change with graph size, sparsity, architecture type, and hardware constraints?**

Secondary questions include:

1. When do Graph Transformers justify their extra compute cost over GNNs?
2. How quickly do training time and memory grow as the number of nodes increases?
3. Which architectures fail first under constrained VRAM?
4. Are efficient Graph Transformers such as SGFormer or GraphGPS-style linear-attention variants more practical than full-attention Graph Transformers?
5. How significant is preprocessing cost relative to model training?
6. Can benchmark results be used to automatically recommend a model for a given hardware budget?

---

## 6. Research Basis

EfficientGraphBench is motivated by several recent works:

- **OpenGT: A Comprehensive Benchmark for Graph Transformers**
  - Demonstrates the need for standardized, reproducible comparisons across Graph Transformer designs.
  - Studies task type, graph structure, positional encodings, and efficiency.

- **GraphGPS**
  - Combines local message passing with global attention.
  - Supports scalable attention variants such as Performer and BigBird.
  - Highlights the trade-off between full attention and scalable attention.

- **SGFormer**
  - Shows that a simplified Graph Transformer can scale efficiently to very large graphs.
  - Demonstrates strong runtime and memory improvements over heavier Transformer baselines.

- **MIT Thesis: Benchmarking Graph Transformers Toward Scalability for Large Graphs**
  - Directly compares accuracy, runtime, and GPU memory.
  - Shows that a more complex Graph Transformer does not automatically outperform simpler GNN baselines.

- **Graph Transformers: A Survey**
  - Identifies scalability, memory cost, attention complexity, generalization, and efficiency as major open challenges.

- **Relational Graph Transformer**
  - Shows that preprocessing and positional encoding costs can be significant.
  - Motivates tracking preprocessing overhead separately from training cost.

---

## 7. Product Scope

### Phase 1 — Minimum Viable Benchmark

Supported models:

- GCN
- GraphSAGE
- GAT
- SGFormer or GraphGPS

Supported datasets:

- Cora
- PubMed
- ogbn-arxiv or another medium OGB dataset

Metrics:

- Accuracy / task metric
- Training time
- Inference latency
- Peak GPU memory
- CPU RAM usage
- Parameter count
- Node count
- Edge count

Output:

- CSV/JSON benchmark results
- CLI summary table

### Phase 2 — Scaling Experiments

Add:

- Controlled graph-size experiments
- Memory growth analysis
- Runtime growth analysis
- Out-of-memory threshold tracking
- Sparsity/density experiments

### Phase 3 — Hardware-Aware Model Selection

Inputs:

- Node count
- Edge count
- Available GPU memory
- CPU-only or GPU deployment
- Maximum inference latency
- Minimum acceptable accuracy

Output:

- Feasible models
- Best-performing model under constraints
- Alternative models
- Reason for recommendation

### Phase 4 — Dashboard

Add:

- Model comparison
- Dataset comparison
- Accuracy vs memory plots
- Accuracy vs training time
- Graph size vs memory
- Graph size vs latency
- OOM boundary visualization
- Model recommendation panel

### Phase 5 — Advanced Models

Optional models:

- GraphGPS
- LightGCN
- Masked Graph Transformer
- Kernelized Graph Transformer
- RELGT
- Other OpenGT-supported architectures

---

## 8. Functional Requirements

### FR-1 Dataset Loading

The system shall:

- Load supported graph datasets
- Extract:
  - node count
  - edge count
  - feature count
  - task type
  - class count
  - graph density

### FR-2 Model Registry

The system shall maintain a model registry containing:

- model name
- architecture family
- parameter count
- attention type
- supported task type
- training function
- inference function

### FR-3 Benchmark Runner

The benchmark runner shall:

- Initialize a selected model
- Train using a reproducible configuration
- Evaluate on a fixed split
- Record runtime
- Record memory usage
- Record inference latency
- Save results

### FR-4 Profiler

The profiler shall capture:

- total training time
- per-epoch training time
- validation time
- inference latency
- peak GPU memory
- peak CPU memory
- preprocessing time

### FR-5 Reproducibility

Every run shall store:

- random seed
- dataset split
- model configuration
- optimizer
- learning rate
- number of epochs
- batch size
- hidden dimension
- hardware info
- library versions

### FR-6 Result Storage

Results shall be stored in:

- CSV for easy analysis
- JSON for structured access
- Optional SQLite/PostgreSQL in later versions

### FR-7 Hardware Constraint Filtering

The system shall allow filtering based on:

- maximum GPU memory
- maximum CPU memory
- maximum inference latency
- maximum training time
- minimum accuracy

### FR-8 Recommendation Engine

The recommendation engine shall:

1. Remove models violating hard constraints
2. Rank feasible models
3. Return:
   - recommended model
   - alternatives
   - metric comparison
   - explanation

---

## 9. Non-Functional Requirements

### Reproducibility

- Fixed seeds
- Versioned configurations
- Saved experiment metadata
- Repeatable command-line runs

### Extensibility

New models should be addable without rewriting the benchmark engine.

### Fairness of Comparison

Models should use:

- same dataset split
- same hardware
- transparent hyperparameter tuning
- comparable training budgets where possible

### Usability

A researcher should be able to benchmark a model using one command.

Example:

```bash
egbench run --dataset cora --model graphsage
```

---

## 10. Benchmark Metrics

### Predictive Metrics

Depending on task:

- Accuracy
- ROC-AUC
- F1
- Average Precision
- MAE

### Efficiency Metrics

- Training time
- Training time per epoch
- Inference latency
- Peak GPU memory
- Peak CPU memory
- Parameter count
- Preprocessing time

### Scalability Metrics

- Runtime growth with node count
- Memory growth with node count
- Maximum graph size before OOM
- Performance under different sparsity levels

---

## 11. Initial Model Set

### GCN

Purpose:
- Simple message-passing baseline

### GraphSAGE

Purpose:
- Scalable neighborhood-sampling baseline

### GAT

Purpose:
- Attention-based GNN baseline

### SGFormer

Purpose:
- Efficient Graph Transformer baseline with scalable global attention

### GraphGPS

Purpose:
- Hybrid local-message-passing + global-attention baseline

---

## 12. Initial Datasets

### Cora

Use:
- Small-scale debugging
- Fast iteration

### PubMed

Use:
- Medium-scale experiment

### ogbn-arxiv

Use:
- Larger benchmark
- More meaningful scalability experiment

### Future Datasets

- Amazon2M
- OGB large-scale datasets
- Long Range Graph Benchmark datasets
- RelBench datasets

---

## 13. Example User Flow

1. User selects dataset
2. User selects models
3. User specifies resource limits
4. Benchmark runner trains selected models
5. Profiler records metrics
6. Results are saved
7. Dashboard displays comparisons
8. Recommendation engine suggests suitable model

---

## 14. Example Recommendation

Input:

```text
Dataset: ogbn-arxiv
GPU Memory: 8 GB
Maximum latency: 40 ms
Minimum accuracy: 72%
```

Possible output:

```text
Recommended: SGFormer

Reason:
- Meets accuracy threshold
- Fits within memory budget
- Lower latency than full Graph Transformer

Alternative:
GraphSAGE
- Lower memory
- Slightly lower accuracy
```

---

## 15. Success Criteria

The MVP is successful if it can:

1. Benchmark at least 4 models
2. Run on at least 2 datasets
3. Record accuracy, runtime, latency, and memory
4. Reproduce results across multiple seeds
5. Generate a consistent comparison table
6. Filter models using hardware constraints
7. Recommend at least one feasible model from benchmark history

---

## 16. Out of Scope for MVP

Do not include initially:

- Full arbitrary graph upload
- Energy measurement
- Mobile deployment
- Distributed training
- Dynamic graphs
- Full relational database support
- Dozens of Transformer variants
- Large production frontend
- Automatic hyperparameter search

---

## 17. Key Risks

### Risk: Project becomes too broad

Mitigation:
- Start with 4 models and 2 datasets

### Risk: Fair comparison becomes difficult

Mitigation:
- Store every configuration
- Use consistent splits and hardware

### Risk: Large models cause OOM

Mitigation:
- Start with small datasets
- Treat OOM as a benchmark result

### Risk: Existing benchmarks overlap with the idea

Mitigation:
- Differentiate through hardware-aware profiling and deployment-oriented model recommendation

---

## 18. Final Deliverables

### Required

- Python benchmarking package
- Model registry
- Dataset registry
- Profiler
- Experiment runner
- CSV/JSON output
- Reproducible configs
- Benchmark report

### Recommended

- Streamlit dashboard
- Constraint-based recommendation engine
- Scaling plots
- Research report

### Advanced

- Upload-your-own-graph support
- More Graph Transformer architectures
- Hardware prediction model
- Energy profiling

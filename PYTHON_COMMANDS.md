# EfficientGraphBench — notebook guide

**Already installed? Start at Step 1. You do not need to run every example in this file.**

Choose the notebook kernel **EfficientGraphBench (CUDA GPU)**.

## 1. Import the library

Run this once after opening or restarting your notebook:

```python
from efficientgraphbench import Benchmark
```

## 2. Run your test

This tests all primary models on Cora using your GPU:

```python
results = Benchmark(
    dataset="cora",
    models="all",
    device="cuda",
    epochs=200,
    seeds=[1, 2, 3],
).run()
```

| Setting | What it means | What you can change |
|---|---|---|
| `dataset` | Data to test | For example, `"cora"` or `"pubmed"` |
| `models` | Models to compare | `"all"` or a list such as `["gcn", "gat"]` |
| `device` | Where training runs | `"cuda"` for GPU; `"auto"` to allow CPU fallback |
| `epochs` | Maximum training rounds per run | `200` for a benchmark; `2` for a quick check |
| `seeds` | Repeated runs with different random starts | `[1, 2, 3]` runs each model three times; `[42]` runs once |

`"all"` includes the primary models, not experimental variants. Official Graphormer will show `UNSUPPORTED_TASK` for this task; other models continue. GraphGPS and SGFormer require their separate environments to be installed.

## 3. See the results

Run this after Step 2. `display` renders a table in Jupyter:

```python
from IPython.display import display

display(results.summary()[["model", "runs", "accuracy_mean", "status"]])
```

- `runs`: number of successful runs.
- `accuracy_mean`: average accuracy; `0.81` means **81%**.
- `status`: `SUCCESS`, a failure reason category, or skipped runs.

**That is the basic workflow: import → run → view.** The examples below are optional.

---

## Choose another task

| I want to… | Go to |
|---|---|
| Test only selected models | [Selected models](#selected-models) |
| Test one model | [One model](#one-model) |
| Test my CSV files | [My CSV files](#my-csv-files) |
| Increase graph size until OOM | [Size test](#size-test) |
| See speed and memory | [Speed and memory](#speed-and-memory) |
| Save my results | [Save results](#save-results) |
| Understand a failed run | [Check a failure](#check-a-failure) |

## Selected models

Use this **instead of Step 2**, then run Step 3:

```python
results = Benchmark(
    dataset="cora",
    models=["gcn", "graphsage", "sgformer", "graphgps"],
    device="cuda",
    epochs=200,
    seeds=[1, 2, 3],
).run()
```

## One model

```python
result = Benchmark.run_model(
    model="gcn",
    dataset="cora",
    device="cuda",
    epochs=200,
)

print("Status:", result.status)
print("Accuracy:", result.accuracy)
```

`result` is one run. `results` in the other examples contains multiple runs.

## My CSV files

Your prepared folder should contain `nodes.csv` and `edges.csv`:

```text
data/
    Mydata/
        nodes.csv
        edges.csv
```

```python
results = Benchmark(
    dataset="custom",
    dataset_path="data/Mydata",
    models="all",
    device="cuda",
    epochs=200,
    seeds=[1, 2, 3],
).run()
```

Then run Step 3. These must be prepared graph tables: nodes need `node_id`, `feature_*`, `label`, `split`; edges need `source`, `target`. An arbitrary spreadsheet needs preparation first.

Paths are relative to the notebook's working directory. To see that directory:

```python
from pathlib import Path
print(Path.cwd())
```

If necessary, set `dataset_path` to your folder's full path.

## Size test

Tests 2,000 → 4,000 → 6,000 nodes and so on, up to the full dataset:

```python
from efficientgraphbench import ScalingBenchmark

results = ScalingBenchmark(
    dataset="pubmed",
    models="all",
    step=2000,
    device="cuda",
    epochs=200,
    seeds=[1, 2, 3],
).run()

display(results.to_dataframe()[["model", "num_nodes", "accuracy", "status", "oom_nodes"]])
```

Shows **one final row per model**. After OOM, retains its last successful result and shows where OOM occurred. Remaining seeds and larger sizes for that model are skipped. Each size trains a fresh model on a subgraph; this is not mini-batch training.

## Speed and memory

After a comparison, run:

```python
display(results.summary()[[
    "model",
    "training_time_sec",
    "inference_latency_median_ms",
    "peak_gpu_allocated_mib",
]])
```

Columns show average training seconds, average per-run median inference milliseconds, and average peak allocated GPU memory in MiB.

## Save results

Choose the format you need; you do not need to run all three:

```python
results.save_csv("my_results.csv")        # Open as a spreadsheet
results.save_json("my_results.json")      # Complete records and settings
results.save_markdown("my_report.md")     # Readable report
```

Normal runs also save records, logs and checkpoints under `experiments/` by default. To choose a different folder, add `output_dir="experiments/my_test"` inside `Benchmark(...)`.

## Check a failure

For a comparison:

```python
for result in results:
    if result.status != "SUCCESS":
        print(result.model, result.status, result.failure_reason)
```

For a single-model run, use `print(result.failure_reason)`.

| Message | What to do |
|---|---|
| CUDA unavailable | Select the CUDA kernel, or use `device="auto"` |
| Environment not installed | Set up that reference model; see below |
| OOM | Use the size test to find a smaller working graph |
| Unsupported task | That implementation cannot run the current task |
| Skipped after a failure | Read the first failure; the repeat was not executed |

---

## Installation — only when needed

**Your current laptop is already set up. Skip this section there.**

For a new notebook environment, install the wheel in a notebook cell:

```python
%pip install "path/to/efficientgraphbench-0.1.0-py3-none-any.whl"
```

Restart the kernel afterward. For reference models, install Git on the computer, then run these commands in a **terminal**, not as Python code:

```bash
python -m pip install uv
egbench setup sgformer
egbench setup graphgps
```

Reference setup currently supports Windows x86-64.

## Full reference

Need advanced settings, provenance, JSONL exports, or terminal commands? Open [PYTHON_API_REFERENCE.md](PYTHON_API_REFERENCE.md). It contains the complete reference; this guide covers everyday notebook use.
